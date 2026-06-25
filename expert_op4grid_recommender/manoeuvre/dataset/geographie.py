"""
manoeuvre/dataset/geographie.py — Localisation géographique des postes.

Le dataset RTE 7000 (instantanés XIIDM) **ne porte pas** de coordonnées
géographiques (vérifié : pas d'extension ``substationPosition``, pas de
lat/lon). L'IHM « explorer la journée » a besoin de placer chaque poste sur une
carte ; on résout les coordonnées par une **chaîne de sources**, du plus fiable
au plus dépendant du réseau :

1. **embarquées** dans l'instantané (extension pypowsybl ``substationPosition``)
   — *absent du dataset RTE 7000 actuel*, mais gratuit et exact si un jour
   présent ;
2. **instantané committé** ``data/postes_rte_geo.json`` — **indexé par
   ``substation_id``** (clé pypowsybl), donc résolution runtime = simple lookup.
   Produit hors-ligne par ``scripts/fetch_postes_geo.py`` (qui réalise
   l'appariement ODRE ↔ postes **une fois**, là où ODRE est joignable, et le
   valide) ;
3. **ODRE en direct** (``odre.opendatasoft.com``, dataset
   ``postes-electriques-rte``) — repli runtime si le Space autorise l'accès
   sortant ; mis en cache localement.

L'appariement ODRE → ``substation_id`` est **best-effort** (normalisation des
mnémoniques RTE — points/espaces de tête, casse) : il est calculé et **mesuré**
par le script de fetch, pas à chaud. Si aucune coordonnée n'est résolue, l'IHM
reste utile : elle présente le **classement** des postes les plus actifs en
liste (la carte n'est qu'une présentation géographique de ce classement).

Stdlib pur (urllib) côté HTTP, comme ``dataset/source.py``.
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

#: dataset ODRE des postes électriques RTE (avec coordonnées + tension).
ODRE_DATASET = "postes-electriques-rte"
ODRE_EXPORT = ("https://odre.opendatasoft.com/api/explore/v2.1/catalog/"
               "datasets/{dataset}/exports/json")

#: emplacement par défaut de l'instantané committé (résolution sans réseau).
SNAPSHOT_DEFAUT = "data/postes_rte_geo.json"

# Codes HTTP **transitoires** d'ODRE (on retente). ``403`` en est **exclu** :
# côté ODRE c'est un refus (auth / politique de sortie bloquée), pas un aléa —
# on échoue alors immédiatement et on bascule sur le classement en liste, sans
# bloquer « Explorer la journée » sur des retries inutiles.
_RETRIABLE = {429, 500, 502, 503, 504}


# ===========================================================================
# 1. Coordonnées embarquées dans le réseau (extension pypowsybl)
# ===========================================================================

def positions_xiidm(net) -> dict[str, dict]:
    """``{substation_id: {'lat', 'lon', 'source': 'xiidm'}}`` depuis l'extension
    ``substationPosition`` du réseau, si présente. ``{}`` sinon (cas du dataset
    RTE 7000 actuel). Best-effort — ne lève jamais."""
    try:
        ext = net.get_extensions("substationPosition")
    except Exception:
        return {}
    if ext is None or len(ext) == 0:
        return {}
    out: dict[str, dict] = {}
    if "latitude" in ext.columns and "longitude" in ext.columns:
        for sid, lat, lon in zip(ext.index, ext["latitude"], ext["longitude"]):
            try:
                out[str(sid)] = {"lat": float(lat), "lon": float(lon),
                                 "source": "xiidm"}
            except (TypeError, ValueError):
                continue
    return out


# ===========================================================================
# 2. Instantané committé (indexé par substation_id)
# ===========================================================================

def charger_snapshot(path: str | Path = SNAPSHOT_DEFAUT) -> dict[str, dict]:
    """Charge l'instantané committé ``{substation_id: {lat, lon, nom?,
    tension?}}``. ``{}`` si le fichier est absent ou illisible.

    Tolère deux formes : un dict déjà indexé par ``substation_id`` ; ou une
    liste d'enregistrements portant une clé ``substation_id``/``code``."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict] = {}
    items = (data.items() if isinstance(data, dict)
             else ((r.get("substation_id") or r.get("code"), r)
                   for r in data if isinstance(r, dict)))
    for sid, rec in items:
        if not sid or not isinstance(rec, dict):
            continue
        geo = _coord(rec)
        if geo is None:
            continue
        out[str(sid)] = {"lat": geo[0], "lon": geo[1], "source": "snapshot",
                         **{k: rec[k] for k in ("nom", "tension")
                            if k in rec}}
    return out


def _coord(rec: dict) -> Optional[tuple[float, float]]:
    """``(lat, lon)`` d'un enregistrement, tolérant : champs ``lat``/``lon``
    directs, ``geo_point_2d`` (dict / liste / chaîne ``'lat,lon'``)."""
    if rec.get("lat") is not None and rec.get("lon") is not None:
        try:
            return float(rec["lat"]), float(rec["lon"])
        except (TypeError, ValueError):
            pass
    return _extraire_geo(rec)


# ===========================================================================
# 3. ODRE en direct (fetch + cache) — repli runtime
# ===========================================================================

def _http_get(url: str, timeout: int = 120, essais: int = 4,
              token: Optional[str] = None) -> bytes:
    """GET avec retries exponentiels sur les codes transitoires (cf. source.py)."""
    headers = {"User-Agent": "expert-op4grid-geo/1.0"}
    if token:
        headers["Authorization"] = f"Apikey {token}"
    req = urllib.request.Request(url, headers=headers)
    for essai in range(essais):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRIABLE or essai == essais - 1:
                raise
            time.sleep(2 ** essai + random.uniform(0, 1))
        except (urllib.error.URLError, TimeoutError):
            if essai == essais - 1:
                raise
            time.sleep(2 ** essai + random.uniform(0, 1))
    raise AssertionError("unreachable")


def fetch_odre_records(
    cache_dir: str | Path | None = None,
    token: Optional[str] = None,
    force: bool = False,
    essais: int = 4,
    timeout: int = 120,
) -> list[dict]:
    """Tous les enregistrements du dataset ODRE des postes (API *exports/json*).

    Mis en cache sous ``cache_dir/odre_postes.json`` (repris tel quel si présent
    et ``force=False``). Lève si ODRE est injoignable et qu'aucun cache n'existe.

    ``essais``/``timeout`` : la résolution **runtime** les abaisse (échec rapide
    si la sortie vers ODRE est bloquée → repli sur le classement en liste) ; le
    script de fetch hors-ligne garde les valeurs par défaut (plus robustes)."""
    cache = Path(cache_dir) / "odre_postes.json" if cache_dir else None
    if cache and cache.exists() and not force:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    url = ODRE_EXPORT.format(dataset=ODRE_DATASET)
    raw = _http_get(url, token=token, essais=essais, timeout=timeout)
    records = json.loads(raw.decode("utf-8", "replace"))
    if not isinstance(records, list):
        records = records.get("results", []) if isinstance(records, dict) else []
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(records, ensure_ascii=False),
                         encoding="utf-8")
    return records


# --- détection tolérante des champs ODRE -----------------------------------

_CLES_GEO = ("geo_point_2d", "geo_point", "geopoint", "coordonnees",
             "geo_shape", "point", "geom")
_CLES_CODE = ("code_poste", "codeposte", "code", "code_gdo", "code_site",
              "poste", "code_poste_source")
_CLES_NOM = ("nom_poste", "libelle_poste", "nom", "libelle", "nom_site",
             "intitule")
_CLES_TENSION = ("tension", "niveau_tension", "tension_kv")


def _extraire_geo(rec: dict) -> Optional[tuple[float, float]]:
    """``(lat, lon)`` depuis un enregistrement ODRE (formes multiples)."""
    for k in _CLES_GEO:
        v = rec.get(k)
        if v is None:
            continue
        if isinstance(v, dict):
            lat = v.get("lat") if v.get("lat") is not None else v.get("latitude")
            lon = v.get("lon") if v.get("lon") is not None else v.get("longitude")
            if lat is None and "coordinates" in v:  # GeoJSON [lon, lat]
                try:
                    lon, lat = v["coordinates"][0], v["coordinates"][1]
                except (IndexError, TypeError):
                    lat = lon = None
            if lat is not None and lon is not None:
                try:
                    return float(lat), float(lon)
                except (TypeError, ValueError):
                    continue
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            # ODRE geo_point_2d est [lat, lon] ; GeoJSON serait [lon, lat].
            try:
                return float(v[0]), float(v[1])
            except (TypeError, ValueError):
                continue
        elif isinstance(v, str) and "," in v:
            try:
                a, b = v.split(",")[:2]
                return float(a), float(b)
            except ValueError:
                continue
    return None


def _premier(rec: dict, cles: Iterable[str]) -> Optional[str]:
    for k in cles:
        v = rec.get(k)
        if v not in (None, ""):
            return str(v)
    return None


# ===========================================================================
# Appariement ODRE ↔ substation_id (best-effort, mesuré hors-ligne)
# ===========================================================================

def normaliser_mnemonique(s: str) -> str:
    """Normalise un mnémonique de poste pour l'appariement : majuscules, retrait
    des points/espaces/tirets de tête et de tout caractère non alphanumérique
    (``'.G.RO'`` → ``'GRO'``, ``'.CTLH'`` → ``'CTLH'``)."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _index_par(records: list[dict], cles: Iterable[str]) -> dict[str, dict]:
    """Index ``{mnémonique normalisé: {lat, lon, nom, tension}}`` à partir du
    **premier champ renseigné** de ``cles`` (code OU nom), pour les enregistrements
    géolocalisés. La 1ʳᵉ occurrence d'une clé gagne."""
    idx: dict[str, dict] = {}
    for rec in records:
        geo = _extraire_geo(rec)
        if geo is None:
            continue
        val = _premier(rec, cles)
        if not val:
            continue
        norm = normaliser_mnemonique(val)
        if norm and norm not in idx:
            idx[norm] = {"lat": geo[0], "lon": geo[1],
                         "nom": _premier(rec, _CLES_NOM) or val,
                         "tension": _premier(rec, _CLES_TENSION)}
    return idx


def _match_prefixe(norm: str, index: dict[str, dict]) -> Optional[dict]:
    """Unique entrée d'``index`` dont la clé est préfixe de ``norm`` ou inversement
    (≥ 4 caractères communs). ``None`` si zéro ou plusieurs candidats (ambigu)."""
    trouve = None
    for k, v in index.items():
        if (k.startswith(norm) or norm.startswith(k)) and min(len(k), len(norm)) >= 4:
            if trouve is not None:
                return None        # ambigu
            trouve = v
    return trouve


def apparier_odre(
    records: list[dict],
    substation_ids: Iterable[str],
    prefix_fallback: bool = True,
) -> tuple[dict[str, dict], dict]:
    """Apparie les enregistrements ODRE aux ``substation_ids`` du réseau.

    Retourne ``(positions, stats)``. ``positions`` : ``{substation_id: {'lat',
    'lon', 'nom', 'tension', 'source': 'odre'}}``. ``stats`` mesure la qualité
    **et** aide au diagnostic (``n_odre``, ``n_apparies``, ``taux``,
    ``sample_fields``, ``sample_codes``, ``sample_noms``, ``sample_subs``,
    ``sample_non_apparies``).

    Stratégie : on indexe ODRE **par code** (``_CLES_CODE``) **et par nom**
    (``_CLES_NOM``), mnémoniques normalisés ; pour chaque ``substation_id`` : match
    **exact** (code puis nom), puis repli optionnel par **préfixe** (un côté
    préfixe de l'autre, ≥ 4 car.). Couvre le cas où ODRE n'expose qu'un **nom**
    (``CONCARNEAU``) là où le réseau porte un **mnémonique** (``CONCA``). Pure."""
    code_index = _index_par(records, _CLES_CODE)
    name_index = _index_par(records, _CLES_NOM)
    sub_ids = list(dict.fromkeys(map(str, substation_ids)))
    positions: dict[str, dict] = {}
    n_exact = n_prefixe = 0
    non_apparies: list[str] = []
    for sid in sub_ids:
        norm = normaliser_mnemonique(sid)
        rec = code_index.get(norm) or name_index.get(norm)
        if rec is not None:
            n_exact += 1
        elif prefix_fallback and len(norm) >= 4:
            rec = _match_prefixe(norm, code_index) or _match_prefixe(norm, name_index)
            if rec is not None:
                n_prefixe += 1
        if rec is not None:
            positions[sid] = {"lat": rec["lat"], "lon": rec["lon"],
                              "nom": rec["nom"], "tension": rec.get("tension"),
                              "source": "odre"}
        else:
            non_apparies.append(sid)
    n = len(sub_ids)
    # Échantillons de diagnostic (pourquoi 0 apparié ? quel champ ODRE ?).
    sample_fields = sorted(records[0].keys())[:40] if records else []
    sample_codes = [_premier(r, _CLES_CODE) for r in records[:6]]
    sample_noms = [_premier(r, _CLES_NOM) for r in records[:6]]
    stats = {"n_substations": n, "n_odre": len(records),
             "n_index_code": len(code_index), "n_index_nom": len(name_index),
             "n_apparies": len(positions), "n_exact": n_exact,
             "n_prefixe": n_prefixe,
             "taux": round(len(positions) / n, 3) if n else 0.0,
             "sample_fields": sample_fields, "sample_codes": sample_codes,
             "sample_noms": sample_noms, "sample_subs": sub_ids[:6],
             "sample_non_apparies": non_apparies[:6]}
    return positions, stats


# ===========================================================================
# Résolution en chaîne (runtime)
# ===========================================================================

def ecrire_snapshot(path: str | Path, positions: dict[str, dict]) -> None:
    """Écrit un instantané committable ``{substation_id: {lat, lon, nom?,
    tension?}}`` (best-effort — ne lève pas si l'écriture échoue, p. ex. FS en
    lecture seule)."""
    try:
        out = {sid: {k: v for k, v in (("lat", r.get("lat")), ("lon", r.get("lon")),
                                       ("nom", r.get("nom")), ("tension", r.get("tension")))
                     if v is not None}
               for sid, r in positions.items()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def resoudre(
    substation_ids: Iterable[str],
    net=None,
    snapshot_path: str | Path = SNAPSHOT_DEFAUT,
    cache_dir: str | Path | None = None,
    autoriser_odre: bool = True,
    token: Optional[str] = None,
    essais: int = 2,
    timeout: int = 30,
    persist_path: str | Path | None = None,
    stats_out: Optional[dict] = None,
) -> tuple[dict[str, dict], str]:
    """Résout les coordonnées des ``substation_ids`` par chaîne de sources.

    Retourne ``(positions, source)`` où ``source`` ∈ {``'xiidm'``,
    ``'snapshot'``, ``'odre'``, ``'aucune'``}. Best-effort : un échec ODRE
    (réseau bloqué) n'interrompt pas — on renvoie ce qui a pu être résolu (au
    pire ``{}`` + ``'aucune'``, l'IHM bascule alors sur le classement en liste).

    ``persist_path`` : après un appariement ODRE réussi, **écrit l'instantané**
    (indexé par ``substation_id``) à ce chemin → les explorations suivantes
    repassent par la source ``snapshot`` (instantanée, sans re-fetch) et le
    fichier est committable. ``stats_out`` (dict muté) reçoit les statistiques
    d'appariement ODRE (``n_apparies``, ``taux``…) pour affichage."""
    ids = list(dict.fromkeys(map(str, substation_ids)))

    if net is not None:
        emb = positions_xiidm(net)
        emb = {s: emb[s] for s in ids if s in emb}
        if emb:
            return emb, "xiidm"

    snap = charger_snapshot(snapshot_path)
    snap = {s: snap[s] for s in ids if s in snap}
    if snap:
        return snap, "snapshot"

    if autoriser_odre:
        try:
            records = fetch_odre_records(cache_dir, token=token, essais=essais,
                                         timeout=timeout)
            positions, stats = apparier_odre(records, ids)
            if stats_out is not None:
                stats_out.update(stats)
            if positions:
                if persist_path:
                    ecrire_snapshot(persist_path, positions)
                return positions, "odre"
        except Exception as exc:
            # Diagnostic : distinguer « ODRE injoignable » de « 0 apparié ».
            if stats_out is not None:
                stats_out.update({"odre_error": f"{type(exc).__name__}: {exc}"})

    return {}, "aucune"
