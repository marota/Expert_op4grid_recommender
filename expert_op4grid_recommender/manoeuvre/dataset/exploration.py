"""
manoeuvre/dataset/exploration.py — Exploration de l'**intérêt d'une journée**.

Pour une date donnée, on charge **trois situations** du réseau France (par
défaut **minuit**, **midi** et **23 h**) et on estime, **par poste**, les
**changements d'organes de coupure (OC)** observés au cours de la journée :
le **nombre d'OC dont l'état diffère** entre les trois instantanés, **ventilé
par type d'OC** (``BREAKER`` = disjoncteur, ``DISCONNECTOR`` = sectionneur).

Les postes les plus « actifs » (le plus de changements) sont mis en évidence :
ce sont les journées/postes intéressants à inspecter dans l'IHM de manœuvre.

Découpage :

- **extraction** (``extraire_etats_kinds``, ``structure_reseau``) : dépend de
  pypowsybl — lit un réseau chargé ;
- **cœur d'agrégation** (``changements_par_vl``, ``agreger_par_poste``,
  ``classer_postes``) : **Python pur** (dicts/listes), testable sans pypowsybl.

Un « poste » géographique = une **substation** pypowsybl (regroupe les voltage
levels de ses différents niveaux de tension). L'IHM affiche un disque par
substation ; le détail par voltage level reste disponible (un VL = un « poste »
au sens de la vue topologique historique).
"""
from __future__ import annotations

from typing import Iterable, Optional

#: heures par défaut visées pour caractériser une journée (instantané le plus
#: proche choisi dans le dataset, pas de 5 min).
HEURES_DEFAUT: tuple[str, ...] = ("00:00", "12:00", "23:00")

#: types d'OC ventilés (alignés sur ``get_switches().kind`` de pypowsybl) :
#: ``BREAKER`` = disjoncteur (DJ), ``DISCONNECTOR`` = sectionneur (SA/SL/SS),
#: ``LOAD_BREAK_SWITCH`` = interrupteur.
TYPES_OC: tuple[str, ...] = ("BREAKER", "DISCONNECTOR", "LOAD_BREAK_SWITCH")


# ===========================================================================
# Extraction (pypowsybl)
# ===========================================================================

def extraire_etats_kinds(
    net, vl_filter: Optional[set[str]] = None,
) -> tuple[dict[str, dict[str, bool]], dict[str, str]]:
    """``(etats_par_vl, kinds)`` d'un réseau chargé.

    - ``etats_par_vl`` : ``{voltage_level_id: {switch_id: ouvert ?}}`` ;
    - ``kinds``        : ``{switch_id: 'BREAKER' | 'DISCONNECTOR' | …}``.

    Une seule passe ``get_switches(all_attributes=True)`` (colonnes
    ``voltage_level_id``, ``open``, ``kind``)."""
    sw = net.get_switches(all_attributes=True)
    for col in ("voltage_level_id", "open", "kind"):
        if col not in sw.columns:
            raise ValueError(
                f"Colonne '{col}' absente de get_switches() — version de "
                f"pypowsybl ? Colonnes : {sorted(map(str, sw.columns))}")
    if vl_filter:
        sw = sw[sw["voltage_level_id"].isin(vl_filter)]
    etats: dict[str, dict[str, bool]] = {}
    kinds: dict[str, str] = {}
    vls = sw["voltage_level_id"].tolist()
    opens = sw["open"].tolist()
    knds = sw["kind"].tolist()
    ids = sw.index.tolist()
    for sid, vl, op, kd in zip(ids, vls, opens, knds):
        sid = str(sid)
        etats.setdefault(str(vl), {})[sid] = bool(op)
        kinds[sid] = str(kd)
    return etats, kinds


def structure_reseau(net) -> tuple[dict[str, dict], dict[str, str]]:
    """Structure (invariante dans la journée) du réseau :

    - ``vl_meta`` : ``{vl: {'substation', 'nominal_v', 'name'}}`` (VL
      NODE_BREAKER uniquement — ceux inspectables en topologie détaillée) ;
    - ``sub_name`` : ``{substation_id: nom_affiché}``.
    """
    vlt = net.get_voltage_levels(all_attributes=True)
    if "topology_kind" in vlt.columns:
        vlt = vlt[vlt["topology_kind"] == "NODE_BREAKER"]
    vl_meta: dict[str, dict] = {}
    subs = vlt["substation_id"].tolist() if "substation_id" in vlt else []
    nvs = vlt["nominal_v"].tolist() if "nominal_v" in vlt else []
    names = vlt["name"].tolist() if "name" in vlt else []
    ids = vlt.index.tolist()
    for i, vl in enumerate(ids):
        vl_meta[str(vl)] = {
            "substation": str(subs[i]) if i < len(subs) and subs[i] else "",
            "nominal_v": float(nvs[i]) if i < len(nvs) and nvs[i] == nvs[i] else 0.0,
            "name": str(names[i]) if i < len(names) and names[i] else str(vl),
        }
    sub_name: dict[str, str] = {}
    try:
        st = net.get_substations(all_attributes=True)
        snames = st["name"].tolist() if "name" in st else []
        for i, sid in enumerate(st.index.tolist()):
            nm = str(snames[i]) if i < len(snames) and snames[i] else str(sid)
            sub_name[str(sid)] = nm
    except Exception:
        pass
    return vl_meta, sub_name


# ===========================================================================
# Cœur d'agrégation (Python pur — testable sans pypowsybl)
# ===========================================================================

def _a_change(etats: list[Optional[bool]]) -> bool:
    """Un organe a-t-il changé d'état dans la journée ? (≥ 2 valeurs distinctes
    parmi les états **renseignés** — l'absence d'un instantané = pas d'info,
    pas un changement)."""
    vus = {e for e in etats if e is not None}
    return len(vus) > 1


def changements_par_vl(
    situations: list[dict[str, dict[str, bool]]],
    kinds: dict[str, str],
) -> dict[str, dict]:
    """Changements d'OC **par voltage level** sur la journée.

    ``situations`` : un ``{vl: {switch_id: ouvert ?}}`` par instantané (ordre
    chronologique). Pour chaque VL, on compte les organes dont l'état n'est pas
    constant sur la journée, **ventilés par type d'OC**.

    Retourne ``{vl: {'total': int, 'BREAKER': int, 'DISCONNECTOR': int,
    'n_oc': int, 'changed': [switch_id, …]}}``. Fonction pure."""
    vls: list[str] = []
    vus: set[str] = set()
    for situ in situations:
        for vl in situ:
            if vl not in vus:
                vus.add(vl)
                vls.append(vl)

    out: dict[str, dict] = {}
    for vl in vls:
        sw_ids: list[str] = []
        seen: set[str] = set()
        for situ in situations:
            for sid in situ.get(vl, {}):
                if sid not in seen:
                    seen.add(sid)
                    sw_ids.append(sid)
        par_type: dict[str, int] = {t: 0 for t in TYPES_OC}
        changed: list[str] = []
        for sid in sw_ids:
            etats = [situ.get(vl, {}).get(sid) for situ in situations]
            if _a_change(etats):
                changed.append(sid)
                kd = kinds.get(sid, "AUTRE")
                par_type[kd] = par_type.get(kd, 0) + 1
        out[vl] = {
            "total": len(changed),
            "n_oc": len(sw_ids),
            "changed": changed,
            **{t: par_type.get(t, 0) for t in TYPES_OC},
            **{k: v for k, v in par_type.items() if k not in TYPES_OC},
        }
    return out


def agreger_par_poste(
    changes_par_vl: dict[str, dict],
    vl_meta: dict[str, dict],
    sub_name: Optional[dict[str, str]] = None,
) -> dict[str, dict]:
    """Agrège les changements **par poste (substation)**.

    Retourne ``{substation: {'name', 'nominal_v_max', 'total', 'BREAKER',
    'DISCONNECTOR', 'n_oc', 'vls': [{'vl', 'nominal_v', 'total', 'BREAKER',
    'DISCONNECTOR', 'n_oc'}, …]}}`` (les VL d'un poste triés par changements
    décroissants). Fonction pure."""
    sub_name = sub_name or {}
    postes: dict[str, dict] = {}
    for vl, ch in changes_par_vl.items():
        meta = vl_meta.get(vl, {})
        sub = meta.get("substation") or vl
        nv = meta.get("nominal_v", 0.0)
        p = postes.get(sub)
        if p is None:
            p = postes[sub] = {
                "substation": sub,
                "name": sub_name.get(sub) or sub,
                "nominal_v_max": 0.0,
                "total": 0, "n_oc": 0,
                **{t: 0 for t in TYPES_OC},
                "vls": [],
            }
        p["nominal_v_max"] = max(p["nominal_v_max"], nv)
        p["total"] += ch["total"]
        p["n_oc"] += ch["n_oc"]
        for t in TYPES_OC:
            p[t] += ch.get(t, 0)
        p["vls"].append({
            "vl": vl, "nominal_v": nv,
            "name": meta.get("name") or vl,
            "total": ch["total"], "n_oc": ch["n_oc"],
            **{t: ch.get(t, 0) for t in TYPES_OC},
        })
    for p in postes.values():
        p["vls"].sort(key=lambda d: (-d["total"], -d["nominal_v"], d["vl"]))
    return postes


def classer_postes(postes: dict[str, dict], n_top: int = 10) -> list[str]:
    """Identifiants des ``n_top`` postes les plus actifs (changements totaux
    décroissants ; départage par tension max puis nom). Fonction pure."""
    actifs = [s for s, p in postes.items() if p["total"] > 0]
    actifs.sort(key=lambda s: (-postes[s]["total"],
                               -postes[s]["nominal_v_max"],
                               postes[s]["name"]))
    return actifs[:n_top]


def vl_le_plus_actif(poste: dict) -> Optional[str]:
    """Voltage level le plus changeant d'un poste (pour ouvrir la vue
    topologique la plus « intéressante » au double-clic). ``None`` si aucun."""
    vls = poste.get("vls") or []
    return vls[0]["vl"] if vls else None
