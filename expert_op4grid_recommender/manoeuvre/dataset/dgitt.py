"""
manoeuvre/dataset/dgitt.py — Adaptateur de lecture du dataset
``OpenSynth/D-GITT-RTE7000-2021`` (Hugging Face) vers les chronologies du
pipeline (``TimelinePoste``).

Téléchargement (hors de ce module — nécessite l'accès réseau à huggingface.co) ::

    pip install -U huggingface_hub
    hf download OpenSynth/D-GITT-RTE7000-2021 --repo-type dataset \
        --local-dir data/dgitt_rte7000_2021

Puis ::

    python scripts/build_rte7000_blocks.py --input data/dgitt_rte7000_2021 ...

**Écrit défensivement** : le schéma exact du dataset n'ayant pas pu être
inspecté depuis cet environnement (huggingface.co hors allowlist réseau),
l'adaptateur normalise un format **long** générique — une ligne = un état (ou
changement d'état) d'organe :

    (horodatage, poste/voltage level, organe, état ouvert ?)

avec auto-détection des noms de colonnes usuels et conversion des valeurs
(bool, 0/1, OPEN/CLOSED…). À la première exécution sur les vraies données :
si l'auto-détection échoue, l'erreur liste les colonnes trouvées — compléter
alors les alias ci-dessous (`_ALIASES`) ou passer un ``mapping`` explicite.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from .timeline import Snapshot, TimelinePoste

logger = logging.getLogger(__name__)

#: alias de colonnes reconnus (insensibles à la casse), par champ normalisé
_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "datetime", "date", "t", "instant",
                  "horodatage"),
    "voltage_level": ("voltage_level_id", "voltage_level", "vl", "vl_id",
                      "substation", "substation_id", "poste", "sub_id"),
    "switch": ("switch_id", "switch", "organe", "organe_id", "breaker_id",
               "id_organe", "element_id"),
    "open": ("open", "is_open", "etat", "state", "status", "value", "position"),
}

_TRUE = {"true", "1", "open", "ouvert", "opened", "o"}
_FALSE = {"false", "0", "closed", "ferme", "fermé", "close", "f", "c"}


def _to_open(v) -> bool:
    """Normalise une valeur d'état en « ouvert ? » (booléen)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"État d'organe non interprétable : {v!r}")


def _detecter_colonnes(colonnes, mapping: Optional[dict] = None) -> dict:
    """Mappe les champs normalisés vers les colonnes réelles (les ``mapping``
    explicites priment ; sinon premiers alias trouvés, insensibles à la casse)."""
    lower = {str(c).lower(): c for c in colonnes}
    out: dict[str, str] = {}
    mapping = mapping or {}
    for champ, alias in _ALIASES.items():
        if champ in mapping:
            out[champ] = mapping[champ]
            continue
        for a in alias:
            if a in lower:
                out[champ] = lower[a]
                break
    manquants = set(_ALIASES) - set(out)
    if manquants:
        raise ValueError(
            f"Colonnes non identifiées : {sorted(manquants)}. "
            f"Colonnes du fichier : {sorted(map(str, colonnes))}. "
            "Complétez _ALIASES (manoeuvre/dataset/dgitt.py) ou passez un "
            "mapping explicite {champ: colonne}.")
    return out


def _fichiers_tabulaires(input_dir: Path) -> list[Path]:
    exts = {".parquet", ".csv", ".csv.gz"}
    return sorted(p for p in Path(input_dir).rglob("*")
                  if p.is_file() and (p.suffix in exts
                                      or "".join(p.suffixes[-2:]) in exts))


def charger_timelines(
    input_dir: Path,
    vl_filter: Optional[set[str]] = None,
    mapping: Optional[dict] = None,
) -> Iterator[TimelinePoste]:
    """Itère les ``TimelinePoste`` des postes du dataset (un par VL).

    Lit tous les fichiers parquet/CSV de ``input_dir`` (récursif), concatène,
    trie par horodatage et reconstruit, par poste, les snapshots successifs en
    appliquant les lignes sur un état courant (fonctionne pour un format
    « événements » — seules les bascules — comme pour un format « snapshots »
    — l'état complet à chaque pas).
    """
    import pandas as pd   # paresseux : le cœur du package n'en dépend pas

    fichiers = _fichiers_tabulaires(input_dir)
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier parquet/CSV sous {input_dir} — le dataset "
            "est-il téléchargé ? (cf. docstring de manoeuvre/dataset/dgitt.py)")

    frames = []
    cols: Optional[dict] = None
    for f in fichiers:
        df = (pd.read_parquet(f) if f.suffix == ".parquet"
              else pd.read_csv(f))
        if cols is None:
            cols = _detecter_colonnes(df.columns, mapping)
            logger.info("Colonnes détectées : %s", cols)
        frames.append(df[[cols["timestamp"], cols["voltage_level"],
                          cols["switch"], cols["open"]]])
    data = pd.concat(frames, ignore_index=True)
    assert cols is not None

    data = data.sort_values([cols["voltage_level"], cols["timestamp"]],
                            kind="stable")
    for vl, grp in data.groupby(cols["voltage_level"], sort=True):
        vl = str(vl)
        if vl_filter and vl not in vl_filter:
            continue
        snapshots: list[Snapshot] = []
        etat: dict[str, bool] = {}
        for ts, sous in grp.groupby(cols["timestamp"], sort=True):
            etat = dict(etat)
            for _, row in sous.iterrows():
                etat[str(row[cols["switch"]])] = _to_open(row[cols["open"]])
            snapshots.append(Snapshot(timestamp=str(ts), etats=etat))
        if snapshots:
            yield TimelinePoste(vl, snapshots)
