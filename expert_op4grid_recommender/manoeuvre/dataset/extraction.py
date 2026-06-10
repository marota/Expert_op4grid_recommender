"""
manoeuvre/dataset/extraction.py — Conversion des blocs de transition en
artefacts exploitables (phase 4 du plan) :

- **scénario** au format de l'IHM / des tests (``tests/manoeuvre/scenarios``) :
  ``{voltage_level_id, name, depart, cible, depart_nodale?, cible_nodale?}``
  + bloc ``meta`` (tags, horodatages, ids de topologie, manœuvres observées) —
  directement consommable par le séquenceur via la façade pluggable ;
- **séquence observée** au format ``tests/manoeuvre/sequences`` : la séquence
  réelle approchée (manœuvres ordonnées dérivées de l'historique), pour
  comparaison algo vs opérateur ;
- écriture du dataset (un JSON par bloc) + statistiques agrégées.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

from ..topologie import PosteTopologique
from ..plugins import CibleDetaillee
from .timeline import BlocTransition


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()) or "bloc"


def _nodale(poste: PosteTopologique, etats: dict[str, bool]) -> list[list[str]]:
    """Partition nodale (groupes de départs) d'un état détaillé."""
    topo = CibleDetaillee(poste.voltage_level_id, etats).topologie_nodale(poste)
    return [sorted(n.equipment_ids) for n in topo.noeuds.values()]


def bloc_to_scenario(
    bloc: BlocTransition,
    poste: Optional[PosteTopologique] = None,
    name: Optional[str] = None,
) -> dict:
    """Scénario départ → cible au format du dépôt. Avec ``poste``, les
    partitions nodales sont jointes (sinon omises)."""
    name = name or _safe(
        f"{bloc.voltage_level_id}_{bloc.t_cible}_"
        f"{(bloc.tags or ['bloc'])[0]}")
    out = {
        "voltage_level_id": bloc.voltage_level_id,
        "name": name,
        "depart": dict(bloc.etats_depart),
        "cible": dict(bloc.etats_cible),
        "meta": {
            "source": "historique",
            "tags": list(bloc.tags),
            "t_depart": bloc.t_depart,
            "t_cible": bloc.t_cible,
            "topologie_depart_id": bloc.topologie_depart_id,
            "topologie_cible_id": bloc.topologie_cible_id,
            "nb_organes_changes": bloc.nb_organes_changes,
            "nb_manoeuvres_observees": len(bloc.manoeuvres_observees),
            "nb_etats_transitoires": len(bloc.transitoires),
            "duree_stable_avant": bloc.duree_stable_avant,
            "duree_stable_apres": bloc.duree_stable_apres,
            "retour_observe": bloc.retour_observe,
        },
    }
    if poste is not None:
        out["depart_nodale"] = _nodale(poste, bloc.etats_depart)
        out["cible_nodale"] = _nodale(poste, bloc.etats_cible)
    return out


def bloc_to_sequence_observee(
    bloc: BlocTransition,
    name: Optional[str] = None,
) -> dict:
    """Séquence **réelle approchée** du bloc (manœuvres ordonnées dérivées des
    snapshots), au format des séquences sauvegardées du dépôt — la référence
    « opérateur » du benchmark."""
    name = name or _safe(f"{bloc.voltage_level_id}_{bloc.t_cible}_observee")
    return {
        "voltage_level_id": bloc.voltage_level_id,
        "name": name,
        "scenario": None,
        "edited": False,
        "mode": "observee",
        "depart": dict(bloc.etats_depart),
        "cible": dict(bloc.etats_cible),
        "nb_manoeuvres": len(bloc.manoeuvres_observees),
        "manoeuvres": [
            {"ordre": i + 1, "switch_id": m["switch_id"],
             "action": m["action"], "raison": f"observée ({m['timestamp']})",
             "boucle": None}
            for i, m in enumerate(bloc.manoeuvres_observees)
        ],
        "meta": {"tags": list(bloc.tags), "t_depart": bloc.t_depart,
                 "t_cible": bloc.t_cible},
    }


def ecrire_dataset(
    blocs: Iterable[BlocTransition],
    out_dir: Path,
    postes: Optional[dict[str, PosteTopologique]] = None,
    avec_sequences: bool = True,
) -> list[Path]:
    """Écrit un JSON de scénario par bloc dans ``out_dir/scenarios`` (et la
    séquence observée dans ``out_dir/sequences`` si ``avec_sequences`` et que
    des manœuvres ont été observées). Retourne les chemins écrits."""
    postes = postes or {}
    scen_dir = Path(out_dir) / "scenarios"
    seq_dir = Path(out_dir) / "sequences"
    scen_dir.mkdir(parents=True, exist_ok=True)
    ecrits: list[Path] = []
    for bloc in blocs:
        sc = bloc_to_scenario(bloc, postes.get(bloc.voltage_level_id))
        p = scen_dir / f"{sc['name']}.json"
        p.write_text(json.dumps(sc, indent=2, ensure_ascii=False))
        ecrits.append(p)
        if avec_sequences and bloc.manoeuvres_observees:
            seq_dir.mkdir(parents=True, exist_ok=True)
            sq = bloc_to_sequence_observee(bloc)
            q = seq_dir / f"{sq['name']}.json"
            q.write_text(json.dumps(sq, indent=2, ensure_ascii=False))
            ecrits.append(q)
    return ecrits


def stats_blocs(blocs: Iterable[BlocTransition]) -> dict:
    """Statistiques agrégées (contenu descriptif du papier) : volumes par tag,
    par poste, et distribution des tailles de transition."""
    blocs = list(blocs)
    par_tag: dict[str, int] = {}
    par_poste: dict[str, int] = {}
    tailles: list[int] = []
    observees: int = 0
    for b in blocs:
        for t in (b.tags or ["inclasse"]):
            par_tag[t] = par_tag.get(t, 0) + 1
        par_poste[b.voltage_level_id] = par_poste.get(b.voltage_level_id, 0) + 1
        tailles.append(b.nb_organes_changes)
        observees += bool(b.manoeuvres_observees)
    return {
        "nb_blocs": len(blocs),
        "nb_postes": len(par_poste),
        "blocs_avec_sequence_observee": observees,
        "par_tag": dict(sorted(par_tag.items(), key=lambda kv: -kv[1])),
        "par_poste": dict(sorted(par_poste.items(), key=lambda kv: -kv[1])),
        "organes_changes": {
            "min": min(tailles) if tailles else 0,
            "max": max(tailles) if tailles else 0,
            "moyenne": round(sum(tailles) / len(tailles), 2) if tailles else 0,
        },
    }
