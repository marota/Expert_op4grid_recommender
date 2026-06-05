"""
tests/manoeuvre/test_golden_sequences.py
------------------------------------------
**Golden / test de caractérisation** du séquenceur (phase 2).

But : figer le **comportement observable** de ``determiner_manoeuvres_cible_detaillee``
sur tout le corpus de scénarios sauvegardés (``tests/manoeuvre/scenarios/*.json``),
dans les deux modes (``smooth`` / ``aggressive``), afin de servir de **filet de
sécurité aux refactors à iso-comportement** (indexation des lookups, passe de
rejeu unique des vérificateurs…).

Pour chaque (scénario, mode) on sérialise un instantané canonique
``tests/manoeuvre/goldens/<scenario>__<mode>.golden.json`` :

- ``manoeuvres`` : **séquence ordonnée exacte** ``[switch_id, action, type_boucle]``
  (l'ordre est la propriété la plus fragile d'un refactor du séquenceur) ;
- ``is_verified`` / ``is_verified_detaillee`` ;
- ``ecarts`` : liste **ordonnée** des écarts détaillés résiduels ;
- ``noeuds_non_realisables`` : **canonique** (trié) — dérivé d'ensembles ;
- ``partition_obtenue`` : **canonique** (trié) — le regroupement est sémantique,
  pas l'ordre ni le nom des nœuds.

Régénération **consciente** après un changement de comportement assumé :

    UPDATE_GOLDENS=1 pytest tests/manoeuvre/test_golden_sequences.py

Déterminisme : la sortie a été vérifiée stable d'un process à l'autre
(``PYTHONHASHSEED`` variable) ; les champs dérivés d'ensembles de chaînes sont
néanmoins canonicalisés par prudence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique
from expert_op4grid_recommender.manoeuvre.algo import (
    determiner_manoeuvres_cible_detaillee,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

SCEN_DIR = Path(__file__).parent / "scenarios"
GOLDEN_DIR = Path(__file__).parent / "goldens"
MODES = ("smooth", "aggressive")

_UPDATE = os.environ.get("UPDATE_GOLDENS") == "1"


def _scenarios() -> list[Path]:
    if not SCEN_DIR.exists():
        return []
    return sorted(SCEN_DIR.glob("*.json"))


pytestmark = pytest.mark.skipif(
    not _scenarios(), reason="Aucun scénario sauvegardé.")


def _graph_from_states(vl: str, states: dict):
    """Graphe de fixture avec les états d'organes du scénario appliqués."""
    G = build_graph_from_fixture(vl)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


def _canon_groups(groups) -> list[list[str]]:
    """Forme canonique (triée) d'une collection de groupes d'équipements."""
    return sorted(sorted(g) for g in groups)


def _snapshot(path: Path, mode: str) -> dict:
    d = json.loads(path.read_text())
    vl = d["voltage_level_id"]
    poste = PosteTopologique.from_graph(_graph_from_states(vl, d["depart"]), vl)
    cible_graph = _graph_from_states(vl, d["cible"])
    res = determiner_manoeuvres_cible_detaillee(poste, cible_graph, mode=mode)

    partition = (sorted(sorted(grp) for grp in res.topo_obtenue.partition())
                 if res.topo_obtenue is not None else [])
    return {
        "voltage_level_id": vl,
        "mode": mode,
        "n_manoeuvres": res.nb_manoeuvres,
        "manoeuvres": [[m.switch_id, m.action, m.type_boucle]
                       for m in res.manoeuvres],
        "is_verified": bool(res.is_verified),
        "is_verified_detaillee": bool(res.is_verified_detaillee),
        "ecarts": list(res.ecarts),
        "noeuds_non_realisables": _canon_groups(res.noeuds_non_realisables),
        "partition_obtenue": partition,
    }


def _dump(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _cases() -> list[tuple[Path, str]]:
    return [(p, mode) for p in _scenarios() for mode in MODES]


@pytest.mark.parametrize(
    "path,mode", _cases(),
    ids=lambda v: (v.stem if isinstance(v, Path) else v),
)
def test_golden_sequence(path: Path, mode: str):
    """La séquence produite (et l'état de vérification) doit correspondre, octet
    pour octet, au golden versionné. Une divergence signale soit une régression
    d'un refactor censé être à iso-comportement, soit un changement de
    comportement à valider (régénérer avec ``UPDATE_GOLDENS=1``)."""
    vl = json.loads(path.read_text())["voltage_level_id"]
    if vl not in list_available_fixtures():
        pytest.skip(f"Fixture {vl} absente")
    # Les organes du scénario doivent exister dans la fixture.
    known = {dd.get("switch_id")
             for _u, _v, dd in build_graph_from_fixture(vl).edges(data=True)}
    depart = json.loads(path.read_text())["depart"]
    if not set(depart) <= known:
        pytest.skip(f"Organes du scénario absents de la fixture {vl}")

    snap = _snapshot(path, mode)
    golden_path = GOLDEN_DIR / f"{path.stem}__{mode}.golden.json"

    if _UPDATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(_dump(snap))

    assert golden_path.exists(), (
        f"Golden absent : {golden_path.name}. "
        "Le générer avec : UPDATE_GOLDENS=1 pytest "
        "tests/manoeuvre/test_golden_sequences.py")

    expected = json.loads(golden_path.read_text())
    # Round-trip JSON pour comparer des types homogènes (tuples -> listes).
    snap = json.loads(json.dumps(snap))

    if snap != expected:
        diffs = [k for k in set(snap) | set(expected)
                 if snap.get(k) != expected.get(k)]
        msg = [f"Divergence golden {golden_path.name} — champs: {sorted(diffs)}"]
        if "manoeuvres" in diffs:
            cur, old = snap["manoeuvres"], expected["manoeuvres"]
            msg.append(f"  manoeuvres: {len(cur)} obtenue(s) vs {len(old)} golden")
            for i in range(min(len(cur), len(old))):
                if cur[i] != old[i]:
                    msg.append(f"  1ère divergence idx {i}: "
                               f"obtenu={cur[i]} golden={old[i]}")
                    break
        assert snap == expected, "\n".join(msg)
