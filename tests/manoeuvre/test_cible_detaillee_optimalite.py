"""
tests/manoeuvre/test_cible_detaillee_optimalite.py
--------------------------------------------------
**Optimalité de la détermination « topologie nodale cible → topologie détaillée »**
sur les postes 400 kV à 3 jeux de barres.

Pour chaque scénario 3 JdB (``scenarios/*.json`` : ``depart`` + ``cible``
détaillée visée), on prend la **topologie nodale cible sous-jacente** (partition
de la cible détaillée) et on vérifie que ``determiner_topo_complete_cible``
(l'algorithme utilisé par l'IHM via ``/api/nodale_to_detaillee`) :

1. **retourne bien une topologie détaillée** réalisant la cible nodale
   (``is_verified`` + bon nombre de nœuds) ;
2. est **au moins aussi optimale en manœuvres** que la cible détaillée *faite
   main* du test : ``nb(T_ALGO) ≤ nb(T_VISÉE)``.

Le point (2) verrouille le correctif de placement « origin-preserving » (candidat
de coût brut minimal, pénalité multi-barres désactivée, retenu transactionnellement
quand il vérifie la cible avec moins de manœuvres) — cf. ``algo/placement.py``
(``_placement_automatique(..., penaliser_multibarre=False)``) et
``algo/targets.py`` (``determiner_topo_complete_cible``).

Avant le correctif, ``TAVELP7`` plaçait le grand nœud sur une seule barre
(4 ré-aiguillages, 13 manœuvres) là où la cible visée n'en demande qu'1 (barres
1B+3A+3B couplées, 7 manœuvres). Le correctif fait converger ``T_ALGO`` vers
**exactement** la cible visée (7 manœuvres). Sans dépendance pypowsybl (fixtures).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    TopologieNodale,
    determiner_topo_complete_cible,
    determiner_manoeuvres_cible_detaillee,
)
from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _set_switch

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

_SCEN_DIR = pathlib.Path(__file__).parent / "scenarios"

# Scénarios 3 JdB ayant une **cible détaillée explicite** (depart + cible).
_SCENARIOS_3B = [
    "CHESNP7_cible_3noeuds.json",
    "TAVELP7_cible_3noeuds.json",
    "TRI.PP7_cible_3_noeuds.json",
]


def _graph(stem: str, states: dict):
    G = build_graph_from_fixture(stem)
    for sid, op in states.items():
        _set_switch(G, sid, op)
    return G


def _cout_visee_exacte(poste, cible_graph) -> int | None:
    """Coût MINIMAL (nb manœuvres) pour atteindre EXACTEMENT la cible détaillée
    visée, parmi les modes qui la vérifient (``is_verified_detaillee``)."""
    couts = []
    for mode in ("smooth", "aggressive"):
        r = determiner_manoeuvres_cible_detaillee(poste, cible_graph, mode=mode)
        if r.is_verified_detaillee:
            couts.append(r.nb_manoeuvres)
    return min(couts) if couts else None


@pytest.mark.parametrize("seqfile", _SCENARIOS_3B)
def test_topo_detaillee_retournee_et_optimale(seqfile):
    path = _SCEN_DIR / seqfile
    if not path.exists():
        pytest.skip(f"Scénario absent : {seqfile}")
    d = json.loads(path.read_text())
    vl = d["voltage_level_id"]
    stem = vl.replace(".", "_")
    if stem not in list_available_fixtures():
        pytest.skip(f"Fixture {stem} absente")

    poste = PosteTopologique.from_graph(_graph(stem, d["depart"]), vl)
    nodal = TopologieNodale.from_graph(_graph(stem, d["cible"]), vl)
    assert nodal.nb_noeuds >= 3, "cible attendue à ≥ 3 nœuds (poste 3 JdB)"

    # (1) L'algorithme RETOURNE une topologie détaillée réalisant la cible nodale.
    res = determiner_topo_complete_cible(poste, nodal)
    assert res.is_verified, f"{vl}: cible nodale non réalisée — {res.message}"
    assert res.topo_obtenue is not None
    assert res.topo_obtenue.nb_noeuds == nodal.nb_noeuds
    assert res.nb_manoeuvres > 0

    # (2) Optimalité : T_ALGO ≤ T_VISÉE (la cible auto n'est jamais plus coûteuse
    #     que la cible détaillée faite main, à partition nodale égale).
    n_visee = _cout_visee_exacte(poste, _graph(stem, d["cible"]))
    assert n_visee is not None, (
        f"{vl}: la cible détaillée visée n'est atteinte exactement par aucun mode "
        "(smooth/aggressive) — pré-requis du test invalide")
    assert res.nb_manoeuvres <= n_visee, (
        f"{vl}: T_ALGO ({res.nb_manoeuvres} manœuvres) PLUS coûteuse que la cible "
        f"visée ({n_visee}). Régression du placement « origin-preserving ».")


def test_tavelp7_converge_vers_la_cible_visee():
    """Cas-témoin du correctif : sur TAVELP7, ``determiner_topo_complete_cible``
    atteint **exactement** la même topologie détaillée que la cible visée (le grand
    nœud couple les barres 1B+3A+3B au lieu de tout ré-aiguiller sur une seule
    barre), en **7 manœuvres** (au lieu de 13 avant le correctif)."""
    seqfile = "TAVELP7_cible_3noeuds.json"
    path = _SCEN_DIR / seqfile
    if not path.exists() or "TAVELP7" not in list_available_fixtures():
        pytest.skip("Fixture/scénario TAVELP7 absent")
    d = json.loads(path.read_text())
    vl = d["voltage_level_id"]
    poste = PosteTopologique.from_graph(_graph("TAVELP7", d["depart"]), vl)
    nodal = TopologieNodale.from_graph(_graph("TAVELP7", d["cible"]), vl)

    res = determiner_topo_complete_cible(poste, nodal)
    n_visee = _cout_visee_exacte(poste, _graph("TAVELP7", d["cible"]))

    assert res.is_verified
    # Convergence : même coût que la cible détaillée visée (placement optimal).
    assert res.nb_manoeuvres == n_visee, (
        f"TAVELP7: T_ALGO={res.nb_manoeuvres} ≠ cible visée={n_visee} "
        "(le placement origin-preserving devrait converger).")
