"""
tests/manoeuvre/test_noviop3_3noeuds.py
---------------------------------------
Test du poste **NOVIOP3 → 3 nœuds** (2 barres sectionnées A/B, couplage
`COUPL.A`). La cible scinde le poste en 3 nœuds ; la séquence attendue (5
manœuvres) ré-aiguille ASNIE.1 (boucle longue) puis ouvre le couplage `COUPL.A`.

La séquence de référence est sauvegardée dans
``tests/manoeuvre/sequences/NOVIOP3_cible_3noeuds.json`` (générée via l'IHM).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique
from expert_op4grid_recommender.manoeuvre.algo import (
    determiner_manoeuvres_cible_detaillee,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

SEQ = Path(__file__).parent / "sequences" / "NOVIOP3_cible_3noeuds.json"
COUPL_A = "NOVIOP3_NOVIO3COUPL.A DJ_OC"

pytestmark = pytest.mark.skipif(
    not SEQ.exists() or "NOVIOP3" not in list_available_fixtures(),
    reason="Séquence/fixture NOVIOP3 absente",
)


def _graph_from_states(vl, states):
    G = build_graph_from_fixture(vl)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


@pytest.fixture
def data():
    return json.loads(SEQ.read_text())


@pytest.fixture
def resultat(data):
    vl = data["voltage_level_id"]
    poste = PosteTopologique.from_graph(_graph_from_states(vl, data["depart"]), vl)
    cible_graph = _graph_from_states(vl, data["cible"])
    return determiner_manoeuvres_cible_detaillee(poste, cible_graph)


def test_trois_noeuds_atteints_et_verifies(resultat):
    assert resultat.is_verified, resultat.message
    assert resultat.is_verified_detaillee, resultat.ecarts
    assert resultat.ecarts == []
    assert resultat.topo_obtenue.nb_noeuds == 3


def test_couplage_ouvert(resultat):
    """Le couplage `COUPL.A` est ouvert pour séparer les barres."""
    actions = {(m.switch_id, m.action) for m in resultat.manoeuvres}
    assert (COUPL_A, "OPEN") in actions


def test_reaiguillage_boucle_longue_avant_ouverture_couplage(resultat):
    """ASNIE.1 est ré-aiguillé en boucle longue AVANT l'ouverture du couplage
    (R9/R11) : aucune fermeture de SA cible tant que l'ancien SA est fermé."""
    ms = resultat.manoeuvres
    i_coupl = next(i for i, m in enumerate(ms) if m.switch_id == COUPL_A)
    longues = [m for m in ms[:i_coupl] if m.type_boucle == "LONGUE"]
    assert longues, "Ré-aiguillage boucle longue attendu avant le couplage"


def test_sequence_correspond_a_la_reference(resultat, data):
    """La séquence calculée correspond à la séquence de référence sauvegardée
    (mêmes organes, mêmes actions, même ordre)."""
    ref = data["manoeuvres"]
    assert len(resultat.manoeuvres) == len(ref), (
        f"{len(resultat.manoeuvres)} manœuvres calculées vs {len(ref)} référence")
    for i, (calc, r) in enumerate(zip(resultat.manoeuvres, ref)):
        assert calc.switch_id == r["switch_id"], (
            f"Manœuvre {i + 1}: {calc.switch_id} != {r['switch_id']}")
        assert calc.action == r["action"], (
            f"Manœuvre {i + 1}: {calc.action} != {r['action']}")
