"""
tests/manoeuvre/test_ssavop3_modes.py
-------------------------------------
Deux **modes de dé-énergisation** du séquenceur, sur SSAVOP3 → 4 nœuds (gros
poste à demi-barres A/B, deux sectionnements `SS.1A23` / `SS.2A23`) :

- **smooth** (défaut) : dé-énergise au plus près, en place (clignotement DJ),
  sans déplacer-puis-ramener (double-déplacement supprimé) ;
- **aggressive** : dé-énergise **en lot** (ouvre plusieurs DJ d'un coup,
  commute les SA hors tension, ré-alimente une seule fois) — bien moins de
  manœuvres, davantage d'ouvrages momentanément hors tension.

Les deux modes doivent atteindre la **même** topologie détaillée cible, sans
écart de sectionneur. Avant optimisation, le smooth comptait **62** manœuvres
(double-déplacement de BETTI/LAND5/TR311).
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

SCEN = Path(__file__).parent / "scenarios" / "SSAVOP3_cible_4noeuds.json"

pytestmark = pytest.mark.skipif(
    not SCEN.exists() or "SSAVOP3" not in list_available_fixtures(),
    reason="Scénario/fixture SSAVOP3 absent",
)


def _graph_from_states(vl, states):
    G = build_graph_from_fixture(vl)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


def _run(mode):
    d = json.loads(SCEN.read_text())
    vl = d["voltage_level_id"]
    poste = PosteTopologique.from_graph(_graph_from_states(vl, d["depart"]), vl)
    cible = _graph_from_states(vl, d["cible"])
    return determiner_manoeuvres_cible_detaillee(poste, cible, mode=mode)


@pytest.fixture
def smooth():
    return _run("smooth")


@pytest.fixture
def aggressive():
    return _run("aggressive")


def test_smooth_atteint_la_cible_detaillee(smooth):
    assert smooth.is_verified, smooth.message
    assert smooth.is_verified_detaillee, smooth.ecarts
    assert smooth.ecarts == []


def test_aggressive_atteint_la_cible_detaillee(aggressive):
    assert aggressive.is_verified, aggressive.message
    assert aggressive.is_verified_detaillee, aggressive.ecarts
    assert aggressive.ecarts == []


def test_les_deux_modes_atteignent_la_meme_topologie(smooth, aggressive):
    assert smooth.topo_obtenue.meme_topologie(aggressive.topo_obtenue)


def test_smooth_un_seul_ouvrage_hors_tension_a_la_fois():
    """R10ter : le mode smooth ne déconnecte jamais plus d'un ouvrage à la fois
    par ré-aiguillage (parking un par un) — aucune violation relevée."""
    from expert_op4grid_recommender.manoeuvre.algo import (
        _verifier_un_seul_hors_tension)
    d = json.loads(SCEN.read_text())
    vl = d["voltage_level_id"]
    poste = PosteTopologique.from_graph(_graph_from_states(vl, d["depart"]), vl)
    res = _run("smooth")
    assert _verifier_un_seul_hors_tension(poste, res.manoeuvres) == []


def test_aggressive_plus_court_que_smooth(smooth, aggressive):
    """Le mode agressif (batch) n'est jamais plus long que le smooth ici, et
    reste dans l'ordre de grandeur de la séquence experte (≈ 20)."""
    assert aggressive.nb_manoeuvres <= smooth.nb_manoeuvres
    assert aggressive.nb_manoeuvres < 40


def test_aucun_sectionneur_sous_tension_dans_les_deux_modes(smooth, aggressive):
    for res in (smooth, aggressive):
        for m in res.manoeuvres:
            if "SEC." in m.switch_id and m.action == "OPEN":
                assert "sous tension" not in m.raison.lower(), m.raison
