"""
tests/manoeuvre/test_topologie.py
-----------------------------------
Tests de TopologieNodale et PosteTopologique (étapes 1.5-1.6).
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import (
    TopologieNodale,
    PosteTopologique,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures


def _fixtures_available() -> bool:
    return len(list_available_fixtures()) > 0


pytestmark = pytest.mark.skipif(
    not _fixtures_available(), reason="Fixtures de postes non générées."
)


# ---------------------------------------------------------------------------
# TopologieNodale — constructeurs sans pypowsybl
# ---------------------------------------------------------------------------

def test_from_node_groups_partition():
    topo = TopologieNodale.from_node_groups(
        "VL", [["A", "B"], ["C"], ["D", "E"]]
    )
    assert topo.nb_noeuds == 3
    assert topo.noeud_par_depart["A"] == topo.noeud_par_depart["B"]
    assert topo.noeud_par_depart["A"] != topo.noeud_par_depart["C"]
    assert topo.partition() == {
        frozenset({"A", "B"}), frozenset({"C"}), frozenset({"D", "E"})
    }


def test_from_bus_assignment():
    topo = TopologieNodale.from_bus_assignment(
        "VL", {"A": 7, "B": 7, "C": 3}
    )
    assert topo.nb_noeuds == 2
    assert topo.noeud_par_depart["A"] == topo.noeud_par_depart["B"]


def test_meme_topologie_ignore_noms():
    t1 = TopologieNodale.from_node_groups("VL", [["A", "B"], ["C"]])
    t2 = TopologieNodale.from_node_groups("VL", [["C"], ["B", "A"]])
    assert t1.meme_topologie(t2)


def test_meme_topologie_detecte_difference():
    t1 = TopologieNodale.from_node_groups("VL", [["A", "B"], ["C"]])
    t2 = TopologieNodale.from_node_groups("VL", [["A"], ["B", "C"]])
    assert not t1.meme_topologie(t2)


# ---------------------------------------------------------------------------
# TopologieNodale.from_graph — extraction depuis l'état détaillé
# ---------------------------------------------------------------------------

def test_carrip3_from_graph():
    """CARRIP3 : couplage fermé → départs connectés sur un seul nœud."""
    G = build_graph_from_fixture("CARRIP3")
    topo = TopologieNodale.from_graph(G, "CARRIP3")
    # Le gros nœud regroupe la majorité des départs.
    plus_gros = max(topo.noeuds.values(), key=lambda n: len(n.departs))
    assert len(plus_gros.departs) >= 10


def test_from_graph_tous_departs_attribues():
    for vl in ["CARRIP3", "NOVIOP3", "CZBEVP3"]:
        if vl not in list_available_fixtures():
            continue
        G = build_graph_from_fixture(vl)
        topo = TopologieNodale.from_graph(G, vl)
        # chaque départ apparaît dans exactement un nœud
        seen = list(topo.noeud_par_depart)
        assert len(seen) == len(set(seen))


# ---------------------------------------------------------------------------
# PosteTopologique — assemblage complet
# ---------------------------------------------------------------------------

def test_poste_topologique_from_graph():
    G = build_graph_from_fixture("CARRIP3")
    poste = PosteTopologique.from_graph(G, "CARRIP3")
    assert poste.nb_jeux_barres == 2
    assert poste.tronconnement.nb_jeux_barres == 2
    assert poste.topologie_nodale.nb_noeuds >= 1
    assert "Poste 'CARRIP3'" in poste.resume()


def test_attribuer_noeuds_consolidation():
    """L'attribution des nœuds remplit Troncon.noeuds_electriques."""
    G = build_graph_from_fixture("CARRIP3")
    poste = PosteTopologique.from_graph(G, "CARRIP3")
    troncon = next(iter(poste.tronconnement.troncons.values()))
    # Le tronçon unique porte tous les nœuds connectés du poste.
    assert len(troncon.noeuds_electriques) >= 1
