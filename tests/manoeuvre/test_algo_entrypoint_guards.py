"""
tests/manoeuvre/test_algo_entrypoint_guards.py
----------------------------------------------
Garde-fous des points d'entrée publics d'``algo.py`` (faisabilité / pré-conditions),
épinglés avant l'éclatement du module (#7) :

- **graphe absent** sur le poste → message explicite, aucune manœuvre ;
- **départs cibles absents** du poste (cible incohérente) → faisabilité refusée,
  topologie courante conservée.

Ces branches sont indépendantes de la taille du poste ; on utilise CARRIP3.
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    TopologieNodale,
    determiner_topo_complete_cible,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures


pytestmark = pytest.mark.skipif(
    "CARRIP3" not in list_available_fixtures(),
    reason="Fixture CARRIP3 absente.",
)

VL = "CARRIP3"


def _poste():
    return PosteTopologique.from_graph(build_graph_from_fixture(VL), VL)


def test_graph_absent_returns_explicit_message():
    poste = _poste()
    cible = poste.topologie_nodale
    poste.graph = None    # simule une vue de poste sans graphe support

    res = determiner_topo_complete_cible(poste, cible)

    assert res.nb_manoeuvres == 0
    assert res.is_verified is False
    assert "Graphe absent" in res.message


def test_target_with_unknown_feeder_is_infeasible():
    poste = _poste()
    # Cible référençant un départ inexistant → faisabilité refusée.
    groups = [sorted(n.equipment_ids) for n in poste.topologie_nodale.noeuds.values()]
    groups.append(["GHOST_FEEDER_42"])
    cible = TopologieNodale.from_node_groups(VL, groups)

    res = determiner_topo_complete_cible(poste, cible)

    assert res.nb_manoeuvres == 0
    assert res.is_verified is False
    assert "absents du poste" in res.message
    assert "GHOST_FEEDER_42" in res.message
    # La topologie obtenue reste la topologie courante (aucune manœuvre tentée).
    assert res.topo_obtenue is poste.topologie_nodale
