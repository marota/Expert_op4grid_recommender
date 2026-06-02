"""
tests/manoeuvre/test_scenarios_sauvegardes.py
-----------------------------------------------
Tests sur les **scénarios sauvegardés** (topologies détaillées départ/cible
créées via l'IHM) : le calcul de séquence détaillée doit atteindre la
topologie détaillée visée (barre exacte de chaque départ).

Les scénarios sont des fixtures JSON ``tests/manoeuvre/scenarios/<nom>.json``
(``depart`` / ``cible`` = états des organes, ``*_nodale`` = partitions).
Ils sont rejoués sur les fixtures de poste (sans pypowsybl).
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

SCEN_DIR = Path(__file__).parent / "scenarios"


def _scenarios():
    if not SCEN_DIR.exists():
        return []
    return sorted(SCEN_DIR.glob("*.json"))


def _graph_from_states(vl, states):
    G = build_graph_from_fixture(vl)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


pytestmark = pytest.mark.skipif(
    not _scenarios(), reason="Aucun scénario sauvegardé.")


@pytest.mark.parametrize("path", _scenarios(), ids=lambda p: p.stem)
def test_scenario_atteint_topologie_detaillee(path):
    """Chaque scénario sauvegardé doit mener à sa topologie détaillée cible."""
    d = json.loads(path.read_text())
    vl = d["voltage_level_id"]
    if vl not in list_available_fixtures():
        pytest.skip(f"Fixture {vl} absente")
    # les ids d'organes doivent exister dans la fixture
    known = {dd.get("switch_id")
             for _u, _v, dd in build_graph_from_fixture(vl).edges(data=True)}
    if not set(d["depart"]) <= known:
        pytest.skip(f"Organes du scénario absents de la fixture {vl}")

    poste = PosteTopologique.from_graph(_graph_from_states(vl, d["depart"]), vl)
    cible_graph = _graph_from_states(vl, d["cible"])
    res = determiner_manoeuvres_cible_detaillee(poste, cible_graph)

    assert res.is_verified, f"{path.stem} : nodale non atteinte — {res.message}"
    assert res.is_verified_detaillee, \
        f"{path.stem} : détaillée non atteinte — écarts {res.ecarts}"
    assert res.ecarts == []


def test_carrip3_1noeud_requinconcage():
    """Scénario CARRIP3 → 1 nœud : la section 1.2 est dé-énergisée pour fermer
    le sectionnement, PUIS ses départs sont **requinçonçés** (ramenés) sur 1.2
    en boucle courte (manœuvres supplémentaires) pour atteindre exactement la
    topologie détaillée cible."""
    path = SCEN_DIR / "CARRIP3_cible_1noeud.json"
    if not path.exists() or "CARRIP3" not in list_available_fixtures():
        pytest.skip("Scénario/fixture CARRIP3 absent")
    d = json.loads(path.read_text())
    poste = PosteTopologique.from_graph(
        _graph_from_states("CARRIP3", d["depart"]), "CARRIP3")
    cible_graph = _graph_from_states("CARRIP3", d["cible"])
    res = determiner_manoeuvres_cible_detaillee(poste, cible_graph)

    assert res.is_verified_detaillee, res.ecarts

    # On ferme le sectionnement après dé-énergisation...
    idx_sect = next(i for i, m in enumerate(res.manoeuvres)
                    if m.action == "CLOSE" and "sectionnement" in m.raison.lower())
    # ... puis on requinçonçe des départs (ré-aiguillage boucle courte APRÈS)
    # pour les ramener sur leur barre cible (1.2).
    retours = [m for m in res.manoeuvres[idx_sect + 1:]
               if m.type_boucle == "COURTE" and m.action == "CLOSE"
               and "1.2" in m.raison]
    assert retours, "Aucun requinçonçage (retour boucle courte) vers 1.2"
    # La vérification détaillée (ecarts == []) garantit déjà que chaque départ
    # finit sur sa barre exacte imposée par la cible.
