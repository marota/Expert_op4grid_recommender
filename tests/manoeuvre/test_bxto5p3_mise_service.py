"""
tests/manoeuvre/test_bxto5p3_mise_service.py
--------------------------------------------
Régression : **mise en service d'un départ (DJ ouvert → fermé)** combinée à
une recomposition nodale par sectionnement, sur le poste multi-sections
BXTO5P3 (2 barres × 3 sections = 6 SJB).

Le scénario ``BXTO5P3_cible_2noeuds_miseServiceSectionnement`` reconnecte le
départ TR317 (``BXTO53Y317``, DJ ``TR317 DJ.HT`` ouvert → fermé) tout en
fermant le sectionnement ``SS.2.23`` et en ouvrant le couplage ``COUPL.1``.

Bug d'origine : l'algorithme ignorait le changement d'état du DJ de départ et
ne générait que 2 manœuvres (fermer SS.2.12, ouvrir COUPL.1) ⇒ topologie
obtenue à **8 nœuds** au lieu des **7** visés.

Correctif attendu :
- la topologie **nodale** cible (7 nœuds) est atteinte et vérifiée ;
- la topologie **détaillée** est atteinte (aucun écart) ;
- la **règle du sectionneur** est respectée : le sectionnement ``SS.2.23`` est
  fermé pendant que sa section est hors tension, c.-à-d. **avant** la fermeture
  du DJ de mise en service de TR317.
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

SCEN = (Path(__file__).parent / "scenarios"
        / "BXTO5P3_cible_2noeuds_miseServiceSectionnement.json")

DJ_MISE_EN_SERVICE = "BXTO5P3_BXTO53TR317 DJ.HT_OC"
SECTIONNEMENT = "BXTO5P3_BXTO53SEC..23 SS.2.23_OC"
COUPLAGE = "BXTO5P3_BXTO53COUPL.1 DJ_OC"

pytestmark = pytest.mark.skipif(
    not SCEN.exists() or "BXTO5P3" not in list_available_fixtures(),
    reason="Scénario/fixture BXTO5P3 absent",
)


def _graph_from_states(vl, states):
    G = build_graph_from_fixture(vl)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


@pytest.fixture
def resultat():
    d = json.loads(SCEN.read_text())
    vl = d["voltage_level_id"]
    poste = PosteTopologique.from_graph(_graph_from_states(vl, d["depart"]), vl)
    cible_graph = _graph_from_states(vl, d["cible"])
    return determiner_manoeuvres_cible_detaillee(poste, cible_graph)


def test_topologie_nodale_cible_atteinte(resultat):
    """Le cœur du bug : 7 nœuds atteints (et non 8)."""
    assert resultat.is_verified, resultat.message
    assert resultat.topo_obtenue is not None
    assert resultat.topo_obtenue.nb_noeuds == 7


def test_topologie_detaillee_sans_ecart(resultat):
    """La barre exacte de chaque départ est conforme à la cible."""
    assert resultat.is_verified_detaillee, resultat.ecarts
    assert resultat.ecarts == []


def test_dj_mise_en_service_present_et_ferme(resultat):
    """Le DJ de départ TR317 (ouvert → fermé) est bien manœuvré (CLOSE)."""
    djs = [m for m in resultat.manoeuvres
           if m.switch_id == DJ_MISE_EN_SERVICE]
    assert len(djs) == 1, "Le DJ de mise en service doit être manœuvré une fois"
    assert djs[0].action == "CLOSE"


def test_sectionnement_ferme_avant_mise_en_service(resultat):
    """Règle du sectionneur : on ferme ``SS.2.23`` (section hors tension) AVANT
    de remettre TR317 sous tension par fermeture de son DJ."""
    ids = [m.switch_id for m in resultat.manoeuvres]
    assert SECTIONNEMENT in ids, "SS.2.23 doit être fermé pour recomposer le nœud"
    assert DJ_MISE_EN_SERVICE in ids
    i_sect = ids.index(SECTIONNEMENT)
    i_dj = ids.index(DJ_MISE_EN_SERVICE)
    assert i_sect < i_dj, (
        "Le sectionnement doit être fermé avant la mise en service du départ "
        "(section morte au moment de la fermeture du sectionneur)."
    )


def test_sectionnement_ferme_et_couplage_ouvert(resultat):
    """La recomposition combine fermeture de sectionnement et ouverture de
    couplage."""
    actions = {(m.switch_id, m.action) for m in resultat.manoeuvres}
    assert (SECTIONNEMENT, "CLOSE") in actions
    assert (COUPLAGE, "OPEN") in actions


def test_dj_mise_en_service_est_la_derniere_manoeuvre(resultat):
    """La mise sous tension du départ reconnecté intervient en dernier, une fois
    la topologie de barres établie."""
    assert resultat.manoeuvres[-1].switch_id == DJ_MISE_EN_SERVICE
    assert resultat.manoeuvres[-1].action == "CLOSE"
