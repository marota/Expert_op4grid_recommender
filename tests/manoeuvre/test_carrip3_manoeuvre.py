"""
tests/manoeuvre/test_carrip3_manoeuvre.py
-------------------------------------------
Test de bout en bout du poste **CARRIP3** : passage d'une topologie nodale
courante à une topologie nodale cible, avec génération et vérification de la
séquence de manœuvres (phase 2 de l'algorithme nodale → détaillée).

Note sur les données
~~~~~~~~~~~~~~~~~~~~~
La fixture CARRIP3 de ce dépôt est extraite d'un réseau RTE *réduit* : le poste
y est modélisé en **2 jeux de barres** (1.1/1.2 et 2.1/2.2) avec 17 départs
(``BERT L31CARRI``, ``CARRIL31RANTI``, ``CARRIY631``…). La topologie cible à
3 nœuds décrite dans la demande initiale (départs ``CARR8L3x``, ``…SNCF``,
``…ZRAN5``, ``…ZSSSE``) correspond au poste RTE *complet* (≥ 3 barres), absent
des réseaux livrés ici. Ce test exprime donc une **cible 2 barres
représentative** (réalisable sur la fixture) en réutilisant la même mécanique
« noeud n°0 / noeud n°1 » que la cible demandée.
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import (
    TopologieNodale,
    PosteTopologique,
)
from expert_op4grid_recommender.manoeuvre.algo import (
    determiner_topo_complete_cible,
    ResultatManoeuvres,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures


pytestmark = pytest.mark.skipif(
    "CARRIP3" not in list_available_fixtures(),
    reason="Fixture CARRIP3 absente.",
)

VL = "CARRIP3"

# --- Topologie nodale cible (2 barres, réalisable sur la fixture) ----------
# noeud n°0 (barre 1) : départs lignes « ouest »
# noeud n°1 (barre 2) : transformateurs + reste
# Les générateurs isolés (CARRIINF/CARRIING) restent sur leurs nœuds propres.
CIBLE_NOEUD_0 = [
    "BERT L31CARRI",
    "BRENOL31CARRI",
    "CARRIL31PERSA",
    "CARRIL31RANTI",
    "CARRIL31U.MON",
    "CARRIL31VALES",
    "CARRIL32U.MON",
]
CIBLE_NOEUD_1 = [
    "BARR6L31CARRI",
    "CARRI3T312",
    "CARRI3T313",
    "CARRI3T314",
    "CARRIL31V.PAU",
    "CARRIY631",
    "CARRIY632",
    "CARRIY633",
]


def _poste() -> PosteTopologique:
    G = build_graph_from_fixture(VL)
    return PosteTopologique.from_graph(G, VL)


def _cible(poste: PosteTopologique) -> TopologieNodale:
    """Cible 2 nœuds + nœuds des équipements isolés tels quels."""
    groupes = [list(CIBLE_NOEUD_0), list(CIBLE_NOEUD_1)]
    connectes = set(CIBLE_NOEUD_0) | set(CIBLE_NOEUD_1)
    for noeud in poste.topologie_nodale.noeuds.values():
        restes = noeud.equipment_ids - connectes
        if restes and not (noeud.equipment_ids & connectes):
            groupes.append(sorted(restes))
    return TopologieNodale.from_node_groups(VL, groupes)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_carrip3_structure():
    """CARRIP3 est un poste double barre : 2 barres, 1 tronçon, couplage DJ."""
    poste = _poste()
    assert poste.nb_jeux_barres == 2
    assert len(poste.tronconnement.troncons) == 1
    troncon = next(iter(poste.tronconnement.troncons.values()))
    assert troncon.couplage_breakers, "Le DJ de couplage doit être identifié"


def test_carrip3_sequence_atteint_la_cible():
    """La séquence de manœuvres atteint et vérifie la topologie cible."""
    poste = _poste()
    cible = _cible(poste)
    res = determiner_topo_complete_cible(poste, cible)

    assert res.is_changed
    assert res.is_verified, res.message
    assert cible.meme_topologie(res.topo_obtenue)
    assert res.nb_manoeuvres > 0


def test_carrip3_sequence_ouvre_le_couplage():
    """Passer de 1 nœud (couplage fermé) à 2 nœuds ouvre le DJ de couplage."""
    poste = _poste()
    res = determiner_topo_complete_cible(poste, _cible(poste))
    couplage = [
        m for m in res.manoeuvres
        if m.action == "OPEN" and "couplage" in m.raison.lower()
    ]
    assert couplage and any("COUPL" in m.switch_id for m in couplage)


def test_carrip3_reaiguillages_en_boucle_courte():
    """Le couplage étant initialement fermé, les ré-aiguillages sont sûrs
    (boucle courte : le départ reste sous tension)."""
    poste = _poste()
    res = determiner_topo_complete_cible(poste, _cible(poste))
    boucles = [m for m in res.manoeuvres if m.type_boucle is not None]
    assert boucles
    assert all(m.type_boucle == "COURTE" for m in boucles)


def test_carrip3_sequence_minimale_et_coherente():
    """Chaque ré-aiguillage = exactement 1 fermeture + 1 ouverture de SA,
    et seuls les départs réellement déplacés sont manœuvrés."""
    poste = _poste()
    res = determiner_topo_complete_cible(poste, _cible(poste))
    fermetures_sa = [m for m in res.manoeuvres
                     if m.action == "CLOSE" and m.type_boucle == "COURTE"]
    ouvertures_sa = [m for m in res.manoeuvres
                     if m.action == "OPEN" and m.type_boucle == "COURTE"]
    assert len(fermetures_sa) == len(ouvertures_sa) == len(res.departs_reaiguilles)


# ---------------------------------------------------------------------------
# Aide à l'inspection manuelle (non-test)
# ---------------------------------------------------------------------------

def format_sequence(res: ResultatManoeuvres) -> str:  # pragma: no cover
    """Formatage lisible de la séquence (utilisé en debug/CLI)."""
    return res.resume()


if __name__ == "__main__":  # pragma: no cover
    poste = _poste()
    cible = _cible(poste)
    print("== TOPOLOGIE COURANTE ==")
    print(poste.topologie_nodale.resume())
    print("\n== TOPOLOGIE CIBLE ==")
    print(cible.resume())
    res = determiner_topo_complete_cible(poste, cible)
    print("\n== SEQUENCE DE MANOEUVRES ==")
    print(res.resume())
    print("\n== TOPOLOGIE OBTENUE ==")
    print(res.topo_obtenue.resume())
