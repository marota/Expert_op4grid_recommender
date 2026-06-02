"""
tests/manoeuvre/test_securite_sectionneurs.py
---------------------------------------------
Tests unitaires des briques de **sûreté des sectionneurs de barre** ajoutées à
``algo.py`` :

- ``_live_graph_sans`` : sous-graphe des switches fermés, sectionneur retiré ;
- ``_ouvrages_energises_sur`` : ouvrages énergisant un côté, par connectivité
  électrique réelle (capte aussi les ouvrages raccordés directement par DJ sans
  sectionneur d'aiguillage, ex. côté HT d'un transformateur) ;
- ``_verifier_securite_sectionneurs`` : rejoue une séquence et signale toute
  ouverture de sectionneur laissant **deux côtés sous tension**.

Support : poste CZBEVP3 (1 barre / 3 sections chaînées par SS.1.12 et SS.1.23,
sans couplage ; chaque section porte un transfo raccordé par DJ direct + une
ligne raccordée par SA+DJ).
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique
from expert_op4grid_recommender.manoeuvre.algo import (
    Manoeuvre,
    _live_graph_sans,
    _ouvrages_energises_sur,
    _verifier_securite_sectionneurs,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

VL = "CZBEVP3"
SS_12 = "CZBEVP3_CZBEV3CBO.1 SS.1.12_OC"
SS_23 = "CZBEVP3_CZBEV3CBO.1 SS.1.23_OC"
TR311 = "CZBEVP3_CZBEV3TR311 DJ.HT_OC"      # transfo, raccordé par DJ direct (sans SA)
FALLO2 = "CZBEVP3_CZBEV3FALLO.2 DJ_OC"      # ligne, section 1.1
TR312 = "CZBEVP3_CZBEV3TR312 DJ.HT_OC"
DANT5 = "CZBEVP3_CZBEV3DANT5.1 DJ_OC"

pytestmark = pytest.mark.skipif(
    VL not in list_available_fixtures(), reason="Fixture CZBEVP3 absente")


def _poste_depart():
    """Poste CZBEVP3 dans l'état « tout fermé » (1 nœud)."""
    G = build_graph_from_fixture(VL)
    return PosteTopologique.from_graph(G, VL)


def _sjb_node(poste, sjb_id):
    G = poste.graph
    for n in poste.tronconnement.barre_par_busbar:
        if G.nodes[n].get("busbar_section_id") == sjb_id:
            return n
    raise AssertionError(f"SJB {sjb_id} introuvable")


# ---------------------------------------------------------------------------
# _live_graph_sans
# ---------------------------------------------------------------------------

def test_live_graph_sans_retire_le_sectionneur():
    """Retirer SS.1.12 sépare la section 1.1 du reste (graphe live)."""
    poste = _poste_depart()
    n11 = _sjb_node(poste, "CZBEVP3_1.1")
    n12 = _sjb_node(poste, "CZBEVP3_1.2")
    # Tout fermé : 1.1 et 1.2 reliés.
    import networkx as nx
    H_plein = _live_graph_sans(poste.graph, [])
    assert nx.has_path(H_plein, n11, n12)
    # Sans SS.1.12 : 1.1 isolé de 1.2 (SS.1.23 garde 1.2-1.3 ensemble).
    H = _live_graph_sans(poste.graph, [SS_12])
    assert not nx.has_path(H, n11, n12)


# ---------------------------------------------------------------------------
# _ouvrages_energises_sur
# ---------------------------------------------------------------------------

def test_ouvrages_energises_capte_transfo_sans_SA():
    """Le côté 1.1 est énergisé par TR311 (DJ direct, **sans SA**) et FALLO.2.
    Régression : l'ancien contrôle basé sur le câblage SA ratait TR311."""
    poste = _poste_depart()
    n11 = _sjb_node(poste, "CZBEVP3_1.1")
    H = _live_graph_sans(poste.graph, [SS_12])
    side_11 = {n11}
    ouvrages = _ouvrages_energises_sur(poste.graph, poste.cellules, side_11, H)
    eqs = {eq for eq, _ in ouvrages}
    assert "CZBEV3T311" in eqs, "Transfo TR311 (DJ direct) doit être détecté"
    assert "CZBEVL32FALLO" in eqs, "Ligne FALLO.2 doit être détectée"
    # Chaque ouvrage expose son DJ (dé-énergisable).
    for eq, brk in ouvrages:
        assert brk, f"{eq} devrait avoir un DJ propre"


def test_ouvrages_energises_cote_oppose_distinct():
    """Avec SS.1.12 retiré, le côté {1.2,1.3} ne compte pas les ouvrages de 1.1
    (la distinction des deux côtés est correcte)."""
    poste = _poste_depart()
    n12 = _sjb_node(poste, "CZBEVP3_1.2")
    n13 = _sjb_node(poste, "CZBEVP3_1.3")
    H = _live_graph_sans(poste.graph, [SS_12])
    cote = _ouvrages_energises_sur(poste.graph, poste.cellules, {n12, n13}, H)
    eqs = {eq for eq, _ in cote}
    assert "CZBEV3T311" not in eqs and "CZBEVL32FALLO" not in eqs
    assert {"CZBEV3T312", "CZBEVL31DANT5", "CZBEV3T313",
            "CZBEVL31FALLO"} <= eqs


# ---------------------------------------------------------------------------
# _verifier_securite_sectionneurs
# ---------------------------------------------------------------------------

def test_verifier_signale_ouverture_sous_tension():
    """Ouvrir SS.1.12 directement (deux côtés énergisés) est signalé en écart."""
    poste = _poste_depart()
    seq = [Manoeuvre(SS_12, "OPEN", "ouverture directe (test)")]
    ecarts = _verifier_securite_sectionneurs(poste, seq)
    assert len(ecarts) == 1
    assert SS_12 in ecarts[0]
    assert "sous tension" in ecarts[0].lower()


def test_verifier_ok_si_section_de_energisee_avant():
    """Si la section 1.1 est mise hors tension (DJ ouverts) avant l'ouverture du
    sectionneur, aucun écart n'est signalé."""
    poste = _poste_depart()
    seq = [
        Manoeuvre(FALLO2, "OPEN", "mise hors tension"),
        Manoeuvre(TR311, "OPEN", "mise hors tension"),
        Manoeuvre(SS_12, "OPEN", "ouverture sectionnement (hors tension)"),
        Manoeuvre(FALLO2, "CLOSE", "remise sous tension"),
        Manoeuvre(TR311, "CLOSE", "remise sous tension"),
    ]
    assert _verifier_securite_sectionneurs(poste, seq) == []


def test_verifier_les_deux_sectionneurs_sous_tension():
    """Ouvrir les deux sectionnements sans dé-énergisation produit deux écarts
    (séquence « naïve » d'origine)."""
    poste = _poste_depart()
    seq = [
        Manoeuvre(SS_12, "OPEN", "ouverture directe"),
        Manoeuvre(SS_23, "OPEN", "ouverture directe"),
    ]
    ecarts = _verifier_securite_sectionneurs(poste, seq)
    assert len(ecarts) == 2


def test_verifier_aucun_sectionneur_aucune_alerte():
    """Une séquence ne touchant aucun sectionneur ne produit aucun écart."""
    poste = _poste_depart()
    seq = [Manoeuvre(TR311, "OPEN", "x"), Manoeuvre(TR311, "CLOSE", "x")]
    assert _verifier_securite_sectionneurs(poste, seq) == []
