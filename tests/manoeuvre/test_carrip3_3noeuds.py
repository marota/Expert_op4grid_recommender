"""
tests/manoeuvre/test_carrip3_3noeuds.py
-----------------------------------------
Test de la **règle du sectionneur de barre** sur CARRIP3 : créer un 3ème nœud
électrique en ouvrant un sectionnement de barre.

Règle métier vérifiée
~~~~~~~~~~~~~~~~~~~~~~~
Un sectionneur de barre ne se manœuvre que **hors charge**. Pour scinder une
barre en deux nœuds :
1. ré-aiguiller (boucle courte) tous les départs de la section à isoler sur
   l'autre barre, afin de la laisser **hors tension** ;
2. ouvrir le sectionnement de barre (sûr car la section est morte) ;
3. ouvrir le couplage (DJ) pour séparer les barres ;
4. ré-aiguiller (boucle longue) les départs du 3ème nœud sur la section
   désormais isolée.

Cible (réalisable sur ce poste 2 barres / 4 SJB)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- nœud 0 : barre 1 entière (1.1 + 1.2) — nœud « mixte »
- nœud 1 : section 2.1 (barre 2, section 1)
- nœud 2 : section 2.2 (barre 2, section 2) — **isolée en ouvrant SS.2.12**

(Trois nœuds *tous mixtes* — comme la cible opérationnelle initiale à 3 nœuds —
exigeraient 6 SJB / 3 barres, absentes de ce modèle réduit de CARRIP3.)
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique
from expert_op4grid_recommender.manoeuvre.topologie import TopologieNodale
from expert_op4grid_recommender.manoeuvre.algo import (
    determiner_manoeuvres_avec_sections,
    determiner_topo_complete_cible,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures


pytestmark = pytest.mark.skipif(
    "CARRIP3" not in list_available_fixtures(),
    reason="Fixture CARRIP3 absente.",
)

VL = "CARRIP3"

# Classes d'aiguillage (chaque départ atteint 2 SJB : une par barre, à sa section)
CLASSE_A = [  # atteignent 1.1 / 2.1 (section 1)
    "BERT L31CARRI", "CARRIL31VALES", "CARRI3T314", "BARR6L31CARRI",
    "BRENOL31CARRI", "CARRIL31U.MON", "CARRIY631", "CARRI3T313",
]
CLASSE_B = [  # atteignent 1.2 / 2.2 (section 2)
    "CARRIY632", "CARRIL31RANTI", "CARRIL31V.PAU", "CARRIL31PERSA",
    "CARRIL32U.MON", "CARRIY633", "CARRI3T312",
]

NODE_1 = {"BERT L31CARRI", "BARR6L31CARRI"}        # -> 2.1 (classe A)
NODE_2 = {"CARRIL31RANTI", "CARRIL31PERSA"}         # -> 2.2 (classe B, section isolée)
NODE_0 = set(CLASSE_A + CLASSE_B) - NODE_1 - NODE_2  # -> barre 1


def _poste() -> PosteTopologique:
    return PosteTopologique.from_graph(build_graph_from_fixture(VL), VL)


def _placement():
    return [
        (NODE_0, {"CARRIP3_1.1", "CARRIP3_1.2"}),
        (NODE_1, {"CARRIP3_2.1"}),
        (NODE_2, {"CARRIP3_2.2"}),
    ]


def _run():
    return determiner_manoeuvres_avec_sections(_poste(), _placement())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_3noeuds_atteint_et_verifie():
    """Le 3ème nœud est créé et la topologie cible est vérifiée."""
    res = _run()
    assert res.is_verified, res.message
    assert res.is_changed
    # 3 nœuds placés + 2 générateurs isolés = 5 nœuds obtenus
    assert res.topo_obtenue.nb_noeuds == 5


def test_sectionnement_ouvert_hors_tension():
    """Le sectionnement de barre est ouvert, section hors tension."""
    res = _run()
    sect = [m for m in res.manoeuvres
            if m.action == "OPEN" and "sectionnement" in m.raison.lower()]
    assert sect, "Un sectionnement de barre doit être ouvert"
    assert any("SS.2.12" in m.switch_id for m in sect)
    # La règle de sécurité : section hors tension au moment de l'ouverture.
    assert all("hors tension" in m.raison for m in sect), \
        "Le sectionnement ne doit être ouvert qu'une fois la section morte"


def test_ordre_sectionnement_avant_couplage():
    """Le sectionnement (hors charge) est ouvert avant le couplage (DJ)."""
    res = _run()
    idx_sect = next(i for i, m in enumerate(res.manoeuvres)
                    if "sectionnement" in m.raison.lower())
    idx_coupl = next(i for i, m in enumerate(res.manoeuvres)
                     if "couplage" in m.raison.lower())
    assert idx_sect < idx_coupl


def test_boucle_courte_avant_sectionnement_longue_apres():
    """Avant l'ouverture du sectionnement : uniquement des boucles courtes.
    Le ré-aiguillage final (vers la section isolée) est en boucle longue."""
    res = _run()
    idx_sect = next(i for i, m in enumerate(res.manoeuvres)
                    if "sectionnement" in m.raison.lower())
    avant = res.manoeuvres[:idx_sect]
    apres = res.manoeuvres[idx_sect:]
    # Avant : pas de boucle longue (la section est dé-énergisée en boucle courte)
    assert all(m.type_boucle != "LONGUE" for m in avant)
    # Après : on trouve des boucles longues (mise HT / bascule SA / remise ST)
    assert any(m.type_boucle == "LONGUE" for m in apres)


def test_point_entree_unique_cree_3eme_noeud_automatiquement():
    """`determiner_topo_complete_cible` (placement automatique) atteint une
    cible 3 nœuds en ouvrant un sectionnement, sans placement explicite."""
    poste = _poste()
    # Cible exprimée uniquement en topologie nodale (pas de SJB explicite) :
    # un gros nœud + deux petits nœuds mono-section (un par classe).
    n1 = ["BARR6L31CARRI"]            # classe A -> section seule
    n2 = ["CARRI3T312"]              # classe B -> section isolée (sectionnement)
    autres = sorted((set(CLASSE_A) | set(CLASSE_B)) - set(n1) - set(n2))
    cible = TopologieNodale.from_node_groups(
        VL, [autres, n1, n2, ["CARRIINF"], ["CARRIING"]]
    )
    res = determiner_topo_complete_cible(poste, cible)
    assert res.is_verified, res.message
    sect = [m for m in res.manoeuvres
            if m.action == "OPEN" and "sectionnement" in m.raison.lower()]
    assert sect and all("hors tension" in m.raison for m in sect)


def test_infaisable_trois_noeuds_mixtes():
    """Trois nœuds *tous mixtes* (chacun classe A+B) sont impossibles sur ce
    poste 2 barres : le placement automatique le détecte."""
    poste = _poste()
    g = [["BERT L31CARRI", "CARRIL31RANTI"],   # A + B
         ["BARR6L31CARRI", "CARRIL31PERSA"],   # A + B
         ["CARRI3T314", "CARRI3T312"]]         # A + B
    autres = sorted((set(CLASSE_A) | set(CLASSE_B)) - {e for grp in g for e in grp})
    cible = TopologieNodale.from_node_groups(VL, g + [autres])
    res = determiner_topo_complete_cible(poste, cible)
    assert not res.is_verified


def test_boucle_longue_ouvre_ancien_sa_avant_fermer_nouveau():
    """R9 (sécurité court-circuit) : en boucle longue, l'ancien sectionneur est
    ouvert AVANT la fermeture du sectionneur cible — jamais de pont entre deux
    barres de potentiels différents."""
    res = _run()

    def cell_of(sid):
        return sid.split(" SA")[0].split(" DJ")[0]

    by_cell = {}
    for i, m in enumerate(res.manoeuvres):
        if m.type_boucle == "LONGUE":
            by_cell.setdefault(cell_of(m.switch_id), []).append((i, m))

    assert by_cell, "Le scénario doit comporter des ré-aiguillages boucle longue"
    for cell, items in by_cell.items():
        opens_old = [i for i, m in items
                     if m.action == "OPEN" and "quitte" in m.raison]
        closes_new = [i for i, m in items
                      if m.action == "CLOSE" and "vers" in m.raison]
        assert opens_old and closes_new
        assert max(opens_old) < min(closes_new), (
            f"{cell}: le SA cible est fermé avant l'ouverture de l'ancien SA "
            "(risque de court-circuit)")


def test_jamais_de_pont_court_circuit():
    """Invariant de sécurité : en rejouant la séquence, aucun sectionneur ne
    ponte deux barres qui ne seraient pas déjà au même potentiel par ailleurs.
    Un départ ne « court-circuite » jamais deux barres déconnectées (en boucle
    courte, le pont temporaire est sûr car les barres sont reliées par le
    couplage fermé)."""
    import networkx as nx
    from expert_op4grid_recommender.manoeuvre import algo
    poste = _poste()
    res = determiner_manoeuvres_avec_sections(poste, _placement())
    G = poste.graph.copy()
    cells = poste.cellules
    departs = [c.equipment_id for c in cells.cellules_depart]

    def closed_graph():
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            if not d.get("open", False):
                H.add_edge(u, v)
        return H

    for m in res.manoeuvres:
        algo._set_switch(G, m.switch_id, m.action == "OPEN")
        H = closed_graph()
        for eq in departs:
            cell = cells.get_cellule_depart(eq)
            wired = [bb for bb in cell.busbar_nodes
                     if algo._sa_path_to_sjb(cell, bb)
                     and all(not algo._is_open(G, s)
                             for s in algo._sa_path_to_sjb(cell, bb))]
            if len(wired) < 2:
                continue
            # Les barres pontées doivent rester connectées même si l'on retire
            # les nœuds internes de la cellule (sinon = pont créé par le départ).
            internes = [n for n in cell.all_nodes if n not in cell.busbar_nodes]
            H2 = H.copy()
            H2.remove_nodes_from(internes)
            comp = (nx.node_connected_component(H2, wired[0])
                    if wired[0] in H2 else set())
            assert all(bb in comp for bb in wired), (
                f"{eq}: pont entre barres déconnectées {wired} après "
                f"{m.action} {m.switch_id} (court-circuit)")


def test_boucle_longue_ouvre_seulement_le_dj_de_cellule():
    """R7 : pour ré-aiguiller un départ d'une cellule omnibus, on n'ouvre que
    le **DJ d'ensemble de la cellule** (côté sélecteurs de barre), jamais les
    disjoncteurs propres aux équipements en aval (transfo / groupe)."""
    from expert_op4grid_recommender.manoeuvre.cellules import detecter_cellules
    from expert_op4grid_recommender.manoeuvre import algo
    from expert_op4grid_recommender.manoeuvre.models import NodeType

    G = build_graph_from_fixture(VL)
    cells = detecter_cellules(G, VL)
    sjb = {G.nodes[n].get("busbar_section_id"): n
           for n, d in G.nodes(data=True)
           if d.get("node_type") == NodeType.BUSBAR_SECTION}

    # Cellule omnibus : transfo CARRI3T312 + groupe CARRIINF (DJ de cellule
    # commun "TR312 DJ.HT_OC", puis DJ propres en aval).
    cell = cells.get_cellule_depart("CARRI3T312")
    djs = algo._own_breakers_to_sjb(cell, sjb["CARRIP3_1.2"], "CARRI3T312")
    assert djs == ["CARRIP3_CARRI3TR312 DJ.HT_OC"]
    assert not any("DJ.HT_OC.CARRI" in d for d in djs)   # pas le DJ du transfo
    assert not any("Disj CARRIINF" in d for d in djs)     # pas le DJ du groupe

    # Cellule simple : un seul DJ propre.
    simple = cells.get_cellule_depart("CARRIL31RANTI")
    assert algo._own_breakers_to_sjb(simple, sjb["CARRIP3_1.2"], "CARRIL31RANTI") \
        == ["CARRIP3_CARRI3RANTI.1 DJ_OC"]


def test_departs_du_3eme_noeud_en_boucle_longue():
    """Les départs du 3ème nœud (section 2.2 isolée) sont ré-aiguillés en
    boucle longue, donc avec ouverture/fermeture de leur disjoncteur."""
    res = _run()
    longues = [m for m in res.manoeuvres if m.type_boucle == "LONGUE"]
    eqs = {m.switch_id.split()[0] for m in longues}  # préfixe ~ cellule
    # Au moins un DJ ouvert puis refermé pour les départs du nœud 2
    opens = [m for m in longues if m.action == "OPEN" and "hors tension" in m.raison]
    closes = [m for m in longues if m.action == "CLOSE" and "sous tension" in m.raison]
    assert opens and closes
    assert len(opens) == len(closes)
