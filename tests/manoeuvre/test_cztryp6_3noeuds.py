"""
tests/manoeuvre/test_cztryp6_3noeuds.py
-----------------------------------------
Test de la **règle du sectionneur de barre** sur CZTRYP6 : créer un 3ème nœud
électrique en ouvrant un sectionnement de barre sur un poste à **4 sections**.

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

Cible (réalisable sur ce poste 2 barres / 8 SJB — 4 sections par barre)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- nœud 0 : barre 1 sections 1–3 (CHESNL61CZTRY, CHESNL63CZTRY, CZTRYY633)
- nœud 1 : barre 2 sections 1–4 (CHESNL62CZTRY, CZTRYL61MOISE, CZTRYY632)
- nœud 2 : barre 1 section 4 isolée (CZTRYL61PLISO, CZTRYL61SENAR, CZTRYY634)

La séquence de référence (20 manœuvres) est sauvegardée dans
``tests/manoeuvre/sequences/CZTRYP6_cible_3noeuds.json``.
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


pytestmark = pytest.mark.skipif(
    "CZTRYP6" not in list_available_fixtures(),
    reason="Fixture CZTRYP6 absente.",
)

VL = "CZTRYP6"
SEQ_PATH = Path(__file__).parent / "sequences" / "CZTRYP6_cible_3noeuds.json"

# Départs attendus sur chaque nœud cible
NODE_0 = {"CHESNL61CZTRY", "CHESNL63CZTRY", "CZTRYY633"}
NODE_1 = {"CHESNL62CZTRY", "CZTRYL61MOISE", "CZTRYY632"}
NODE_2 = {"CZTRYL61PLISO", "CZTRYL61SENAR", "CZTRYY634"}


def _load_sequence():
    return json.loads(SEQ_PATH.read_text())


def _graph_from_states(states):
    G = build_graph_from_fixture(VL)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


def _run():
    seq = _load_sequence()
    poste = PosteTopologique.from_graph(_graph_from_states(seq["depart"]), VL)
    cible_graph = _graph_from_states(seq["cible"])
    return determiner_manoeuvres_cible_detaillee(poste, cible_graph)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_3noeuds_atteint_et_verifie():
    """La topologie cible à 3 nœuds est atteinte et vérifiée (nodale + détaillée)."""
    res = _run()
    assert res.is_verified, res.message
    assert res.is_changed
    assert res.is_verified_detaillee, f"écarts: {res.ecarts}"
    assert res.ecarts == []


def test_3noeuds_topologie_obtenue():
    """La topologie obtenue comporte bien 3 nœuds connectés avec les bons départs."""
    res = _run()
    topo = res.topo_obtenue
    # Collecter les nœuds non-singleton (> 1 départ ou départs nommés)
    noeuds_connectes = [
        n.equipment_ids for n in topo.noeuds.values()
        if len(n.equipment_ids) > 1
    ]
    eq_sets = {frozenset(s) for s in noeuds_connectes}
    assert frozenset(NODE_0) in eq_sets, f"Nœud 0 manquant, obtenu: {eq_sets}"
    assert frozenset(NODE_1) in eq_sets, f"Nœud 1 manquant, obtenu: {eq_sets}"
    assert frozenset(NODE_2) in eq_sets, f"Nœud 2 manquant, obtenu: {eq_sets}"


def test_sectionnement_ouvert_hors_tension():
    """Le sectionnement de barre SS.1.34 est ouvert, section hors tension."""
    res = _run()
    sect = [m for m in res.manoeuvres
            if m.action == "OPEN" and "sectionnement" in m.raison.lower()]
    assert sect, "Un sectionnement de barre doit être ouvert"
    assert any("SS.1.34" in m.switch_id for m in sect), \
        f"SS.1.34 attendu, organes ouverts: {[m.switch_id for m in sect]}"
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
    # Avant : pas de boucle longue
    assert all(m.type_boucle != "LONGUE" for m in avant)
    # Après : on trouve des boucles longues (mise HT / bascule SA / remise ST)
    assert any(m.type_boucle == "LONGUE" for m in apres)


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


def test_departs_du_3eme_noeud_en_boucle_longue():
    """Les départs du 3ème nœud (section 1.4 isolée) sont ré-aiguillés en
    boucle longue, donc avec ouverture/fermeture de leur disjoncteur."""
    res = _run()
    longues = [m for m in res.manoeuvres if m.type_boucle == "LONGUE"]
    # Au moins un DJ ouvert puis refermé pour les départs du nœud 2
    opens = [m for m in longues if m.action == "OPEN" and "hors tension" in m.raison]
    closes = [m for m in longues if m.action == "CLOSE" and "sous tension" in m.raison]
    assert opens and closes
    assert len(opens) == len(closes)


def test_nb_manoeuvres_coherent():
    """Le nombre de manœuvres calculé correspond à la séquence de référence."""
    seq = _load_sequence()
    res = _run()
    assert res.nb_manoeuvres == seq["nb_manoeuvres"]


def test_jamais_de_pont_court_circuit():
    """Invariant de sécurité : en rejouant la séquence, aucun sectionneur ne
    ponte deux barres qui ne seraient pas déjà au même potentiel par ailleurs.
    Un départ ne « court-circuite » jamais deux barres déconnectées (en boucle
    courte, le pont temporaire est sûr car les barres sont reliées par le
    couplage fermé)."""
    import networkx as nx
    from expert_op4grid_recommender.manoeuvre import algo

    seq = _load_sequence()
    poste = PosteTopologique.from_graph(_graph_from_states(seq["depart"]), VL)
    cible_graph = _graph_from_states(seq["cible"])
    res = determiner_manoeuvres_cible_detaillee(poste, cible_graph)
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
            internes = [n for n in cell.all_nodes if n not in cell.busbar_nodes]
            H2 = H.copy()
            H2.remove_nodes_from(internes)
            comp = (nx.node_connected_component(H2, wired[0])
                    if wired[0] in H2 else set())
            assert all(bb in comp for bb in wired), (
                f"{eq}: pont entre barres déconnectées {wired} après "
                f"{m.action} {m.switch_id} (court-circuit)")


def test_sequence_correspond_a_la_reference():
    """La séquence calculée correspond à la séquence de référence sauvegardée
    (même organes, mêmes actions, même ordre)."""
    seq = _load_sequence()
    res = _run()
    ref_manoeuvres = seq["manoeuvres"]
    assert len(res.manoeuvres) == len(ref_manoeuvres), (
        f"Nombre de manœuvres : {len(res.manoeuvres)} calculé "
        f"vs {len(ref_manoeuvres)} référence")
    for i, (calc, ref) in enumerate(zip(res.manoeuvres, ref_manoeuvres)):
        assert calc.switch_id == ref["switch_id"], (
            f"Manœuvre {i+1}: switch {calc.switch_id} != {ref['switch_id']}")
        assert calc.action == ref["action"], (
            f"Manœuvre {i+1}: action {calc.action} != {ref['action']}")
