"""
tests/manoeuvre/test_cztryp6_3noeuds.py
-----------------------------------------
Test de la **règle du sectionneur de barre** sur CZTRYP6 : créer un 3ème nœud
électrique en ouvrant un sectionnement de barre sur un poste à **4 sections**.

Règle métier vérifiée
~~~~~~~~~~~~~~~~~~~~~~~
Un sectionneur de barre ne se manœuvre que **hors charge**. Les départs du 3ème
nœud **restent** sur leur section cible (1.4) ; le **mode smooth optimisé** les
**dé-énergise en place** (clignotement DJ) plutôt que de les déplacer puis les
ramener :
1. ouvrir les **DJ d'ouvrage** des départs de la section à isoler (mise hors
   tension en place) ;
2. ouvrir le **sectionnement** de barre (sûr car la section est morte) ;
3. **refermer** ces DJ (ré-alimentation sur la même section) ;
4. ouvrir le **couplage** (DJ) pour séparer les barres.
Aucun ré-aiguillage aller-retour (plus de double-déplacement). Le ré-aiguillage
**boucle longue** (R9) reste démontré par les départs qui **changent** de barre
(cf. ``test_noviop3_3noeuds.py``).

Cible (réalisable sur ce poste 2 barres / 8 SJB — 4 sections par barre)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- nœud 0 : barre 1 sections 1–3 (CHESNL61CZTRY, CHESNL63CZTRY, CZTRYY633)
- nœud 1 : barre 2 sections 1–4 (CHESNL62CZTRY, CZTRYL61MOISE, CZTRYY632)
- nœud 2 : barre 1 section 4 isolée (CZTRYL61PLISO, CZTRYL61SENAR, CZTRYY634)

La séquence de référence (8 manœuvres, mode smooth optimisé) est sauvegardée dans
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


def test_departs_du_3eme_noeud_de_energises_en_place():
    """Mode smooth optimisé : les départs du 3ème nœud (section 1.4) **restent**
    sur leur barre cible et sont **dé-énergisés en place** (clignotement DJ : DJ
    ouvert avant l'ouverture du sectionneur, refermé après) — sans ré-aiguillage
    aller-retour inutile (plus de double-déplacement)."""
    res = _run()
    idx_sect = next(i for i, m in enumerate(res.manoeuvres)
                    if "sectionnement" in m.raison.lower())
    avant = res.manoeuvres[:idx_sect]
    apres = res.manoeuvres[idx_sect:]
    # Avant le sectionneur : ouverture des DJ des départs du 3ème nœud (mise HT).
    mises_ht = [m for m in avant
                if m.action == "OPEN" and "hors tension" in m.raison]
    assert len(mises_ht) >= 1, "Les départs de la section sont dé-énergisés avant"
    # Après : refermeture de ces mêmes DJ (remise sous tension), même nombre.
    remises = [m for m in apres
               if m.action == "CLOSE" and "sous tension" in m.raison]
    assert len(remises) == len(mises_ht), (
        "Chaque départ dé-énergisé est ré-alimenté après l'ouverture du sectionneur")
    # Ce sont bien des DJ d'ouvrage (clignotement), pas des SA.
    assert all("DJ" in m.switch_id for m in mises_ht + remises)


def test_pas_de_double_deplacement():
    """Optimisation smooth : aucun ouvrage n'est déplacé puis ramené — aucun DJ
    n'est manœuvré plus de deux fois (un clignotement = 1 OPEN + 1 CLOSE)."""
    res = _run()
    from collections import Counter
    dj = Counter(m.switch_id for m in res.manoeuvres if "DJ" in m.switch_id)
    pires = {k: v for k, v in dj.items() if v > 2}
    assert not pires, f"DJ manœuvrés plus de 2 fois (double-déplacement) : {pires}"


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
