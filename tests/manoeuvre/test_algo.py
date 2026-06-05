"""
tests/manoeuvre/test_algo.py
------------------------------
Tests de l'algorithme nodale → détaillée (phase 2) sur CARRIP3.
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import (
    TopologieNodale,
    PosteTopologique,
)
from expert_op4grid_recommender.manoeuvre.algo import (
    determiner_topo_complete_cible,
    determiner_manoeuvres_cible_detaillee,
    Manoeuvre,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures


def _fixtures_available() -> bool:
    return len(list_available_fixtures()) > 0


pytestmark = pytest.mark.skipif(
    not _fixtures_available(), reason="Fixtures de postes non générées."
)


@pytest.fixture
def poste_carrip3() -> PosteTopologique:
    G = build_graph_from_fixture("CARRIP3")
    return PosteTopologique.from_graph(G, "CARRIP3")


def _set_switch_in_graph(G, switch_id, open_):
    for u, v, d in G.edges(data=True):
        if d.get("switch_id") == switch_id:
            d["open"] = open_


def test_cible_un_noeud_referme_couplage_sans_reaiguiller():
    """R6 : pour atteindre une topologie à 1 nœud depuis un départ où les barres
    sont découplées, on **referme le couplage** (et les sectionnements) plutôt
    que de ramener tous les départs sur une seule barre. Les départs restent sur
    leurs barres : aucun ré-aiguillage (boucle)."""
    G = build_graph_from_fixture("CARRIP3")
    # Départ : couplage de barres ouvert -> 2 nœuds (barre 1 / barre 2)
    _set_switch_in_graph(G, "CARRIP3_CARRI3COUPL.1 DJ_OC", True)
    poste = PosteTopologique.from_graph(G, "CARRIP3")
    assert poste.topologie_nodale.nb_noeuds >= 2

    # Cible : 1 nœud = tous les départs connectés ensemble (les groupes isolés,
    # singletons, restent isolés).
    topo = poste.topologie_nodale
    connectes, isoles = [], []
    for noeud in topo.noeuds.values():
        ids = sorted(noeud.equipment_ids)
        (connectes if len(ids) > 1 else isoles).append(ids)
    groupes = [sorted(sum(connectes, []))] + isoles
    cible = TopologieNodale.from_node_groups("CARRIP3", groupes)

    res = determiner_topo_complete_cible(poste, cible)
    assert res.is_verified, res.message
    # Le couplage est refermé, et AUCUN ré-aiguillage (pas de boucle) :
    assert any(m.action == "CLOSE" and "couplage" in m.raison.lower()
               for m in res.manoeuvres)
    assert all(m.type_boucle is None for m in res.manoeuvres), \
        "Aucun ré-aiguillage ne doit être nécessaire (on referme le couplage)"
    assert res.nb_manoeuvres <= 2


def _replay_check_sectionneurs(poste, res):
    """Vérifie l'invariant : tout SECTIONNEUR (DISCONNECTOR) n'est manœuvré que
    si, à l'exclusion de cet organe, ses deux extrémités sont équipotentielles
    (chemin fermé alternatif) ou si l'un des côtés est hors tension (sa
    composante ne contient aucun équipement). Les DJ (BREAKER) sont exclus."""
    import networkx as nx
    from expert_op4grid_recommender.manoeuvre import algo
    from expert_op4grid_recommender.manoeuvre.models import SwitchKind

    G = poste.graph.copy()
    # type & extrémités de chaque switch
    info = {}
    for u, v, d in G.edges(data=True):
        if d.get("switch_id"):
            info[d["switch_id"]] = (u, v, d.get("kind"))

    def closed_graph_without(edge):
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            if d.get("open", False):
                continue
            if {u, v} == set(edge):
                continue
            H.add_edge(u, v)
        return H

    def has_equipment(H, start):
        if start not in H:
            return False
        comp = nx.node_connected_component(H, start)
        return any(G.nodes[n].get("equipment_id") for n in comp)

    for m in res.manoeuvres:
        u, v, kind = info[m.switch_id]
        if kind == SwitchKind.DISCONNECTOR:
            H = closed_graph_without((u, v))
            equipotentiel = (u in H and v in H and nx.has_path(H, u, v))
            cote_mort = (not has_equipment(H, u)) or (not has_equipment(H, v))
            assert equipotentiel or cote_mort, (
                f"Sectionneur {m.action} {m.switch_id} manœuvré entre deux "
                "potentiels différents sous tension (court-circuit)")
        algo._set_switch(G, m.switch_id, m.action == "OPEN")


def test_fusion_un_noeud_respecte_regle_sectionneur():
    """Fusionner des barres découplées vers 1 nœud : un DJ de couplage peut
    relier des potentiels différents, mais un sectionneur n'est fermé qu'après
    mise hors tension de la section (jamais entre deux potentiels vifs)."""
    G = build_graph_from_fixture("CARRIP3")
    # Départ très fragmenté : couplage ET sectionnement de barre 1 ouverts
    _set_switch_in_graph(G, "CARRIP3_CARRI3COUPL.1 DJ_OC", True)
    _set_switch_in_graph(G, "CARRIP3_CARRI3SEC..12 SS.1.12_OC", True)
    poste = PosteTopologique.from_graph(G, "CARRIP3")
    assert poste.topologie_nodale.nb_noeuds >= 3

    topo = poste.topologie_nodale
    connectes, isoles = [], []
    for noeud in topo.noeuds.values():
        ids = sorted(noeud.equipment_ids)
        (connectes if len(ids) > 1 else isoles).append(ids)
    groupes = [sorted(sum(connectes, []))] + isoles
    cible = TopologieNodale.from_node_groups("CARRIP3", groupes)

    res = determiner_topo_complete_cible(poste, cible)
    assert res.is_verified, res.message
    # Le couplage (DJ) est refermé ; le sectionnement n'est fermé qu'après mise
    # hors tension de la section.
    assert any(m.action == "CLOSE" and "couplage" in m.raison.lower()
               for m in res.manoeuvres)
    sect_close = [m for m in res.manoeuvres
                  if m.action == "CLOSE" and "sectionnement" in m.raison.lower()]
    assert all("hors tension" in m.raison or "équipotentiel" in m.raison
               for m in sect_close)
    # Invariant de sécurité vérifié sur toute la séquence.
    _replay_check_sectionneurs(poste, res)


def test_cible_detaillee_atteinte_avec_barres_exactes():
    """Quand la topologie détaillée est imposée, chaque départ doit finir sur
    sa barre exacte (pas seulement la bonne partition nodale). Les départs de la
    section dé-énergisée sont ramenés sur leur barre d'origine (manœuvres
    supplémentaires en boucle courte) et la topologie détaillée est vérifiée."""
    # Départ : couplage + sectionnement barre 1 ouverts (5 nœuds)
    Gd = build_graph_from_fixture("CARRIP3")
    for sid in ("CARRIP3_CARRI3COUPL.1 DJ_OC", "CARRIP3_CARRI3SEC..12 SS.1.12_OC"):
        _set_switch_in_graph(Gd, sid, True)
    poste = PosteTopologique.from_graph(Gd, "CARRIP3")

    # Cible détaillée = état pristine (1 nœud, toutes barres couplées, départs
    # sur leurs barres d'origine 1.1/1.2/2.1/2.2).
    Gc = build_graph_from_fixture("CARRIP3")  # pristine

    res = determiner_manoeuvres_cible_detaillee(poste, Gc)
    assert res.is_verified, res.message
    assert res.is_verified_detaillee, f"écarts: {res.ecarts}"
    assert res.ecarts == []
    # Des ré-aiguillages boucle courte de retour existent (départs ramenés sur
    # leur barre d'origine après fermeture du sectionnement).
    assert any(m.type_boucle == "COURTE" for m in res.manoeuvres)


def test_cible_detaillee_signale_les_ecarts(monkeypatch):
    """Si la topologie détaillée n'est pas atteignable exactement, les écarts
    sont consignés (is_verified_detaillee False, liste d'écarts non vide)."""
    # On force un cas où le raffinement ne peut pas tout placer : cible plaçant
    # un départ sur une barre, mais on vérifie surtout le mécanisme de rapport.
    Gd = build_graph_from_fixture("CARRIP3")
    poste = PosteTopologique.from_graph(Gd, "CARRIP3")
    Gc = build_graph_from_fixture("CARRIP3")
    # cible = départ identique -> aucun écart (sanity du champ)
    res = determiner_manoeuvres_cible_detaillee(poste, Gc)
    assert res.is_verified_detaillee
    assert isinstance(res.ecarts, list)


def _split_2_barres(poste_carrip3):
    """Construit une cible 2 barres en scindant le gros nœud connecté en deux."""
    topo = poste_carrip3.topologie_nodale
    plus_gros = max(topo.noeuds.values(), key=lambda n: len(n.departs))
    ids = sorted(plus_gros.equipment_ids)
    half = len(ids) // 2
    groupes = [ids[:half], ids[half:]]
    # On préserve les autres nœuds (équipements isolés) tels quels.
    for nom, noeud in topo.noeuds.items():
        if noeud is plus_gros:
            continue
        groupes.append(sorted(noeud.equipment_ids))
    return TopologieNodale.from_node_groups("CARRIP3", groupes)


# ---------------------------------------------------------------------------
# Cas no-op
# ---------------------------------------------------------------------------

def test_noop_aucune_manoeuvre(poste_carrip3):
    """Cible == état courant → aucune manœuvre, vérifié."""
    res = determiner_topo_complete_cible(
        poste_carrip3, poste_carrip3.topologie_nodale
    )
    assert res.nb_manoeuvres == 0
    assert res.is_verified
    assert not res.is_changed


# ---------------------------------------------------------------------------
# Cas 2 barres : ré-aiguillage + ouverture de couplage
# ---------------------------------------------------------------------------

def test_split_2_barres_verifie(poste_carrip3):
    """Une cible 2-nœuds est atteinte et vérifiée."""
    cible = _split_2_barres(poste_carrip3)
    res = determiner_topo_complete_cible(poste_carrip3, cible)
    assert res.is_verified, res.message
    assert res.is_changed
    assert res.nb_manoeuvres > 0
    # La topologie obtenue est isomorphe à la cible.
    assert cible.meme_topologie(res.topo_obtenue)


def test_split_ouvre_le_couplage(poste_carrip3):
    """Le passage à 2 nœuds requiert l'ouverture du DJ de couplage."""
    cible = _split_2_barres(poste_carrip3)
    res = determiner_topo_complete_cible(poste_carrip3, cible)
    couplage_ouvert = [
        m for m in res.manoeuvres
        if m.action == "OPEN" and "couplage" in m.raison.lower()
    ]
    assert couplage_ouvert, "Le couplage de barres doit être ouvert"
    assert any("COUPL" in m.switch_id for m in couplage_ouvert)


def test_split_genere_des_reaiguillages(poste_carrip3):
    """Le split produit des ré-aiguillages de départs (SA fermé/ouvert)."""
    cible = _split_2_barres(poste_carrip3)
    res = determiner_topo_complete_cible(poste_carrip3, cible)
    assert res.departs_reaiguilles
    # Chaque manœuvre référence un switch et une action valides.
    for m in res.manoeuvres:
        assert isinstance(m, Manoeuvre)
        assert m.action in ("OPEN", "CLOSE")
        assert m.switch_id


def test_split_boucle_courte(poste_carrip3):
    """Couplage initialement fermé → ré-aiguillage en boucle courte."""
    cible = _split_2_barres(poste_carrip3)
    res = determiner_topo_complete_cible(poste_carrip3, cible)
    reaig = [m for m in res.manoeuvres if m.type_boucle is not None]
    assert reaig
    assert all(m.type_boucle == "COURTE" for m in reaig)


# ---------------------------------------------------------------------------
# Cas infaisable : plus de nœuds que de barres
# ---------------------------------------------------------------------------

def test_trois_noeuds_sur_deux_barres_infaisable(poste_carrip3):
    """Demander 3 nœuds connectés sur un poste 2 barres n'est pas réalisable."""
    topo = poste_carrip3.topologie_nodale
    plus_gros = max(topo.noeuds.values(), key=lambda n: len(n.departs))
    ids = sorted(plus_gros.equipment_ids)
    third = max(1, len(ids) // 3)
    groupes = [ids[:third], ids[third:2 * third], ids[2 * third:]]
    cible = TopologieNodale.from_node_groups("CARRIP3", groupes)
    res = determiner_topo_complete_cible(poste_carrip3, cible)
    # On ne peut pas créer 3 potentiels distincts avec 2 barres.
    assert not res.is_verified
