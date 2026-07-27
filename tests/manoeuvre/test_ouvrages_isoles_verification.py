"""
tests/manoeuvre/test_ouvrages_isoles_verification.py
-----------------------------------------------------
Régression : un **ouvrage isolé** (déconnecté — composante sans barre) ne doit
pas faire échouer la vérification d'une cible nodale qui ne le mentionne pas.

Contexte du bug : ``TopologieNodale.from_graph`` matérialise **une entrée de
``noeuds`` par composante portant un équipement**, y compris les composantes
**sans barre** (ouvrages déconnectés). La cible nodale éditée dans l'IHM, elle,
ne décrit que les ouvrages à placer sur un nœud. La comparaison stricte de
partitions rendait donc *toute* cible « non réalisable » dès qu'un ouvrage était
déjà déconnecté au départ, d'où le faux avertissement de l'IHM :

    ⚠ Cible partiellement réalisable (obtenu 2 nœud(s) + 5 ouvrage(s) isolé(s)
      / visé 2 nœud(s))

alors que la cible était bel et bien atteinte et que les ouvrages déjà
déconnectés au départ le restaient dans la topologie cible.

Trois niveaux couverts :
1. cœur — ``TopologieNodale`` (``noeuds_isoles`` / ``nb_noeuds_reels`` /
   ``meme_topologie``), sur graphe NetworkX pur ;
2. algo — ``PlanificateurTopologie.identifier_topologie_detaillee`` ;
3. IHM  — ``Session.nodale_to_detaillee``.
"""

from __future__ import annotations

import importlib.util
import pathlib

import networkx as nx
import pytest

from expert_op4grid_recommender.manoeuvre.models import NodeType
from expert_op4grid_recommender.manoeuvre.topologie import TopologieNodale


# ---------------------------------------------------------------------------
# 1. Cœur — TopologieNodale (graphe NX pur, ni Flask ni pypowsybl)
# ---------------------------------------------------------------------------

def _graphe(deconnecte=("L3",)) -> nx.Graph:
    """Poste jouet : une barre, ``L1``/``L2`` raccordés, ``L3`` déconnecté."""
    G = nx.Graph()
    G.add_node(0, node_type=NodeType.BUSBAR_SECTION, equipment_id="BB")
    for i, eq in enumerate(("L1", "L2", "L3"), start=1):
        G.add_node(i, node_type=NodeType.EQUIPMENT, equipment_id=eq)
        G.add_edge(0, i, open=eq in deconnecte, switch_id=f"S{i}")
    return G


def test_from_graph_marque_les_composantes_sans_barre():
    topo = TopologieNodale.from_graph(_graphe(), "VL")
    isole = {nom for nom in topo.noeuds_isoles}
    assert len(isole) == 1
    (nom_isole,) = isole
    assert topo.noeuds[nom_isole].equipment_ids == {"L3"}
    # ``noeuds`` reste exhaustif (compatibilité : le placement s'appuie dessus).
    assert topo.nb_noeuds == 2
    assert topo.nb_noeuds_reels == 1


def test_from_node_groups_na_pas_de_noeud_isole():
    """Une cible construite depuis une partition n'a pas la notion d'isolé."""
    cible = TopologieNodale.from_node_groups("VL", [["L1", "L2"]])
    assert cible.noeuds_isoles == set()
    assert cible.nb_noeuds_reels == cible.nb_noeuds == 1


def test_partition_reste_exhaustive():
    """``partition()`` (utilisée par les goldens) est inchangée : elle inclut
    toujours les ouvrages isolés."""
    topo = TopologieNodale.from_graph(_graphe(), "VL")
    assert topo.partition() == {frozenset({"L1", "L2"}), frozenset({"L3"})}


def test_meme_topologie_ignore_un_isole_hors_cible():
    """Le cas du bug : la cible ne parle pas de ``L3`` (déjà déconnecté)."""
    obtenue = TopologieNodale.from_graph(_graphe(), "VL")
    cible = TopologieNodale.from_node_groups("VL", [["L1", "L2"]])
    assert obtenue.meme_topologie(cible)
    assert cible.meme_topologie(obtenue)          # symétrique


def test_meme_topologie_reste_stricte_si_la_cible_mentionne_lisole():
    """L'expert demande explicitement ``L3`` sur le nœud → reconnexion exigée,
    la cible n'est donc **pas** réalisée."""
    obtenue = TopologieNodale.from_graph(_graphe(), "VL")
    cible = TopologieNodale.from_node_groups("VL", [["L1", "L2", "L3"]])
    assert not obtenue.meme_topologie(cible)
    assert not cible.meme_topologie(obtenue)


def test_meme_topologie_isole_declare_comme_noeud_a_part():
    """Cible mentionnant ``L3`` comme nœud à lui seul : comparaison stricte,
    satisfaite (comportement historique préservé)."""
    obtenue = TopologieNodale.from_graph(_graphe(), "VL")
    cible = TopologieNodale.from_node_groups("VL", [["L1", "L2"], ["L3"]])
    assert obtenue.meme_topologie(cible)


def test_meme_topologie_detecte_une_vraie_difference_avec_un_isole():
    """La relaxation ne masque pas une partition réellement fausse."""
    obtenue = TopologieNodale.from_graph(_graphe(), "VL")
    cible = TopologieNodale.from_node_groups("VL", [["L1"], ["L2"]])
    assert not obtenue.meme_topologie(cible)


def test_meme_topologie_detecte_un_ouvrage_de_la_cible_qui_finit_isole():
    """``L2`` est visé sur un nœud mais se retrouve déconnecté → non réalisé."""
    obtenue = TopologieNodale.from_graph(_graphe(deconnecte=("L2", "L3")), "VL")
    cible = TopologieNodale.from_node_groups("VL", [["L1", "L2"]])
    assert not obtenue.meme_topologie(cible)


# ---------------------------------------------------------------------------
# 2 & 3. Algo + IHM (nécessitent pypowsybl / flask)
# ---------------------------------------------------------------------------

pytest.importorskip("flask")
pytest.importorskip("pypowsybl")

import pypowsybl as pp  # noqa: E402

from expert_op4grid_recommender.manoeuvre.topologie import (  # noqa: E402
    PosteTopologique,
)
from expert_op4grid_recommender.manoeuvre.plugins import (  # noqa: E402
    PlanificateurTopologie,
)

_IHM_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "scripts" / "manoeuvre_ihm.py")
VL = "S1VL2"


def _load_ihm():
    spec = importlib.util.spec_from_file_location("manoeuvre_ihm_iso_mod", _IHM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ihm = _load_ihm()


@pytest.fixture()
def session_avec_isole():
    """Session dont l'état de **départ** comporte un ouvrage déjà déconnecté.

    Renvoie ``(session, ouvrage_isolé, [départs connectés])``."""
    net = pp.network.create_four_substations_node_breaker_network()
    s = ihm.Session(net)
    s.load(VL)
    feeders = sorted({eq for g in s.groups_of(s.initial) for eq in g})
    iso_eq = feeders[0]
    s.initial = ihm.Session._isoler_dans_etat(s, s.initial, [iso_eq])
    s.current = dict(s.initial)
    s._graph_cache.clear()
    s._topo_cache.clear()
    assert s.nodale_state(s.initial)["isolated"] == [iso_eq]
    return s, iso_eq, [e for e in feeders if e != iso_eq]


def _cible_deux_noeuds(connectes):
    moitie = len(connectes) // 2
    return [connectes[:moitie], connectes[moitie:]]


def test_algo_identifie_la_cible_malgre_un_ouvrage_deja_deconnecte(session_avec_isole):
    s, _iso, connectes = session_avec_isole
    s.apply(s.initial)
    poste = PosteTopologique.from_graph(s._graph(s.initial), s.vl)
    cible = TopologieNodale.from_node_groups(VL, _cible_deux_noeuds(connectes))

    ident = PlanificateurTopologie().identifier_topologie_detaillee(poste, cible)

    assert ident.is_realisable, ident.message
    assert ident.noeuds_non_realisables == []


def test_ihm_pas_de_faux_avertissement_ouvrages_isoles(session_avec_isole):
    """Cas du bug, ouvrages isolés **déclarés** par l'IHM."""
    s, iso, connectes = session_avec_isole
    res = s.nodale_to_detaillee(_cible_deux_noeuds(connectes), [iso])

    assert res["is_verified"] is True
    assert res["nb_obtenu"] == res["nb_vise"] == 2
    assert res["nb_isoles"] == 1
    assert res["message"] == ""
    assert res["noeuds_non_realisables"] == []
    assert iso in res["nodale"]["isolated"]        # reste déconnecté


def test_ihm_ouvrages_deja_deconnectes_non_declares(session_avec_isole):
    """Même cas mais sans liste ``isolated`` : un ouvrage **déjà déconnecté au
    départ** et non replacé sur un nœud reste hors de la partition cible (il
    n'est plus réinjecté comme nœud « orphelin », qui gonflait ``nb_vise``)."""
    s, iso, connectes = session_avec_isole
    res = s.nodale_to_detaillee(_cible_deux_noeuds(connectes), [])

    assert res["is_verified"] is True
    assert res["nb_obtenu"] == res["nb_vise"] == 2
    assert res["nb_isoles"] == 1
    assert iso in res["nodale"]["isolated"]


def test_ihm_reconnexion_demandee_reste_dans_le_perimetre(session_avec_isole):
    """Glisser un ouvrage isolé sur un nœud = demande de **reconnexion** : il
    reste dans le périmètre de la cible (``nb_vise`` couvre tous les départs) et
    n'est pas re-déconnecté d'office."""
    s, iso, connectes = session_avec_isole
    groupes = _cible_deux_noeuds(connectes)
    groupes[0] = groupes[0] + [iso]
    res = s.nodale_to_detaillee(groupes, [])

    assert res["nb_vise"] == 2
    # La reconnexion d'un départ déconnecté est hors de portée de l'algo (limite
    # documentée) : le verdict est négatif — mais **expliqué**, jamais nu.
    assert res["is_verified"] is False
    assert res["message"]
    assert iso in res["nodale"]["isolated"]         # pas reconnecté


def test_ihm_verdict_negatif_reste_diagnostique(session_avec_isole):
    """Une cible réellement inatteignable garde son diagnostic. Avant le
    correctif, la branche « ouvrages isolés » de l'IHM renvoyait un
    avertissement **nu** (``message``/``ecarts``/``noeuds_non_realisables``
    forcés à vide)."""
    s, iso, connectes = session_avec_isole
    # Un nœud par départ : irréalisable sur un poste à 2 jeux de barres.
    res = s.nodale_to_detaillee([[eq] for eq in connectes], [iso])

    assert res["is_verified"] is False
    assert res["message"]
    assert res["noeuds_non_realisables"]
