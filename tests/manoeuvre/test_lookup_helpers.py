"""
tests/manoeuvre/test_lookup_helpers.py
----------------------------------------
**Filet de sécurité ciblé pour le refactor « indexation des lookups »** (#1).

``_is_open`` / ``_set_switch`` / ``_eq_node`` retrouvent aujourd'hui un organe
ou un nœud par **scan linéaire** du graphe. #1 remplacera ces scans par un
index ``switch_id -> arête`` / ``equipment_id -> nœud`` porté par le poste.

Ces tests **figent le contrat observable** que l'index devra reproduire :

1. **Comportement sur identifiant inconnu** (mode d'échec actuel, *silencieux*) :
   ``_is_open`` → ``True`` (considéré ouvert) ; ``_set_switch`` → no-op sans
   exception ; ``_eq_node`` → ``None``.
   ⚠ Si #1 choisit de **durcir** ce contrat (lever ``KeyError`` sur id inconnu),
   ces trois tests doivent être mis à jour **consciemment** — c'est le but.
2. **Validité sur graphe dérivé** : l'index sera construit sur ``poste.graph``
   mais utilisé sur des **copies** (``poste.graph.copy()``, postes virtuels).
   Les coordonnées topologiques (nœuds, arêtes) sont stables par copie : les
   lookups doivent rester corrects, et muter une copie ne doit pas toucher
   l'original.
3. **Invariant d'intégrité** : tout ``switch_id`` émis dans une séquence existe
   réellement comme arête (sinon ``_set_switch`` no-op silencieusement et la
   séquence est faussée sans bruit).
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import (
    PosteTopologique,
    TopologieNodale,
)
from expert_op4grid_recommender.manoeuvre.algo import (
    determiner_topo_complete_cible,
    _is_open,
    _set_switch,
    _eq_node,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

VL = "CARRIP3"

pytestmark = pytest.mark.skipif(
    VL not in list_available_fixtures(), reason="Fixture CARRIP3 absente.")


def _graph():
    return build_graph_from_fixture(VL)


def _switch_ids(G) -> list[str]:
    return [d["switch_id"] for _u, _v, d in G.edges(data=True)
            if d.get("switch_id")]


def _equipment_ids(G) -> list[str]:
    return [d["equipment_id"] for _n, d in G.nodes(data=True)
            if d.get("equipment_id")]


# ---------------------------------------------------------------------------
# 1. Contrat sur identifiant inconnu (mode d'échec actuel, à figer)
# ---------------------------------------------------------------------------

def test_is_open_unknown_id_returns_true():
    """Contrat actuel : un organe inconnu est considéré **ouvert**."""
    G = _graph()
    assert _is_open(G, "ORGANE_INEXISTANT") is True


def test_set_switch_unknown_id_is_silent_noop():
    """Contrat actuel : ``_set_switch`` sur id inconnu ne lève pas et ne modifie
    **rien** (no-op silencieux). #1 peut choisir de durcir ce contrat — auquel
    cas ce test devra être actualisé sciemment."""
    G = _graph()
    before = {d["switch_id"]: d["open"]
              for _u, _v, d in G.edges(data=True) if d.get("switch_id")}
    _set_switch(G, "ORGANE_INEXISTANT", True)   # ne doit pas lever
    after = {d["switch_id"]: d["open"]
             for _u, _v, d in G.edges(data=True) if d.get("switch_id")}
    assert after == before


def test_eq_node_unknown_returns_none():
    G = _graph()
    assert _eq_node(G, "EQUIPEMENT_INEXISTANT") is None


# ---------------------------------------------------------------------------
# 2. Lookups corrects pour des identifiants réels
# ---------------------------------------------------------------------------

def test_is_open_reflects_state():
    """``_is_open`` reflète l'état stocké sur l'arête, dans les deux sens."""
    G = _graph()
    sid = _switch_ids(G)[0]
    _set_switch(G, sid, True)
    assert _is_open(G, sid) is True
    _set_switch(G, sid, False)
    assert _is_open(G, sid) is False


def test_set_switch_toggles_known_id():
    G = _graph()
    sid = _switch_ids(G)[0]
    _set_switch(G, sid, True)
    assert _is_open(G, sid) is True


def test_eq_node_finds_equipment():
    G = _graph()
    eq = _equipment_ids(G)[0]
    node = _eq_node(G, eq)
    assert node is not None
    assert G.nodes[node].get("equipment_id") == eq


# ---------------------------------------------------------------------------
# 3. Validité sur graphe dérivé (copie)
# ---------------------------------------------------------------------------

def test_lookups_valides_sur_copie_et_original_intact():
    """Muter une **copie** via ``_set_switch`` n'affecte pas l'original, et les
    lookups restent corrects sur la copie (coordonnées topologiques stables)."""
    G = _graph()
    sid = _switch_ids(G)[0]
    _set_switch(G, sid, False)            # état connu sur l'original
    H = G.copy()
    _set_switch(H, sid, True)             # on ne touche que la copie
    assert _is_open(H, sid) is True
    assert _is_open(G, sid) is False, "la copie ne doit pas affecter l'original"
    # equipment_id -> node reste valide sur la copie
    eq = _equipment_ids(G)[0]
    assert _eq_node(H, eq) == _eq_node(G, eq)


# ---------------------------------------------------------------------------
# 4. Invariant d'intégrité : tout switch_id émis existe
# ---------------------------------------------------------------------------

def _cible_fusion_un_noeud(poste):
    """Cible 1 nœud (départs connectés fusionnés) — produit une séquence non
    triviale de ré-aiguillages et de manœuvres de couplage."""
    connectes, isoles = [], []
    for noeud in poste.topologie_nodale.noeuds.values():
        ids = sorted(noeud.equipment_ids)
        (connectes if len(ids) > 1 else isoles).append(ids)
    groupes = [sorted(sum(connectes, []))] + isoles
    return TopologieNodale.from_node_groups(poste.voltage_level_id, groupes)


def test_sequence_switch_ids_existent_tous():
    """Tout ``switch_id`` d'une séquence produite correspond à une arête réelle
    (garde-fou contre le no-op silencieux de ``_set_switch``)."""
    G = _graph()
    # Couplage ouvert au départ -> ≥ 2 nœuds, fusion non triviale ensuite.
    _set_switch(G, "CARRIP3_CARRI3COUPL.1 DJ_OC", True)
    poste = PosteTopologique.from_graph(G, VL)
    res = determiner_topo_complete_cible(poste, _cible_fusion_un_noeud(poste))

    assert res.manoeuvres, "séquence attendue non vide"
    valides = set(_switch_ids(poste.graph))
    inconnus = sorted({m.switch_id for m in res.manoeuvres} - valides)
    assert not inconnus, f"switch_id émis sans arête correspondante : {inconnus}"


def test_rejeu_sur_copie_est_reproductible():
    """Rejouer la séquence sur une **copie** du graphe est déterministe et
    laisse l'original intact."""
    G = _graph()
    _set_switch(G, "CARRIP3_CARRI3COUPL.1 DJ_OC", True)
    poste = PosteTopologique.from_graph(G, VL)
    res = determiner_topo_complete_cible(poste, _cible_fusion_un_noeud(poste))

    def rejouer():
        H = poste.graph.copy()
        for m in res.manoeuvres:
            _set_switch(H, m.switch_id, m.action == "OPEN")
        return {d["switch_id"]: d["open"]
                for _u, _v, d in H.edges(data=True) if d.get("switch_id")}

    etat_initial = {d["switch_id"]: d["open"]
                    for _u, _v, d in poste.graph.edges(data=True)
                    if d.get("switch_id")}
    assert rejouer() == rejouer()                      # déterministe
    # L'original n'a pas bougé pendant les rejeux sur copie.
    assert {d["switch_id"]: d["open"]
            for _u, _v, d in poste.graph.edges(data=True)
            if d.get("switch_id")} == etat_initial
