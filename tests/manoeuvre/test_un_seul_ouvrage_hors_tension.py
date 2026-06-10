"""
tests/manoeuvre/test_un_seul_ouvrage_hors_tension.py
----------------------------------------------------
**Sémantique exacte** du vérificateur public ``ouvrages_simultanement_hors_tension``
(règle R10ter, bonne pratique du mode *smooth* : ne déconnecter **qu'un seul
ouvrage à la fois**), testée sur des **séquences synthétiques** déterministes
(sans pypowsybl) construites sur les DJ réels d'une fixture.

On vérifie qu'il compte une **coupure temporaire** (DJ propre initialement fermé,
ouvert **puis refermé**) ssi **plus d'un** ouvrage l'est simultanément, et qu'il
**exempte** : (a) les ouvrages **déjà déconnectés** au départ (DJ ouvert initial),
et (b) les ouvrages **mis hors service** (DJ final ouvert — coupure *définitive*,
pas temporaire). Cette sémantique distingue une vraie séquence « un par un » (→ 0
alerte) d'un *batch* de section (→ alertes), défaut historiquement non détecté.
"""
from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    ouvrages_simultanement_hors_tension,
)
from expert_op4grid_recommender.manoeuvre.algo.results import Manoeuvre
from expert_op4grid_recommender.manoeuvre.algo.graph_ops import (
    _inter_sjb_couplers, _set_switch)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

_STEM = "CARRIP3"


def _poste(open_djs: tuple[str, ...] = ()):
    """Poste CARRIP3 (pristine), avec d'éventuels DJ forcés ouverts au départ."""
    if _STEM not in list_available_fixtures():
        pytest.skip("Fixture CARRIP3 absente")
    G = build_graph_from_fixture(_STEM)
    for sid in open_djs:
        _set_switch(G, sid, True)
    return PosteTopologique.from_graph(G, _STEM)


def _deux_djs(poste):
    """Deux DJ d'ouvrage (hors couplage) de deux départs distincts."""
    coup = {s for cp in _inter_sjb_couplers(poste) for s in cp.switch_ids}
    djs = []
    for c in poste.cellules.cellules_depart:
        brk = [b.switch_id for b in c.breakers if b.switch_id not in coup]
        if brk:
            djs.append(brk[0])
        if len(djs) == 2:
            break
    if len(djs) < 2:
        pytest.skip("CARRIP3 : moins de 2 départs à DJ")
    return djs[0], djs[1]


def _O(sid):
    return Manoeuvre(sid, "OPEN", "test")


def _C(sid):
    return Manoeuvre(sid, "CLOSE", "test")


def test_un_par_un_aucune_alerte():
    """Déconnexion **un par un** (ouvrir/refermer avant le suivant) → 0 alerte."""
    poste = _poste()
    a, b = _deux_djs(poste)
    seq = [_O(a), _C(a), _O(b), _C(b)]
    assert ouvrages_simultanement_hors_tension(poste, seq) == []


def test_deux_simultanement_leve_une_alerte():
    """Deux ouvrages hors tension **en même temps** (ouvrir a, ouvrir b avant de
    refermer) → alerte levée, nommant les deux ouvrages."""
    poste = _poste()
    a, b = _deux_djs(poste)
    seq = [_O(a), _O(b), _C(a), _C(b)]
    viol = ouvrages_simultanement_hors_tension(poste, seq)
    assert viol, "deux ouvrages hors tension simultanément doivent être signalés"
    assert all("plus d'un ouvrage" in v for v in viol)


def test_ouvrage_deja_deconnecte_est_exempte():
    """Un ouvrage **déjà déconnecté** au départ (DJ ouvert initial) ne compte pas :
    ouvrir/refermer un autre ouvrage pendant ce temps → 0 alerte."""
    poste = _poste()
    a, b = _deux_djs(poste)
    poste_a_ouvert = _poste(open_djs=(a,))      # a déjà déconnecté au départ
    seq = [_O(b), _C(b)]                         # b temporaire ; a reste ouvert
    assert ouvrages_simultanement_hors_tension(poste_a_ouvert, seq) == []


def test_mise_hors_service_est_exemptee():
    """Un ouvrage **mis hors service** (DJ final **ouvert**, jamais refermé) est une
    coupure *définitive*, pas temporaire → exempté : il ne crée pas d'alerte même
    si un autre ouvrage est momentanément hors tension en même temps."""
    poste = _poste()
    a, b = _deux_djs(poste)
    # a ouvert et NON refermé (mise hors service) ; b temporaire (ouvert/refermé).
    seq = [_O(a), _O(b), _C(b)]
    assert ouvrages_simultanement_hors_tension(poste, seq) == []


def test_alerte_compte_chaque_moment_de_chevauchement():
    """Trois ouvrages temporaires se chevauchant → l'alerte est levée (≥1 moment),
    et disparaît dès qu'on revient à un seul à la fois."""
    poste = _poste()
    coup = {s for cp in _inter_sjb_couplers(poste) for s in cp.switch_ids}
    djs = [b.switch_id for c in poste.cellules.cellules_depart
           for b in c.breakers if b.switch_id not in coup][:3]
    if len(djs) < 3:
        pytest.skip("CARRIP3 : moins de 3 départs à DJ")
    a, b, c = djs
    # chevauchement (a et b ouverts ensemble) PUIS un-par-un (c seul).
    chevauche = [_O(a), _O(b), _C(a), _C(b)]
    propre = [_O(c), _C(c)]
    assert ouvrages_simultanement_hors_tension(poste, chevauche)
    assert ouvrages_simultanement_hors_tension(poste, propre) == []
