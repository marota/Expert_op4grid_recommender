"""
tests/manoeuvre/test_muhlbp7_sequence_minimale.py
-------------------------------------------------
**Non-régression « manœuvres inutiles »** sur ``MUHLBP7`` (poste 400 kV à 5 jeux
de barres). La cible (``scenarios/MUHLBP7_cible.json``) n'est qu'un **petit delta**
de l'état de départ : ré-aiguiller 2 départs (SIERE.1, SIERE.2) + fermer un
sectionnement (4.1_4.2). La séquence experte de référence
(``sequences/MUHLBP7_cible_expert.json``) y suffit en **9 manœuvres**.

Avant le candidat *diff minimal* (cf. ``algo/targets.py``), la voie placement +
alignement produisait **37 manœuvres** : elle **fermait** une dizaine de couplers
(LIAIS/COUPL/MC…) pour reconstruire la partition, puis les **rouvrait** à
l'alignement (ferme-puis-rouvre inutile). Ce test verrouille le correctif :

1. la cible détaillée est atteinte **exactement** (``is_verified_detaillee``) ;
2. en **au plus** le nombre de manœuvres de la séquence experte (9) ;
3. **aucun organe de couplage n'est manœuvré inutilement** : tout organe manipulé
   doit changer d'état entre le départ et la cible — seuls les **DJ de cellule de
   départ** font l'aller-retour (ré-aiguillage boucle longue : ouvrir puis refermer).

Sans dépendance pypowsybl (fixture). VL ``MUHLBP7`` (5 JdB → voie multi-barres).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    determiner_manoeuvres_cible_detaillee,
)
from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _set_switch, _is_open

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

_SCEN = pathlib.Path(__file__).parent / "scenarios" / "MUHLBP7_cible.json"
_EXPERT = pathlib.Path(__file__).parent / "sequences" / "MUHLBP7_cible_expert.json"
_STEM = "MUHLBP7"


def _graph(states: dict):
    G = build_graph_from_fixture(_STEM)
    for sid, op in states.items():
        _set_switch(G, sid, op)
    return G


def _skip_if_absent():
    if not _SCEN.exists() or _STEM not in list_available_fixtures():
        pytest.skip("Scénario/fixture MUHLBP7 absent")


@pytest.mark.parametrize("mode", ["smooth", "aggressive"])
def test_muhlbp7_pas_de_manoeuvres_inutiles(mode):
    _skip_if_absent()
    d = json.loads(_SCEN.read_text())
    vl = d["voltage_level_id"]
    poste = PosteTopologique.from_graph(_graph(d["depart"]), vl)
    depart_G = _graph(d["depart"])
    cible_G = _graph(d["cible"])

    res = determiner_manoeuvres_cible_detaillee(poste, _graph(d["cible"]), mode=mode)

    # (1) cible détaillée atteinte exactement.
    assert res.is_verified_detaillee, f"{mode}: écarts {res.ecarts}"
    assert res.ecarts == []

    # (2) au plus le nombre de manœuvres de la séquence experte (réf. 9).
    n_expert = (len(json.loads(_EXPERT.read_text())["manoeuvres"])
                if _EXPERT.exists() else 9)
    assert res.nb_manoeuvres <= n_expert, (
        f"{mode}: {res.nb_manoeuvres} manœuvres > experte {n_expert} "
        f"— régression « ferme-puis-rouvre » de couplers ?")

    # (3) aucun organe manœuvré inutilement : tout organe touché doit changer
    #     d'état entre départ et cible, SAUF les DJ de cellule de départ (qui font
    #     l'aller-retour du ré-aiguillage boucle longue).
    diff = {sid for _u, _v, dd in cible_G.edges(data=True)
            if (sid := dd.get("switch_id")) is not None
            and _is_open(depart_G, sid) != _is_open(cible_G, sid)}
    dj_depart = {sw.switch_id for c in poste.cellules.cellules_depart
                 for sw in c.breakers}
    for m in res.manoeuvres:
        if m.switch_id in dj_depart:
            continue  # aller-retour de DJ (ré-aiguillage) : légitime
        assert m.switch_id in diff, (
            f"{mode}: organe '{m.switch_id}' manœuvré inutilement "
            f"({m.action}) — il ne diffère pas entre départ et cible")


def test_muhlbp7_couplers_non_touches_inutilement():
    """Les **faisceaux de couplage** (LIAIS/COUPL/MC…) qui ne changent pas d'état
    entre départ et cible ne doivent **jamais** être manœuvrés (cœur du correctif
    : plus de fermeture-puis-réouverture pour reconstruire la partition)."""
    _skip_if_absent()
    from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _inter_sjb_couplers
    d = json.loads(_SCEN.read_text())
    vl = d["voltage_level_id"]
    poste = PosteTopologique.from_graph(_graph(d["depart"]), vl)
    depart_G, cible_G = _graph(d["depart"]), _graph(d["cible"])

    coupler_sids = {s for cp in _inter_sjb_couplers(poste) for s in cp.switch_ids}
    inchanges = {s for s in coupler_sids
                 if _is_open(depart_G, s) == _is_open(cible_G, s)}

    res = determiner_manoeuvres_cible_detaillee(poste, _graph(d["cible"]), mode="smooth")
    touches_inutiles = sorted({m.switch_id for m in res.manoeuvres} & inchanges)
    assert not touches_inutiles, (
        f"Couplers manœuvrés inutilement (état inchangé entre départ et cible) : "
        f"{touches_inutiles}")
