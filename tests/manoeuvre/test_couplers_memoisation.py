"""
tests/manoeuvre/test_couplers_memoisation.py
----------------------------------------------
**Filet de sécurité ciblé pour le refactor « mémoïsation de
``_inter_sjb_couplers`` »** (#2).

``_inter_sjb_couplers(poste)`` est aujourd'hui recalculé ~10× par analyse. #2
le mémoïsera sur le ``PosteTopologique``. Trois **préconditions** rendent un tel
cache correct ; ce module les fige (elles ne l'étaient pas) :

1. **Invariance à l'état des organes** : le résultat ne dépend que de la
   *topologie* du poste (graphe + tronçonnement), pas de l'état ouvert/fermé des
   switches. Inverser tous les organes ne change pas les liaisons inter-SJB.
   → c'est ce qui autorise à cacher sur le poste sans clé d'état.
2. **Pureté** : l'appel ne mute pas le poste ; deux appels successifs renvoient
   une structure identique (idempotence).
3. **Non-mutation de ``poste.graph`` par le pipeline** : les points d'entrée
   publics travaillent sur une *copie* du graphe ; ``poste.graph`` reste intact,
   donc un résultat caché reste valide pendant toute l'analyse.

Plus une caractérisation unitaire de la détection des **couplages parallèles**
(MORBRP6 : ``COUPL.B`` masqué par la liaison ``SELF.1``), structure subtile que
le cache devra reproduire à l'identique.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import PosteTopologique
from expert_op4grid_recommender.manoeuvre.algo import (
    _inter_sjb_couplers,
    determiner_manoeuvres_cible_detaillee,
)

from .fixture_loader import build_graph_from_fixture, list_available_fixtures

SCEN_DIR = Path(__file__).parent / "scenarios"

# Postes (≥ 2 jeux de barres) portant des liaisons inter-SJB à recenser.
POSTES = ["CARRIP3", "CARRIP6", "CZTRYP6", "PALUNP3", "SSAVOP3", "MORBRP6"]


def _available(postes):
    fx = set(list_available_fixtures())
    return [p for p in postes if p in fx]


def _flip_all_switches(G):
    """Inverse l'état ouvert/fermé de **tous** les switches réels (les internal
    connections, ``switch_id`` None, restent fermées)."""
    for _u, _v, d in G.edges(data=True):
        if d.get("switch_id") is not None:
            d["open"] = not d.get("open", False)
    return G


def _switch_state(G) -> dict:
    return {d["switch_id"]: bool(d.get("open", False))
            for _u, _v, d in G.edges(data=True) if d.get("switch_id")}


def _canon_couplers(couplers) -> list[tuple]:
    """Forme canonique, comparable et indépendante de l'ordre, d'une liste de
    liaisons inter-SJB (gère les couplages parallèles via ``switch_ids``)."""
    return sorted(
        (cp.sjb_a, cp.sjb_b,
         tuple(sorted(cp.switch_ids)), tuple(sorted(cp.breaker_ids)),
         cp.is_sectionnement)
        for cp in couplers
    )


@pytest.mark.parametrize("vl", _available(POSTES))
def test_couplers_invariants_a_l_etat_des_organes(vl):
    """Précondition #1 du cache : les liaisons inter-SJB ne dépendent QUE de la
    topologie. Inverser tous les organes ne doit rien changer."""
    poste_a = PosteTopologique.from_graph(build_graph_from_fixture(vl), vl)
    poste_b = PosteTopologique.from_graph(
        _flip_all_switches(build_graph_from_fixture(vl)), vl)

    couplers_a = _canon_couplers(_inter_sjb_couplers(poste_a))
    couplers_b = _canon_couplers(_inter_sjb_couplers(poste_b))

    assert couplers_a == couplers_b, (
        f"{vl}: les couplers dépendent de l'état des organes — "
        "un cache sur le poste serait incorrect")


@pytest.mark.parametrize("vl", _available(POSTES))
def test_couplers_idempotents_et_purs(vl):
    """Préconditions #2 : appel pur (ne mute pas le poste) et idempotent (deux
    appels donnent la même structure)."""
    poste = PosteTopologique.from_graph(build_graph_from_fixture(vl), vl)
    avant = _switch_state(poste.graph)

    c1 = _canon_couplers(_inter_sjb_couplers(poste))
    c2 = _canon_couplers(_inter_sjb_couplers(poste))

    assert c1 == c2, f"{vl}: _inter_sjb_couplers non idempotent"
    assert _switch_state(poste.graph) == avant, \
        f"{vl}: _inter_sjb_couplers a muté poste.graph"


def test_couplers_paralleles_detectes_morbrp6():
    """Caractérisation : MORBRP6 détecte le **couplage parallèle** ``COUPL.B``
    (masqué par la liaison ``SELF.1``). Le cache devra reproduire cette
    structure ; on pin sa présence au niveau unitaire."""
    if "MORBRP6" not in list_available_fixtures():
        pytest.skip("Fixture MORBRP6 absente")
    poste = PosteTopologique.from_graph(
        build_graph_from_fixture("MORBRP6"), "MORBRP6")
    couplers = _inter_sjb_couplers(poste)

    tous_sids = {sid for cp in couplers for sid in cp.switch_ids}
    assert any("COUPL.B" in sid for sid in tous_sids), \
        "le couplage parallèle COUPL.B n'est pas recensé"
    # Au moins une paire de SJB reliée par ≥ 2 liaisons (parallélisme effectif).
    from collections import Counter
    paires = Counter((cp.sjb_a, cp.sjb_b) for cp in couplers)
    assert any(n >= 2 for n in paires.values()), \
        "aucune liaison parallèle détectée sur MORBRP6"


# ---------------------------------------------------------------------------
# Précondition #3 : le pipeline ne mute pas poste.graph
# ---------------------------------------------------------------------------

def _graph_from_states(vl: str, states: dict):
    G = build_graph_from_fixture(vl)
    for _u, _v, d in G.edges(data=True):
        sid = d.get("switch_id")
        if sid in states:
            d["open"] = states[sid]
    return G


def _scenarios_pipeline():
    names = ["CARRIP3_cible_3noeuds", "PALUNP3_cible_4noeuds",
             "MORBRP6_cible_4noeuds"]
    out = []
    for n in names:
        p = SCEN_DIR / f"{n}.json"
        if p.exists():
            out.append(p)
    return out


@pytest.mark.parametrize("path", _scenarios_pipeline(), ids=lambda p: p.stem)
@pytest.mark.parametrize("mode", ["smooth", "aggressive"])
def test_pipeline_ne_mute_pas_poste_graph(path, mode):
    """Précondition #3 du cache : ``determiner_manoeuvres_cible_detaillee`` ne
    doit pas altérer ``poste.graph`` (il travaille sur une copie). Un résultat de
    ``_inter_sjb_couplers`` mémoïsé sur le poste reste donc valide tout du long."""
    d = json.loads(path.read_text())
    vl = d["voltage_level_id"]
    if vl not in list_available_fixtures():
        pytest.skip(f"Fixture {vl} absente")
    poste = PosteTopologique.from_graph(_graph_from_states(vl, d["depart"]), vl)
    cible_graph = _graph_from_states(vl, d["cible"])

    avant = _switch_state(poste.graph)
    determiner_manoeuvres_cible_detaillee(poste, cible_graph, mode=mode)
    assert _switch_state(poste.graph) == avant, \
        f"{path.stem}/{mode}: le pipeline a muté poste.graph"
