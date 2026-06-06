"""
tests/manoeuvre/test_ssv_op7_3jdb.py
------------------------------------
Cas d'étude **3 jeux de barres** sur un poste 400 kV réel : ``SSV.OP7``
(extrait du réseau France 28/08/2024). Contrairement aux fixtures
``postes_departs_multiples`` (CORNIP3/GUARBP6/MORBRP6 — en réalité *4* barres
« double-barre + 2 barres de transfert »), ``SSV.OP7`` est un **vrai poste à
3 jeux de barres** : SJB nommées ``_1A/_1B``, ``_2A/_2B``, ``_3A/_3B`` →
entiers de tête {1, 2, 3}.

Topologie (état initial du snapshot) :
- 3 barres × 2 demi-rames (A/B), graphe de couplage en **triangle** (toute
  paire de barres est couplable) ;
- 14 départs 400 kV, répartis 7/7 sur les barres 1 et 2 ; **barre 3 = réserve**
  (aucun départ), tous les départs atteignent les 3 barres.

Ces tests pinnent le comportement **cible** de l'Étape 1+2 (placement
automatique généralisé à N jeux de barres) : ``determiner_topo_complete_cible``
doit réaliser une cible nodale à **3 nœuds** sur ce poste, sans dégradation
« niveaux de barres supplémentaires ».
"""
from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    TopologieNodale,
    determiner_topo_complete_cible,
)
from expert_op4grid_recommender.manoeuvre.algo.placement import _placement_automatique

from .fixture_loader import (
    build_graph_from_fixture,
    get_fixture_metadata,
    list_available_fixtures,
)

FIXTURE = "SSV_OP7"

pytestmark = pytest.mark.skipif(
    FIXTURE not in list_available_fixtures(),
    reason="Fixture SSV_OP7 (poste 400 kV à 3 jeux de barres) absente.",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _poste() -> PosteTopologique:
    """Construit le poste avec le **vrai** voltage_level_id (avec point) pour
    que la détection des barres par nommage fonctionne (cf. bug RAN.PP6 :
    passer le *stem* underscore ferait retomber sur le repli structurel)."""
    vl = get_fixture_metadata(FIXTURE)["voltage_level_id"]
    return PosteTopologique.from_graph(build_graph_from_fixture(FIXTURE), vl)


def _switch_states(G):
    return {d["switch_id"]: bool(d.get("open", False))
            for _, _, d in G.edges(data=True) if d.get("switch_id")}


def _known_switch_ids(G):
    return {d.get("switch_id") for _, _, d in G.edges(data=True) if d.get("switch_id")}


def _departs_par_barre(poste):
    """{barre -> [equipment_id...]} d'après la barre actuellement câblée."""
    from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _wired_busbar
    bp = poste.tronconnement.barre_par_busbar
    out: dict[int, list[str]] = {}
    for c in poste.cellules.cellules_depart:
        wb = _wired_busbar(c, poste.graph)
        b = bp.get(wb) if wb is not None else None
        if b is not None:
            out.setdefault(b, []).append(c.equipment_id)
    return out


# --------------------------------------------------------------------------
# Pré-requis : le poste est bien un 3-JdB
# --------------------------------------------------------------------------

def test_ssv_op7_est_bien_un_poste_3_barres():
    poste = _poste()
    barres = set(poste.tronconnement.barre_par_busbar.values())
    assert len(barres) == 3, f"SSV.OP7 doit avoir 3 barres, obtenu {sorted(barres)}"
    # 14 départs, 6 SJB (3 barres × 2 demi-rames).
    assert len(poste.cellules.cellules_depart) == 14
    assert len(poste.tronconnement.barre_par_busbar) == 6


# --------------------------------------------------------------------------
# Étape 1+2 : cible nodale à 3 nœuds réalisée (plus de scoping 2-JdB)
# --------------------------------------------------------------------------

def _cible_3_noeuds(poste) -> TopologieNodale:
    """Construit une cible à 3 nœuds : on garde les 2 paquets courants (barres 1
    et 2) et on **isole un départ sur un 3ᵉ nœud** (qui devra atterrir sur la
    barre de réserve / une demi-rame). Sollicite le placement sur 3 barres."""
    par_barre = _departs_par_barre(poste)
    barres_peuplees = sorted(par_barre, key=lambda b: -len(par_barre[b]))
    assert len(barres_peuplees) >= 2
    g_a = list(par_barre[barres_peuplees[0]])
    g_b = list(par_barre[barres_peuplees[1]])
    assert len(g_a) >= 2, "besoin d'au moins 2 départs pour en détacher un"
    isole = g_a.pop()                       # un départ part sur le 3ᵉ nœud
    groups = [sorted(g_a), sorted(g_b), [isole]]
    return TopologieNodale.from_node_groups(poste.voltage_level_id, groups)


def test_placement_realise_la_cible_3_barres():
    """**Cœur de l'Étape 1.** Le placement automatique (``_placement_automatique``)
    réalise une cible nodale à 3 nœuds sur les **3 jeux de barres** — là où
    l'ancien scoping « 2 jeux de barres » abandonnait le 3ᵉ nœud à l'opérateur."""
    poste = _poste()
    cible = _cible_3_noeuds(poste)
    assert cible.nb_noeuds == 3

    placement, faisable, msg, non_places = _placement_automatique(poste, cible)

    # 1. Affectation complète et réalisable (plus de dégradation par scoping).
    assert faisable is True, msg
    assert msg == "OK"
    assert non_places == []
    assert len(placement) == 3

    # 2. Les 3 nœuds occupent des SJB disjointes couvrant les **3 barres**.
    bp = poste.tronconnement.barre_par_busbar
    node_par_sjbid = {poste.graph.nodes[n].get("busbar_section_id"): n for n in bp}
    barres_utilisees: set[int] = set()
    sjb_vues: set[str] = set()
    for _deps, sjbs in placement:
        assert sjbs, "chaque nœud occupe au moins une SJB"
        assert sjb_vues.isdisjoint(sjbs), "les nœuds occupent des SJB disjointes"
        sjb_vues |= sjbs
        barres_utilisees |= {bp[node_par_sjbid[s]] for s in sjbs}
    assert len(barres_utilisees) == 3, "les 3 jeux de barres sont exploités"


def test_determiner_topo_complete_cible_ne_degrade_plus_par_scoping():
    """À 3 barres, l'entrée nodale ne renvoie plus la dégradation « niveaux de
    barres supplémentaires » : le 3ᵉ jeu de barres est désormais géré."""
    poste = _poste()
    before = _switch_states(poste.graph)
    cible = _cible_3_noeuds(poste)

    res = determiner_topo_complete_cible(poste, cible)

    assert res.noeuds_non_realisables == []
    assert "niveaux de barres supplémentaires" not in res.message
    # Des manœuvres sont émises et référencent des organes existants.
    assert res.nb_manoeuvres > 0
    known = _known_switch_ids(poste.graph)
    assert all(m.switch_id in known for m in res.manoeuvres)
    # Invariant : le graphe du poste n'est pas muté.
    assert _switch_states(poste.graph) == before


@pytest.mark.xfail(
    reason="Réalisation SÉQUENCÉE d'une cible > 2 nœuds via un couplage "
           "multi-barres partagé (cellule LIAIS de SSV.OP7) : frontière du "
           "séquenceur (étape suivante), hors périmètre Étape 1+2 (placement).",
    strict=False,
)
def test_cible_3_noeuds_realisee_de_bout_en_bout():
    poste = _poste()
    cible = _cible_3_noeuds(poste)
    res = determiner_topo_complete_cible(poste, cible)
    assert res.is_verified is True, res.message
    assert res.topo_obtenue is not None and res.topo_obtenue.nb_noeuds == 3


def test_cible_identite_triviale_inchangee():
    """Une cible == topologie courante reste triviale (court-circuit), même à
    3 barres : aucune manœuvre."""
    poste = _poste()
    res = determiner_topo_complete_cible(poste, poste.topologie_nodale)
    assert res.is_verified is True
    assert res.nb_manoeuvres == 0
    assert "satisfait déjà" in res.message
