"""
tests/manoeuvre/test_postes_caracteristiques_particulieres.py
-------------------------------------------------------------
Postes **réels** (réseau France 28/08/2024) choisis pour leurs **caractéristiques
particulières**, complétant la couverture du module Manoeuvre au-delà des 7 postes
400 kV à 3 JdB. Identifiés par un balayage des 4018 VL NODE_BREAKER du réseau
(``scripts`` ad hoc), puis extraits en fixtures (sans pypowsybl à l'exécution) :

| Fixture   | VL réel   | Caractéristique particulière                              |
|-----------|-----------|----------------------------------------------------------|
| `_OBER_7` | `.OBER 7` | **8 jeux de barres** (max), faisceaux partagés massifs    |
| `_VANY_7` | `.VANY 7` | **7 jeux de barres**, anneau de couplers                  |
| `_ZAND_7` | `.ZAND 7` | 6 barres, **12 SJB** (multi-section), organes internes     |
| `MUHLBP7` | `MUHLBP7` | 5 barres, 10 SJB, faisceaux partagés                      |
| `_LAUF_7` | `.LAUF 7` | **24 SJB** (sectionnement extrême), 4 barres, anneau       |
| `P_GASP6` | `P.GASP6` | **26 départs** (max), 225 kV, faisceaux partagés          |
| `CPNIEP6` | `CPNIEP6` | **organe interne 2 bornes** (self/réactance), 4 barres     |
| `ROMAIP6` | `ROMAIP6` | **omnibus** (départs multiples ×4), 8 SJB, 225 kV          |
| `REICHP3` | `REICHP3` | **14 SJB** (sectionnement extrême), 63 kV, omnibus         |
| `_MUHL_6` | `.MUHL 6` | **10 départs déconnectés** (nœuds 0-barre), 3 barres       |

Deux niveaux d'exigence :
1. **Caractérisation structurelle** — chaque poste présente bien la caractéristique
   visée (verrouille le corpus et documente *pourquoi* il est intéressant).
2. **Innocuité** — sur une variété de cibles, ``determiner_topo_complete_cible``
   reste **sûr** quel que soit le degré de réalisation : graphe du poste non muté,
   manœuvres sur des organes existants, pas de dégradation par scoping « 2 JdB »,
   vérificateur de sectionneurs aligné, et ``is_verified`` ⇒ topologie exacte.

Robuste au nommage point/espace (VL ``.OBER 7`` ↔ fixture ``_OBER_7``) : le VL
réel est lu dans la fixture, la fixture est chargée par son *stem*.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import pytest

from expert_op4grid_recommender.manoeuvre import (
    PosteTopologique,
    TopologieNodale,
    determiner_topo_complete_cible,
    sectionneurs_sous_charge_par_manoeuvre,
)
from expert_op4grid_recommender.manoeuvre.algo.graph_ops import (
    _inter_sjb_couplers,
    _wired_busbar,
)
from expert_op4grid_recommender.manoeuvre.algo.targets import _organes_internes_2bornes

from .fixture_loader import (
    build_graph_from_fixture,
    get_fixture_metadata,
    list_available_fixtures,
)

# Stem -> caractéristiques attendues (bornes **inférieures** : ``>=``).
POSTES: dict[str, dict] = {
    "_OBER_7": dict(barres=8, ring=True, shared=1),
    "_VANY_7": dict(barres=7, ring=True, shared=1),
    "_ZAND_7": dict(barres=6, sjb=12, shared=1, self2b=1),
    "MUHLBP7": dict(barres=5, sjb=10, shared=1, self2b=1),
    "_LAUF_7": dict(barres=4, sjb=24, ring=True),
    "P_GASP6": dict(barres=3, departs=24, shared=1),
    "CPNIEP6": dict(barres=4, self2b=1, departs=15),
    "ROMAIP6": dict(barres=2, omnibus=3, sjb=8),
    "REICHP3": dict(barres=2, sjb=14, omnibus=1),
    "_MUHL_6": dict(barres=3, deco=8),
}

_DISPONIBLES = [s for s in POSTES if s in list_available_fixtures()]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _poste(stem: str) -> PosteTopologique:
    vl = get_fixture_metadata(stem)["voltage_level_id"]
    return PosteTopologique.from_graph(build_graph_from_fixture(stem), vl)


def _switch_states(G) -> dict:
    return {d["switch_id"]: bool(d.get("open", False))
            for _, _, d in G.edges(data=True) if d.get("switch_id")}


def _known(G) -> set:
    return {d.get("switch_id") for _, _, d in G.edges(data=True) if d.get("switch_id")}


def _metrics(poste: PosteTopologique) -> dict:
    G = poste.graph
    bp = poste.tronconnement.barre_par_busbar
    cps = _inter_sjb_couplers(poste)
    sid: dict = defaultdict(int)
    for cp in cps:
        for s in cp.switch_ids:
            sid[s] += 1
    BG = nx.Graph()
    for cp in cps:
        ba, bb = bp.get(cp.sjb_a), bp.get(cp.sjb_b)
        if ba is not None and bb is not None and ba != bb:
            BG.add_edge(ba, bb)
    return dict(
        barres=len(set(bp.values())),
        sjb=len(bp),
        noeuds=poste.topologie_nodale.nb_noeuds,
        departs=len(poste.cellules.cellules_depart),
        shared=sum(1 for c in sid.values() if c >= 2),
        omnibus=sum(1 for c in poste.cellules.cellules_depart if c.is_multiple),
        deco=sum(1 for c in poste.cellules.cellules_depart if _wired_busbar(c, G) is None),
        self2b=len(_organes_internes_2bornes(poste)),
        ring=BG.number_of_nodes() >= 3 and BG.number_of_edges() >= BG.number_of_nodes(),
    )


def _cible_separer_barres(poste) -> list[list[str]]:
    """Cible « séparer les barres couplées » : scinde le plus grand nœud par barre
    câblée, et préserve les autres nœuds tels quels."""
    bp = poste.tronconnement.barre_par_busbar
    noeuds = list(poste.topologie_nodale.noeuds.values())
    big = max(noeuds, key=lambda n: len(n.equipment_ids))
    groups = [sorted(n.equipment_ids) for n in noeuds if n is not big]
    sub: dict = defaultdict(list)
    for eq in sorted(big.equipment_ids):
        cell = poste.cellules.get_cellule_depart(eq)
        wb = _wired_busbar(cell, poste.graph) if cell else None
        sub[bp.get(wb) if wb is not None else f"iso_{eq}"].append(eq)
    groups += [sorted(v) for v in sub.values()]
    return [g for g in groups if g]


def _cible_round_robin(poste, k: int) -> list[list[str]]:
    d = sorted(e for n in poste.topologie_nodale.noeuds.values()
               for e in n.equipment_ids)
    return [g for g in (d[i::k] for i in range(k)) if g]


def _cibles(poste) -> dict[str, list[list[str]]]:
    out = {"separer": _cible_separer_barres(poste),
           "rr3": _cible_round_robin(poste, 3),
           "rr4": _cible_round_robin(poste, 4)}
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------
# 1. Caractérisation structurelle
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste particulier.")
@pytest.mark.parametrize("stem", _DISPONIBLES)
def test_caracteristique_presente(stem):
    """Chaque poste présente bien la caractéristique particulière qui justifie son
    ajout au corpus (bornes inférieures sur les métriques topologiques)."""
    spec = POSTES[stem]
    poste = _poste(stem)
    m = _metrics(poste)
    for key in ("barres", "sjb", "departs", "shared", "omnibus", "deco", "self2b"):
        if key in spec:
            assert m[key] >= spec[key], (
                f"{stem}: {key}={m[key]} < attendu {spec[key]} — la fixture a-t-elle "
                "changé ? (métriques: " + ", ".join(f"{k}={v}" for k, v in m.items()) + ")")
    if spec.get("ring"):
        assert m["ring"], f"{stem}: topologie de couplers attendue en anneau"


@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste particulier.")
@pytest.mark.parametrize("stem", _DISPONIBLES)
def test_construction_poste_ne_mute_pas_le_graphe(stem):
    """``PosteTopologique.from_graph`` + caractérisation ne mutent pas le graphe."""
    G = build_graph_from_fixture(stem)
    before = _switch_states(G)
    vl = get_fixture_metadata(stem)["voltage_level_id"]
    poste = PosteTopologique.from_graph(G, vl)
    _metrics(poste)
    assert _switch_states(poste.graph) == before


# --------------------------------------------------------------------------
# 2. Innocuité du moteur sur ces postes (toutes cibles)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _DISPONIBLES, reason="Aucune fixture de poste particulier.")
@pytest.mark.parametrize("stem", _DISPONIBLES)
@pytest.mark.parametrize("shape", ["separer", "rr3", "rr4"])
def test_innocuite_cible(stem, shape):
    """Pour une variété de cibles sur ces postes atypiques, ``determiner_topo_
    complete_cible`` reste **sûr** quel que soit le degré de réalisation."""
    poste = _poste(stem)
    cibles = _cibles(poste)
    if shape not in cibles:
        pytest.skip(f"forme {shape} indisponible pour {stem}")
    before = _switch_states(poste.graph)
    known = _known(poste.graph)
    cible = TopologieNodale.from_node_groups(poste.voltage_level_id, cibles[shape])

    res = determiner_topo_complete_cible(poste, cible)

    # (a) graphe du poste jamais muté.
    assert _switch_states(poste.graph) == before
    # (b) toute manœuvre porte sur un organe existant.
    assert all(m.switch_id in known for m in res.manoeuvres)
    # (c) plus aucune dégradation par scoping « 2 jeux de barres ».
    assert "niveaux de barres supplémentaires" not in res.message
    assert "algorithme 2 jeux de barres" not in res.message
    # (d) vérificateur de sectionneurs : sortie alignée, pas d'exception.
    viol = sectionneurs_sous_charge_par_manoeuvre(poste, res.manoeuvres)
    assert len(viol) == len(res.manoeuvres)
    assert all(v is None or isinstance(v, str) for v in viol)
    # (e) cohérence : si vérifié, alors topologie exacte et aucun écart.
    assert res.topo_obtenue is not None
    if res.is_verified:
        assert res.topo_obtenue.nb_noeuds == cible.nb_noeuds
        assert res.ecarts == []


# --------------------------------------------------------------------------
# 3. Comportements ciblés sur les caractéristiques rares
# --------------------------------------------------------------------------

@pytest.mark.skipif("_MUHL_6" not in _DISPONIBLES, reason="Fixture _MUHL_6 absente.")
def test_departs_deconnectes_ne_sont_pas_places():
    """`.MUHL 6` a ~10 départs **déconnectés** (DJ propre ouvert). Le placement ne
    place que les départs **connectés** : aucune manœuvre ne réénergise un départ
    déconnecté (cf. limite documentée « reconnexion de départs déconnectés »)."""
    poste = _poste("_MUHL_6")
    m = _metrics(poste)
    assert m["deco"] >= 8, f"attendu ≥8 départs déconnectés, obtenu {m['deco']}"
    # Cible = topologie nodale courante (identité) → aucune manœuvre nécessaire.
    res = determiner_topo_complete_cible(poste, poste.topologie_nodale)
    assert res.is_verified
    assert res.nb_manoeuvres == 0


@pytest.mark.skipif("P_GASP6" not in _DISPONIBLES, reason="Fixture P_GASP6 absente.")
def test_gros_poste_separer_barres_realise():
    """`P.GASP6` (26 départs, 225 kV, faisceaux partagés) : la cible « séparer les
    barres » est **réalisée et vérifiée** — preuve que le réalisateur connectivité
    tient sur un gros poste à faisceaux partagés (≠ 400 kV 3 JdB)."""
    poste = _poste("P_GASP6")
    before = _switch_states(poste.graph)
    cible = TopologieNodale.from_node_groups(
        poste.voltage_level_id, _cible_separer_barres(poste))
    res = determiner_topo_complete_cible(poste, cible)
    assert res.is_verified, f"P.GASP6 séparer-barres non réalisé : {res.message}"
    assert res.topo_obtenue.nb_noeuds == cible.nb_noeuds
    assert res.ecarts == []
    assert _switch_states(poste.graph) == before


@pytest.mark.skipif(
    not ({"CPNIEP6", "_ZAND_7"} & set(_DISPONIBLES)),
    reason="Aucune fixture à organe interne.")
@pytest.mark.parametrize("stem", [s for s in ("CPNIEP6", "_ZAND_7") if s in _DISPONIBLES])
def test_organes_internes_2bornes_detectes(stem):
    """Les postes à **organe interne 2 bornes** (self/réactance reliant deux SJB du
    même VL) sont détectés par ``_organes_internes_2bornes`` — ces organes seront
    passés en ``organes_fixes`` au séquenceur (ni ré-aiguillés ni signalés en écart)."""
    poste = _poste(stem)
    organes = _organes_internes_2bornes(poste)
    assert organes, f"{stem}: aucun organe interne 2 bornes détecté"
