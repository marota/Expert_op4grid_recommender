"""
tests/manoeuvre/test_placement_decomposition.py
-----------------------------------------------
**Étape 2 — décomposition récursive du placement le long du graphe de couplage.**

Quand l'énumération exacte excède le garde-fou combinatoire
(``MAX_COMBINAISONS_PLACEMENT``), ``_placement_automatique`` bascule sur
``_placement_decompose`` qui réutilise la primitive exacte 2-JdB
(``_recherche_exhaustive``) sur des **sous-ensembles** de jeux de barres :
- par **composantes connexes** du graphe de couplage (séparable → exact) ;
- par **bissection** récursive d'une composante unique trop grosse.

On force ici le garde-fou à 1 (``monkeypatch``) pour exercer ces chemins même
sur de petits postes, et on vérifie les invariants de **faisabilité**.
"""
from __future__ import annotations

import networkx as nx
import pytest

from expert_op4grid_recommender.manoeuvre import PosteTopologique, TopologieNodale
from expert_op4grid_recommender.manoeuvre.algo import placement as pmod
from expert_op4grid_recommender.manoeuvre.algo.graph_ops import _InterSjbCoupler

from .fixture_loader import (
    build_graph_from_fixture,
    get_fixture_metadata,
    list_available_fixtures,
)


# --------------------------------------------------------------------------
# (0) Pénalité multi-barres : **gated > 2 barres** (cas 2-JdB inchangés)
# --------------------------------------------------------------------------

def _scenario_chaine_4sjb():
    """4 SJB reliées en chaîne 0-1-2-3, 2 nœuds atteignant toutes les SJB."""
    sjb_nodes = [0, 1, 2, 3]
    R = {"a": frozenset({0, 1, 2, 3}), "b": frozenset({0, 1, 2, 3})}
    wired_sjb = {"a": 0, "b": 3}
    nodes = [["a"], ["b"]]
    couplers = [
        _InterSjbCoupler(0, 1, ["c01"], ["c01"]),
        _InterSjbCoupler(1, 2, ["c12"], ["c12"]),
        _InterSjbCoupler(2, 3, ["c23"], ["c23"]),
    ]
    cp_closed = [True, True, True]
    CG = nx.path_graph([0, 1, 2, 3])
    groupe_connexe = lambda s: nx.is_connected(CG.subgraph(s))  # noqa: E731
    return sjb_nodes, R, wired_sjb, nodes, couplers, cp_closed, groupe_connexe


def test_penalite_multibarre_gated_sur_2_barres():
    """La pénalité multi-barres ne s'applique **qu'aux postes > 2 barres**. Sur un
    poste à 2 barres, passer ``barre_par`` à ``_recherche_exhaustive`` ne change
    **pas** le résultat (mêmes coût et affectation lex-min) — garantie de
    non-régression du comportement 2-JdB."""
    sjb_nodes, R, wired_sjb, nodes, couplers, cp_closed, gc = _scenario_chaine_4sjb()
    barre_par_2 = {0: 0, 1: 0, 2: 1, 3: 1}  # 2 barres -> pénalité inactive

    sans = pmod._recherche_exhaustive(
        nodes, sjb_nodes, R, wired_sjb, couplers, cp_closed, gc, None)
    avec = pmod._recherche_exhaustive(
        nodes, sjb_nodes, R, wired_sjb, couplers, cp_closed, gc, barre_par_2)

    assert sans is not None and avec is not None
    assert sans == avec, "la pénalité doit être inactive (gated) à 2 barres"


def test_penalite_multibarre_reduit_les_noeuds_multibarres():
    """Sur **3 barres**, à coût pénalisé minimal, l'affectation retenue ne crée
    pas plus de nœuds multi-barres que nécessaire : ici chaque nœud peut tenir sur
    une barre, la recherche pénalisée doit donc renvoyer une affectation
    **mono-barre** (aucun nœud à cheval sur 2 barres)."""
    # 3 barres mono-SJB (0,1,2) ; 3 nœuds épinglés chacun sur une barre.
    sjb_nodes = [0, 1, 2]
    R = {"a": frozenset({0}), "b": frozenset({1}), "c": frozenset({2})}
    wired_sjb = {"a": 0, "b": 1, "c": 2}
    nodes = [["a"], ["b"], ["c"]]
    couplers = [
        _InterSjbCoupler(0, 1, ["c01"], ["c01"]),
        _InterSjbCoupler(1, 2, ["c12"], ["c12"]),
        _InterSjbCoupler(0, 2, ["c02"], ["c02"]),
    ]
    cp_closed = [True, True, True]
    CG = nx.Graph(); CG.add_nodes_from(sjb_nodes)
    CG.add_edges_from([(0, 1), (1, 2), (0, 2)])
    gc = lambda s: nx.is_connected(CG.subgraph(s))  # noqa: E731
    barre_par = {0: 0, 1: 1, 2: 2}  # 3 barres -> pénalité active

    best = pmod._recherche_exhaustive(
        nodes, sjb_nodes, R, wired_sjb, couplers, cp_closed, gc, barre_par)
    assert best is not None
    assign = best[1]
    node_sjbs: dict = {}
    for j, ni in enumerate(assign):
        node_sjbs.setdefault(ni, set()).add(sjb_nodes[j])
    for sset in node_sjbs.values():
        assert len({barre_par[s] for s in sset}) == 1, "nœud mono-barre attendu"


# --------------------------------------------------------------------------
# (1) Décomposition par composantes connexes — exacte (test synthétique)
# --------------------------------------------------------------------------

def test_decomposition_par_composantes_est_exacte(monkeypatch):
    """Deux composantes de couplage disjointes {0,1} et {2,3} : chaque nœud
    n'occupe qu'une composante ; la décomposition place chacun dans la sienne.
    Le coût étant séparable d'une composante à l'autre, le résultat est exact."""
    sjb_nodes = [0, 1, 2, 3]
    R = {"a": frozenset({0, 1}), "b": frozenset({2, 3})}
    wired_sjb = {"a": 0, "b": 2}
    nodes = [["a"], ["b"]]
    couplers = [
        _InterSjbCoupler(0, 1, ["s01"], ["s01"]),   # couplage (DJ) intra-comp A
        _InterSjbCoupler(2, 3, ["s23"], ["s23"]),   # couplage (DJ) intra-comp B
    ]
    cp_closed = [True, True]
    CG = nx.Graph()
    CG.add_nodes_from(sjb_nodes)
    CG.add_edge(0, 1)
    CG.add_edge(2, 3)
    groupe_connexe = lambda s: nx.is_connected(CG.subgraph(s))  # noqa: E731
    barre_par = {0: 0, 1: 0, 2: 1, 3: 1}  # SJB 0,1 -> barre 0 ; 2,3 -> barre 1

    # Garde-fou à 1 → la voie exacte globale est désactivée → décomposition.
    monkeypatch.setattr(pmod, "MAX_COMBINAISONS_PLACEMENT", 1)
    amap = pmod._placement_complet(
        nodes, sjb_nodes, R, wired_sjb, couplers, cp_closed, groupe_connexe, barre_par)

    assert amap is not None, "la décomposition doit trouver un placement complet"
    grp: dict[int, set[int]] = {}
    for s, ni in amap.items():
        grp.setdefault(ni, set()).add(s)
    assert grp[0] <= {0, 1}, "nœud 'a' placé dans sa composante {0,1}"
    assert grp[1] <= {2, 3}, "nœud 'b' placé dans sa composante {2,3}"
    assert pmod._placement_est_faisable(amap, nodes, R, groupe_connexe)


# --------------------------------------------------------------------------
# Helpers fixtures
# --------------------------------------------------------------------------

def _poste(name):
    vl = get_fixture_metadata(name)["voltage_level_id"]
    return PosteTopologique.from_graph(build_graph_from_fixture(name), vl)


def _split_en_3(poste):
    """Cible : le plus gros nœud courant scindé pour obtenir ~3 nœuds."""
    groups = [sorted(n.equipment_ids) for n in poste.topologie_nodale.noeuds.values()]
    big = max(groups, key=len)
    rest = [g for g in groups if g is not big]
    half = max(1, len(big) // 2)
    newg = rest + [big[:half], big[half:half + 1], big[half + 1:]]
    newg = [g for g in newg if g]
    return TopologieNodale.from_node_groups(poste.voltage_level_id, newg)


# --------------------------------------------------------------------------
# (2) Bissection : la faisabilité est préservée sur un vrai 3-JdB (SSV.OP7)
# --------------------------------------------------------------------------

def test_bissection_sur_chaine_de_barres(monkeypatch):
    """Bissection (composante unique) sur une **chaîne** de 4 barres mono-SJB
    0-1-2-3, 4 nœuds épinglés chacun sur une barre. Garde-fou à 1 → la
    bissection coupe la chaîne et place chaque nœud sur sa barre."""
    sjb_nodes = [0, 1, 2, 3]
    R = {f"d{i}": frozenset({i}) for i in range(4)}
    wired_sjb = {f"d{i}": i for i in range(4)}
    nodes = [["d0"], ["d1"], ["d2"], ["d3"]]
    couplers = [
        _InterSjbCoupler(0, 1, ["c01"], ["c01"]),
        _InterSjbCoupler(1, 2, ["c12"], ["c12"]),
        _InterSjbCoupler(2, 3, ["c23"], ["c23"]),
    ]
    cp_closed = [True, True, True]
    CG = nx.path_graph([0, 1, 2, 3])
    groupe_connexe = lambda s: nx.is_connected(CG.subgraph(s))  # noqa: E731
    barre_par = {i: i for i in range(4)}  # chaque SJB est sa propre barre

    monkeypatch.setattr(pmod, "MAX_COMBINAISONS_PLACEMENT", 1)
    amap = pmod._placement_complet(
        nodes, sjb_nodes, R, wired_sjb, couplers, cp_closed, groupe_connexe, barre_par)

    assert amap is not None, "la bissection doit placer les 4 nœuds de la chaîne"
    assert pmod._placement_est_faisable(amap, nodes, R, groupe_connexe)
    # chaque nœud i atterrit sur la SJB i (seule atteignable).
    assert {amap[i] for i in range(4)} == {0, 1, 2, 3}


@pytest.mark.skipif(
    "SSV_OP7" not in list_available_fixtures(),
    reason="Fixture SSV_OP7 (poste 400 kV à 3 jeux de barres) absente.",
)
def test_decomposition_jamais_de_placement_invalide(monkeypatch):
    """Contrat de sûreté : sous garde-fou forcé (décomposition), le placement est
    **soit complet et valide, soit une dégradation gracieuse** — jamais une
    affectation invalide, jamais d'exception. (Le cas exact, lui, réussit.)

    NB : sur SSV.OP7 chaque départ n'atteint **qu'une demi-rame par barre** ; un
    nœud « mixte » exige une barre entière. La recherche exacte gère ce cas ; la
    bissection naïve peut légitimement dégrader — d'où ce contrat de sûreté."""
    poste = _poste("SSV_OP7")
    cible = _split_en_3(poste)

    # Voie exacte (garde-fou nominal) : doit réussir.
    pl_exact, ok_exact, _m, _n = pmod._placement_automatique(poste, cible)
    assert ok_exact is True

    # Voie décomposition forcée : valide-ou-dégradée, jamais invalide.
    monkeypatch.setattr(pmod, "MAX_COMBINAISONS_PLACEMENT", 1)
    pl_dec, ok_dec, _m2, np2 = pmod._placement_automatique(poste, cible)
    if ok_dec:
        # si complet : SJB disjointes, partition de départs == cible.
        vues: set[str] = set()
        for _d, sjbs in pl_dec:
            assert sjbs and vues.isdisjoint(sjbs)
            vues |= sjbs
        assert {frozenset(d) for d, _ in pl_dec} == {frozenset(d) for d, _ in pl_exact}
    else:
        # dégradation gracieuse : placement partiel + nœuds laissés à l'opérateur.
        assert isinstance(np2, list)


# --------------------------------------------------------------------------
# (3) Sur 2 barres, la décomposition reste cohérente avec l'exact
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    "CARRIP3" not in list_available_fixtures(),
    reason="Fixture CARRIP3 absente.",
)
def test_decomposition_coherente_sur_2_barres(monkeypatch):
    poste = _poste("CARRIP3")  # poste à 2 barres
    cible = _split_en_3(poste)

    pl_exact, ok_exact, _m, _n = pmod._placement_automatique(poste, cible)
    monkeypatch.setattr(pmod, "MAX_COMBINAISONS_PLACEMENT", 1)
    pl_dec, ok_dec, _m2, _n2 = pmod._placement_automatique(poste, cible)

    # Même verdict de faisabilité et même nombre de nœuds placés.
    assert ok_dec == ok_exact
    assert len(pl_dec) == len(pl_exact)
    assert {frozenset(d) for d, _ in pl_dec} == {frozenset(d) for d, _ in pl_exact}
