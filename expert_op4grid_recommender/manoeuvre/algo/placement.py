"""
manoeuvre/algo/placement.py — Phases 2.2-2.4 : placement nœud → sections de barres (énumération par partitions connexes, best-effort, glouton).
"""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Optional
import networkx as nx

from ..cellules import CelluleDepart
from ..cellules import SwitchInfo
from ..topologie import TopologieNodale, PosteTopologique
from ._constants import POIDS_REAIGUILLAGE, POIDS_MANOEUVRE_COUPLER, POIDS_OUVERTURE_SECTIONNEMENT, POIDS_NOEUD_MULTIBARRE, MAX_COMBINAISONS_PLACEMENT, MAX_COMBINAISONS_BEST_EFFORT
from .graph_ops import _inter_sjb_couplers, _is_open, _sa_path_to_sjb, _set_switch


def _assignations_connexes(sjb_nodes, k, est_connexe):
    """Génère les affectations SJB->nœud (tuples de longueur n) où **chaque nœud
    reçoit un groupe de SJB connexe non vide**, couvrant toutes les SJB.

    Équivalent **en ensemble** à ``itertools.product(range(k), repeat=n)`` filtré
    sur « tous les nœuds non vides et connexes », mais généré **par construction**
    (partitions en k blocs connexes × bijections bloc->nœud) plutôt qu'en filtrant
    k^n affectations. Le coût minimal exploré est donc identique ; seul l'ordre
    d'énumération (et donc le tie-breaking en cas d'égalité) diffère.

    ``est_connexe(frozenset[int]) -> bool`` : prédicat de connexité (mémoïsé).
    """
    n = len(sjb_nodes)
    pos = {s: i for i, s in enumerate(sjb_nodes)}

    # Sous-ensembles connexes non vides, groupés par leur plus petite SJB (indice).
    connexes_par_ancre: dict[int, list[frozenset]] = {}
    for r in range(1, n + 1):
        for combo in itertools.combinations(sjb_nodes, r):
            fs = frozenset(combo)
            if est_connexe(fs):
                ancre = min(pos[s] for s in fs)
                connexes_par_ancre.setdefault(ancre, []).append(fs)

    # Partitions en exactement k blocs connexes, ancrées sur la plus petite SJB
    # non couverte -> chaque partition (non ordonnée) générée une seule fois.
    partitions: list[list[frozenset]] = []

    def _rec(remaining: frozenset, blocks: list):
        if not remaining:
            if len(blocks) == k:
                partitions.append(blocks)
            return
        if len(blocks) >= k:
            return
        ancre = min(pos[s] for s in remaining)
        for block in connexes_par_ancre.get(ancre, ()):
            if (block <= remaining
                    and len(remaining) - len(block) >= (k - len(blocks) - 1)):
                _rec(remaining - block, blocks + [block])

    _rec(frozenset(sjb_nodes), [])

    # Affectation des blocs aux nœuds : toutes les bijections (nœuds distinguables
    # — départs distincts, donc coûts distincts selon le nœud porteur du bloc).
    for blocks in partitions:
        for perm in itertools.permutations(range(k)):
            assign = [0] * n
            for block, node in zip(blocks, perm):
                for s in block:
                    assign[pos[s]] = node
            yield tuple(assign)


def _main_busbar_sjb(poste: PosteTopologique) -> tuple[set[int], set[int]]:
    """
    Pour un poste à > 2 jeux de barres, identifie les **2 jeux de barres
    principaux** (système double-barre classique) et les niveaux supplémentaires.

    Heuristique : on regroupe les SJB par jeu de barres (``barre_par_busbar``) et
    on retient les **2 jeux de barres portant le plus de sections** (départage par
    index croissant). Les SJB des autres jeux de barres (typiquement des niveaux à
    section unique, ex. 3B/4B) sont « supplémentaires » et laissés à l'opérateur.

    Returns
    -------
    (main_sjb, extra_sjb)  ensembles de nœuds SJB.
    """
    barre_par = poste.tronconnement.barre_par_busbar
    by_barre: dict[int, set[int]] = {}
    for s, b in barre_par.items():
        by_barre.setdefault(b, set()).add(s)
    ordered = sorted(by_barre, key=lambda b: (-len(by_barre[b]), b))
    main_barres = set(ordered[:2])
    main_sjb = {s for b in main_barres for s in by_barre[b]}
    extra_sjb = set(barre_par) - main_sjb
    return main_sjb, extra_sjb


def _scoping_raison(
    sjb_id: dict[int, Optional[str]],
    extra_sjb: set[int],
    forced_non_places: list[list[str]],
) -> str:
    """Raison de dégradation liée au scoping > 2 jeux de barres."""
    noms = ", ".join(str(sjb_id.get(s, s)) for s in sorted(extra_sjb))
    return (f"niveaux de barres supplémentaires ({noms}) non gérés par "
            f"l'algorithme 2 jeux de barres : {len(forced_non_places)} nœud(s) "
            "laissé(s) à l'opérateur")


def _message_non_realisable(raisons: list[str]) -> str:
    """Assemble le message de dégradation à partir des raisons collectées."""
    if not raisons:
        return "Aucune affectation de SJB réalisable (topologie impossible)."
    return "Topologie cible non réalisable sur ce poste : " + " ; ".join(raisons) + "."


def _diagnostic_infaisabilite(
    nodes: list[list[str]],
    R: dict[str, frozenset],
    sjb_id: dict[int, Optional[str]],
) -> list[str]:
    """
    Explique **pourquoi** aucune affectation SJB complète n'existe (option 2).

    Retourne une **liste de raisons** (sans préfixe), parmi les deux modes
    d'échec réels :
    1. *Sur-réservation d'une classe d'aiguillage* (condition de Hall) : plus de
       nœuds confinés à un ensemble de sections que de sections disponibles ;
    2. *Organe interne à 2 bornes* (self/réactance ``LINE_SIDE1``+``LINE_SIDE2``)
       présent sur plusieurs nœuds alors qu'il n'atteint qu'une seule section.
    """
    def _names(sset) -> str:
        return "{" + ", ".join(str(sjb_id.get(s, s)) for s in sorted(sset)) + "}"

    raisons: list[str] = []

    # 1. Sur-réservation (Hall) : pour chaque ensemble de sections T atteignable
    #    par un départ, compter les nœuds *forcés* d'utiliser une section de T
    #    (un de leurs départs n'atteint que des sections de T).
    classes = {R[eq] for node in nodes for eq in node if R.get(eq)}
    vus: set[frozenset] = set()
    for T in sorted(classes, key=lambda t: (len(t), sorted(t))):
        forces = [node for node in nodes
                  if any(R.get(eq, frozenset()) <= T for eq in node)]
        if len(forces) > len(T) and T not in vus:
            vus.add(T)
            raisons.append(
                f"{len(forces)} nœuds nécessitent une section parmi {_names(T)} "
                f"({len(T)} disponible(s))")

    # 2. Organe interne à 2 bornes éclaté sur plusieurs nœuds
    occurrences = Counter(eq for node in nodes for eq in node)
    for eq, nb in sorted(occurrences.items()):
        atteignables = R.get(eq, frozenset())
        if nb >= 2 and len(atteignables) < nb:
            raisons.append(
                f"organe interne '{eq}' présent sur {nb} nœuds mais seulement "
                f"{len(atteignables)} section(s) atteignable(s) {_names(atteignables)} "
                "— selfs/réactances à 2 bornes non gérées côté par côté")

    return raisons


def _placement_greedy(
    nodes: list[list[str]],
    R: dict[str, frozenset],
    sjb_nodes: list[int],
    sjb_id: dict[int, Optional[str]],
) -> tuple[list[tuple[set[str], set[str]]], list[list[str]]]:
    """Repli glouton (postes très volumineux) : une SJB simple par nœud, les
    nœuds les plus contraints d'abord."""
    libres = set(sjb_nodes)
    placement: list[tuple[set[str], set[str]]] = []
    places: set[int] = set()
    ordre = sorted(
        range(len(nodes)),
        key=lambda i: min((len(R.get(eq, ())) for eq in nodes[i]), default=99),
    )
    for i in ordre:
        cand = [s for s in libres
                if all(s in R.get(eq, frozenset()) for eq in nodes[i])]
        if cand:
            s = min(cand)
            libres.discard(s)
            places.add(i)
            placement.append((set(nodes[i]), {sjb_id[s]}))
    non_places = [list(nodes[i]) for i in range(len(nodes)) if i not in places]
    return placement, non_places


def _placement_best_effort(
    nodes: list[list[str]],
    R: dict[str, frozenset],
    sjb_nodes: list[int],
    sjb_id: dict[int, Optional[str]],
    CG: nx.Graph,
    wired_sjb: dict[str, Optional[int]],
) -> tuple[list[tuple[set[str], set[str]]], list[list[str]]]:
    """
    Placement **partiel** « best-effort » (option 4 — dégradation gracieuse).

    Quand aucune affectation complète n'existe, place le **plus grand nombre
    possible** de nœuds cibles (chacun entièrement satisfait : sections connexes
    + tous ses départs atteignant une section de leur groupe) et laisse les
    nœuds restants à compléter par l'opérateur.

    Returns
    -------
    (placement, noeuds_non_places)
        ``placement`` = ``[(departs, sjb_ids)]`` des nœuds réalisés ;
        ``noeuds_non_places`` = liste des départs des nœuds non réalisés.
    """
    k = len(nodes)
    n_sjb = len(sjb_nodes)
    if k == 0:
        return [], []

    # Garde-fou : chaque SJB → un nœud OU « inutilisée » (k+1 choix par SJB).
    if (k + 1) ** n_sjb > MAX_COMBINAISONS_BEST_EFFORT:
        return _placement_greedy(nodes, R, sjb_nodes, sjb_id)

    # Connexité d'un groupe de SJB : **mémoïsée** (cf. ``_placement_automatique``).
    _conn: dict[frozenset, bool] = {}

    def _groupe_connexe(sjb_set: set[int]) -> bool:
        key = frozenset(sjb_set)
        r = _conn.get(key)
        if r is None:
            r = nx.is_connected(CG.subgraph(key))
            _conn[key] = r
        return r

    best = None  # (score, placement, non_places)
    for assign in itertools.product(range(k + 1), repeat=n_sjb):
        node_sjbs: dict[int, set[int]] = {i: set() for i in range(k)}
        for j, ni in enumerate(assign):
            if ni < k:
                node_sjbs[ni].add(sjb_nodes[j])
        places: list[int] = []
        valide = True
        for i in range(k):
            s = node_sjbs[i]
            if not s:
                continue  # nœud abandonné (laissé à l'opérateur)
            if not _groupe_connexe(s):
                valide = False
                break
            if any(not (R.get(eq, frozenset()) & s) for eq in nodes[i]):
                valide = False
                break
            places.append(i)
        if not valide or not places:
            continue
        reaig = sum(
            1 for i in places for eq in nodes[i]
            if wired_sjb.get(eq) not in node_sjbs[i]
        )
        score = (len(places), -reaig)
        if best is None or score > best[0]:
            placement = [
                (set(nodes[i]), {sjb_id[s] for s in node_sjbs[i]}) for i in places
            ]
            non_places = [list(nodes[i]) for i in range(k) if i not in places]
            best = (score, placement, non_places)

    if best is None:
        return [], [list(n) for n in nodes]
    return best[1], best[2]


def _within_guard(k: int, n_sjb: int) -> bool:
    """Vrai si l'énumération exhaustive (~k^nb_SJB) tient dans le garde-fou."""
    return 1 <= k <= n_sjb and k ** n_sjb <= MAX_COMBINAISONS_PLACEMENT


def _recherche_exhaustive(
    nodes, sjb_nodes, R, wired_sjb, couplers, cp_closed, groupe_connexe,
    barre_par=None, penaliser_multibarre=True,
) -> Optional[tuple[int, tuple]]:
    """Recherche **exacte** de l'affectation complète SJB->nœud de coût minimal.

    Énumère les partitions connexes (``_assignations_connexes``) et retient le
    **lex-min** parmi les affectations de coût minimal — reproduisant exactement
    l'ancienne énumération ``itertools.product``. Le **garde-fou combinatoire
    est de la responsabilité de l'appelant** (cf. ``_within_guard``).

    Couvre N jeux de barres sans changement : ``sjb_nodes`` et le graphe de
    connexité (via ``groupe_connexe``) sont génériques. Le filtre
    ``na is None``/``nb is None`` permet l'appel sur un **sous-ensemble** de SJB
    (décomposition) ; sur l'ensemble complet il n'écarte jamais aucun coupler,
    donc le comportement 2-JdB est strictement préservé.

    ``barre_par`` (optionnel) : map SJB->barre. Si fourni, on **départage à coût
    égal en faveur du moins de nœuds « multi-barres »** (nœuds dont les SJB
    couvrent plusieurs jeux de barres). Cela évite les nœuds « exotiques »
    (demi-rames croisées de barres différentes, ex. ``{1A,2B}``) que la
    décomposition par paires de ``_inter_sjb_couplers`` rend faussement
    réalisables à travers un faisceau de couplage **partagé** : à coût égal, on
    préfère des nœuds tenant sur une seule barre (ou barres entières), réalisables
    par le séquenceur. **Optimalité en coût préservée** → goldens à coût unique
    inchangés.

    Returns ``(cost, assign)`` ou ``None`` si aucune affectation faisable.
    """
    k = len(nodes)
    n_sjb = len(sjb_nodes)
    if k == 0 or k > n_sjb:
        return None
    # Pénalité multi-barres : uniquement sur les postes > 2 barres (cas 2-JdB
    # strictement inchangés). On juge le nombre de barres sur l'ensemble **complet**
    # du poste (``barre_par``), pas sur le sous-ensemble courant.
    penalise_multibarre = (
        penaliser_multibarre and barre_par is not None
        and len(set(barre_par.values())) > 2)
    best = None  # interne : (cost, assign)  — coût pénalisé inclus
    for assign in _assignations_connexes(sjb_nodes, k, groupe_connexe):
        node_sjbs: dict[int, set[int]] = {i: set() for i in range(k)}
        for j, ni in enumerate(assign):
            node_sjbs[ni].add(sjb_nodes[j])
        if any(not s for s in node_sjbs.values()):
            continue  # chaque nœud doit avoir ≥ 1 SJB
        if not all(groupe_connexe(s) for s in node_sjbs.values()):
            continue
        ok = True
        for i, deps in enumerate(nodes):
            sset = node_sjbs[i]
            if any(not (R[eq] & sset) for eq in deps):
                ok = False
                break
        if not ok:
            continue
        reaig = sum(
            1 for i, deps in enumerate(nodes) for eq in deps
            if wired_sjb.get(eq) not in node_sjbs[i]
        )
        node_of_sjb = {sjb_nodes[j]: assign[j] for j in range(n_sjb)}
        cpl = 0
        sect = 0
        for cp, currently_closed in zip(couplers, cp_closed):
            na = node_of_sjb.get(cp.sjb_a)
            nb = node_of_sjb.get(cp.sjb_b)
            if na is None or nb is None:
                continue  # coupler hors du sous-ensemble courant
            same = na == nb
            if same and not currently_closed:
                cpl += 1
            elif (not same) and currently_closed:
                cpl += 1
                if cp.is_sectionnement:
                    sect += 1
        cost = (POIDS_REAIGUILLAGE * reaig
                + POIDS_MANOEUVRE_COUPLER * cpl
                + POIDS_OUVERTURE_SECTIONNEMENT * sect)
        # Pénalité **dominante** des nœuds multi-barres (réalisabilité > coût).
        if penalise_multibarre:
            multibarre = sum(
                max(0, len({barre_par[s] for s in node_sjbs[i] if s in barre_par}) - 1)
                for i in range(k)
            )
            cost += POIDS_NOEUD_MULTIBARRE * multibarre
        # Tie-break **lex-min** sur ``assign`` (à coût pénalisé égal) — reproduit
        # l'ancienne énumération ``itertools.product`` sur les cas non pénalisés.
        key = (cost, assign)
        if best is None or key < best:
            best = key
    return best


def _placement_est_faisable(amap, nodes, R, groupe_connexe) -> bool:
    """Vérifie qu'un dict ``{sjb_node -> index_nœud}`` réalise **tous** les nœuds
    (groupes connexes + chaque départ atteint une SJB de son groupe)."""
    if amap is None:
        return False
    node_sjbs: dict[int, set[int]] = {}
    for s, ni in amap.items():
        node_sjbs.setdefault(ni, set()).add(s)
    if len(node_sjbs) != len(nodes):
        return False
    for i, deps in enumerate(nodes):
        sset = node_sjbs.get(i)
        if not sset or not groupe_connexe(sset):
            return False
        if any(not (R.get(eq, frozenset()) & sset) for eq in deps):
            return False
    return True


def _couper_graphe_couplage(CG: nx.Graph):
    """Coupe un graphe de couplage **connexe** en 2 sous-ensembles de SJB **connexes
    et équilibrés**. ``None`` si impossible.

    Préférence à un **pont** (le plus équilibré) ; sinon **bipartition équilibrée**
    (Kernighan-Lin) si les deux moitiés restent connexes ; sinon **moitié/moitié**
    (cas dense, p.ex. graphe complet d'un poste triple-barre entièrement maillé) ;
    en dernier recours, **coupe d'arêtes minimale**. Un cut équilibré est crucial :
    une coupe min isole souvent **un seul** sommet sur un graphe dense, ce qui
    empêche de répartir les nœuds (sous-capacité)."""
    n = CG.number_of_nodes()
    if n < 2:
        return None

    # 1) pont le plus équilibré.
    best = None
    for u, v in nx.bridges(CG):
        H = CG.copy()
        H.remove_edge(u, v)
        comps = list(nx.connected_components(H))
        if len(comps) == 2:
            bal = min(len(comps[0]), len(comps[1]))
            if best is None or bal > best[0]:
                best = (bal, set(comps[0]), set(comps[1]))
    if best is not None:
        return best[1], best[2]

    # 2) bipartition équilibrée (Kernighan-Lin), si les 2 moitiés sont connexes.
    try:
        a, b = nx.algorithms.community.kernighan_lin_bisection(CG, seed=0)
        if a and b and nx.is_connected(CG.subgraph(a)) and nx.is_connected(CG.subgraph(b)):
            return set(a), set(b)
    except Exception:
        pass

    # 3) moitié/moitié (les deux moitiés sont connexes si le graphe est dense).
    nodesl = sorted(CG.nodes())
    mid = n // 2
    A, B = set(nodesl[:mid]), set(nodesl[mid:])
    if A and B and nx.is_connected(CG.subgraph(A)) and nx.is_connected(CG.subgraph(B)):
        return A, B

    # 4) repli : coupe d'arêtes minimale (peut être déséquilibrée).
    try:
        cut = nx.minimum_edge_cut(CG)
    except Exception:
        return None
    H = CG.copy()
    H.remove_edges_from(cut)
    comps = list(nx.connected_components(H))
    if len(comps) >= 2:
        return set(comps[0]), set().union(*comps[1:])
    return None


def _couper_par_barres(CGsub: nx.Graph, sjb_set: set, barre_par: dict) -> Optional[tuple]:
    """Coupe l'ensemble de SJB en 2 en **gardant les SJB d'une même barre du même
    côté** (les départs raccordent des *barres*, pas des demi-rames isolées :
    couper à l'intérieur d'une barre casserait la réacheminabilité).

    Coupe le **graphe au niveau barre** (barres adjacentes si un coupler relie
    leurs SJB), puis retransforme en ensembles de SJB. Repli sur une coupe SJB
    générique quand l'ensemble ne couvre qu'une seule barre."""
    barres: dict[int, set[int]] = {}
    for s in sjb_set:
        barres.setdefault(barre_par.get(s), set()).add(s)
    if len(barres) >= 2:
        BG = nx.Graph()
        BG.add_nodes_from(barres)
        for u, v in CGsub.edges():
            bu, bv = barre_par.get(u), barre_par.get(v)
            if bu != bv:
                BG.add_edge(bu, bv)
        parts = _couper_graphe_couplage(BG)
        if parts is not None:
            ga, gb = parts
            A = {s for s in sjb_set if barre_par.get(s) in ga}
            B = {s for s in sjb_set if barre_par.get(s) in gb}
            if A and B:
                return A, B
    # Une seule barre (ou découpe-barre impossible) : coupe SJB générique.
    return _couper_graphe_couplage(CGsub)


def _placement_decompose(
    node_idx, sjb_set, nodes, R, wired_sjb, couplers, cp_closed,
    groupe_connexe, barre_par, depth=0, penaliser_multibarre=True,
):
    """**Étape 2 — décomposition récursive le long du graphe de couplage.**

    Réutilise la primitive exacte 2-JdB (``_recherche_exhaustive``) sur des
    **sous-ensembles** de jeux de barres lorsque le poste complet excède le
    garde-fou combinatoire :

    1. **cas de base** : sous-ensemble assez petit → recherche exacte ;
    2. **composantes connexes** du graphe de couplage : chaque nœud n'occupe
       qu'**une** composante (groupe connexe) ; le coût est séparable d'une
       composante à l'autre → décomposition **exacte** ;
    3. **bissection** d'une composante unique trop grosse : coupe en 2 demi-
       graphes connexes, affecte chaque nœud au côté atteint par ses départs
       (décision binaire = primitive 2-JdB), puis **récursion** sur chaque côté.

    ``node_idx`` : indices (globaux dans ``nodes``) des nœuds à placer.
    Retourne un dict ``{sjb_node -> index_nœud}`` ou ``None``.
    """
    if not node_idx:
        return {}
    sjb_set = set(sjb_set)
    n_sjb = len(sjb_set)
    k = len(node_idx)
    if k > n_sjb:
        return None
    if _within_guard(k, n_sjb):
        sub_nodes = [nodes[g] for g in node_idx]
        sjb_sorted = sorted(sjb_set)
        best = _recherche_exhaustive(
            sub_nodes, sjb_sorted, R, wired_sjb, couplers, cp_closed, groupe_connexe,
            barre_par, penaliser_multibarre)
        if best is None:
            return None
        return {sjb_sorted[j]: node_idx[local] for j, local in enumerate(best[1])}
    if depth > n_sjb:
        return None  # garde-fou anti-récursion

    CGsub = nx.Graph()
    CGsub.add_nodes_from(sjb_set)
    for cp in couplers:
        if cp.sjb_a in sjb_set and cp.sjb_b in sjb_set:
            CGsub.add_edge(cp.sjb_a, cp.sjb_b)
    comps = [set(c) for c in nx.connected_components(CGsub)]

    if len(comps) > 1:
        # (2) composantes connexes : affectation séparable, exacte.
        node_comp: dict[int, int] = {}
        for gi in node_idx:
            deps = nodes[gi]
            cands = [ci for ci, c in enumerate(comps)
                     if all(R.get(eq, frozenset()) & c for eq in deps)]
            if not cands:
                return None
            node_comp[gi] = max(
                cands,
                key=lambda ci: (sum(1 for eq in deps if wired_sjb.get(eq) in comps[ci]),
                                -min(comps[ci])),
            )
        out: dict[int, int] = {}
        for ci, comp in enumerate(comps):
            sub_idx = [gi for gi in node_idx if node_comp[gi] == ci]
            if not sub_idx:
                continue
            sub = _placement_decompose(
                sub_idx, comp, nodes, R, wired_sjb, couplers, cp_closed,
                groupe_connexe, barre_par, depth + 1, penaliser_multibarre)
            if sub is None:
                return None
            out.update(sub)
        return out

    # (3) composante unique trop grosse : bissection récursive (primitive 2-JdB).
    # Coupe **au niveau barre** (SJB d'une même barre du même côté) pour préserver
    # la réacheminabilité départ -> barre.
    parts = _couper_par_barres(CGsub, sjb_set, barre_par)
    if parts is None:
        return None
    side_a, side_b = parts

    # Affectation des nœuds aux deux côtés, **en respectant la capacité** : un
    # côté à ``|side|`` SJB héberge au plus ``|side|`` nœuds (groupes disjoints).
    forced_a: list[int] = []
    forced_b: list[int] = []
    both: list[int] = []
    for gi in node_idx:
        deps = nodes[gi]
        ok_a = all(R.get(eq, frozenset()) & side_a for eq in deps)
        ok_b = all(R.get(eq, frozenset()) & side_b for eq in deps)
        if ok_a and ok_b:
            both.append(gi)
        elif ok_a:
            forced_a.append(gi)
        elif ok_b:
            forced_b.append(gi)
        else:
            return None  # nœud à cheval sur la coupe : non réalisable ainsi
    cap_a = len(side_a) - len(forced_a)
    cap_b = len(side_b) - len(forced_b)
    if cap_a < 0 or cap_b < 0:
        return None
    a_idx, b_idx = list(forced_a), list(forced_b)
    # Répartit les nœuds « indifférents » : côté **câblé** de préférence, sinon là
    # où il reste de la capacité (départage déterministe par index).
    for gi in sorted(both):
        deps = nodes[gi]
        wa = sum(1 for eq in deps if wired_sjb.get(eq) in side_a)
        wb = sum(1 for eq in deps if wired_sjb.get(eq) in side_b)
        prefer_a = wa >= wb
        if prefer_a and cap_a > 0:
            a_idx.append(gi)
            cap_a -= 1
        elif (not prefer_a) and cap_b > 0:
            b_idx.append(gi)
            cap_b -= 1
        elif cap_a > 0:
            a_idx.append(gi)
            cap_a -= 1
        elif cap_b > 0:
            b_idx.append(gi)
            cap_b -= 1
        else:
            return None  # plus de capacité d'aucun côté

    out = {}
    for sub_idx, side in ((a_idx, side_a), (b_idx, side_b)):
        if not sub_idx:
            continue
        sub = _placement_decompose(
            sub_idx, side, nodes, R, wired_sjb, couplers, cp_closed,
            groupe_connexe, barre_par, depth + 1, penaliser_multibarre)
        if sub is None:
            return None
        out.update(sub)
    return out


def _placement_complet(
    nodes, sjb_nodes, R, wired_sjb, couplers, cp_closed, groupe_connexe, barre_par,
    penaliser_multibarre=True,
):
    """Affectation **complète** SJB->nœud (tous les nœuds placés) ou ``None``.

    Exacte (lex-min) si le garde-fou combinatoire le permet, sinon Étape 2
    (décomposition récursive). Le résultat est revérifié faisable avant d'être
    déclaré complet (on ne renvoie jamais une affectation partielle ici)."""
    n_sjb = len(sjb_nodes)
    k = len(nodes)
    amap = None
    if _within_guard(k, n_sjb):
        best = _recherche_exhaustive(
            nodes, sjb_nodes, R, wired_sjb, couplers, cp_closed, groupe_connexe,
            barre_par, penaliser_multibarre)
        if best is not None:
            amap = {sjb_nodes[j]: best[1][j] for j in range(n_sjb)}
    else:
        amap = _placement_decompose(
            list(range(k)), set(sjb_nodes), nodes, R, wired_sjb, couplers,
            cp_closed, groupe_connexe, barre_par, 0, penaliser_multibarre)
    return amap if _placement_est_faisable(amap, nodes, R, groupe_connexe) else None


def _placement_automatique(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
    penaliser_multibarre: bool = True,
) -> tuple[list[tuple[set[str], set[str]]], bool, str, list[list[str]]]:
    """
    Calcule un placement ``[(departs, sjb_ids)]`` réalisant ``topo_cible``.

    Si la cible est **réalisable**, retourne le placement complet
    (``faisable=True``). Sinon (option 2/4) : retourne un **diagnostic explicite**
    de l'infaisabilité dans ``message`` et un **placement partiel** « best-effort »
    plaçant le plus de nœuds possible, ``noeuds_non_places`` listant les départs
    des nœuds laissés à l'opérateur.

    ``penaliser_multibarre`` (postes > 2 barres) : quand ``True`` (défaut), pénalise
    de façon **dominante** les nœuds multi-barres (évite les nœuds « exotiques »
    demi-rames croisées). Quand ``False``, on cherche le placement de **coût brut
    minimal** (ré-aiguillage), qui peut légitimement préférer un nœud multi-barres
    quand cela **évite des ré-aiguillages** (barres entièrement couplées). L'appelant
    (``determiner_topo_complete_cible``) essaie les deux et **retient, de façon
    transactionnelle, la réalisation vérifiée la moins coûteuse en manœuvres**.

    Returns
    -------
    (placement, faisable, message, noeuds_non_places)
    """
    G = poste.graph
    barre_par = poste.tronconnement.barre_par_busbar
    sjb_nodes = sorted(barre_par)
    sjb_id = {n: G.nodes[n].get("busbar_section_id") for n in sjb_nodes}

    # Connexité / câblage courant **par équipement** : la connexité électrique
    # est jugée sur un chemin de switches fermés depuis le nœud propre de
    # l'équipement (gère les omnibus : un groupe isolé, DJ propre ouvert, n'est
    # pas connecté même si son co-locataire l'est).
    eq_node = {data.get("equipment_id"): n for n, data in G.nodes(data=True)
               if data.get("equipment_id")}
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if not d.get("open", False):
            H.add_edge(u, v)

    R: dict[str, frozenset] = {}        # SJB atteignables (classe de position)
    connected: dict[str, bool] = {}
    wired_sjb: dict[str, Optional[int]] = {}   # SJB où le départ est câblé (SA fermé)
    for c in poste.cellules.cellules_depart:
        for eq in {c.equipment_id} | set(c.shared_equipment_ids):
            # Union : un organe interne à 2 bornes (self/réactance) possède deux
            # cellules — on cumule les SJB atteignables des deux côtés.
            R[eq] = R.get(eq, frozenset()) | frozenset(c.busbar_nodes)
            en = eq_node.get(eq)
            connected[eq] = connected.get(eq, False) or bool(
                en is not None and en in H
                and any(s in H and nx.has_path(H, en, s) for s in R[eq])
            )
            if wired_sjb.get(eq) is None:
                wired = [bb for bb in c.busbar_nodes
                         if _sa_path_to_sjb(c, bb)
                         and all(not _is_open(G, s) for s in _sa_path_to_sjb(c, bb))]
                wired_sjb[eq] = wired[0] if wired else None

    # Nœuds à placer : ceux ayant ≥ 1 départ actuellement connecté.
    nodes: list[list[str]] = []
    for noeud in topo_cible.noeuds.values():
        deps = [e for e in noeud.equipment_ids if e in R and connected[e]]
        if deps:
            nodes.append(deps)

    if not nodes:
        return [], True, "Aucun nœud connecté à placer.", []

    # --- Graphe des couplers entre SJB (contrainte de connexité des groupes) -
    # **Généralisé à N jeux de barres** (Étape 1) : on construit CG sur **toutes**
    # les SJB du poste et **tous** les couplers — plus aucune restriction aux
    # « 2 jeux de barres principaux ». Pour un poste à 2 barres, ``barre_par``
    # ne contient que 2 barres, donc le comportement est strictement préservé.
    couplers = _inter_sjb_couplers(poste)
    CG = nx.Graph()
    CG.add_nodes_from(sjb_nodes)
    for cp in couplers:
        CG.add_edge(cp.sjb_a, cp.sjb_b)

    k = len(nodes)

    # Connexité d'un groupe de SJB dans CG : **mémoïsée**. Le même sous-ensemble
    # réapparaît dans des milliers d'affectations ; on passe de ~k^n appels à
    # ``nx.is_connected`` à ≤ 2^(nb SJB) calculs distincts.
    _conn: dict[frozenset, bool] = {}

    def _groupe_connexe(sjb_set: set[int]) -> bool:
        key = frozenset(sjb_set)
        r = _conn.get(key)
        if r is None:
            r = nx.is_connected(CG.subgraph(key))
            _conn[key] = r
        return r

    # État courant (fermé ?) de chaque coupler : **invariant** de la recherche,
    # calculé une seule fois (au lieu d'être recalculé à chaque itération).
    cp_closed = [all(not _is_open(G, s) for s in cp.switch_ids) for cp in couplers]

    # Affectation complète : recherche exacte (lex-min) dans le garde-fou, sinon
    # décomposition récursive le long du graphe de couplage (Étape 2).
    assign_map = _placement_complet(
        nodes, sjb_nodes, R, wired_sjb, couplers, cp_closed, _groupe_connexe,
        barre_par, penaliser_multibarre)

    full_placement = None
    if assign_map is not None:
        node_sjbs = {i: set() for i in range(k)}
        for s, ni in assign_map.items():
            node_sjbs[ni].add(s)
        full_placement = [
            (set(nodes[i]), {sjb_id[s] for s in node_sjbs[i]})
            for i in range(k)
        ]

    # Cas nominal : affectation complète réalisable (sur N jeux de barres).
    if full_placement is not None:
        return full_placement, True, "OK", []

    # --- Dégradation : placement partiel + diagnostic explicite -------------
    raisons = _diagnostic_infaisabilite(nodes, R, sjb_id)
    placement, partial_non_places = _placement_best_effort(
        nodes, R, sjb_nodes, sjb_id, CG, wired_sjb)
    return placement, False, _message_non_realisable(raisons), partial_non_places


def _placement_avec_reconnexions(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
    topo_cible: TopologieNodale,
    reconnections: list[tuple[CelluleDepart, SwitchInfo]],
) -> tuple[list[tuple[set[str], set[str]]], bool, str, list[list[str]]]:
    """
    Calcule le placement nœud→SJB en tenant compte des **reconnexions** de
    départ (DJ ouvert → fermé dans la cible).

    Le placement est calculé sur un poste **virtuel** où les DJ reconnectés sont
    fermés et leurs SA positionnés comme dans la cible : les départs reconnectés
    sont ainsi vus *connectés sur leur barre cible*, ce qui force l'affectation
    de leur SJB cible au bon nœud (via le coût de ré-aiguillage). Le séquenceur,
    lui, travaillera sur le poste **réel** (DJ encore ouverts), pour que les
    sections cibles restent hors tension lors des manœuvres de sectionnement.
    """
    vl = poste.voltage_level_id
    G_virt = poste.graph.copy()
    coupling_sids: set[str] = set()
    for cp in _inter_sjb_couplers(poste):
        coupling_sids.update(cp.switch_ids)

    for cell, dj_sw in reconnections:
        for sa in cell.disconnectors:
            if sa.switch_id in coupling_sids:
                continue
            _set_switch(G_virt, sa.switch_id, _is_open(cible_graph, sa.switch_id))
        _set_switch(G_virt, dj_sw.switch_id, False)

    poste_virt = PosteTopologique.from_graph(G_virt, vl)
    return _placement_automatique(poste_virt, topo_cible)


def _departure_dj_changes(
    poste: PosteTopologique,
    cible_graph: nx.Graph,
) -> tuple[list[tuple[CelluleDepart, SwitchInfo]],
           list[tuple[CelluleDepart, SwitchInfo]]]:
    """
    Identifie les changements d'état des **DJ de départ** entre l'état initial
    (``poste.graph``) et la cible (``cible_graph``).

    Les DJ de couplage (inter-SJB) sont exclus — ils sont gérés par le
    séquenceur général.

    Returns
    -------
    (reconnections, disconnections)
        Chaque élément est une liste de ``(cellule_départ, switch_info_du_DJ)``.
        - reconnection : DJ ouvert → fermé (mise en service)
        - disconnection : DJ fermé → ouvert (mise hors service)
    """
    G = poste.graph
    coupling_sids: set[str] = set()
    for cp in _inter_sjb_couplers(poste):
        coupling_sids.update(cp.switch_ids)

    reconnections: list[tuple[CelluleDepart, SwitchInfo]] = []
    disconnections: list[tuple[CelluleDepart, SwitchInfo]] = []

    for cell in poste.cellules.cellules_depart:
        for sw in cell.breakers:
            if sw.switch_id in coupling_sids:
                continue
            initial_open = _is_open(G, sw.switch_id)
            cible_open = _is_open(cible_graph, sw.switch_id)
            if initial_open and not cible_open:
                reconnections.append((cell, sw))
            elif not initial_open and cible_open:
                disconnections.append((cell, sw))

    return reconnections, disconnections
