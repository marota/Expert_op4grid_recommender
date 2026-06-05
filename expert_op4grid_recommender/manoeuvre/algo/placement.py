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
from ._constants import POIDS_REAIGUILLAGE, POIDS_MANOEUVRE_COUPLER, POIDS_OUVERTURE_SECTIONNEMENT, MAX_COMBINAISONS_PLACEMENT, MAX_COMBINAISONS_BEST_EFFORT
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


def _placement_automatique(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
) -> tuple[list[tuple[set[str], set[str]]], bool, str, list[list[str]]]:
    """
    Calcule un placement ``[(departs, sjb_ids)]`` réalisant ``topo_cible``.

    Si la cible est **réalisable**, retourne le placement complet
    (``faisable=True``). Sinon (option 2/4) : retourne un **diagnostic explicite**
    de l'infaisabilité dans ``message`` et un **placement partiel** « best-effort »
    plaçant le plus de nœuds possible, ``noeuds_non_places`` listant les départs
    des nœuds laissés à l'opérateur.

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

    # --- Scoping > 2 jeux de barres -----------------------------------------
    # L'algorithme de placement est conçu pour le double jeu de barres classique.
    # Sur un poste à > 2 jeux de barres (ex. niveaux supplémentaires 3B/4B), on se
    # restreint aux **2 jeux de barres principaux** ; les nœuds dont au moins un
    # départ n'atteint que les niveaux supplémentaires sont laissés à l'opérateur.
    forced_non_places: list[list[str]] = []
    extra_sjb: set[int] = set()
    if len(set(barre_par.values())) > 2:
        main_sjb, extra_sjb = _main_busbar_sjb(poste)
        sjb_nodes = [s for s in sjb_nodes if s in main_sjb]
        R = {eq: (rs & main_sjb) for eq, rs in R.items()}
        wired_sjb = {eq: (b if b in main_sjb else None)
                     for eq, b in wired_sjb.items()}
        kept: list[list[str]] = []
        for nd in nodes:
            if all(R.get(eq) for eq in nd):
                kept.append(nd)              # nœud entièrement sur le 2-JdB
            else:
                forced_non_places.append(nd)  # touche un niveau supplémentaire
        nodes = kept

    # Graphe des couplers entre SJB (pour la contrainte de connexité des groupes),
    # restreint aux SJB retenues (2 jeux de barres principaux si scoping).
    main_set = set(sjb_nodes)
    couplers = [cp for cp in _inter_sjb_couplers(poste)
                if cp.sjb_a in main_set and cp.sjb_b in main_set]
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

    # Recherche exhaustive d'une affectation **complète** (tous les nœuds placés),
    # uniquement si elle est possible (k ≤ nb SJB) et tient dans le garde-fou.
    best = None  # (cost, assign tuple)
    if k <= len(sjb_nodes) and k ** len(sjb_nodes) <= MAX_COMBINAISONS_PLACEMENT:
        for assign in _assignations_connexes(sjb_nodes, k, _groupe_connexe):
            node_sjbs: dict[int, set[int]] = {i: set() for i in range(k)}
            for j, ni in enumerate(assign):
                node_sjbs[ni].add(sjb_nodes[j])
            if any(not s for s in node_sjbs.values()):
                continue  # chaque nœud doit avoir ≥ 1 SJB
            # Groupes connexes dans le graphe des couplers
            if not all(_groupe_connexe(s) for s in node_sjbs.values()):
                continue
            # Faisabilité : chaque départ atteint une SJB de son nœud
            ok = True
            for i, deps in enumerate(nodes):
                sset = node_sjbs[i]
                if any(not (R[eq] & sset) for eq in deps):
                    ok = False
                    break
            if not ok:
                continue
            # Coût : ré-aiguillages (départ dont la barre câblée n'est pas dans son
            # groupe) + manœuvres de couplers (sectionnements pénalisés).
            reaig = sum(
                1 for i, deps in enumerate(nodes) for eq in deps
                if wired_sjb.get(eq) not in node_sjbs[i]
            )
            node_of_sjb = {sjb_nodes[j]: assign[j] for j in range(len(sjb_nodes))}
            cpl = 0
            sect = 0
            for cp, currently_closed in zip(couplers, cp_closed):
                same = node_of_sjb[cp.sjb_a] == node_of_sjb[cp.sjb_b]
                if same and not currently_closed:
                    cpl += 1
                elif (not same) and currently_closed:
                    cpl += 1
                    if cp.is_sectionnement:
                        sect += 1
            cost = (POIDS_REAIGUILLAGE * reaig
                    + POIDS_MANOEUVRE_COUPLER * cpl
                    + POIDS_OUVERTURE_SECTIONNEMENT * sect)
            # Tie-break **lex-min** sur ``assign`` : reproduit exactement le choix
            # de l'ancienne énumération ``itertools.product`` (premier min-coût en
            # ordre lexicographique). La sortie est donc strictement identique,
            # quel que soit l'ordre d'énumération des partitions connexes.
            if (best is None or cost < best[0]
                    or (cost == best[0] and assign < best[1])):
                best = (cost, assign)

    full_placement = None
    if best is not None:
        assign = best[1]
        node_sjbs = {i: set() for i in range(k)}
        for j, ni in enumerate(assign):
            node_sjbs[ni].add(sjb_nodes[j])
        full_placement = [
            (set(nodes[i]), {sjb_id[s] for s in node_sjbs[i]})
            for i in range(k)
        ]

    # Cas nominal : affectation complète ET aucun niveau supplémentaire écarté.
    if full_placement is not None and not forced_non_places:
        return full_placement, True, "OK", []

    # --- Dégradation : placement partiel + diagnostic explicite -------------
    raisons: list[str] = []
    if forced_non_places:
        raisons.append(_scoping_raison(sjb_id, extra_sjb, forced_non_places))
    if full_placement is not None:
        placement, partial_non_places = full_placement, []
    elif nodes:
        raisons.extend(_diagnostic_infaisabilite(nodes, R, sjb_id))
        placement, partial_non_places = _placement_best_effort(
            nodes, R, sjb_nodes, sjb_id, CG, wired_sjb)
    else:
        placement, partial_non_places = [], []
    non_places = partial_non_places + forced_non_places
    return placement, False, _message_non_realisable(raisons), non_places


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
