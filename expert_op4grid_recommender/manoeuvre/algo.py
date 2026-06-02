"""
manoeuvre/algo.py  —  Phase 2 : algorithme nodale → détaillée
---------------------------------------------------------------
Calcule la séquence d'organes de coupure (OC) à manœuvrer pour passer de
l'état détaillé courant d'un poste à une topologie nodale cible.

Correspondance C++ : ``Topologie::determineTopoCompleteCible`` (TOPOPoste.cc:3944)
et ses sous-routines (``connectAndDeconnectOuvrageHS``, ``evalueEtatCouplage``,
``identifySuperTronconnement``, ``getTronconnementBesoinReaiguillage2barres``,
``reaiguillage2barres``, ``listeDordre``).

Point d'entrée unique
~~~~~~~~~~~~~~~~~~~~~~~
``determiner_topo_complete_cible(poste, topo_cible)`` intègre toute la chaîne :
faisabilité → **placement automatique** nœud→sections de barres
(``_placement_automatique``) → séquenceur général
(``determiner_manoeuvres_avec_sections``) → vérification. Il gère
indifféremment 1 barre, 2 barres, et la **création de nœuds supplémentaires**
par ouverture de sectionnement, sans placement explicite.

Couverture
~~~~~~~~~~
- Postes 1 barre et 2 barres standard (couplage + ré-aiguillage boucle
  courte/longue).                                                        [OK]
- Création d'un nœud au-delà du nombre de barres par **ouverture de
  sectionnement de barre** (dé-énergisation préalable de la section).    [OK]
- Ordonnancement type ``listeDordre`` : fermeture des couplages d'abord,
  ré-aiguillages boucle courte, ouverture des sectionnements (hors tension),
  ouverture des couplages, ré-aiguillages boucle longue, puis suppression
  des manœuvres sans effet.                                              [OK]
- Contrôle de court-circuit avant fermeture d'un couplage.               [OK]
- Vérification post-manœuvre (recalcul de la topologie nodale).          [OK]

Règle du sectionneur de barre
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Un sectionneur de barre ne se manœuvre que hors charge. Pour scinder une barre
en deux nœuds, la section à isoler est d'abord mise hors tension (ses départs
ré-aiguillés sur l'autre barre en boucle courte), puis le sectionnement est
ouvert, puis les départs du nouveau nœud y sont ré-aiguillés en boucle longue.

Limites connues (documentées, cf. doc C++) :
- Ré-aiguillage d'omnibus complexes (départs multiples scindés)          [partiel]
- Contrôle de court-circuit : vérifie l'égalité de nœud cible avant
  fermeture de couplage (pas de calcul de potentiel fin)                 [simplifié]
- Postes ≥ 3 barres physiques / topologies multi-tronçons non chaînées   [partiel]
- Nœuds mêlant départs connectés et déconnectés                          [partiel]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import networkx as nx

from .models import SwitchKind
from .cellules import CelluleDepart, SwitchInfo
from .topologie import (
    TopologieNodale,
    PosteTopologique,
)
from .troncons import Troncon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de sortie
# ---------------------------------------------------------------------------

@dataclass
class Manoeuvre:
    """Une opération unitaire sur un organe de coupure."""
    switch_id: str
    action: Literal["OPEN", "CLOSE"]
    raison: str
    type_boucle: Optional[Literal["COURTE", "LONGUE"]] = None

    def __repr__(self) -> str:  # pragma: no cover - debug
        b = f" [{self.type_boucle}]" if self.type_boucle else ""
        return f"{self.action:5s} {self.switch_id}  ({self.raison}){b}"


@dataclass
class ResultatManoeuvres:
    """Résultat complet de l'algorithme."""
    voltage_level_id: str
    topo_initiale: TopologieNodale
    topo_cible: TopologieNodale
    manoeuvres: list[Manoeuvre] = field(default_factory=list)
    departs_reaiguilles: set[str] = field(default_factory=set)
    couplages_modifies: list[str] = field(default_factory=list)
    is_changed: bool = False
    is_verified: bool = False
    topo_obtenue: Optional[TopologieNodale] = None
    message: str = ""

    @property
    def nb_manoeuvres(self) -> int:
        return len(self.manoeuvres)

    def resume(self) -> str:
        lines = [
            f"Manœuvres VL '{self.voltage_level_id}' : {self.nb_manoeuvres} OC, "
            f"changed={self.is_changed}, verified={self.is_verified}"
        ]
        for i, m in enumerate(self.manoeuvres, 1):
            lines.append(f"  {i:2d}. {m!r}")
        if self.message:
            lines.append(f"  -> {self.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def determiner_topo_complete_cible(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
) -> ResultatManoeuvres:
    """
    Calcule la séquence de manœuvres pour atteindre ``topo_cible`` depuis
    l'état détaillé courant de ``poste``.

    Parameters
    ----------
    poste :
        Vue complète du poste (cellules, tronçonnement, topologie nodale
        courante), construite par ``PosteTopologique.from_graph``.
    topo_cible :
        Topologie nodale visée.

    Returns
    -------
    ResultatManoeuvres
    """
    res = ResultatManoeuvres(
        voltage_level_id=poste.voltage_level_id,
        topo_initiale=poste.topologie_nodale,
        topo_cible=topo_cible,
    )

    # On travaille sur une copie du graphe pour simuler les manœuvres.
    G = poste.graph.copy() if poste.graph is not None else None
    if G is None:
        res.message = "Graphe absent : impossible de calculer les manœuvres."
        return res

    # --- Court-circuit : la topologie courante satisfait déjà la cible ----
    if poste.topologie_nodale.meme_topologie(topo_cible):
        res.is_changed = False
        res.is_verified = True
        res.topo_obtenue = poste.topologie_nodale
        res.message = "La topologie courante satisfait déjà la cible (aucune manœuvre)."
        return res

    # --- Phase 2.1 : faisabilité (départs cibles présents) ----------------
    departs_poste = {c.equipment_id for c in poste.cellules.cellules_depart}
    for c in poste.cellules.cellules_depart:
        departs_poste |= set(c.shared_equipment_ids)
    manquants = set(topo_cible.noeud_par_depart) - departs_poste
    if manquants:
        res.topo_obtenue = poste.topologie_nodale
        res.message = f"Départs cibles absents du poste : {sorted(manquants)}"
        return res

    # --- Phases 2.2-2.4 : placement automatique nœud -> sections de barres -
    placement, faisable, msg = _placement_automatique(poste, topo_cible)
    if not faisable:
        res.topo_obtenue = poste.topologie_nodale
        res.message = msg
        return res

    # --- Délégation au séquenceur général (couplage + sectionnement) -------
    core = determiner_manoeuvres_avec_sections(poste, placement)
    core.topo_initiale = poste.topologie_nodale
    core.topo_cible = topo_cible
    core.is_verified = bool(
        core.topo_obtenue and topo_cible.meme_topologie(core.topo_obtenue)
    )
    core.is_changed = bool(core.manoeuvres)
    core.message = (
        "Topologie cible atteinte et vérifiée." if core.is_verified
        else "La topologie obtenue ne correspond pas à la cible "
             f"(obtenu {core.topo_obtenue.nb_noeuds if core.topo_obtenue else 0} "
             f"nœud(s), visé {topo_cible.nb_noeuds})."
    )
    return core


# ---------------------------------------------------------------------------
# Phases 2.2-2.4 — Placement automatique des nœuds sur les sections de barres
# ---------------------------------------------------------------------------
#
# Généralise ``evalueEtatCouplage`` + ``identifySuperTronconnement`` +
# ``getTronconnementBesoinReaiguillage2barres`` du C++ : à partir d'une
# topologie nodale cible, on attribue à chaque nœud un ensemble de SJB.
#
# Modèle (double barre RTE) :
# - chaque départ atteint une « classe de position » = l'ensemble des SJB
#   qu'il peut rejoindre (une SJB par barre, à sa section) ;
# - un nœud occupe, pour chacune de ses positions, **une seule barre** (ses SJB
#   restent ainsi connectées via les sectionnements internes à la barre) ;
# - deux nœuds différents sur la même barre ⇒ ouverture du sectionnement ;
# - nb de nœuds « mixtes » (≥ 2 positions) ≤ nb de barres.
#
# On choisit l'affectation barre↔nœud qui minimise (ré-aiguillages + pénalité
# d'ouverture de sectionnement), en respectant les départs fixes.
# ---------------------------------------------------------------------------

def _placement_automatique(
    poste: PosteTopologique,
    topo_cible: TopologieNodale,
) -> tuple[list[tuple[set[str], set[str]]], bool, str]:
    """
    Calcule un placement ``[(departs, sjb_ids)]`` réalisant ``topo_cible``.

    Returns
    -------
    (placement, faisable, message)
    """
    import itertools
    from collections import defaultdict

    G = poste.graph
    barre_par = poste.tronconnement.barre_par_busbar
    barres = sorted(set(barre_par.values()))
    nb_barres = len(barres)
    sjb_id = {n: G.nodes[n].get("busbar_section_id") for n in barre_par}

    # Connexité courante **par équipement** (et non par cellule) : pour les
    # omnibus, chaque départ a son propre disjoncteur ; un groupe isolé ne doit
    # pas hériter de la connexité de son co-locataire.
    sjb_nodes = set(barre_par)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if not d.get("open", False):
            H.add_edge(u, v)
    eq_node = {data.get("equipment_id"): n for n, data in G.nodes(data=True)
               if data.get("equipment_id")}

    # Classe de position (SJB atteignables) + connexité courante par départ
    R: dict[str, frozenset] = {}
    connected: dict[str, bool] = {}
    cur_sjb: dict[str, Optional[int]] = {}
    for c in poste.cellules.cellules_depart:
        for eq in {c.equipment_id} | set(c.shared_equipment_ids):
            R[eq] = frozenset(c.busbar_nodes)
            en = eq_node.get(eq)
            reached = ({s for s in sjb_nodes if en is not None and en in H
                        and nx.has_path(H, en, s)} if en is not None else set())
            connected[eq] = bool(reached)
            cur_sjb[eq] = min(reached) if reached else None

    def slot(cls: frozenset, barre: int) -> Optional[int]:
        for s in cls:
            if barre_par.get(s) == barre:
                return s
        return None

    # Nœuds à placer : ceux ayant ≥ 1 départ actuellement connecté.
    # Les nœuds entièrement déconnectés (ex. groupes isolés) sont laissés tels
    # quels (ils restent sur leur nœud courant).
    nodes: list[dict] = []
    for noeud in topo_cible.noeuds.values():
        deps = [e for e in noeud.equipment_ids if e in R and connected[e]]
        if not deps:
            continue
        positions = {R[e] for e in deps}
        fixed: dict[frozenset, int] = {}
        for e in deps:
            barres_e = {barre_par[s] for s in R[e] if s in barre_par}
            if len(barres_e) == 1:
                fixed[R[e]] = next(iter(barres_e))
        nodes.append({"departs": deps, "positions": positions, "fixed": fixed})

    if not nodes:
        return [], True, "Aucun nœud connecté à placer."

    # Faisabilité globale
    nb_mixtes = sum(1 for nd in nodes if len(nd["positions"]) >= 2)
    if nb_mixtes > nb_barres:
        return ([], False,
                f"{nb_mixtes} nœuds mixtes pour {nb_barres} barre(s) : "
                "topologie impossible (il faudrait plus de jeux de barres).")
    demand: dict[frozenset, int] = defaultdict(int)
    for nd in nodes:
        for p in nd["positions"]:
            demand[p] += 1
    for p, d in demand.items():
        if d > nb_barres:
            return ([], False,
                    f"{d} nœuds requièrent la même section pour {nb_barres} "
                    "barre(s) : topologie impossible.")

    # Recherche de la meilleure affectation (une barre par nœud)
    best = None
    for combo in itertools.product(barres, repeat=len(nodes)):
        slots_used: dict[tuple, int] = {}
        ok = True
        for i, nd in enumerate(nodes):
            b = combo[i]
            if any(fb != b for fb in nd["fixed"].values()):
                ok = False
                break
            for p in nd["positions"]:
                if slot(p, b) is None:
                    ok = False
                    break
                key = (p, b)
                if key in slots_used:
                    ok = False
                    break
                slots_used[key] = i
            if not ok:
                break
        if not ok:
            continue
        # Coût : ré-aiguillages + pénalité ouverture de sectionnement
        reaig = 0
        for i, nd in enumerate(nodes):
            b = combo[i]
            for e in nd["departs"]:
                s = slot(R[e], b)
                if s is not None and cur_sjb.get(e) != s:
                    reaig += 1
        per_barre: dict[int, set] = defaultdict(set)
        for (p, b), i in slots_used.items():
            per_barre[b].add(i)
        sect = sum(len(idxs) - 1 for idxs in per_barre.values() if len(idxs) > 1)
        cost = reaig + 10 * sect
        if best is None or cost < best[0]:
            best = (cost, combo)

    if best is None:
        return [], False, "Aucune affectation de barres réalisable."

    combo = best[1]
    placement: list[tuple[set[str], set[str]]] = []
    for i, nd in enumerate(nodes):
        b = combo[i]
        sjbs = {sjb_id[slot(p, b)] for p in nd["positions"] if slot(p, b) is not None}
        placement.append((set(nd["departs"]), sjbs))
    return placement, True, "OK"


def _set_switch(G: nx.Graph, switch_id: str, open_: bool) -> None:
    """Modifie l'état d'un switch (par son id) dans le graphe simulé."""
    for u, v, d in G.edges(data=True):
        if d.get("switch_id") == switch_id:
            d["open"] = open_
            return


# ===========================================================================
# Placement sur sections de barres — règle du sectionnement de barre
# ===========================================================================
#
# Un **sectionneur de barre** (sectionnement, DISCONNECTOR entre deux SJB) ne se
# manœuvre que **hors charge**. Pour l'ouvrir afin de créer un nœud
# supplémentaire (au-delà du nombre de jeux de barres), la section à isoler doit
# d'abord être mise **hors tension** : tous ses départs sont ré-aiguillés sur
# l'autre barre (boucle courte, tant que le couplage est fermé), puis le
# sectionnement est ouvert, puis les départs du nouveau nœud sont ré-aiguillés
# (boucle longue) sur la section désormais isolée.
#
# Un **couplage** (travée à DJ/BREAKER) coupe le courant nominal : son DJ
# s'ouvre directement.
#
# Cette logique permet de réaliser des cibles à *plus de nœuds que de barres*.
# ---------------------------------------------------------------------------


@dataclass
class _InterSjbCoupler:
    """Liaison entre deux SJB : sectionnement (SA seuls) ou couplage (avec DJ)."""
    sjb_a: int
    sjb_b: int
    switch_ids: list[str]
    breaker_ids: list[str]

    @property
    def is_sectionnement(self) -> bool:
        return not self.breaker_ids


def _inter_sjb_couplers(poste: PosteTopologique) -> list[_InterSjbCoupler]:
    """
    Recense les liaisons inter-SJB (sectionnements et couplages) d'un poste,
    en contractant les nœuds intermédiaires du sous-graphe de couplage.
    """
    G = poste.graph
    bb_nodes = set(poste.tronconnement.barre_par_busbar)

    # Sous-graphe de couplage : SJB + nœuds hors cellules de départ
    departure_internal: set[int] = set()
    for c in poste.cellules.cellules_depart:
        departure_internal.update(n for n in c.all_nodes if n not in bb_nodes)
    coupler_nodes = bb_nodes | (set(G.nodes()) - departure_internal)
    coupler_G = G.subgraph(coupler_nodes)

    couplers: list[_InterSjbCoupler] = []
    seen: set[frozenset] = set()
    bb_list = sorted(bb_nodes)
    for i, a in enumerate(bb_list):
        for b in bb_list[i + 1:]:
            key = frozenset((a, b))
            if key in seen:
                continue
            # Chemin entre a et b ne traversant aucune autre SJB
            others = bb_nodes - {a, b}
            H = coupler_G.subgraph(coupler_nodes - others)
            try:
                path = nx.shortest_path(H, a, b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            sw_ids, brk_ids = [], []
            for u, v in zip(path, path[1:]):
                d = H.edges[u, v]
                sid = d.get("switch_id")
                if sid is None:
                    continue
                sw_ids.append(sid)
                if d.get("kind") == SwitchKind.BREAKER:
                    brk_ids.append(sid)
            if sw_ids:
                couplers.append(_InterSjbCoupler(a, b, sw_ids, brk_ids))
                seen.add(key)
    return couplers


def determiner_manoeuvres_avec_sections(
    poste: PosteTopologique,
    placement: list[tuple[set[str], set[str]]],
) -> ResultatManoeuvres:
    """
    Calcule la séquence de manœuvres pour réaliser un **placement explicite**
    de nœuds sur des sections de jeux de barres, en respectant la règle du
    sectionnement de barre (dé-énergisation avant ouverture).

    Parameters
    ----------
    poste :
        Vue complète du poste.
    placement :
        Liste de ``(departs, sjb_ids)`` : chaque entrée décrit un nœud cible,
        l'ensemble de ses départs et l'ensemble des SJB (``busbar_section_id``)
        qu'il occupe. Les départs non cités restent inchangés.

    Returns
    -------
    ResultatManoeuvres
    """
    vl = poste.voltage_level_id
    G = poste.graph.copy()
    cells = poste.cellules

    # Map busbar_section_id -> node
    sjb_node_par_id = {
        G.nodes[n].get("busbar_section_id"): n
        for n in poste.tronconnement.barre_par_busbar
    }

    # --- cible nodale (pour la vérification) -------------------------------
    # Les départs placés forment les nœuds explicites ; les départs non cités
    # conservent leur nœud courant (ex. générateurs isolés).
    groupes = [sorted(d) for d, _ in placement]
    places = {eq for d, _ in placement for eq in d}
    for noeud in poste.topologie_nodale.noeuds.values():
        reste = sorted(noeud.equipment_ids - places)
        if reste:
            groupes.append(reste)
    topo_cible = TopologieNodale.from_node_groups(vl, groupes)
    res = ResultatManoeuvres(
        voltage_level_id=vl,
        topo_initiale=poste.topologie_nodale,
        topo_cible=topo_cible,
    )

    # --- résolution départ -> SJB cible ------------------------------------
    target_sjb: dict[str, int] = {}
    node_de_sjb: dict[int, int] = {}  # sjb -> index de nœud
    for idx, (departs, sjb_ids) in enumerate(placement):
        sjb_set = {sjb_node_par_id[s] for s in sjb_ids if s in sjb_node_par_id}
        for s in sjb_set:
            node_de_sjb[s] = idx
        for eq in departs:
            cell = cells.get_cellule_depart(eq)
            if cell is None:
                continue
            reachable = cell.busbar_nodes & sjb_set
            if not reachable:
                res.message = (
                    f"Départ '{eq}' ne peut atteindre aucune SJB de son nœud."
                )
                return res
            target_sjb[eq] = min(reachable)

    # --- couplers à ouvrir / fermer ----------------------------------------
    # - à ouvrir  : entre SJB de nœuds différents,
    # - à fermer  : entre SJB d'un même nœud actuellement séparées (fusion).
    couplers = _inter_sjb_couplers(poste)
    to_open: list[_InterSjbCoupler] = []
    to_close: list[_InterSjbCoupler] = []
    for cp in couplers:
        na, nb = node_de_sjb.get(cp.sjb_a), node_de_sjb.get(cp.sjb_b)
        if na is None or nb is None:
            continue
        if na != nb:
            to_open.append(cp)
        elif any(_is_open(G, sid) for sid in cp.switch_ids):
            to_close.append(cp)

    # --- groupes SJB finaux (couplers gardés fermés) -----------------------
    sjb_graph = nx.Graph()
    sjb_graph.add_nodes_from(poste.tronconnement.barre_par_busbar)
    open_ids = {sid for cp in to_open for sid in cp.switch_ids}
    for cp in couplers:
        if cp not in to_open:
            sjb_graph.add_edge(cp.sjb_a, cp.sjb_b)
    groupe_sjb = {}
    for gid, comp in enumerate(nx.connected_components(sjb_graph)):
        for s in comp:
            groupe_sjb[s] = gid

    # Référence = groupe portant le plus de départs cibles (le « tronc »)
    from collections import Counter
    poids = Counter(groupe_sjb[s] for s in target_sjb.values() if s in groupe_sjb)
    ref_group = poids.most_common(1)[0][0] if poids else None
    ref_sjbs = {s for s, g in groupe_sjb.items() if g == ref_group}

    # SJB « derrière un sectionnement » : au moment où l'on ouvre les
    # sectionnements (phase C), les couplages (DJ) sont *encore fermés*. On
    # calcule donc la connectivité en gardant fermés tous les couplers SAUF les
    # sectionnements à ouvrir. La composante ne contenant pas la référence est
    # la section à mettre hors tension.
    to_open_sect_ids = {sid for cp in to_open if cp.is_sectionnement
                        for sid in cp.switch_ids}
    energ_graph = nx.Graph()
    energ_graph.add_nodes_from(poste.tronconnement.barre_par_busbar)
    for cp in couplers:
        if any(sid in to_open_sect_ids for sid in cp.switch_ids):
            continue  # sectionnement ouvert en phase C
        energ_graph.add_edge(cp.sjb_a, cp.sjb_b)
    ref_repr = next(iter(ref_sjbs), None)
    energises = (nx.node_connected_component(energ_graph, ref_repr)
                 if ref_repr is not None else set())
    sjb_isoles = set(poste.tronconnement.barre_par_busbar) - energises

    manoeuvres: list[Manoeuvre] = []
    reaiguilles: set[str] = set()

    def parking_sjb(eq: str, target: int) -> Optional[int]:
        """SJB tampon (hors section isolée) accessible par le départ."""
        cell = cells.get_cellule_depart(eq)
        for bb in cell.busbar_nodes:
            if bb != target and bb not in sjb_isoles:
                return bb
        return None

    # --- Phase 0 : fermeture des couplages nécessaires (listeDordre §1) -----
    # On ferme d'abord les couplages requis (fusion de barres dans un même
    # nœud) pour préparer les ré-aiguillages en boucle courte. Contrôle de
    # court-circuit : on ne ferme que si les deux SJB visent le même nœud
    # (même potentiel cible), sinon on signale le risque et on s'abstient.
    for cp in to_close:
        if node_de_sjb.get(cp.sjb_a) != node_de_sjb.get(cp.sjb_b):
            logger.warning(
                "Fermeture de couplage %s ignorée : risque de court-circuit "
                "(SJB de nœuds cibles différents).", cp.switch_ids,
            )
            continue
        for sid in cp.switch_ids:
            if _is_open(G, sid):
                _set_switch(G, sid, False)
                manoeuvres.append(Manoeuvre(
                    switch_id=sid, action="CLOSE",
                    raison="fermeture couplage de barres (préparation)",
                ))

    # --- Phase A/B : ré-aiguillages boucle courte (couplage encore fermé) ---
    parkings: dict[str, int] = {}
    for eq, tgt in sorted(target_sjb.items()):
        if tgt in sjb_isoles:
            buf = parking_sjb(eq, tgt)
            if buf is None:
                res.message = f"Pas de SJB tampon pour '{eq}'."
                return res
            parkings[eq] = tgt
            if _reaiguiller_vers_sjb(G, cells, eq, buf, "COURTE", manoeuvres):
                reaiguilles.add(eq)
        else:
            if _reaiguiller_vers_sjb(G, cells, eq, tgt, "COURTE", manoeuvres):
                reaiguilles.add(eq)

    # --- Phase C : ouverture des sectionnements (section hors tension) -----
    for cp in to_open:
        if not cp.is_sectionnement:
            continue
        # Vérifier que la section isolée (côté hors tension) ne porte plus de
        # départ câblé.
        cote = cp.sjb_a if cp.sjb_a in sjb_isoles else cp.sjb_b
        encore = [eq for eq in target_sjb
                  if cote in _wired_sjbs(G, cells, eq)]
        for sid in cp.switch_ids:
            if not _is_open(G, sid):
                _set_switch(G, sid, True)
                etat = "hors tension" if not encore else "ATTENTION sous tension"
                manoeuvres.append(Manoeuvre(
                    switch_id=sid, action="OPEN",
                    raison=f"ouverture sectionnement de barre (section {etat})",
                ))

    # --- Phase D : ouverture des couplages (DJ) ----------------------------
    for cp in to_open:
        if cp.is_sectionnement:
            continue
        for sid in cp.breaker_ids:
            if not _is_open(G, sid):
                _set_switch(G, sid, True)
                manoeuvres.append(Manoeuvre(
                    switch_id=sid, action="OPEN",
                    raison="ouverture couplage de barres",
                ))

    # --- Phase E : ré-aiguillage boucle longue vers sections isolées -------
    for eq in sorted(parkings):
        if _reaiguiller_vers_sjb(G, cells, eq, parkings[eq], "LONGUE", manoeuvres):
            reaiguilles.add(eq)

    # --- Optimisation : suppression des manœuvres sans effet (listeDordre) -
    manoeuvres = _optimiser_sequence(poste, manoeuvres)

    res.manoeuvres = manoeuvres
    res.departs_reaiguilles = reaiguilles
    res.couplages_modifies = [sid for cp in (to_open + to_close)
                              for sid in cp.switch_ids]

    # --- Vérification ------------------------------------------------------
    topo_obtenue = TopologieNodale.from_graph(G, vl)
    res.topo_obtenue = topo_obtenue
    res.is_verified = topo_cible.meme_topologie(topo_obtenue)
    res.is_changed = bool(manoeuvres)
    res.message = (
        "Topologie cible atteinte et vérifiée."
        if res.is_verified
        else f"Cible non atteinte (obtenu {topo_obtenue.nb_noeuds} nœuds, "
             f"visé {topo_cible.nb_noeuds})."
    )
    return res


# ---------------------------------------------------------------------------
# Helpers bas niveau opérant sur le graphe « live »
# ---------------------------------------------------------------------------

def _optimiser_sequence(
    poste: PosteTopologique, manoeuvres: list[Manoeuvre]
) -> list[Manoeuvre]:
    """
    Supprime les manœuvres sans effet en rejouant la séquence depuis l'état
    initial : une manœuvre qui place un OC dans l'état où il se trouve déjà est
    redondante et retirée. Les bascules réelles (ex. ouverture/fermeture d'un DJ
    en boucle longue) sont conservées.
    """
    G = poste.graph.copy()
    out: list[Manoeuvre] = []
    for m in manoeuvres:
        want_open = (m.action == "OPEN")
        if _is_open(G, m.switch_id) == want_open:
            continue
        _set_switch(G, m.switch_id, want_open)
        out.append(m)
    return out


def _is_open(G: nx.Graph, switch_id: str) -> bool:
    for _, _, d in G.edges(data=True):
        if d.get("switch_id") == switch_id:
            return bool(d.get("open", False))
    return True


def _eq_node(G: nx.Graph, eq_id: str) -> Optional[int]:
    for n, d in G.nodes(data=True):
        if d.get("equipment_id") == eq_id:
            return n
    return None


def _sa_path_to_sjb(cell: CelluleDepart, sjb_node: int) -> list[str]:
    """IDs des sectionneurs (SA) sur le chemin départ -> SJB."""
    return [s.switch_id for s in cell.disconnectors_vers_barre(sjb_node)]


def _own_breakers_to_sjb(
    cell: CelluleDepart, sjb_node: int, eq_id: str | None = None
) -> list[str]:
    """
    **Disjoncteur d'ensemble de la cellule** à manœuvrer pour mettre le départ
    hors tension lors d'un ré-aiguillage en boucle longue.

    Règle (cf. docs/manoeuvre_regles.md, R7) : on n'ouvre que le DJ situé
    **côté sélecteurs de barre** (entre les sectionneurs d'aiguillage et le
    reste de la cellule). Ouvrir ce seul disjoncteur dé-énergise la cellule et
    suffit pour basculer ensuite les sectionneurs ; on n'ouvre **pas** les
    disjoncteurs propres aux équipements situés en aval (cas omnibus : un même
    DJ de cellule alimente plusieurs équipements).

    Méthode : on parcourt le chemin équipement → SJB depuis le **côté barre**,
    on saute les sectionneurs (sélecteurs de barre), puis on collecte le(s)
    disjoncteur(s) en série jusqu'à un nœud de branchement (degré > 2, signe
    d'un point omnibus partagé) ou l'équipement.
    """
    if cell.subgraph is None:
        return [b.switch_id for b in cell.breakers]
    sg = cell.subgraph
    eq_id = eq_id or cell.equipment_id
    eq_node = next((n for n, d in sg.nodes(data=True)
                    if d.get("equipment_id") == eq_id), None)
    if eq_node is None:
        return [b.switch_id for b in cell.breakers]
    try:
        path = nx.shortest_path(sg, eq_node, sjb_node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    res: list[str] = []
    collecting = False
    # Parcours depuis le côté barre (path[::-1] = [sjb, …, équipement])
    rev = path[::-1]
    for a, b in zip(rev, rev[1:]):
        d = sg.edges[a, b]
        kind = d.get("kind")
        sid = d.get("switch_id")
        if kind == SwitchKind.BREAKER and sid:
            res.append(sid)
            collecting = True
            # Nœud de branchement omnibus atteint -> DJ de cellule identifié.
            if b != eq_node and sg.degree(b) > 2:
                break
        elif kind == SwitchKind.DISCONNECTOR:
            if collecting:
                break          # sectionneur en aval du DJ -> fin
            continue            # sélecteurs de barre en amont -> on saute
        elif collecting:
            break
    return list(dict.fromkeys(res))


def _wired_sjbs(G: nx.Graph, cells, eq_id: str) -> set[int]:
    """SJB vers lesquelles le départ est *câblé* (SA fermés), indépendamment
    de la connectivité électrique globale."""
    cell = cells.get_cellule_depart(eq_id)
    if cell is None:
        return set()
    res = set()
    for bb in cell.busbar_nodes:
        sa = _sa_path_to_sjb(cell, bb)
        if sa and all(not _is_open(G, s) for s in sa):
            res.add(bb)
    return res


def _reaiguiller_vers_sjb(
    G: nx.Graph,
    cells,
    eq_id: str,
    target_sjb: int,
    boucle: Literal["COURTE", "LONGUE"],
    manoeuvres: list[Manoeuvre],
) -> bool:
    """
    Ré-aiguille un départ vers une SJB cible. Retourne True si des manœuvres
    ont été générées.

    - COURTE : ferme le SA cible, ouvre les autres SA (départ sous tension).
    - LONGUE : ouvre le DJ propre, bascule les SA, referme le DJ.
    """
    cell = cells.get_cellule_depart(eq_id)
    if cell is None:
        return False

    sjb_id = (G.nodes[target_sjb].get("busbar_section_id") or str(target_sjb))
    sa_cible = _sa_path_to_sjb(cell, target_sjb)
    if not sa_cible:
        return False
    # Déjà câblé sur la cible et nulle part ailleurs ?
    deja = all(not _is_open(G, s) for s in sa_cible) and all(
        _is_open(G, s)
        for bb in cell.busbar_nodes if bb != target_sjb
        for s in _sa_path_to_sjb(cell, bb)
    )
    if deja:
        return False

    n_before = len(manoeuvres)
    djs = _own_breakers_to_sjb(cell, target_sjb, eq_id)

    if boucle == "LONGUE":
        for dj in djs:
            if not _is_open(G, dj):
                _set_switch(G, dj, True)
                manoeuvres.append(Manoeuvre(
                    switch_id=dj, action="OPEN",
                    raison=f"mise hors tension '{eq_id}' (boucle longue)",
                    type_boucle="LONGUE",
                ))

    for sa in sa_cible:
        if _is_open(G, sa):
            _set_switch(G, sa, False)
            manoeuvres.append(Manoeuvre(
                switch_id=sa, action="CLOSE",
                raison=f"ré-aiguillage '{eq_id}' vers {sjb_id}",
                type_boucle=boucle,
            ))
    for bb in cell.busbar_nodes:
        if bb == target_sjb:
            continue
        for sa in _sa_path_to_sjb(cell, bb):
            if not _is_open(G, sa):
                _set_switch(G, sa, True)
                bb_id = G.nodes[bb].get("busbar_section_id") or str(bb)
                manoeuvres.append(Manoeuvre(
                    switch_id=sa, action="OPEN",
                    raison=f"'{eq_id}' quitte {bb_id}",
                    type_boucle=boucle,
                ))

    if boucle == "LONGUE":
        for dj in djs:
            _set_switch(G, dj, False)
            manoeuvres.append(Manoeuvre(
                switch_id=dj, action="CLOSE",
                raison=f"remise sous tension '{eq_id}' (boucle longue)",
                type_boucle="LONGUE",
            ))

    return len(manoeuvres) > n_before
