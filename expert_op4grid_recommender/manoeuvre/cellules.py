"""
manoeuvre/cellules.py  —  Étape 1.2
-------------------------------------
Détection des cellules de départ et de couplage à partir du graphe
node/breaker d'un voltage level.

Contexte
~~~~~~~~
Dans le modèle C++ NF/TOPO, une **cellule** (``CelluleDepartTopo``,
``CelluleBarresTopo``) est un sous-graphe préfabriqué par la bibliothèque NF.
En pypowsybl, ces cellules n'existent pas comme objets de premier ordre :
elles doivent être reconstruites par parcours de graphe.

Algorithme de détection
~~~~~~~~~~~~~~~~~~~~~~~
On part du graphe node/breaker complet du VL (cf. graph.py) et on partitionne
ses nœuds en cellules selon la règle suivante :

1. Tout nœud de type EQUIPMENT est le **point de départ** d'une cellule de
   départ. On effectue un BFS/DFS depuis ce nœud en traversant **tous** les
   switches et internal connections (qu'ils soient ouverts ou fermés),
   jusqu'à atteindre des nœuds de type BUSBAR_SECTION.

   - Les nœuds BUSBAR_SECTION sont inclus dans la cellule (ils en sont les
     bornes haute-tension) mais le BFS ne les traverse pas (on s'arrête à eux).
   - Deux équipements dont les BFS se chevauchent partagent la même cellule
     (cas des départs multiples / omnibus).

2. Les switches reliant **exclusivement** des nœuds BUSBAR_SECTION (sans passer
   par un nœud EQUIPMENT intermédiaire) forment les **cellules de couplage**.

3. Les nœuds restants (non atteints par aucun BFS) sont des nœuds INTERNAL
   isolés : probablement des nœuds neutres ou non utilisés.

Correspondance avec le C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``CelluleDepart``    ↔  ``CelluleDepartTopo``  (avec son graphe Boost)
- ``CelluleCouplage``  ↔  ``CelluleBarresTopo``  (pour les OC inter-barres)
- ``detecter_cellules``  ↔  appels à ``posteNF->construireCellules()``
  + la boucle ``buildTronconnement`` sur les cellules CELL_T_DEPART
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from .models import EquipmentType, SwitchKind
from .graph import get_node_attrs, busbar_nodes, equipment_nodes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de données des cellules
# ---------------------------------------------------------------------------

@dataclass
class SwitchInfo:
    """Résumé d'un switch appartenant à une cellule."""
    switch_id: str
    node1: int
    node2: int
    kind: SwitchKind
    open: bool

    @property
    def is_closed(self) -> bool:
        return not self.open

    @property
    def is_disconnector(self) -> bool:
        return self.kind == SwitchKind.DISCONNECTOR

    @property
    def is_breaker(self) -> bool:
        return self.kind == SwitchKind.BREAKER


@dataclass
class CelluleDepart:
    """
    Cellule de départ : sous-graphe reliant un équipement à ses SJB accessibles.

    Correspond à ``CelluleDepartTopo`` dans le C++ NF/TOPO.

    Attributes
    ----------
    equipment_id :
        Identifiant de l'équipement principal de la cellule (ligne, transfo,
        groupe, consommation…).
    equipment_type :
        Type d'équipement (LOAD, GENERATOR, LINE_SIDE1, etc.).
    all_nodes :
        Ensemble de tous les connectivity nodes de la cellule
        (équipement + nœuds intermédiaires + SJB atteintes).
    busbar_nodes :
        Nœuds de type BUSBAR_SECTION atteints par la cellule.
        Chaque SJB représente une barre à laquelle l'équipement peut être
        connecté via les switches de la cellule.
    switches :
        Tous les switches (BREAKER, DISCONNECTOR, LOAD_BREAK_SWITCH) de la
        cellule, qu'ils soient ouverts ou fermés.
    connected_busbars :
        Sous-ensemble de ``busbar_nodes`` effectivement connectés via un
        chemin de switches **tous fermés** depuis le nœud d'équipement.
    shared_equipment_ids :
        Si la cellule est de type omnibus/multiple, contient les identifiants
        des autres équipements qui partagent cette cellule.
        Vide dans le cas standard (cellule simple).
    subgraph :
        Sous-graphe NetworkX de la cellule (utilisé pour les analyses fines
        de connectivité, ex. ré-aiguillage, boucle courte).
    """
    equipment_id: str
    equipment_type: EquipmentType
    all_nodes: set[int] = field(default_factory=set)
    busbar_nodes: set[int] = field(default_factory=set)
    switches: list[SwitchInfo] = field(default_factory=list)
    connected_busbars: set[int] = field(default_factory=set)
    shared_equipment_ids: set[str] = field(default_factory=set)
    subgraph: Optional[nx.Graph] = field(default=None, repr=False)
    # Cache des chemins SA (équipement -> SJB) : structurel donc **indépendant de
    # l'état ouvert/fermé** des organes (le sous-graphe de la cellule est figé).
    # Voir ``disconnectors_vers_barre`` (appelé des dizaines de milliers de fois).
    _path_cache: dict = field(default_factory=dict, init=False, repr=False,
                              compare=False)

    # ------------------------------------------------------------------
    # Propriétés utilitaires
    # ------------------------------------------------------------------

    @property
    def busbar_section_ids(self) -> set[str]:
        """IDs (str) des SJB atteintes par cette cellule."""
        if self.subgraph is None:
            return set()
        return {
            self.subgraph.nodes[n].get("busbar_section_id")
            for n in self.busbar_nodes
            if self.subgraph.nodes[n].get("busbar_section_id") is not None
        }

    @property
    def connected_busbar_section_ids(self) -> set[str]:
        """IDs (str) des SJB effectivement connectées (chemin de switches fermés)."""
        if self.subgraph is None:
            return set()
        return {
            self.subgraph.nodes[n].get("busbar_section_id")
            for n in self.connected_busbars
            if self.subgraph.nodes[n].get("busbar_section_id") is not None
        }

    @property
    def nb_barres_accessibles(self) -> int:
        """Nombre de barres (SJB) accessibles depuis cet équipement."""
        return len(self.busbar_nodes)

    @property
    def is_connected(self) -> bool:
        """True si l'équipement est connecté à au moins une barre."""
        return len(self.connected_busbars) > 0

    @property
    def is_multiple(self) -> bool:
        """True si la cellule est partagée avec d'autres équipements (omnibus)."""
        return len(self.shared_equipment_ids) > 0

    @property
    def disconnectors(self) -> list[SwitchInfo]:
        """Liste des sectionneurs (SA) de la cellule."""
        return [s for s in self.switches if s.is_disconnector]

    @property
    def breakers(self) -> list[SwitchInfo]:
        """Liste des disjoncteurs (DJ) de la cellule."""
        return [s for s in self.switches if s.is_breaker]

    @property
    def is_reaiguillage(self) -> bool:
        """
        True si la cellule est en ré-aiguillage simplifié.

        Dans cette topologie RTE, l'équipement est relié directement à la barre
        via un unique sectionneur d'aiguillage (SA), sans disjoncteur propre.
        La cellule a exactement 2 nœuds (SJB + nœud équipement) et un seul
        switch de type DISCONNECTOR.
        """
        return len(self.breakers) == 0 and len(self.switches) == 1 and self.switches[0].is_disconnector

    def disconnectors_vers_barre(self, bbs_node: int) -> list[SwitchInfo]:
        """
        Retourne les sectionneurs sur le chemin menant à un nœud de SJB donné.

        Utilisé dans le ré-aiguillage (step 2.4.4) pour identifier les OC
        à manœuvrer afin de changer de barre.

        **Mémoïsé** par ``bbs_node`` : le chemin est structurel (sous-graphe de
        cellule figé), donc invariant ; la méthode est appelée en très grand
        nombre (``_sa_path_to_sjb``) lors du placement et du séquencement.

        Parameters
        ----------
        bbs_node :
            Nœud de connectivité de la SJB cible.
        """
        if bbs_node not in self._path_cache:
            self._path_cache[bbs_node] = self._compute_disconnectors_vers_barre(bbs_node)
        return self._path_cache[bbs_node]

    def _compute_disconnectors_vers_barre(self, bbs_node: int) -> list[SwitchInfo]:
        if self.subgraph is None or bbs_node not in self.subgraph:
            return []

        # Trouver le nœud d'équipement dans le sous-graphe
        eq_node = _find_equipment_node(self.subgraph, self.equipment_id)
        if eq_node is None:
            return []

        # Chemin le plus court dans le sous-graphe (tous switches confondus)
        try:
            path = nx.shortest_path(self.subgraph, eq_node, bbs_node)
        except nx.NetworkXNoPath:
            return []

        result = []
        for u, v in zip(path[:-1], path[1:]):
            sw_id = self.subgraph.edges[u, v].get("switch_id")
            kind = self.subgraph.edges[u, v].get("kind", SwitchKind.INTERNAL)
            if sw_id is not None and kind == SwitchKind.DISCONNECTOR:
                result.append(SwitchInfo(
                    switch_id=sw_id,
                    node1=u, node2=v,
                    kind=kind,
                    open=self.subgraph.edges[u, v].get("open", False),
                ))
        return result


@dataclass
class CelluleCouplage:
    """
    Cellule de couplage : switch(es) reliant deux sections de jeux de barres.

    Correspond à la partie couplage de ``CelluleBarresTopo`` dans le C++.

    Dans un poste à 2 barres, fermer le couplage électrique relie les deux
    barres, permettant aux départs d'un même nœud d'être répartis sur les
    deux barres.

    Attributes
    ----------
    switches :
        OC de couplage (généralement un BREAKER flanqué de deux DISCONNECTORS).
    busbar_node_1 :
        Nœud de la première SJB reliée par ce couplage.
    busbar_node_2 :
        Nœud de la deuxième SJB reliée par ce couplage.
    subgraph :
        Sous-graphe NetworkX des nœuds et switches de cette cellule.
    """
    switches: list[SwitchInfo] = field(default_factory=list)
    busbar_node_1: Optional[int] = None
    busbar_node_2: Optional[int] = None
    subgraph: Optional[nx.Graph] = field(default=None, repr=False)

    @property
    def is_closed(self) -> bool:
        """True si tous les switches actifs du couplage sont fermés."""
        return all(not s.open for s in self.switches
                   if s.kind in (SwitchKind.BREAKER, SwitchKind.LOAD_BREAK_SWITCH))

    @property
    def main_breaker(self) -> Optional[SwitchInfo]:
        """Disjoncteur principal du couplage (s'il existe)."""
        breakers = [s for s in self.switches if s.is_breaker]
        return breakers[0] if breakers else None


# ---------------------------------------------------------------------------
# Résultat global de la détection
# ---------------------------------------------------------------------------

@dataclass
class CellulesVL:
    """
    Résultat de la détection de cellules pour un voltage level.

    Attributes
    ----------
    voltage_level_id :
        Identifiant du voltage level analysé.
    cellules_depart :
        Liste des cellules de départ détectées, une par équipement
        (ou par groupe d'équipements partagés).
    cellules_couplage :
        Liste des cellules de couplage détectées.
    noeuds_non_attribues :
        Nœuds du graphe n'appartenant à aucune cellule (nœuds neutres /
        non utilisés dans la configuration courante).
    """
    voltage_level_id: str
    cellules_depart: list[CelluleDepart] = field(default_factory=list)
    cellules_couplage: list[CelluleCouplage] = field(default_factory=list)
    noeuds_non_attribues: set[int] = field(default_factory=set)

    def get_cellule_depart(self, equipment_id: str) -> Optional[CelluleDepart]:
        """Retourne la cellule de départ d'un équipement donné, ou None."""
        for c in self.cellules_depart:
            if c.equipment_id == equipment_id or equipment_id in c.shared_equipment_ids:
                return c
        return None

    def resume(self) -> str:
        return (
            f"VL '{self.voltage_level_id}': "
            f"{len(self.cellules_depart)} départ(s), "
            f"{len(self.cellules_couplage)} couplage(s), "
            f"{len(self.noeuds_non_attribues)} nœud(s) non attribué(s)"
        )


# ---------------------------------------------------------------------------
# Algorithme principal de détection
# ---------------------------------------------------------------------------

def detecter_cellules(G: nx.Graph, voltage_level_id: str) -> CellulesVL:
    """
    Détecte et construit les cellules de départ et de couplage du voltage level.

    Parameters
    ----------
    G :
        Graphe node/breaker du voltage level, construit par ``graph.build_vl_graph``.
    voltage_level_id :
        Identifiant du voltage level (pour le logging et le résultat).

    Returns
    -------
    CellulesVL
        Toutes les cellules identifiées, plus les nœuds non attribués.

    Algorithm
    ---------
    Phase A — Cellules de départ :
        Pour chaque nœud EQUIPMENT, BFS dans ``G`` (structural BFS :
        on traverse tous les switches/internal connections quelle que soit leur
        position ouverte/fermée) jusqu'aux BUSBAR_SECTION. Le sous-graphe
        induit constitue la cellule. Deux équipements atteignant les mêmes
        nœuds intermédiaires partagent leur cellule (départs multiples).

    Phase B — Cellules de couplage :
        Les switches dont les deux extrémités sont des BUSBAR_SECTION (ou
        n'ont pas été incluses dans une cellule de départ) forment les
        cellules de couplage.

    Phase C — Nœuds non attribués :
        Nœuds restants non couverts par A ou B.
    """
    result = CellulesVL(voltage_level_id=voltage_level_id)

    all_nodes = set(G.nodes())
    attributed_nodes: set[int] = set()

    # -----------------------------------------------------------------------
    # Phase A : Cellules de départ
    # -----------------------------------------------------------------------
    eq_nodes = equipment_nodes(G)
    bbs_nodes_set = set(busbar_nodes(G))

    # Map node → CelluleDepart pour détecter les cellules partagées
    node_to_cellule: dict[int, CelluleDepart] = {}

    for eq_node in eq_nodes:
        node_attrs = get_node_attrs(G, eq_node)
        eq_id = node_attrs.equipment_id
        eq_type = node_attrs.equipment_type or EquipmentType.UNKNOWN

        # BFS structurel depuis le nœud d'équipement
        visited_nodes, visited_edges = _structural_bfs(G, eq_node, bbs_nodes_set)

        # Vérifier si certains nœuds intermédiaires (non-SJB) visités appartiennent
        # déjà à une cellule de départ — cas des départs partagés / omnibus.
        # On EXCLUT les SJB : elles sont partagées par toutes les cellules et
        # ne doivent PAS déclencher une fusion.
        overlap_cellule: Optional[CelluleDepart] = None
        for n in visited_nodes:
            if n in bbs_nodes_set:
                continue  # SJB partagées : ne pas déclencher de fusion
            if n in node_to_cellule:
                overlap_cellule = node_to_cellule[n]
                break

        if overlap_cellule is not None:
            # Fusion : l'équipement courant partage sa cellule
            overlap_cellule.shared_equipment_ids.add(eq_id)
            overlap_cellule.all_nodes.update(visited_nodes)
            overlap_cellule.busbar_nodes.update(visited_nodes & bbs_nodes_set)
            # Mise à jour des switches et du sous-graphe
            _enrich_cellule(G, overlap_cellule, visited_nodes, visited_edges)
            for n in visited_nodes:
                node_to_cellule[n] = overlap_cellule
            attributed_nodes.update(visited_nodes)
            logger.debug(
                "Équipement '%s' fusionné dans la cellule de '%s' (départ multiple)",
                eq_id, overlap_cellule.equipment_id,
            )
        else:
            # Nouvelle cellule
            cellule = CelluleDepart(
                equipment_id=eq_id,
                equipment_type=eq_type,
                all_nodes=visited_nodes,
                busbar_nodes=visited_nodes & bbs_nodes_set,
            )
            _enrich_cellule(G, cellule, visited_nodes, visited_edges)
            result.cellules_depart.append(cellule)
            for n in visited_nodes:
                node_to_cellule[n] = cellule
            attributed_nodes.update(visited_nodes)
            logger.debug(
                "Nouvelle cellule de départ : équipement '%s', "
                "%d nœuds, %d SJB, %d switches",
                eq_id, len(visited_nodes), len(cellule.busbar_nodes),
                len(cellule.switches),
            )

    # -----------------------------------------------------------------------
    # Phase B : Cellules de couplage
    # -----------------------------------------------------------------------
    # Propriété clé du modèle node/breaker RTE :
    # - Chaque cellule de départ a ses propres nœuds intermédiaires dédiés
    #   (ils ne sont jamais partagés avec une autre cellule de départ).
    # - Les SJB (busbar sections) sont les seuls nœuds partagés entre cellules.
    # - Un couplage relie deux (ou plus) SJB via des nœuds intermédiaires dédiés
    #   qui n'appartiennent à aucune cellule de départ.
    #
    # Algorithme :
    #   1. Calculer "departure_internal_nodes" = nœuds non-SJB appartenant à
    #      au moins une cellule de départ.
    #   2. Construire le "coupler_subgraph" = G restreint à l'ensemble
    #      (SJB nodes) ∪ (nodes - departure_internal_nodes).
    #   3. Chaque composante connexe de coupler_subgraph qui contient ≥ 2 SJB
    #      est une cellule de couplage.

    departure_internal_nodes: set[int] = set()
    for c in result.cellules_depart:
        departure_internal_nodes.update(
            n for n in c.all_nodes if n not in bbs_nodes_set
        )

    coupler_candidate_nodes = bbs_nodes_set | (all_nodes - departure_internal_nodes)
    coupler_G = G.subgraph(coupler_candidate_nodes)

    processed_comp: set[frozenset] = set()

    for component in nx.connected_components(coupler_G):
        comp_frozen = frozenset(component)
        if comp_frozen in processed_comp:
            continue
        processed_comp.add(comp_frozen)

        bbs_in_comp = component & bbs_nodes_set
        if len(bbs_in_comp) < 2:
            # Pas un couplage (moins de 2 barres reliées)
            continue

        bbs_sorted = sorted(bbs_in_comp)
        if len(bbs_in_comp) > 2:
            logger.warning(
                "Composante de couplage avec %d SJB (poste ≥ 3 barres ?) : %s. "
                "Seulement les 2 premières SJB seront enregistrées dans la cellule.",
                len(bbs_in_comp), bbs_in_comp,
            )

        sub = coupler_G.subgraph(component)
        coup_switches = [
            SwitchInfo(
                switch_id=d["switch_id"],
                node1=u, node2=v,
                kind=d.get("kind", SwitchKind.DISCONNECTOR),
                open=d.get("open", False),
            )
            for u, v, d in sub.edges(data=True)
            if d.get("switch_id") is not None
        ]

        cellule_coup = CelluleCouplage(
            busbar_node_1=bbs_sorted[0],
            busbar_node_2=bbs_sorted[1],
            switches=coup_switches,
            subgraph=sub.copy(),
        )
        result.cellules_couplage.append(cellule_coup)
        attributed_nodes.update(component)

        logger.debug(
            "Nouvelle cellule de couplage : SJB %s ↔ %s, %d switch(es)",
            cellule_coup.busbar_node_1, cellule_coup.busbar_node_2,
            len(cellule_coup.switches),
        )

    # -----------------------------------------------------------------------
    # Phase C : Nœuds non attribués
    # -----------------------------------------------------------------------
    result.noeuds_non_attribues = all_nodes - attributed_nodes

    logger.info(result.resume())
    return result


# ---------------------------------------------------------------------------
# Calcul des SJB connectées (chemin de switches fermés)
# ---------------------------------------------------------------------------

def calculer_connected_busbars(cellule: CelluleDepart) -> set[int]:
    """
    Calcule les SJB effectivement connectées à l'équipement via un chemin
    de switches **tous fermés**.

    Met à jour ``cellule.connected_busbars`` et le retourne.

    Correspond à ``numTJBDepartIsConnected()`` / ``numTJBDepartComplexeIsConnected()``
    dans le C++.
    """
    if cellule.subgraph is None:
        cellule.connected_busbars = set()
        return cellule.connected_busbars

    eq_node = _find_equipment_node(cellule.subgraph, cellule.equipment_id)
    if eq_node is None:
        cellule.connected_busbars = set()
        return cellule.connected_busbars

    # Sous-graphe des switches fermés uniquement (+ internal connections)
    closed_edges = [
        (u, v) for u, v, d in cellule.subgraph.edges(data=True)
        if not d.get("open", False)   # closed = conducteur
    ]
    closed_G = nx.Graph()
    closed_G.add_nodes_from(cellule.subgraph.nodes(data=True))
    closed_G.add_edges_from(closed_edges)

    # BFS depuis le nœud d'équipement dans le graphe fermé
    try:
        reachable = nx.node_connected_component(closed_G, eq_node)
    except nx.NodeNotFound:
        cellule.connected_busbars = set()
        return cellule.connected_busbars

    cellule.connected_busbars = reachable & cellule.busbar_nodes
    return cellule.connected_busbars


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _structural_bfs(
    G: nx.Graph,
    start: int,
    stop_nodes: set[int],
) -> tuple[set[int], set[tuple[int, int]]]:
    """
    BFS structurel depuis ``start``, s'arrêtant aux nœuds de ``stop_nodes``.

    On traverse **tous** les switches (ouverts ou fermés) et les internal
    connections, car on analyse la structure du poste, pas son état électrique.
    Les nœuds ``stop_nodes`` (BUSBAR_SECTION) sont inclus dans le résultat
    mais ne sont pas expandés (le BFS s'arrête à eux).

    Returns
    -------
    visited_nodes : set[int]
        Tous les nœuds atteints, stop_nodes inclus.
    visited_edges : set[tuple[int, int]]
        Arêtes traversées (paires (u, v) canoniques, u < v).
    """
    visited_nodes: set[int] = {start}
    visited_edges: set[tuple[int, int]] = set()
    queue: deque[int] = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            edge_key = (min(node, neighbor), max(node, neighbor))
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                visited_edges.add(edge_key)
                # Ne pas expandre les SJB (frontière de la cellule)
                if neighbor not in stop_nodes:
                    queue.append(neighbor)
            elif edge_key not in visited_edges:
                visited_edges.add(edge_key)

    return visited_nodes, visited_edges



def _enrich_cellule(
    G: nx.Graph,
    cellule: CelluleDepart,
    visited_nodes: set[int],
    visited_edges: set[tuple[int, int]],
) -> None:
    """
    Peuple les champs ``switches``, ``subgraph`` et ``connected_busbars``
    d'une cellule de départ.
    """
    # Sous-graphe induit (copié pour isolation)
    cellule.subgraph = G.subgraph(visited_nodes).copy()

    # Switches de la cellule (on exclut les internal connections)
    cellule.switches = [
        SwitchInfo(
            switch_id=G.edges[u, v]["switch_id"],
            node1=u, node2=v,
            kind=G.edges[u, v].get("kind", SwitchKind.DISCONNECTOR),
            open=G.edges[u, v].get("open", False),
        )
        for u, v in visited_edges
        if G.edges[u, v].get("switch_id") is not None
    ]

    # Calcul initial de la connectivité électrique
    calculer_connected_busbars(cellule)


def _find_equipment_node(subgraph: nx.Graph, equipment_id: str) -> Optional[int]:
    """Trouve le nœud de connectivité de l'équipement dans le sous-graphe."""
    for node, data in subgraph.nodes(data=True):
        if data.get("equipment_id") == equipment_id:
            return node
    return None
