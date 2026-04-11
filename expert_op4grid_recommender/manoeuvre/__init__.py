"""
manoeuvre — Module de portage de l'algorithme topo nodale → détaillée
======================================================================

Ce module est le portage Python de la bibliothèque C++ ``libTOPO`` (projet
TOPO Apogée, RTE), en utilisant pypowsybl comme modèle de données réseau.

Structure
---------
models.py    — Structures de données partagées (SwitchKind, NodeType, …)
graph.py     — Étape 1.1 : extraction du graphe node/breaker d'un voltage level
cellules.py  — Étape 1.2 : détection des cellules de départ et de couplage
troncons.py  — Étape 1.3-1.4 : attribution des nœuds et tronçonnement (à venir)
topologie.py — Étape 1.5-1.6 : TopologieNodale, PosteTopologique (à venir)
algo.py      — Phase 2 : algorithme nodale → détaillée (à venir)

Usage rapide
------------
>>> import pypowsybl as pp
>>> from expert_op4grid_recommender.manoeuvre import build_vl_graph, detecter_cellules
>>>
>>> network = pp.network.create_four_substations_node_breaker_network()
>>> G = build_vl_graph(network, "S1VL2")
>>> cellules = detecter_cellules(G, "S1VL2")
>>> print(cellules.resume())
"""

from .models import (
    NodeType,
    EquipmentType,
    SwitchKind,
    CelluleType,
    NodeAttrs,
    EdgeAttrs,
)
from .graph import (
    build_vl_graph,
    get_node_attrs,
    get_edge_attrs,
    busbar_nodes,
    equipment_nodes,
    TopologyError,
)
from .cellules import (
    SwitchInfo,
    CelluleDepart,
    CelluleCouplage,
    CellulesVL,
    detecter_cellules,
    calculer_connected_busbars,
)

__all__ = [
    # models
    "NodeType", "EquipmentType", "SwitchKind", "CelluleType",
    "NodeAttrs", "EdgeAttrs",
    # graph
    "build_vl_graph", "get_node_attrs", "get_edge_attrs",
    "busbar_nodes", "equipment_nodes", "TopologyError",
    # cellules
    "SwitchInfo", "CelluleDepart", "CelluleCouplage", "CellulesVL",
    "detecter_cellules", "calculer_connected_busbars",
]
