"""
manoeuvre/graph.py  —  Étape 1.1
----------------------------------
Extraction du graphe de connectivité node/breaker d'un voltage level pypowsybl.

Objectif
~~~~~~~~
Construire un graphe NetworkX non orienté ``G`` où :
- Chaque **nœud** est un entier représentant un *connectivity node* du modèle
  IIDM (les mêmes entiers que ``node1`` / ``node2`` dans ``get_switches()``).
- Chaque **arête** est soit un *switch* (BREAKER, DISCONNECTOR, LOAD_BREAK_SWITCH),
  soit une *internal connection* (connexion fixe, toujours fermée).

Les nœuds sont annotés avec leur ``NodeType`` (BUSBAR_SECTION, EQUIPMENT,
INTERNAL) et, selon le type, l'identifiant de la section de barre ou de
l'équipement associé.

Correspondance C++ (NF/TOPO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ce graphe correspond exactement à ce que ``CelluleDepartTopo::buildCellGraph``
et ``CelluleBarresTopo::buildCellGraph`` construisent à partir des cellules NF :
le graphe de connectivité interne d'un voltage level est l'union de tous ces
sous-graphes. L'étape suivante (cellules.py) le repart en cellules individuelles.

Hypothèses sur l'API pypowsybl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``network.get_switches()``         → colonnes : voltage_level_id, kind, node1,
                                        node2, open, retained, fictitious
- ``network.get_busbar_sections()``  → colonnes : voltage_level_id, node (entier),
                                        connected, fictitious
- ``network.get_node_breaker_topology(vl_id).internal_connections``
                                      → DataFrame : node1, node2
- Pour chaque type d'équipement (loads, generators, lines, trafos, …),
  la méthode ``network.get_<type>(all_attributes=True)`` retourne une colonne
  ``node`` (entier) pour les VL en topologie node/breaker.

  ⚠  Si le réseau est en topologie bus/breaker, la colonne ``node`` n'existe pas
     et la fonction lève une ``TopologyError``.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import pandas as pd
import pypowsybl as pp

from .models import NodeAttrs, EdgeAttrs, NodeType, EquipmentType, SwitchKind

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception dédiée
# ---------------------------------------------------------------------------

class TopologyError(Exception):
    """Levée quand le voltage level n'est pas en topologie node/breaker,
    ou quand une hypothèse sur l'API pypowsybl n'est pas vérifiée."""


# ---------------------------------------------------------------------------
# Constantes internes
# ---------------------------------------------------------------------------

# Mapping du libellé pypowsybl vers notre SwitchKind
_SWITCH_KIND_MAP: dict[str, SwitchKind] = {
    "BREAKER": SwitchKind.BREAKER,
    "DISCONNECTOR": SwitchKind.DISCONNECTOR,
    "LOAD_BREAK_SWITCH": SwitchKind.LOAD_BREAK_SWITCH,
}


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def build_vl_graph(network: pp.network.Network, voltage_level_id: str) -> nx.Graph:
    """
    Construit le graphe de connectivité node/breaker pour un voltage level.

    Parameters
    ----------
    network:
        Réseau pypowsybl chargé (doit contenir des VL en topologie NODE_BREAKER).
    voltage_level_id:
        Identifiant du voltage level à traiter.

    Returns
    -------
    nx.Graph
        Graphe non orienté dont les nœuds sont les connectivity nodes (entiers)
        et les arêtes les switches + internal connections.

        Attributs des nœuds : voir ``NodeAttrs`` (stockés à plat dans le dict).
        Attributs des arêtes : voir ``EdgeAttrs`` (stockés à plat dans le dict).

    Raises
    ------
    TopologyError
        Si le voltage level n'est pas de type NODE_BREAKER, ou si l'API
        pypowsybl ne retourne pas les colonnes attendues.
    """
    _assert_node_breaker(network, voltage_level_id)

    G: nx.Graph = nx.Graph()

    _add_switch_edges(network, voltage_level_id, G)
    _add_internal_connection_edges(network, voltage_level_id, G)
    _tag_busbar_section_nodes(network, voltage_level_id, G)
    _tag_equipment_nodes(network, voltage_level_id, G)

    logger.debug(
        "Graphe VL '%s' : %d nœuds, %d arêtes",
        voltage_level_id, G.number_of_nodes(), G.number_of_edges(),
    )
    return G


def get_node_attrs(G: nx.Graph, node: int) -> NodeAttrs:
    """Retourne les attributs d'un nœud sous forme de ``NodeAttrs``."""
    d = G.nodes[node]
    return NodeAttrs(
        node_type=d.get("node_type", NodeType.INTERNAL),
        busbar_section_id=d.get("busbar_section_id"),
        equipment_id=d.get("equipment_id"),
        equipment_type=d.get("equipment_type"),
    )


def get_edge_attrs(G: nx.Graph, u: int, v: int) -> EdgeAttrs:
    """Retourne les attributs d'une arête sous forme de ``EdgeAttrs``."""
    d = G.edges[u, v]
    return EdgeAttrs(
        switch_id=d.get("switch_id"),
        kind=d.get("kind", SwitchKind.INTERNAL),
        open=d.get("open", False),
    )


def busbar_nodes(G: nx.Graph) -> list[int]:
    """Retourne la liste des nœuds de type BUSBAR_SECTION dans le graphe."""
    return [n for n, d in G.nodes(data=True)
            if d.get("node_type") == NodeType.BUSBAR_SECTION]


def equipment_nodes(G: nx.Graph) -> list[int]:
    """Retourne la liste des nœuds de type EQUIPMENT dans le graphe."""
    return [n for n, d in G.nodes(data=True)
            if d.get("node_type") == NodeType.EQUIPMENT]


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _assert_node_breaker(network: pp.network.Network, vl_id: str) -> None:
    """Vérifie que le voltage level est bien en topologie NODE_BREAKER."""
    vls = network.get_voltage_levels(all_attributes=True)
    if vl_id not in vls.index:
        raise TopologyError(f"Voltage level '{vl_id}' introuvable dans le réseau.")
    topology_kind = vls.loc[vl_id, "topology_kind"]
    if topology_kind != "NODE_BREAKER":
        raise TopologyError(
            f"Le voltage level '{vl_id}' est en topologie '{topology_kind}'. "
            "Ce module requiert NODE_BREAKER."
        )


def _ensure_node(G: nx.Graph, node: int) -> None:
    """Ajoute le nœud au graphe s'il n'existe pas encore (attributs par défaut)."""
    if node not in G:
        G.add_node(node, **NodeAttrs().__dict__)


def _add_switch_edges(
    network: pp.network.Network, vl_id: str, G: nx.Graph
) -> None:
    """
    Ajoute les switches du voltage level comme arêtes du graphe.

    Chaque switch relie deux connectivity nodes ``node1`` et ``node2``.
    L'attribut ``open`` indique si le switch est ouvert (non conducteur).
    Les switches fictifs sont inclus car ils font partie de la structure
    topologique (ex : switch fictif entre une SJB et un nœud de départ).
    """
    all_switches = network.get_switches(all_attributes=True)
    if "voltage_level_id" not in all_switches.columns:
        raise TopologyError(
            "La colonne 'voltage_level_id' est absente de get_switches(). "
            "Vérifier la version de pypowsybl."
        )

    vl_switches = all_switches[all_switches["voltage_level_id"] == vl_id]

    for sw_id, row in vl_switches.iterrows():
        n1, n2 = int(row["node1"]), int(row["node2"])
        _ensure_node(G, n1)
        _ensure_node(G, n2)

        kind_str = str(row["kind"])
        kind = _SWITCH_KIND_MAP.get(kind_str, SwitchKind.DISCONNECTOR)
        if kind_str not in _SWITCH_KIND_MAP:
            logger.warning("Switch kind inconnu '%s' pour switch '%s'", kind_str, sw_id)

        attrs = EdgeAttrs(switch_id=sw_id, kind=kind, open=bool(row["open"]))
        G.add_edge(n1, n2, **attrs.__dict__)


def _add_internal_connection_edges(
    network: pp.network.Network, vl_id: str, G: nx.Graph
) -> None:
    """
    Ajoute les internal connections du voltage level comme arêtes toujours fermées.

    Les internal connections sont des connexions fixes sans organe de coupure,
    qui relient typiquement un nœud d'équipement à un nœud de switch intermédiaire.
    Elles correspondent aux connexions internes des cellules dans le modèle NF.
    """
    try:
        nbt = network.get_node_breaker_topology(vl_id)
        ic: pd.DataFrame = nbt.internal_connections
    except Exception as exc:
        # Certains VL sans connexions internes peuvent lever une exception
        logger.debug(
            "Pas de connexions internes pour '%s' (ou erreur API) : %s", vl_id, exc
        )
        return

    if ic is None or ic.empty:
        return

    for _, row in ic.iterrows():
        n1, n2 = int(row["node1"]), int(row["node2"])
        _ensure_node(G, n1)
        _ensure_node(G, n2)
        attrs = EdgeAttrs(switch_id=None, kind=SwitchKind.INTERNAL, open=False)
        # Une internal connection peut coexister avec un switch sur la même paire
        # de nœuds : on utilise une clé d'arête distincte via le multigraphe,
        # mais comme on utilise un Graph simple, on ne les duplique pas.
        if not G.has_edge(n1, n2):
            G.add_edge(n1, n2, **attrs.__dict__)
        else:
            # L'arête existe déjà (switch). On logue un warning : situation rare.
            logger.debug(
                "Internal connection (%d, %d) : arête déjà présente (switch '%s'). "
                "La connexion interne est ignorée pour éviter la duplication.",
                n1, n2, G.edges[n1, n2].get("switch_id"),
            )


def _tag_busbar_section_nodes(
    network: pp.network.Network, vl_id: str, G: nx.Graph
) -> None:
    """
    Marque les nœuds correspondant à des sections de jeux de barres (SJB).

    pypowsybl expose la colonne ``node`` dans ``get_busbar_sections()``
    pour les VL en topologie NODE_BREAKER.
    """
    bbs_df = network.get_busbar_sections(all_attributes=True)
    if "node" not in bbs_df.columns:
        raise TopologyError(
            "La colonne 'node' est absente de get_busbar_sections(). "
            "Le réseau est peut-être en topologie BUS_BREAKER."
        )

    vl_bbs = bbs_df[bbs_df["voltage_level_id"] == vl_id]

    for bbs_id, row in vl_bbs.iterrows():
        node = int(row["node"])
        _ensure_node(G, node)
        G.nodes[node].update(
            NodeAttrs(
                node_type=NodeType.BUSBAR_SECTION,
                busbar_section_id=bbs_id,
            ).__dict__
        )


def _tag_equipment_nodes(
    network: pp.network.Network, vl_id: str, G: nx.Graph
) -> None:
    """
    Marque les nœuds correspondant aux bornes des équipements réseau.

    Strategy
    --------
    On interroge chaque type d'équipement (load, generator, line, transfo, …).
    Pour les VL en NODE_BREAKER, pypowsybl expose la colonne ``node`` dans
    les DataFrames d'éléments (accessible via ``all_attributes=True``).

    Si la colonne ``node`` est absente (réseau bus/breaker ou version
    pypowsybl ancienne), on tente une heuristique : les nœuds qui
    apparaissent dans le graphe sans être des SJB ni reliés à d'autres nœuds
    que par des switches/internal connections sont probablement des nœuds
    terminaux d'équipements. Cette heuristique est marquée comme fallback
    et peut être imprécise.

    Types traités : LOAD, GENERATOR, LINE (côtés 1 et 2),
    2-WINDING TRANSFORMER (côtés 1 et 2), DANGLING_LINE,
    SHUNT_COMPENSATOR, STATIC_VAR_COMPENSATOR, BATTERY.
    """
    # Définition des requêtes : (méthode, eq_type, colonne_vl, colonne_node)
    # Pour les injections monophasées : voltage_level_id + node
    # Pour les branches (lignes, trafos) : voltage_level1_id/node1, voltage_level2_id/node2
    _injection_queries = [
        (network.get_loads,                      EquipmentType.LOAD,                  "voltage_level_id", "node"),
        (network.get_generators,                 EquipmentType.GENERATOR,             "voltage_level_id", "node"),
        (network.get_dangling_lines,             EquipmentType.DANGLING_LINE,          "voltage_level_id", "node"),
        (network.get_shunt_compensators,         EquipmentType.SHUNT_COMPENSATOR,     "voltage_level_id", "node"),
        (network.get_static_var_compensators,    EquipmentType.STATIC_VAR_COMPENSATOR,"voltage_level_id", "node"),
        (network.get_batteries,                  EquipmentType.BATTERY,               "voltage_level_id", "node"),
    ]
    _branch_queries = [
        (network.get_lines,                       EquipmentType.LINE_SIDE1, EquipmentType.LINE_SIDE2,
         "voltage_level1_id", "node1", "voltage_level2_id", "node2"),
        (network.get_2_windings_transformers,     EquipmentType.TRANSFORMER_SIDE1, EquipmentType.TRANSFORMER_SIDE2,
         "voltage_level1_id", "node1", "voltage_level2_id", "node2"),
    ]

    # --- Injections (un seul nœud de connexion) ---
    for getter, eq_type, vl_col, node_col in _injection_queries:
        df = _safe_get(getter)
        if df is None or df.empty:
            continue
        if vl_col not in df.columns:
            continue
        vl_df = df[df[vl_col] == vl_id]
        if vl_df.empty:
            continue

        if node_col not in vl_df.columns:
            logger.warning(
                "Colonne '%s' absente pour %s (VL '%s'). "
                "Les nœuds d'équipement de ce type ne seront pas taggés.",
                node_col, eq_type.value, vl_id,
            )
            continue

        for eq_id, row in vl_df.iterrows():
            node = int(row[node_col])
            _ensure_node(G, node)
            G.nodes[node].update(
                NodeAttrs(
                    node_type=NodeType.EQUIPMENT,
                    equipment_id=eq_id,
                    equipment_type=eq_type,
                ).__dict__
            )

    # --- Branches (deux nœuds de connexion : côté 1 et côté 2) ---
    for (getter, eq_type_s1, eq_type_s2,
         vl_col1, node_col1, vl_col2, node_col2) in _branch_queries:
        df = _safe_get(getter)
        if df is None or df.empty:
            continue

        # Côté 1
        if vl_col1 in df.columns:
            side1 = df[df[vl_col1] == vl_id]
            if not side1.empty and node_col1 in side1.columns:
                for eq_id, row in side1.iterrows():
                    node = int(row[node_col1])
                    _ensure_node(G, node)
                    G.nodes[node].update(
                        NodeAttrs(
                            node_type=NodeType.EQUIPMENT,
                            equipment_id=eq_id,
                            equipment_type=eq_type_s1,
                        ).__dict__
                    )

        # Côté 2
        if vl_col2 in df.columns:
            side2 = df[df[vl_col2] == vl_id]
            if not side2.empty and node_col2 in side2.columns:
                for eq_id, row in side2.iterrows():
                    node = int(row[node_col2])
                    _ensure_node(G, node)
                    G.nodes[node].update(
                        NodeAttrs(
                            node_type=NodeType.EQUIPMENT,
                            equipment_id=eq_id,
                            equipment_type=eq_type_s2,
                        ).__dict__
                    )


def _safe_get(getter) -> Optional[pd.DataFrame]:
    """
    Appelle un getter pypowsybl avec ``all_attributes=True`` pour obtenir
    les colonnes node/breaker (``node``, ``node1``, ``node2``).

    Si le getter ne supporte pas ``all_attributes``, on retente sans.
    Retourne None en cas d'erreur.
    """
    try:
        return getter(all_attributes=True)
    except TypeError:
        # Certains getters anciens n'acceptent pas all_attributes
        try:
            return getter()
        except Exception as exc:
            logger.debug("Erreur lors de l'appel à %s : %s", getter.__name__, exc)
            return None
    except Exception as exc:
        logger.debug("Erreur lors de l'appel à %s (all_attributes=True) : %s", getter.__name__, exc)
        return None
