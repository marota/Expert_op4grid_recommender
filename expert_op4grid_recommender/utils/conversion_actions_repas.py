#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender.

"""
REPAS to Grid2Op Action Conversion Module

This module provides optimized functions for converting REPAS actions to Grid2Op format.

Key optimizations:
- **NO VARIANT CLONING**: Uses analytical topology computation via Union-Find algorithm
- Vectorized pandas operations instead of row-by-row iteration
- Pre-cached network topology data for batch processing
- Set-based lookups for O(1) voltage level filtering

The main performance gain comes from avoiding pypowsybl's clone_variant/remove_variant
operations, which are expensive. Instead, we simulate the switch topology changes
in pure Python using a Union-Find (disjoint set) data structure.
"""

import json
from typing import List, Dict, Optional, Set, Tuple
import warnings
from collections import defaultdict

import pandas as pd
from pandas import DataFrame
from pypowsybl.network import Network

from expert_op4grid_recommender.utils import repas


# =============================================================================
# Union-Find (Disjoint Set) for topology computation
# =============================================================================

class UnionFind:
    """
    Union-Find data structure for computing connected bus components.
    
    Used to determine which nodes are electrically connected after switch operations.
    This avoids the need to actually modify the network and read back bus assignments.
    """
    __slots__ = ['parent', 'rank']
    
    def __init__(self, elements: List[str]):
        """Initialize with each element in its own set."""
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
    
    def find(self, x: str) -> str:
        """Find the root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: str, y: str) -> None:
        """Unite the sets containing x and y using union by rank."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
    
    def get_component_mapping(self) -> Dict[str, int]:
        """
        Get mapping from each element to its component number (1-indexed).
        
        Returns
        -------
        Dict[str, int]
            Mapping from element ID to bus number (1, 2, 3, ...)
        """
        roots = {}
        component_num = 1
        result = {}
        
        for elem in self.parent:
            root = self.find(elem)
            if root not in roots:
                roots[root] = component_num
                component_num += 1
            result[elem] = roots[root]
        
        return result


# =============================================================================
# Cached Network Topology Data
# =============================================================================

class NetworkTopologyCache:
    """
    Cache for network switch topology data.
    
    Pre-computes and stores:
    - Switch connectivity (which nodes each switch connects)
    - Element-to-node mappings (which node each load/gen/branch terminal is on)
    - Initial switch states
    
    This allows computing the resulting bus topology for any switch configuration
    without modifying the actual pypowsybl network.
    """
    
    def __init__(self, n: Network):
        """
        Build topology cache from network.
        
        Parameters
        ----------
        n : Network
            The pypowsybl network object.
        """
        self.network = n
        
        # Get switch data with topology information
        self._switches_df = n.get_switches(attributes=[
            'voltage_level_id', 'bus_breaker_bus1_id', 'bus_breaker_bus2_id', 'open'
        ])
        
        # Get element DataFrames with bus connections
        self._loads_df = n.get_loads(attributes=['bus_id', 'voltage_level_id'])
        self._generators_df = n.get_generators(attributes=['bus_id', 'voltage_level_id'])
        self._shunts_df = n.get_shunt_compensators(attributes=['bus_id', 'voltage_level_id'])
        self._branches_df = n.get_branches(attributes=[
            'bus1_id', 'bus2_id', 'voltage_level1_id', 'voltage_level2_id'
        ])
        
        # Pre-compute per-voltage-level data structures
        self._build_vl_topology_data()
    
    def _build_vl_topology_data(self):
        """Build per-voltage-level topology structures for fast lookup."""
        self._vl_switches = defaultdict(list)  # vl_id -> list of (switch_id, node1, node2)
        self._vl_switch_states = {}  # switch_id -> is_open (initial state)
        self._vl_nodes = defaultdict(set)  # vl_id -> set of all nodes
        
        # Process switches
        for switch_id, row in self._switches_df.iterrows():
            vl_id = row['voltage_level_id']
            node1 = row['bus_breaker_bus1_id']
            node2 = row['bus_breaker_bus2_id']
            is_open = row['open']
            
            if pd.notna(node1) and pd.notna(node2):
                self._vl_switches[vl_id].append((switch_id, node1, node2))
                self._vl_switch_states[switch_id] = is_open
                self._vl_nodes[vl_id].add(node1)
                self._vl_nodes[vl_id].add(node2)
        
        # Element-to-node mappings
        self._load_to_node = dict(zip(self._loads_df.index, self._loads_df['bus_id']))
        self._gen_to_node = dict(zip(self._generators_df.index, self._generators_df['bus_id']))
        self._shunt_to_node = dict(zip(self._shunts_df.index, self._shunts_df['bus_id']))
        
        # Branch terminals to nodes
        self._branch_or_to_node = dict(zip(self._branches_df.index, self._branches_df['bus1_id']))
        self._branch_ex_to_node = dict(zip(self._branches_df.index, self._branches_df['bus2_id']))
        
        # Element-to-VL mappings for filtering
        self._load_to_vl = dict(zip(self._loads_df.index, self._loads_df['voltage_level_id']))
        self._gen_to_vl = dict(zip(self._generators_df.index, self._generators_df['voltage_level_id']))
        self._shunt_to_vl = dict(zip(self._shunts_df.index, self._shunts_df['voltage_level_id']))
    
    def compute_bus_assignments(
        self, 
        switch_changes: Dict[str, bool],
        impacted_vl_ids: Set[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Compute bus assignments after applying switch changes.
        
        Uses Union-Find to compute connected components (buses) efficiently
        WITHOUT modifying the pypowsybl network.
        
        Parameters
        ----------
        switch_changes : Dict[str, bool]
            Mapping from switch_id to new open state (True=open, False=closed).
        impacted_vl_ids : Set[str]
            Set of voltage level IDs where topology needs to be recomputed.
            
        Returns
        -------
        Dict[str, Dict[str, int]]
            Nested dict: {vl_id: {node_id: bus_number}}
        """
        result = {}
        
        for vl_id in impacted_vl_ids:
            nodes = self._vl_nodes.get(vl_id, set())
            if not nodes:
                result[vl_id] = {}
                continue
            
            # Initialize Union-Find with all nodes in this VL
            uf = UnionFind(list(nodes))
            
            # Process all switches in this VL
            for switch_id, node1, node2 in self._vl_switches.get(vl_id, []):
                # Determine effective switch state
                if switch_id in switch_changes:
                    is_open = switch_changes[switch_id]
                else:
                    is_open = self._vl_switch_states.get(switch_id, True)
                
                # If switch is closed, unite the two nodes
                if not is_open:
                    uf.union(node1, node2)
            
            # Get the component (bus) number for each node
            result[vl_id] = uf.get_component_mapping()
        
        return result
    
    def get_element_bus_assignments(
        self,
        node_to_bus: Dict[str, Dict[str, int]],
        impacted_vl_ids: Set[str]
    ) -> dict:
        """
        Convert node-to-bus mappings to element-to-bus mappings.
        
        Parameters
        ----------
        node_to_bus : Dict[str, Dict[str, int]]
            Mapping from voltage_level_id to (node_id -> bus_number).
        impacted_vl_ids : Set[str]
            Set of impacted voltage level IDs.
            
        Returns
        -------
        dict
            The set_bus dictionary for Grid2Op format.
        """
        lines_or_id = {}
        lines_ex_id = {}
        loads_id = {}
        generators_id = {}
        shunts_id = {}
        
        # Process branches
        for branch_id, row in self._branches_df.iterrows():
            vl1 = row['voltage_level1_id']
            vl2 = row['voltage_level2_id']
            
            if vl1 in impacted_vl_ids:
                node = self._branch_or_to_node.get(branch_id)
                if node and vl1 in node_to_bus and node in node_to_bus[vl1]:
                    lines_or_id[branch_id] = node_to_bus[vl1][node]
                elif node:
                    # Node not in mapping means disconnected
                    lines_or_id[branch_id] = -1
            
            if vl2 in impacted_vl_ids:
                node = self._branch_ex_to_node.get(branch_id)
                if node and vl2 in node_to_bus and node in node_to_bus[vl2]:
                    lines_ex_id[branch_id] = node_to_bus[vl2][node]
                elif node:
                    lines_ex_id[branch_id] = -1
        
        # Process loads
        for load_id, vl in self._load_to_vl.items():
            if vl in impacted_vl_ids:
                node = self._load_to_node.get(load_id)
                if node and vl in node_to_bus and node in node_to_bus[vl]:
                    loads_id[load_id] = node_to_bus[vl][node]
                elif node:
                    loads_id[load_id] = -1
        
        # Process generators
        for gen_id, vl in self._gen_to_vl.items():
            if vl in impacted_vl_ids:
                node = self._gen_to_node.get(gen_id)
                if node and vl in node_to_bus and node in node_to_bus[vl]:
                    generators_id[gen_id] = node_to_bus[vl][node]
                elif node:
                    generators_id[gen_id] = -1
        
        # Process shunts
        for shunt_id, vl in self._shunt_to_vl.items():
            if vl in impacted_vl_ids:
                node = self._shunt_to_node.get(shunt_id)
                if node and vl in node_to_bus and node in node_to_bus[vl]:
                    shunts_id[shunt_id] = node_to_bus[vl][node]
                elif node:
                    shunts_id[shunt_id] = -1
        
        return {
            'lines_or_id': lines_or_id,
            'lines_ex_id': lines_ex_id,
            'loads_id': loads_id,
            'generators_id': generators_id,
            'shunts_id': shunts_id
        }


# =============================================================================
# Helper functions for DataFrame processing (kept for compatibility)
# =============================================================================

def add_local_num(df: pd.DataFrame, buses: pd.DataFrame, bus_id_attr: str,
                  voltage_level_id_attr: str, local_num_attr: str) -> pd.DataFrame:
    """Add local bus number to a DataFrame by merging with buses."""
    df = df.merge(buses, left_on=bus_id_attr, right_index=True, how='left')
    df[voltage_level_id_attr] = df[voltage_level_id_attr].fillna('')
    df[local_num_attr] = df[local_num_attr].fillna(-1).astype(int)
    return df


def add_injection_local_num(inj_df: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
    """Add local bus number to injection DataFrame."""
    return add_local_num(inj_df, buses, 'bus_id', 'voltage_level_id', 'local_num')


def add_branch_local_num(branches: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
    """Add local bus numbers for both ends of branches."""
    buses_bus1 = buses.rename(columns=lambda x: x + '_bus1')
    buses_bus2 = buses.rename(columns=lambda x: x + '_bus2')
    branches_with_num = add_local_num(branches, buses_bus1, 'bus1_id',
                                      'voltage_level1_id', 'local_num_bus1')
    branches_with_num = add_local_num(branches_with_num, buses_bus2, 'bus2_id',
                                      'voltage_level2_id', 'local_num_bus2')
    return branches_with_num


def check_connectivity(buses: DataFrame, loads: DataFrame, generators: DataFrame,
                       shunts: DataFrame, action_id: str):
    """Check if action breaks network connectivity and emit warnings if so."""
    buses_out_of_cc = buses['connected_component'] >= 1
    if buses_out_of_cc.any():
        warnings.warn(f"Action {action_id} break connectivity, some buses "
                      f"{list(buses[buses_out_of_cc].index)} are out of main component")


# =============================================================================
# Core conversion functions - NO VARIANT CLONING
# =============================================================================

def convert_to_grid2op_action(
    n: Network, 
    action: repas.Action, 
    topology_cache: Optional[NetworkTopologyCache] = None,
    verbose: bool = False
) -> dict:
    """
    Convert a REPAS action to Grid2Op format WITHOUT cloning variants.
    
    This optimized version uses analytical topology computation via Union-Find
    instead of modifying the pypowsybl network. This avoids the expensive
    clone_variant/remove_variant operations.
    
    Parameters
    ----------
    n : Network
        The pypowsybl network object.
    action : repas.Action
        The REPAS action to convert.
    topology_cache : NetworkTopologyCache, optional
        Pre-built topology cache. If None, one will be created.
        For batch processing, pass a shared cache for better performance.
    verbose : bool, optional
        If True, print action ID during processing (default: False).
        
    Returns
    -------
    dict
        The converted action in Grid2Op format.
    """
    if verbose:
        print(f"Converting action: {action._id}")
    
    # Build or use provided cache
    if topology_cache is None:
        topology_cache = NetworkTopologyCache(n)
    
    result = {"description": action._description}
    
    # Flatten all switch changes into a single dict
    switch_changes = {}
    for vl_id, switches in action._switches_by_voltage_level.items():
        switch_changes.update(switches)
    
    # Get impacted voltage levels
    impacted_vl_ids = set(action._switches_by_voltage_level.keys())
    
    # Compute bus assignments analytically (no network modification!)
    node_to_bus = topology_cache.compute_bus_assignments(switch_changes, impacted_vl_ids)
    
    # Convert to element-level bus assignments
    set_bus = topology_cache.get_element_bus_assignments(node_to_bus, impacted_vl_ids)
    
    result['content'] = {'set_bus': set_bus}
    
    return result


def convert_to_grid2op_action_batch(
    n: Network, 
    actions: List[repas.Action],
    verbose: bool = False
) -> List[dict]:
    """
    Convert multiple REPAS actions to Grid2Op format efficiently.
    
    This batch version:
    1. Builds a single topology cache (expensive, but done once)
    2. Processes all actions using analytical computation (cheap, no variants)
    
    Parameters
    ----------
    n : Network
        The pypowsybl network object.
    actions : List[repas.Action]
        List of REPAS actions to convert.
    verbose : bool, optional
        If True, print progress information (default: False).
        
    Returns
    -------
    List[dict]
        List of converted actions in Grid2Op format (same order as input).
    """
    if not actions:
        return []
    
    # Build topology cache once for all actions
    if verbose:
        print("Building topology cache...")
    topology_cache = NetworkTopologyCache(n)
    
    results = []
    for i, action in enumerate(actions):
        if verbose and (i + 1) % 50 == 0:
            print(f"Converting action {i + 1}/{len(actions)}")
        
        result = convert_to_grid2op_action(n, action, topology_cache, verbose=False)
        results.append(result)
    
    return results


# =============================================================================
# Legacy function with variant cloning (for comparison/fallback)
# =============================================================================

def convert_to_grid2op_action_with_variant(
    n: Network, 
    action: repas.Action, 
    verbose: bool = False
) -> dict:
    """
    Convert a REPAS action to Grid2Op format using variant cloning.
    
    This is the original implementation kept for comparison and as a fallback.
    It is slower than the analytical version but guaranteed to be accurate.
    
    Parameters
    ----------
    n : Network
        The pypowsybl network object.
    action : repas.Action
        The REPAS action to convert.
    verbose : bool, optional
        If True, print action ID during processing (default: False).
        
    Returns
    -------
    dict
        The converted action in Grid2Op format.
    """
    if verbose:
        print(f"Converting action (with variant): {action._id}")
    
    result = {"description": action._description}
    
    # Apply topology on a variant
    n.clone_variant("InitialState", action._id, True)
    n.set_working_variant(action._id)
    
    try:
        # Apply all switches in batch per voltage level
        for voltage_level_id, switches in action._switches_by_voltage_level.items():
            switch_ids = list(switches.keys())
            open_values = list(switches.values())
            n.update_switches(id=switch_ids, open=open_values)
        
        # Get updated bus topology with local numbering
        buses = n.get_buses(attributes=['voltage_level_id', 'connected_component'])
        buses['local_num'] = buses.groupby('voltage_level_id').cumcount() + 1
        
        # Fetch network elements
        loads = add_injection_local_num(n.get_loads(attributes=['bus_id']), buses)
        generators = add_injection_local_num(n.get_generators(attributes=['bus_id']), buses)
        shunts = add_injection_local_num(n.get_shunt_compensators(attributes=['bus_id']), buses)
        branches = add_branch_local_num(
            n.get_branches(attributes=['bus1_id', 'voltage_level1_id', 'bus2_id', 'voltage_level2_id']),
            buses
        )
        
        # Convert to set for O(1) lookup
        impacted_vl_ids = set(action._switches_by_voltage_level.keys())
        
        # Vectorized filtering
        or_mask = branches['voltage_level1_id'].isin(impacted_vl_ids)
        ex_mask = branches['voltage_level2_id'].isin(impacted_vl_ids)
        
        lines_or_id = branches.loc[or_mask, 'local_num_bus1'].to_dict()
        lines_ex_id = branches.loc[ex_mask, 'local_num_bus2'].to_dict()
        
        loads_mask = loads['voltage_level_id'].isin(impacted_vl_ids)
        loads_id = loads.loc[loads_mask, 'local_num'].to_dict()
        
        generators_mask = generators['voltage_level_id'].isin(impacted_vl_ids)
        generators_id = generators.loc[generators_mask, 'local_num'].to_dict()
        
        shunts_mask = shunts['voltage_level_id'].isin(impacted_vl_ids)
        shunts_id = shunts.loc[shunts_mask, 'local_num'].to_dict()
        
        result['content'] = {
            'set_bus': {
                'lines_or_id': lines_or_id,
                'lines_ex_id': lines_ex_id,
                'loads_id': loads_id,
                'generators_id': generators_id,
                'shunts_id': shunts_id
            }
        }
        
    finally:
        n.set_working_variant("InitialState")
        n.remove_variant(action._id)
    
    return result


# =============================================================================
# Disconnection/Reconnection action generation
# =============================================================================

def create_dict_disco_reco_lines_disco(
    net: Network,
    filter_voltage_levels: List[float] = [400, 24., 15., 20., 33., 10.],
    verbose: bool = False
) -> Dict[str, dict]:
    """
    Create disconnection and reconnection actions for lines not at filtered voltage levels.
    
    Parameters
    ----------
    net : Network
        The pypowsybl network object.
    filter_voltage_levels : List[float]
        Voltage levels to exclude from action generation.
    verbose : bool
        If True, print filtered lines.
        
    Returns
    -------
    Dict[str, dict]
        Dictionary of disco/reco actions.
    """
    dict_extra_disco_reco_actions = {}
    
    branches_df = net.get_branches()[["voltage_level1_id", "voltage_level2_id"]]
    vl_df = net.get_voltage_levels()
    
    branches_df["voltage_level2"] = vl_df.loc[branches_df["voltage_level2_id"], "nominal_v"].values
    branches_df["voltage_level1"] = vl_df.loc[branches_df["voltage_level1_id"], "nominal_v"].values
    
    filter_set = set(filter_voltage_levels)
    mask = (~branches_df["voltage_level1"].isin(filter_set)) & \
           (~branches_df["voltage_level2"].isin(filter_set))
    
    eligible_lines = branches_df[mask].index
    filtered_lines = branches_df[~mask].index
    
    if verbose:
        for line in filtered_lines:
            print(f"line filtered through voltage level: {line}")
    
    for line in eligible_lines:
        dict_key = f"disco_{line}"
        description = f"deconnection de l'ouvrage {line}"
        content = {'set_bus': {'lines_or_id': {line: -1}, 'lines_ex_id': {line: -1}}}
        dict_extra_disco_reco_actions[dict_key] = {
            "description": description,
            "description_unitaire": description,
            "content": content
        }
        
        dict_key = f"reco_{line}"
        description = f"reconnection de l'ouvrage {line} aux noeuds 1 a chaque extremite"
        content = {'set_bus': {'lines_or_id': {line: 1}, 'lines_ex_id': {line: 1}}}
        dict_extra_disco_reco_actions[dict_key] = {
            "description": description,
            "content": content
        }
    
    return dict_extra_disco_reco_actions


# =============================================================================
# Description generation
# =============================================================================

def get_all_switch_descriptions(switches_by_voltage_level: Dict[str, Dict[str, bool]]) -> str:
    """
    Generate a human-readable description for all switches in an action.

    Parameters
    ----------
    switches_by_voltage_level : Dict[str, Dict[str, bool]]
        Dictionary mapping voltage level to switch states.

    Returns
    -------
    str
        Description with "et" between actions and substation name at the end.
    """
    descriptions = []
    voltage_level = None

    for vl, switches in switches_by_voltage_level.items():
        voltage_level = vl
        for switch_name, switch_value in switches.items():
            action_type = "Ouverture" if switch_value else "Fermeture"
            descriptions.append(f"{action_type} {switch_name}")

    if descriptions:
        actions_str = " et ".join(descriptions)
        return f"{actions_str} dans le poste {voltage_level}"
    return ""


# =============================================================================
# High-level conversion functions
# =============================================================================

def convert_repas_actions_to_grid2op_actions(
    n: Network, 
    actions: List[repas.Action],
    use_batch: bool = True,
    use_analytical: bool = True,
    verbose: bool = False
) -> Dict[str, dict]:
    """
    Convert a list of REPAS actions to Grid2Op format.
    
    Parameters
    ----------
    n : Network
        The pypowsybl network object.
    actions : List[repas.Action]
        List of REPAS actions to convert.
    use_batch : bool, optional
        If True, use batch processing for better performance (default: True).
    use_analytical : bool, optional
        If True, use analytical topology computation without variant cloning.
        Set to False to use the legacy variant-based approach (default: True).
    verbose : bool, optional
        If True, print progress information (default: False).
        
    Returns
    -------
    Dict[str, dict]
        Dictionary mapping action keys to converted actions.
    """
    if not actions:
        return {}
    
    result = {}
    
    if use_batch and len(actions) > 1:
        if use_analytical:
            converted_list = convert_to_grid2op_action_batch(n, actions, verbose=verbose)
        else:
            # Legacy batch with variants
            converted_list = []
            for action in actions:
                converted_list.append(
                    convert_to_grid2op_action_with_variant(n, action, verbose=verbose)
                )
        
        for action, g2o_action in zip(actions, converted_list):
            if g2o_action is not None:
                action_key = action._id + "_" + next(iter(action._switches_by_voltage_level))
                g2o_action["description_unitaire"] = get_all_switch_descriptions(
                    action._switches_by_voltage_level
                )
                
                voltage_levels = list(action._switches_by_voltage_level.keys())
                if len(voltage_levels) == 1:
                    g2o_action["VoltageLevelId"] = voltage_levels[0]
                    
                result[action_key] = g2o_action
    else:
        convert_fn = convert_to_grid2op_action if use_analytical else convert_to_grid2op_action_with_variant
        
        for action in actions:
            g2o_action = convert_fn(n, action, verbose=verbose)
            if g2o_action is not None:
                action_key = action._id + "_" + next(iter(action._switches_by_voltage_level))
                g2o_action["description_unitaire"] = get_all_switch_descriptions(
                    action._switches_by_voltage_level
                )
                
                voltage_levels = list(action._switches_by_voltage_level.keys())
                if len(voltage_levels) == 1:
                    g2o_action["VoltageLevelId"] = voltage_levels[0]
                    
                result[action_key] = g2o_action

    return result


def convert_to_grid2op_actions(
    n: Network, 
    actions: Dict[str, List[repas.Action]],
    verbose: bool = False
) -> Dict[str, Dict[str, dict]]:
    """
    Convert actions organized by contingency to Grid2Op format.
    
    Parameters
    ----------
    n : Network
        The pypowsybl network object.
    actions : Dict[str, List[repas.Action]]
        Dictionary mapping contingency names to lists of actions.
    verbose : bool, optional
        If True, print progress information (default: False).
        
    Returns
    -------
    Dict[str, Dict[str, dict]]
        Dictionary mapping contingency names to converted action dictionaries.
    """
    result = {}
    for contingency, action_list in actions.items():
        if verbose:
            print(f"Converting actions for contingency: {contingency}")
        result[contingency] = convert_repas_actions_to_grid2op_actions(
            n, action_list, use_batch=True,  verbose=verbose, use_analytical=False,#set to true to avoid network clonging variants
        )
    return result
