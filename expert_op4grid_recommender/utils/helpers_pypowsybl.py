# expert_op4grid_recommender/utils/helpers_pypowsybl.py
"""
Helper functions for pypowsybl backend.

These functions provide pypowsybl-specific implementations of helper utilities
that were originally designed for grid2op.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Any, Dict, Optional


def get_disconnected_lines_pypowsybl(env: Any, obs: Any) -> List[str]:
    """
    Get list of lines that are currently disconnected in the network.
    
    A line is considered disconnected if it's not connected at both extremities.
    This is the pypowsybl equivalent of checking maintenance status.
    
    Args:
        env: SimulationEnvironment instance
        obs: PypowsyblObservation instance
        
    Returns:
        List of disconnected line names
    """
    disconnected_lines = []
    
    # Get line status from observation
    # line_status is True if connected, False if disconnected
    line_status = obs.line_status
    
    for i, line_name in enumerate(obs.name_line):
        if not line_status[i]:
            disconnected_lines.append(line_name)
    
    return disconnected_lines


def get_maintenance_timestep_pypowsybl(env: Any, 
                                        obs: Any,
                                        lines_non_reconnectable: List[str],
                                        do_reco_maintenance: bool = False) -> Tuple[Any, List[str]]:
    """
    Determines which disconnected lines could potentially be reconnected.
    
    For pypowsybl (static analysis), we identify lines that are currently 
    disconnected in the network but are not in the non-reconnectable list.
    Unlike grid2op with chronics, there's no time-based maintenance schedule,
    so we simply identify all currently disconnected but reconnectable lines.
    
    Args:
        env: SimulationEnvironment instance
        obs: PypowsyblObservation instance  
        lines_non_reconnectable: List of line names that should never be reconnected
        do_reco_maintenance: If True, creates reconnection action for eligible lines
        
    Returns:
        Tuple containing:
            - act_reco_maintenance: Action object for reconnecting eligible lines
              (empty action if do_reco_maintenance is False)
            - lines_in_maintenance: List of disconnected line names that could be reconnected
    """
    # Get all disconnected lines
    all_disconnected = get_disconnected_lines_pypowsybl(env, obs)
    
    # Filter out non-reconnectable lines
    lines_in_maintenance = [
        line for line in all_disconnected 
        if line not in lines_non_reconnectable
    ]
    
    if lines_in_maintenance:
        print(f"Detected {len(lines_in_maintenance)} disconnected lines that could be reconnected: {lines_in_maintenance}")
    
    # Create reconnection action only if requested
    maintenance_to_reco = []
    if do_reco_maintenance and lines_in_maintenance:
        maintenance_to_reco = lines_in_maintenance
        print(f"Will attempt to reconnect: {maintenance_to_reco}")
    
    # Create the action object
    act_reco_maintenance = env.action_space({
        "set_line_status": [(line, 1) for line in maintenance_to_reco]
    })
    
    return act_reco_maintenance, maintenance_to_reco


def get_theta_node_pypowsybl(obs: Any, sub_id: int, bus: int = 1) -> float:
    """
    Calculates the median voltage angle (theta) for a specific bus within a substation.
    
    Pypowsybl version - retrieves voltage angles from connected lines.
    
    Args:
        obs: PypowsyblObservation instance
        sub_id: The integer index of the target substation
        bus: The bus number within the substation (default 1 for pypowsybl)
        
    Returns:
        float: The median voltage angle in degrees for the specified bus
    """
    # Get objects connected to this substation
    obj_to_sub = obs.get_obj_connect_to(substation_id=sub_id)
    
    # Get angles from connected lines
    thetas = []
    
    for line_idx in obj_to_sub.get('lines_or_id', []):
        if line_idx < len(obs.theta_or):
            theta = obs.theta_or[line_idx]
            if theta != 0 and not np.isnan(theta):
                thetas.append(theta)
    
    for line_idx in obj_to_sub.get('lines_ex_id', []):
        if line_idx < len(obs.theta_ex):
            theta = obs.theta_ex[line_idx]
            if theta != 0 and not np.isnan(theta):
                thetas.append(theta)
    
    return float(np.median(thetas)) if thetas else 0.0


def get_delta_theta_line_pypowsybl(obs: Any, id_line: int) -> float:
    """
    Calculates the voltage angle difference (delta-theta) across a specific power line.
    
    Args:
        obs: PypowsyblObservation instance
        id_line: The integer index of the power line
        
    Returns:
        float: The difference between the voltage angle at the origin and extremity
    """
    # Get angles directly from observation
    theta_or = obs.theta_or[id_line] if id_line < len(obs.theta_or) else 0.0
    theta_ex = obs.theta_ex[id_line] if id_line < len(obs.theta_ex) else 0.0
    
    # Handle NaN values
    if np.isnan(theta_or):
        theta_or = 0.0
    if np.isnan(theta_ex):
        theta_ex = 0.0
    
    return theta_or - theta_ex


def _check_line_side_switches(network, line_id: str, voltage_level_id: str) -> Optional[Tuple[bool, bool]]:
    """
    Check the breaker and disconnector states at one extremity of a line.

    Uses the node-breaker topology to find the breaker directly connected
    to the line node, then finds all disconnectors connected to the
    intermediate node between the breaker and the busbars.

    Args:
        network: A pypowsybl.network.Network instance.
        line_id: The line (or transformer) identifier.
        voltage_level_id: The voltage level at this extremity.

    Returns:
        A tuple (breaker_open, all_disconnectors_open), or None if no
        breaker was found at this extremity.
    """
    topo = network.get_node_breaker_topology(voltage_level_id)
    nodes = topo.nodes
    switches = topo.switches

    # Find the node for this line in the voltage level
    line_nodes = nodes[nodes['connectable_id'] == line_id]
    if len(line_nodes) == 0:
        return None

    line_node = line_nodes.index[0]

    # Find breaker(s) directly connected to the line node
    line_sw = switches[(switches['node1'] == line_node) | (switches['node2'] == line_node)]
    breakers = line_sw[line_sw['kind'] == 'BREAKER']

    if len(breakers) == 0:
        return None

    # Check all breakers connected to the line node
    all_breakers_open = all(breakers['open'])

    # For each breaker, find the intermediate node and its disconnectors
    all_disconnectors_open = True
    for _, br in breakers.iterrows():
        intermediate_node = br['node1'] if br['node2'] == line_node else br['node2']

        disc_sw = switches[
            ((switches['node1'] == intermediate_node) | (switches['node2'] == intermediate_node))
            & (switches['kind'] == 'DISCONNECTOR')
        ]

        if len(disc_sw) > 0 and not all(disc_sw['open']):
            all_disconnectors_open = False
            break

    return (all_breakers_open, all_disconnectors_open)


def _is_non_reconnectable(network, element_id: str, vl1: str, vl2: str) -> bool:
    """
    Check if a disconnected line/transformer is non-reconnectable.

    Args:
        network: A pypowsybl.network.Network instance.
        element_id: Line or transformer ID.
        vl1: Voltage level at side 1.
        vl2: Voltage level at side 2.

    Returns:
        True if the element is non-reconnectable per the heuristic.
    """
    side1 = _check_line_side_switches(network, element_id, vl1)
    side2 = _check_line_side_switches(network, element_id, vl2)

    # If no breaker found at either side, heuristic does not apply
    if side1 is None or side2 is None:
        return False

    breaker_open_s1, all_disc_open_s1 = side1
    breaker_open_s2, all_disc_open_s2 = side2

    # Prerequisite: at least one open breaker
    if not (breaker_open_s1 or breaker_open_s2):
        return False

    # Non-reconnectable if EITHER side is fully isolated:
    # breaker open AND all disconnectors open on that side.
    # A single fully-isolated extremity means reconnection is impossible
    # regardless of the other side's state.
    side1_isolated = breaker_open_s1 and all_disc_open_s1
    side2_isolated = breaker_open_s2 and all_disc_open_s2
    return side1_isolated or side2_isolated


def _check_switches_from_lookups(
    connectable_map: Dict, switch_adj: Dict, line_id: str, vl_id: str
) -> Optional[Tuple[bool, bool]]:
    """Check breaker/disconnector states using pre-built lookup structures.
    """
    line_node_key = connectable_map.get((line_id, vl_id))
    if line_node_key is None:
        return None

    neighbors = switch_adj.get(line_node_key, [])
    breakers = [(other, is_open) for other, kind, is_open in neighbors if kind == 'BREAKER']

    if not breakers:
        return None

    all_breakers_open = all(is_open for _, is_open in breakers)

    # For each breaker, find the intermediate node and its disconnectors
    all_disconnectors_open = True
    for intermediate_node, _ in breakers:
        for _, kind, is_open in switch_adj.get(intermediate_node, []):
            if kind == 'DISCONNECTOR' and not is_open:
                all_disconnectors_open = False
                break
        if not all_disconnectors_open:
            break

    return (all_breakers_open, all_disconnectors_open)


def detect_non_reconnectable_lines(network) -> List[str]:
    """
    Detect non-reconnectable lines based on switch topology in a pypowsybl network.

    Performance Note: This version uses a hybrid vectorized approach.
    It prioritizes a fast global pandas lookup (0.6s) if Grid2Op-enriched 
    metadata is present, and falls back to a robust per-VL topology fetch 
    otherwise.
    """
    lines_df = network.get_lines()
    disconnected_lines = lines_df[~lines_df['connected1'] | ~lines_df['connected2']]

    trafos_df = network.get_2_windings_transformers()
    disconnected_trafos = trafos_df[~trafos_df['connected1'] | ~trafos_df['connected2']]

    if disconnected_lines.empty and disconnected_trafos.empty:
        return []

    # ── 1. Try Global Vectorized approach (FAST! works in Grid2Op) ──
    all_terminals = network.get_terminals()
    all_switches = network.get_switches()

    if 'connectable_id' in all_terminals.columns and 'node_id' in all_terminals.columns \
       and 'node1_id' in all_switches.columns and 'node2_id' in all_switches.columns:
        
        valid_terms = all_terminals.dropna(subset=['connectable_id', 'node_id'])
        # Global node key: (node_id, vl_id)
        connectable_map = {
            (cid, vlid): (nid, vlid) 
            for cid, vlid, nid in zip(valid_terms['connectable_id'], valid_terms['voltage_level_id'], valid_terms['node_id'])
        }

        # Build adjacency with (node_id, vl_id) keys
        adjacency = defaultdict(list)
        n1 = all_switches['node1_id'].values
        n2 = all_switches['node2_id'].values
        vl_ids = all_switches['voltage_level_id'].values
        kinds = all_switches['kind'].values
        opens = all_switches['open'].values

        for i in range(len(n1)):
            k1 = (n1[i], vl_ids[i])
            k2 = (n2[i], vl_ids[i])
            adjacency[k1].append((k2, kinds[i], opens[i]))
            adjacency[k2].append((k1, kinds[i], opens[i]))
        switch_adj = dict(adjacency)

    else:
        # ── 2. Fallback: Robust VL-based fetching (Slower but compatible) ──
        vl_ids = set()
        if not disconnected_lines.empty:
            vl_ids.update(disconnected_lines['voltage_level1_id'].values)
            vl_ids.update(disconnected_lines['voltage_level2_id'].values)
        if not disconnected_trafos.empty:
            vl_ids.update(disconnected_trafos['voltage_level1_id'].values)
            vl_ids.update(disconnected_trafos['voltage_level2_id'].values)

        connectable_map = {}
        adjacency = defaultdict(list)
        for vl_id in vl_ids:
            topo = network.get_node_breaker_topology(vl_id)
            if topo.nodes.empty:
                continue
                
            valid_nodes = topo.nodes[topo.nodes['connectable_id'].notna()]
            for node_idx, row in valid_nodes.iterrows():
                connectable_map[(row['connectable_id'], vl_id)] = (node_idx, vl_id)
                
            sw = topo.switches
            if not sw.empty:
                n1, n2, kinds, opens = sw['node1'].values, sw['node2'].values, sw['kind'].values, sw['open'].values
                for i in range(len(n1)):
                    k1 = (n1[i], vl_id)
                    k2 = (n2[i], vl_id)
                    adjacency[k1].append((k2, kinds[i], opens[i]))
                    adjacency[k2].append((k1, kinds[i], opens[i]))
        switch_adj = dict(adjacency)

    # ── 3. Check all disconnected elements using lookups ──
    non_reconnectable = []
    for df in (disconnected_lines, disconnected_trafos):
        if df.empty:
            continue
        element_ids = df.index.values
        vl1_arr = df['voltage_level1_id'].values
        vl2_arr = df['voltage_level2_id'].values

        for i in range(len(element_ids)):
            eid = element_ids[i]
            vl1, vl2 = vl1_arr[i], vl2_arr[i]

            side1 = _check_switches_from_lookups(connectable_map, switch_adj, eid, vl1)
            if side1 is None:
                continue

            breaker_open_s1, all_disc_open_s1 = side1

            side2 = _check_switches_from_lookups(connectable_map, switch_adj, eid, vl2)
            if side2 is None:
                continue

            breaker_open_s2, all_disc_open_s2 = side2

            if not (breaker_open_s1 or breaker_open_s2):
                continue

            side1_isolated = breaker_open_s1 and all_disc_open_s1
            side2_isolated = breaker_open_s2 and all_disc_open_s2
            if side1_isolated or side2_isolated:
                non_reconnectable.append(eid)

    return non_reconnectable


def _build_connectable_to_node_map(nodes, vl_id: str) -> Dict:
    """Build a mapping from (connectable_id, vl_id) to (node index, vl_id).
    Used for testing and legacy compatibility.
    """
    if nodes.empty:
        return {}
    valid = nodes[nodes['connectable_id'].notna()]
    if valid.empty:
        return {}
    first = valid[~valid['connectable_id'].duplicated(keep='first')]
    return { (cid, vl_id): (idx, vl_id) for cid, idx in zip(first['connectable_id'].values, first.index.values) }


def _build_switch_adjacency(switches, vl_id: str) -> Dict:
    """Build an adjacency dict with (node, vl_id) keys.
    Used for testing and legacy compatibility.
    """
    adjacency = defaultdict(list)
    if switches.empty:
        return adjacency
    n1 = switches['node1'].values
    n2 = switches['node2'].values
    kinds = switches['kind'].values
    opens = switches['open'].values
    for i in range(len(n1)):
        k1 = (n1[i], vl_id)
        k2 = (n2[i], vl_id)
        adjacency[k1].append((k2, kinds[i], opens[i]))
        adjacency[k2].append((k1, kinds[i], opens[i]))
    return dict(adjacency)
