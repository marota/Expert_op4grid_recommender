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

    # Non-reconnectable: breakers open at both sides AND all disconnectors open at both sides
    return breaker_open_s1 and breaker_open_s2 and all_disc_open_s1 and all_disc_open_s2


def _build_connectable_to_node_map(nodes) -> Dict:
    """Build a mapping from connectable_id to node index from topology nodes.

    Args:
        nodes: DataFrame from topo.nodes with connectable_id column.

    Returns:
        Dict mapping connectable_id -> node index (first occurrence).
    """
    if nodes.empty:
        return {}
    valid = nodes[nodes['connectable_id'].notna()]
    if valid.empty:
        return {}
    first = valid[~valid['connectable_id'].duplicated(keep='first')]
    return dict(zip(first['connectable_id'].values, first.index.values))


def _build_switch_adjacency(switches) -> Dict:
    """Build an adjacency dict from topology switches.

    Args:
        switches: DataFrame from topo.switches with node1, node2, kind, open columns.

    Returns:
        Dict mapping node -> list of (other_node, kind, is_open).
    """
    adjacency = defaultdict(list)
    if switches.empty:
        return adjacency
    n1 = switches['node1'].values
    n2 = switches['node2'].values
    kinds = switches['kind'].values
    opens = switches['open'].values
    for i in range(len(n1)):
        adjacency[n1[i]].append((n2[i], kinds[i], opens[i]))
        adjacency[n2[i]].append((n1[i], kinds[i], opens[i]))
    return adjacency


def _check_switches_from_lookups(
    connectable_map: Dict, switch_adj: Dict, line_id: str
) -> Optional[Tuple[bool, bool]]:
    """Check breaker/disconnector states using pre-built lookup structures.

    Equivalent to _check_line_side_switches but avoids repeated topology
    fetches and DataFrame filtering by using pre-computed dicts.

    Args:
        connectable_map: Dict from connectable_id -> node index.
        switch_adj: Dict from node -> list of (other_node, kind, is_open).
        line_id: The line or transformer identifier.

    Returns:
        A tuple (breaker_open, all_disconnectors_open), or None if no
        breaker was found.
    """
    line_node = connectable_map.get(line_id)
    if line_node is None:
        return None

    neighbors = switch_adj.get(line_node, [])
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

    A disconnected line is considered non-reconnectable if:
    1. It has at least one open breaker at its extremity, AND
    2. Line breakers at BOTH extremities are open, AND
    3. All line disconnectors (sectionneurs) at BOTH extremities are also open.

    This means the line is fully isolated at both ends with no path through
    any switch to a busbar, making reconnection impossible without physical
    intervention.

    This function operates directly on a pypowsybl.network.Network object,
    so it can be used with both the pure pypowsybl backend and the grid2op
    backend (via env.backend._grid.network).

    Args:
        network: A pypowsybl.network.Network instance.

    Returns:
        List of line/transformer IDs that are non-reconnectable.
    """
    # Get disconnected elements
    lines_df = network.get_lines()
    disconnected_lines = lines_df[~lines_df['connected1'] | ~lines_df['connected2']]

    trafos_df = network.get_2_windings_transformers()
    disconnected_trafos = trafos_df[~trafos_df['connected1'] | ~trafos_df['connected2']]

    if disconnected_lines.empty and disconnected_trafos.empty:
        return []

    # Collect all unique voltage levels that need topology lookup
    vl_ids = set()
    if not disconnected_lines.empty:
        vl_ids.update(disconnected_lines['voltage_level1_id'].values)
        vl_ids.update(disconnected_lines['voltage_level2_id'].values)
    if not disconnected_trafos.empty:
        vl_ids.update(disconnected_trafos['voltage_level1_id'].values)
        vl_ids.update(disconnected_trafos['voltage_level2_id'].values)

    # Fetch each topology once and build lookup structures
    topo_cache = {}  # vl_id -> (connectable_map, switch_adjacency)
    for vl_id in vl_ids:
        topo = network.get_node_breaker_topology(vl_id)
        conn_map = _build_connectable_to_node_map(topo.nodes)
        sw_adj = _build_switch_adjacency(topo.switches)
        topo_cache[vl_id] = (conn_map, sw_adj)

    # Check all disconnected elements using cached lookups
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

            conn_map1, sw_adj1 = topo_cache[vl1]
            side1 = _check_switches_from_lookups(conn_map1, sw_adj1, eid)
            if side1 is None:
                continue

            breaker_open_s1, all_disc_open_s1 = side1

            conn_map2, sw_adj2 = topo_cache[vl2]
            side2 = _check_switches_from_lookups(conn_map2, sw_adj2, eid)
            if side2 is None:
                continue

            breaker_open_s2, all_disc_open_s2 = side2

            if not (breaker_open_s1 or breaker_open_s2):
                continue

            if breaker_open_s1 and breaker_open_s2 and all_disc_open_s1 and all_disc_open_s2:
                non_reconnectable.append(eid)

    return non_reconnectable
