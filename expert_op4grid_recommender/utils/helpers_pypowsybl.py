# expert_op4grid_recommender/utils/helpers_pypowsybl.py
"""
Helper functions for pypowsybl backend.

These functions provide pypowsybl-specific implementations of helper utilities
that were originally designed for grid2op.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict


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
