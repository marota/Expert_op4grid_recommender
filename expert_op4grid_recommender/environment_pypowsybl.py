# expert_op4grid_recommender/environment_pypowsybl.py
"""
Environment setup using pure pypowsybl backend.

This is the migrated version of environment.py that works without grid2op.
Provides the same interface for easy migration.
"""

import os
import json
import datetime
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

import numpy as np

from expert_op4grid_recommender.pypowsybl_backend import (
    SimulationEnvironment,
    make_simulation_env,
    PypowsyblObservation
)
from expert_op4grid_recommender import config
from expert_op4grid_recommender.data_loader import load_interesting_lines, DELETED_LINE_NAME, load_actions


def get_env_first_obs_pypowsybl(env_folder: Union[str, Path], 
                                  env_name: str,
                                  thermal_limits_file: Optional[str] = None,
                                  is_DC: bool = False,
                                  threshold_thermal_limit: float = 0.95
                                  ) -> Tuple[SimulationEnvironment, PypowsyblObservation, str]:
    """
    Creates a pypowsybl-based simulation environment and retrieves the initial observation.

    This function initializes a SimulationEnvironment using pypowsybl directly,
    without the grid2op dependency.

    Args:
        env_folder (str or Path): Path to the folder containing the environment files.
        env_name (str): The specific name of the environment to load.
        thermal_limits_file (str, optional): Name of thermal limits JSON file.
        is_DC (bool, optional): If True, use DC load flow instead of AC. Defaults to False.
        threshold_thermal_limit (float, optional): Multiplier for thermal limits. Defaults to 0.95.

    Returns:
        tuple: A tuple containing:
            - env (SimulationEnvironment): The initialized environment object.
            - obs (PypowsyblObservation): The initial observation object.
            - path_env (str): The path to the environment.
    """
    env_path = Path(env_folder) / env_name
    
    # Look for network file
    network_file = None
    for ext in ['.xiidm', '.iidm', '.xml']:
        candidates = list(env_path.glob(f"*{ext}"))
        if candidates:
            network_file = candidates[0]
            break
    
    # Also check in grid/ subfolder
    if network_file is None:
        grid_folder = env_path / "grid"
        if grid_folder.exists():
            for ext in ['.xiidm', '.iidm', '.xml']:
                candidates = list(grid_folder.glob(f"*{ext}"))
                if candidates:
                    network_file = candidates[0]
                    break
    
    if network_file is None:
        raise FileNotFoundError(f"No network file found in {env_path}")
    
    # Look for thermal limits
    thermal_limits_path = None
    if thermal_limits_file:
        thermal_limits_path = env_path / thermal_limits_file
        if not thermal_limits_path.exists():
            thermal_limits_path = None
    
    if thermal_limits_path is None:
        for name in ['thermal_limits.json', 'limits.json']:
            candidate = env_path / name
            if candidate.exists():
                thermal_limits_path = candidate
                break
    
    # Create environment
    env = SimulationEnvironment(
        network_path=network_file,
        thermal_limits_path=thermal_limits_path,
        threshold_thermal_limit=threshold_thermal_limit
    )
    
    # Configure DC mode if requested
    if is_DC:
        # Store DC preference for simulations
        env._use_dc = True
        env.network_manager._default_dc = True
    
    # Get initial observation
    obs = env.get_obs()
    
    return env, obs, str(env_path)


def setup_environment_configs_pypowsybl(analysis_date: Optional[datetime.datetime] = None,
                                         env_folder: Optional[Union[str, Path]] = None,
                                         env_name: Optional[str] = None
                                         ) -> Tuple[SimulationEnvironment, PypowsyblObservation, str, str, 
                                                    Optional[List], Dict, List[str], List[str]]:
    """
    Sets up the pypowsybl environment and loads related configuration files.

    This function performs several setup tasks:
    1. Loads the predefined action space from a JSON file specified in the config.
    2. Initializes the pypowsybl environment using parameters from the config.
    3. Retrieves the initial observation.
    4. Loads lists of non-reconnectable lines and lines to monitor from CSV files.

    Args:
        analysis_date: The date for analysis (optional, for compatibility).
        env_folder: Override for environment folder path.
        env_name: Override for environment name.

    Returns:
        tuple: A tuple containing:
            - env (SimulationEnvironment): The initialized environment object.
            - obs (PypowsyblObservation): The initial observation.
            - path_chronic (str): The environment path (for compatibility).
            - chronic_name (str): Name identifier for the analysis.
            - custom_layout (list or None): Grid layout coordinates (if available).
            - dict_action (dict): The loaded action space dictionary.
            - lines_non_reconnectable (list): Lines that cannot be reconnected.
            - lines_we_care_about (list): Lines designated for monitoring.
    """
    # Use config values if not overridden
    if env_folder is None:
        env_folder = config.ENV_FOLDER
    if env_name is None:
        env_name = config.ENV_NAME
    
    # Load action dictionary
    dict_action = load_actions(config.ACTION_FILE_PATH)
    
    # Create environment
    env, obs, env_path = get_env_first_obs_pypowsybl(
        env_folder, 
        env_name,
        is_DC=config.USE_DC_LOAD_FLOW
    )
    
    # For static analysis, use env_name as chronic_name
    chronic_name = env_name
    if analysis_date:
        chronic_name = f"{env_name}_{analysis_date.strftime('%Y%m%d')}"
    
    # Custom layout (not typically available in bare pypowsybl)
    custom_layout = None
    
    # Load non-reconnectable lines from CSV
    lines_non_reconnectable = []
    try:
        lines_non_reconnectable = list(load_interesting_lines(
            path=env_path,
            file_name="non_reconnectable_lines.csv"
        ))
    except FileNotFoundError:
        pass

    # For bare environments (no chronics), detect non-reconnectable lines
    # from switch topology in the grid.xiidm. For chronic environments,
    # the CSV file per chronic should be the authoritative source.
    if analysis_date is None:
        detected_non_reco = env.network_manager.detect_non_reconnectable_lines()
        if detected_non_reco:
            print(f"Detected {len(detected_non_reco)} non-reconnectable lines from grid topology: {detected_non_reco}")
        for line in detected_non_reco:
            if line not in lines_non_reconnectable:
                lines_non_reconnectable.append(line)

    # Add deleted lines
    lines_non_reconnectable += list(DELETED_LINE_NAME)
    
    # Load lines to monitor
    lines_we_care_about = []
    if config.IGNORE_LINES_MONITORING:
        pass # Empty list signals all lines
    else:
        try:
            lines_we_care_about = load_interesting_lines(
                file_name=os.path.join(str(env_folder), "lignes_a_monitorer.csv")
            )
        except FileNotFoundError:
            # If no specific lines, consider all lines
            lines_we_care_about = list(env.name_line)
    
    return (env, obs, env_path, chronic_name, custom_layout, 
            dict_action, lines_non_reconnectable, lines_we_care_about)


def set_thermal_limits_from_network(env: SimulationEnvironment, 
                                     threshold: float = 0.95) -> SimulationEnvironment:
    """
    Set thermal limits from the network's operational limits.

    Args:
        env: SimulationEnvironment instance
        threshold: Multiplier for limits (e.g., 0.95 for 95%)

    Returns:
        The environment with updated thermal limits
    """
    thermal_limits = env.network_manager.get_thermal_limits()
    
    # Apply threshold
    limits_array = np.array([
        thermal_limits.get(ln, 9999.0) * threshold 
        for ln in env.name_line
    ])
    
    env.set_thermal_limit(limits_array)
    return env


def switch_to_dc_load_flow_pypowsybl(env: SimulationEnvironment,
                                       lines_defaut: List[str],
                                       lines_overloaded_ids_kept: List[int],
                                       maintenance_to_reco_at_t: List[str]
                                       ) -> Tuple[SimulationEnvironment, PypowsyblObservation, PypowsyblObservation]:
    """
    Switch to DC load flow mode and re-run simulations.

    This is called when AC load flow fails to converge.

    Args:
        env: Current environment
        lines_defaut: List of contingency line names
        lines_overloaded_ids_kept: Indices of overloaded lines being considered
        maintenance_to_reco_at_t: Lines to reconnect from maintenance

    Returns:
        Tuple of (new_env, new_obs, obs_simu)
    """
    from expert_op4grid_recommender.utils.simulation_pypowsybl import simulate_contingency
    
    print("Switching to DC load flow due to convergence issues.")
    
    # Get the network path from current environment
    network_path = None
    for attr in ['_network_path', 'network_path']:
        if hasattr(env, attr):
            network_path = getattr(env, attr)
            break
    
    if network_path is None:
        # Try to get from network manager
        network_path = env.network_manager.network
    
    # Create new environment with DC mode
    # Note: In the pypowsybl backend, we can set DC mode on the network manager
    env.network_manager._default_dc = True
    
    # Reset and get new observation
    obs = env.reset()
    
    # Create maintenance reconnection action
    act_reco_maintenance = env.action_space(
        {"set_line_status": [(line_reco, 1) for line_reco in maintenance_to_reco_at_t]}
    )
    
    # Simulate contingency in DC mode
    obs_simu, has_converged = simulate_contingency(
        env, obs, lines_defaut, act_reco_maintenance, timestep=0
    )
    
    if not has_converged:
        print("Simulation failed even in DC mode. Cannot build overflow graph.")
        sys.exit(0)
    
    return env, obs, obs_simu


# ============================================================================
# Compatibility layer - allows gradual migration
# ============================================================================

def get_env_first_obs(env_folder, env_name, use_evaluation_config, date=None, is_DC=False):
    """
    Compatibility wrapper that uses pypowsybl backend.
    
    This function provides the same interface as the original grid2op version
    but uses the pypowsybl backend internally.
    
    Note: The date parameter is kept for API compatibility but is not used
    in static analysis mode.
    """
    return get_env_first_obs_pypowsybl(
        env_folder=env_folder,
        env_name=env_name,
        is_DC=is_DC
    )


def setup_environment_configs(analysis_date: datetime.datetime):
    """
    Compatibility wrapper that uses pypowsybl backend.
    
    This function provides the same interface as the original grid2op version.
    """
    return setup_environment_configs_pypowsybl(analysis_date)
