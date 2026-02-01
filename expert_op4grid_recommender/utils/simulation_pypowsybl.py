# expert_op4grid_recommender/utils/simulation_pypowsybl.py
"""
Simulation utilities using pure pypowsybl backend.

This is the migrated version of simulation.py that works without grid2op.
The interface remains the same for drop-in replacement.
"""

import numpy as np
from typing import Callable, List, Tuple, Any, Optional


def create_default_action(action_space: Callable, defauts: List[str]) -> Any:
    """
    Creates an action object that disconnects a specified list of lines.

    This is typically used to represent the initial N-1 contingency by setting
    both the origin and extremity buses of the fault lines (`defauts`) to -1.

    Args:
        action_space (Callable): The action space object, used to create
                                 the action from a dictionary definition.
        defauts (List[str]): A list of line names to be disconnected in the action.

    Returns:
        Any: The action object representing the disconnection of the specified lines.
    """
    return action_space({
        "set_bus": {
            "lines_ex_id": {defaut: -1 for defaut in defauts},
            "lines_or_id": {defaut: -1 for defaut in defauts}
        }
    })


def simulate_contingency(env: Any, obs: Any, lines_defaut: List[str], 
                         act_reco_maintenance: Any, timestep: int) -> Tuple[Any, bool]:
    """
    Simulates the application of an initial N-1 contingency and any maintenance reconnections.

    This function creates an action to disconnect the specified contingency lines (`lines_defaut`),
    combines it with any provided maintenance reconnection action (`act_reco_maintenance`),
    and simulates the resulting state using the observation's `simulate` method.

    Args:
        env (Any): The environment object, providing `env.action_space`.
        obs (Any): The initial observation object *before* the contingency.
        lines_defaut (List[str]): A list of line names representing the N-1 contingency to simulate.
        act_reco_maintenance (Any): An action object representing lines to be reconnected
                                     from maintenance simultaneously with the contingency.
        timestep (int): The simulation timestep index.

    Returns:
        Tuple[Any, bool]: A tuple containing:
            - obs_simu (Any): The observation object representing the grid state *after*
                              simulating the contingency and reconnections.
            - has_converged (bool): True if the simulation completed without raising an exception,
                                    False otherwise.
    """
    # Create the action to disconnect the contingency lines
    act_deco_defaut = env.action_space({"set_line_status": [(line, -1) for line in lines_defaut]})
    
    # Combine contingency disconnection with maintenance reconnection and simulate
    obs_simu, _, _, info = obs.simulate(act_deco_defaut + act_reco_maintenance, time_step=timestep)

    # Check if the simulation raised any exceptions
    if info["exception"]:
        print(f"ERROR: Simulation of contingency {lines_defaut} failed: {info['exception']}")
        return obs_simu, False

    return obs_simu, True


def check_rho_reduction(obs: Any, timestep: int, act_defaut: Any, action: Any, 
                        overload_ids: List[int], act_reco_maintenance: Any, 
                        lines_we_care_about: Optional[np.ndarray] = None,
                        rho_tolerance: float = 0.01) -> Tuple[bool, Optional[Any]]:
    """
    Checks if applying a candidate action reduces line loadings (rho) below a baseline.

    This function simulates two scenarios from the initial observation `obs`:
    1. Baseline: Applying only the default action (`act_defaut`, usually the contingency)
       and maintenance reconnections (`act_reco_maintenance`).
    2. Candidate: Applying the baseline actions *plus* the candidate `action`.

    It then compares the line loading (`rho`) values for the specified `overload_ids`
    between the two scenarios.

    Args:
        obs (Any): The initial observation object *before* any actions are applied.
        timestep (int): The simulation timestep index.
        act_defaut (Any): The baseline action (e.g., N-1 contingency disconnection).
        action (Any): The candidate action whose effectiveness is being tested.
        overload_ids (List[int]): A list of line indices whose rho values should be checked.
        act_reco_maintenance (Any): An action object for maintenance reconnections.
        lines_we_care_about (Optional[np.ndarray], optional): Array of line names to monitor.
        rho_tolerance (float, optional): Minimum required reduction. Defaults to 0.01.

    Returns:
        Tuple[bool, Optional[Any]]: A tuple containing:
            - is_rho_reduction (bool): True if all rho values decreased by more than tolerance.
            - obs_simu_action (Optional[Any]): The observation after applying the candidate action.
    """
    # Simulate the baseline state (contingency + maintenance)
    obs_defaut, _, _, info_defaut = obs.simulate(act_defaut + act_reco_maintenance, time_step=timestep)
    
    if info_defaut["exception"]:
        print(f"ERROR: Baseline simulation failed in check_rho_reduction: {info_defaut['exception']}")
        return False, None

    # Get initial rho values from the baseline state
    rho_init = obs_defaut.rho[overload_ids]

    # Simulate the candidate state (contingency + maintenance + candidate action)
    obs_simu_action, _, _, info_action = obs.simulate(
        action + act_defaut + act_reco_maintenance, time_step=timestep
    )

    if info_action["exception"]:
        print(f"ERROR: Candidate action simulation failed: {info_action['exception']}")
        return False, obs_simu_action

    # Get final rho values
    rho_final = obs_simu_action.rho[overload_ids]

    # Check if all specified rho values decreased by more than the tolerance
    if np.all(rho_final + rho_tolerance < rho_init):
        max_rho_line = "N/A"
        max_rho = 0.0

        if lines_we_care_about is not None and len(lines_we_care_about) > 0:
            care_mask = np.isin(obs_simu_action.name_line, lines_we_care_about)
            if np.any(care_mask):
                rhos_of_interest = obs_simu_action.rho[care_mask]
                max_rho = np.max(rhos_of_interest)
                max_rho_line_idx = np.where(obs_simu_action.rho == max_rho)[0]
                if max_rho_line_idx.size > 0 and max_rho_line_idx[0] < len(obs.name_line):
                    max_rho_line = obs.name_line[max_rho_line_idx[0]]
        else:
            if obs_simu_action.rho.size > 0:
                max_rho_idx = np.argmax(obs_simu_action.rho)
                max_rho = obs_simu_action.rho[max_rho_idx]
                if max_rho_idx < len(obs.name_line):
                    max_rho_line = obs.name_line[max_rho_idx]

        print(
            f"✅ Rho reduction from {np.round(rho_init, 2)} to {np.round(rho_final, 2)}. "
            f"New max rho is {max_rho:.2f} on line {max_rho_line}."
        )
        return True, obs_simu_action

    return False, obs_simu_action


def check_simu_overloads(obs: Any, obs_defaut: Any, action_space: Callable, timestep: int,
                         lines_defaut: List[str], lines_overloaded_ids: List[int],
                         lines_reco_maintenance: List[str]) -> Tuple[bool, bool]:
    """
    Simulates disconnecting all specified overloaded lines simultaneously.

    Checks for:
    1. Simulation convergence
    2. Load shedding (significant load loss)

    Args:
        obs (Any): Initial observation *before* any actions.
        obs_defaut (Any): Observation after initial contingency (for load comparison).
        action_space (Callable): The action space object.
        timestep (int): Simulation timestep index.
        lines_defaut (List[str]): Line names for the initial contingency.
        lines_overloaded_ids (List[int]): Indices of overloaded lines to disconnect.
        lines_reco_maintenance (List[str]): Lines to reconnect from maintenance.

    Returns:
        Tuple[bool, bool]: (has_converged, has_lost_load)
    """
    # Create action to disconnect all overloaded lines
    valid_overload_ids = [line_id for line_id in lines_overloaded_ids 
                          if line_id < len(obs.name_line)]
    act_deco_overloads = action_space(
        {"set_line_status": [(obs.name_line[line_id], -1) for line_id in valid_overload_ids]}
    )
    
    # Create action for initial contingency
    act_deco_defaut = action_space({"set_line_status": [(line, -1) for line in lines_defaut]})
    
    # Create action for maintenance reconnections
    act_reco_maintenance_obj = action_space(
        {"set_line_status": [(line_reco, 1) for line_reco in lines_reco_maintenance]}
    )

    # Simulate the combined action
    obs_simu, _, _, info = obs.simulate(
        act_deco_overloads + act_deco_defaut + act_reco_maintenance_obj, 
        time_step=timestep
    )

    # Check for simulation failure
    if info["exception"]:
        line_names = [obs.name_line[i] for i in valid_overload_ids]
        print(f"ERROR: Simulation failed when disconnecting overloads ({line_names}): {info['exception']}")
        return False, False

    # Check for load shedding
    if obs_simu.load_p.sum() + 1.0 < obs_defaut.load_p.sum():
        print(f"WARNING: Load shedding occurred. "
              f"Load before: {obs_defaut.load_p.sum():.2f}, Load after: {obs_simu.load_p.sum():.2f}")
        return True, True

    return True, False
