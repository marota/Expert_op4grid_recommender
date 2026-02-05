# expert_op4grid_recommender/utils/simulation.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

import numpy as np
from typing import Callable, List, Tuple, Any, Optional

def create_default_action(action_space: Callable, defauts: List[str]) -> Any:
    """
    Creates a Grid2Op action object that disconnects a specified list of lines.

    This is typically used to represent the initial N-1 contingency by setting
    both the origin and extremity buses of the fault lines (`defauts`) to -1.

    Args:
        action_space (Callable): The Grid2Op action space object, used to create
                                 the action from a dictionary definition.
        defauts (List[str]): A list of line names to be disconnected in the action.

    Returns:
        Any: The Grid2Op action object representing the disconnection of the specified lines.
             The exact type depends on the Grid2Op backend.
    """
    return action_space({
        "set_bus": {
            # Set extremity bus to -1 (disconnected) for each default line
            "lines_ex_id": {defaut: -1 for defaut in defauts},
            # Set origin bus to -1 (disconnected) for each default line
            "lines_or_id": {defaut: -1 for defaut in defauts}
        }
    })


def simulate_contingency(env: Any, obs: Any, lines_defaut: List[str], act_reco_maintenance: Any, timestep: int) -> Tuple[Any, bool]:
    """
    Simulates the application of an initial N-1 contingency and any maintenance reconnections.

    This function creates an action to disconnect the specified contingency lines (`lines_defaut`),
    combines it with any provided maintenance reconnection action (`act_reco_maintenance`),
    and simulates the resulting state using the environment's `simulate` method.
    It checks for simulation exceptions.

    Args:
        env (Any): The Grid2Op environment object, providing `env.action_space` and `obs.simulate`.
        obs (Any): The initial Grid2Op observation object *before* the contingency.
        lines_defaut (List[str]): A list of line names representing the N-1 contingency to simulate.
        act_reco_maintenance (Any): A Grid2Op action object representing lines to be reconnected
                                     from maintenance simultaneously with the contingency.
        timestep (int): The simulation timestep index.

    Returns:
        Tuple[Any, bool]: A tuple containing:
            - obs_simu (Any): The Grid2Op observation object representing the grid state *after*
                              simulating the contingency and reconnections. Returns the initial observation
                              if simulation fails.
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
        return obs_simu, False # Return the (potentially invalid) observation and False

    # Simulation successful
    return obs_simu, True


def check_rho_reduction_with_baseline(obs: Any, timestep: int, act_defaut: Any, action: Any,
                                       overload_ids: List[int], act_reco_maintenance: Any,
                                       baseline_rho: np.ndarray,
                                       lines_we_care_about: Optional[np.ndarray] = None,
                                       rho_tolerance: float = 0.01,
                                       verbose: bool = True) -> Tuple[bool, Optional[Any]]:
    """
    Checks if applying a candidate action reduces line loadings (rho) below a pre-computed baseline.

    This optimized version takes a pre-computed baseline rho array, avoiding redundant
    baseline simulations when checking multiple candidate actions.

    Args:
        obs (Any): The initial Grid2Op observation object *before* any actions are applied.
        timestep (int): The simulation timestep index.
        act_defaut (Any): The baseline Grid2Op action (e.g., N-1 contingency disconnection).
        action (Any): The candidate Grid2Op action whose effectiveness is being tested.
        overload_ids (List[int]): A list of line indices whose rho values should be checked.
        act_reco_maintenance (Any): A Grid2Op action object for maintenance reconnections.
        baseline_rho (np.ndarray): Pre-computed rho values from the baseline simulation
                                   for the overload_ids lines.
        lines_we_care_about (Optional[np.ndarray], optional): Array of line names to monitor.
        rho_tolerance (float, optional): Minimum required reduction. Defaults to 0.01.
        verbose (bool, optional): Whether to print success messages. Defaults to True.

    Returns:
        Tuple[bool, Optional[Any]]: A tuple containing:
            - is_rho_reduction (bool): True if all rho values decreased by more than tolerance.
            - obs_simu_action (Optional[Any]): The observation after applying the candidate action.
    """
    # Simulate the candidate state (contingency + maintenance + candidate action)
    obs_simu_action, _, _, info_action = obs.simulate(
        action + act_defaut + act_reco_maintenance, time_step=timestep
    )

    # If candidate simulation fails, return False but still return the observation
    if info_action["exception"]:
        print(f"ERROR: Candidate action simulation failed in check_rho_reduction: {info_action['exception']}")
        return False, obs_simu_action

    # Get final rho values from the candidate state
    rho_final = obs_simu_action.rho[overload_ids]

    # Check if *all* specified rho values decreased by more than the tolerance
    if np.all(rho_final + rho_tolerance < baseline_rho):
        if verbose:
            max_rho_line = "N/A"
            max_rho = 0.0

            # Find the maximum rho specifically among 'lines_we_care_about' if provided
            if lines_we_care_about is not None and lines_we_care_about.size > 0:
                # Create a mask for lines we care about
                care_mask = np.isin(obs_simu_action.name_line, lines_we_care_about)
                if np.any(care_mask):
                    # Filter rho values and find the maximum
                    rhos_of_interest = obs_simu_action.rho[care_mask]
                    max_rho = np.max(rhos_of_interest)
                    # Find the name of the line corresponding to that max_rho
                    max_rho_line_idx = np.where(obs_simu_action.rho == max_rho)[0]
                    # Ensure index is valid before accessing name_line
                    if max_rho_line_idx.size > 0 and max_rho_line_idx[0] < len(obs.name_line):
                        max_rho_line = obs.name_line[max_rho_line_idx[0]]

            # If lines_we_care_about is not specified, find the overall maximum rho
            else:
                if obs_simu_action.rho.size > 0:
                    max_rho_idx = np.argmax(obs_simu_action.rho)
                    max_rho = obs_simu_action.rho[max_rho_idx]
                    # Ensure index is valid before accessing name_line
                    if max_rho_idx < len(obs.name_line):
                        max_rho_line = obs.name_line[max_rho_idx]

            print(
                f"✅ Rho reduction from {np.round(baseline_rho, 2)} to {np.round(rho_final, 2)}. "
                f"New max rho is {max_rho:.2f} on line {max_rho_line}."
            )
        return True, obs_simu_action

    # If rho reduction condition is not met
    return False, obs_simu_action


def compute_baseline_simulation(obs: Any, timestep: int, act_defaut: Any,
                                 act_reco_maintenance: Any,
                                 overload_ids: List[int]) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """
    Computes the baseline simulation once for use with multiple candidate actions.

    Args:
        obs (Any): The initial Grid2Op observation object *before* any actions are applied.
        timestep (int): The simulation timestep index.
        act_defaut (Any): The baseline Grid2Op action (e.g., N-1 contingency disconnection).
        act_reco_maintenance (Any): A Grid2Op action object for maintenance reconnections.
        overload_ids (List[int]): A list of line indices whose rho values should be extracted.

    Returns:
        Tuple[Optional[np.ndarray], Optional[Any]]: A tuple containing:
            - baseline_rho (Optional[np.ndarray]): The rho values for overload_ids from baseline,
                                                   or None if simulation failed.
            - obs_baseline (Optional[Any]): The baseline observation, or None if failed.
    """
    obs_baseline, _, _, info_baseline = obs.simulate(
        act_defaut + act_reco_maintenance, time_step=timestep
    )

    if info_baseline["exception"]:
        print(f"ERROR: Baseline simulation failed: {info_baseline['exception']}")
        return None, None

    baseline_rho = obs_baseline.rho[overload_ids]
    return baseline_rho, obs_baseline


def check_rho_reduction(obs: Any, timestep: int, act_defaut: Any, action: Any, overload_ids: List[int],
                        act_reco_maintenance: Any, lines_we_care_about: Optional[np.ndarray] = None,
                        rho_tolerance: float = 0.01) -> Tuple[bool, Optional[Any]]:
    """
    Checks if applying a candidate action reduces line loadings (rho) below a baseline.

    This function simulates two scenarios from the initial observation `obs`:
    1. Baseline: Applying only the default action (`act_defaut`, usually the contingency)
       and maintenance reconnections (`act_reco_maintenance`).
    2. Candidate: Applying the baseline actions *plus* the candidate `action`.

    It then compares the line loading (`rho`) values for the specified `overload_ids`
    between the two scenarios. The candidate action is considered effective if *all*
    rho values in `overload_ids` decrease by more than `rho_tolerance`.

    Note: For checking multiple actions, use compute_baseline_simulation() once and then
    check_rho_reduction_with_baseline() for each action to avoid redundant simulations.

    Args:
        obs (Any): The initial Grid2Op observation object *before* any actions are applied.
        timestep (int): The simulation timestep index.
        act_defaut (Any): The baseline Grid2Op action (e.g., N-1 contingency disconnection).
        action (Any): The candidate Grid2Op action whose effectiveness is being tested.
        overload_ids (List[int]): A list of line indices whose rho values should be checked
                                  for reduction.
        act_reco_maintenance (Any): A Grid2Op action object representing lines to be reconnected
                                     from maintenance in both baseline and candidate simulations.
        lines_we_care_about (Optional[np.ndarray], optional): An array of line names. If provided,
                                     the function will find and report the maximum rho among these
                                     specific lines after applying the candidate action. Defaults to None.
        rho_tolerance (float, optional): The minimum required reduction in rho for *all* lines
                                         in `overload_ids` for the action to be considered
                                         effective. Defaults to 0.01 (1% loading).

    Returns:
        Tuple[bool, Optional[Any]]: A tuple containing:
            - is_rho_reduction (bool): True if all rho values in `overload_ids` decreased by
                                       more than `rho_tolerance`, False otherwise (including
                                       simulation failures).
            - obs_simu_action (Optional[Any]): The Grid2Op observation object resulting from
                                                simulating the candidate action. Returns None if the
                                                baseline simulation failed. Returns the observation even
                                                if the candidate simulation failed or rho wasn't reduced.
    """
    # Compute baseline
    baseline_rho, obs_baseline = compute_baseline_simulation(
        obs, timestep, act_defaut, act_reco_maintenance, overload_ids
    )

    if baseline_rho is None:
        return False, None

    # Use the optimized function
    return check_rho_reduction_with_baseline(
        obs, timestep, act_defaut, action, overload_ids, act_reco_maintenance,
        baseline_rho, lines_we_care_about, rho_tolerance
    )


def check_simu_overloads(obs: Any, obs_defaut: Any, action_space: Callable, timestep: int, lines_defaut: List[str],
                         lines_overloaded_ids: List[int], lines_reco_maintenance: List[str]) -> Tuple[bool, bool]:
    """
    Simulates disconnecting all specified overloaded lines simultaneously along with contingencies.

    This function checks the grid's stability when facing the combined impact of the initial
    contingency (`lines_defaut`), maintenance reconnections (`lines_reco_maintenance`), and
    the disconnection of *all* specified overloaded lines (`lines_overloaded_ids`).

    It checks for two failure conditions:
    1. Simulation Exception: The power flow calculation fails to converge.
    2. Load Shedding: The total load served after the simulation is significantly less
       than the load served in the baseline `obs_defaut` state (indicating lost load).

    Args:
        obs (Any): The initial Grid2Op observation object *before* any actions.
        obs_defaut (Any): The Grid2Op observation object representing the baseline state
                          (typically after the initial contingency), used for load comparison.
        action_space (Callable): The Grid2Op action space object.
        timestep (int): The simulation timestep index.
        lines_defaut (List[str]): List of line names for the initial contingency.
        lines_overloaded_ids (List[int]): List of indices for *all* overloaded lines to disconnect.
        lines_reco_maintenance (List[str]): List of line names to reconnect from maintenance.

    Returns:
        Tuple[bool, bool]: A tuple containing:
            - has_converged (bool): True if the simulation completed without exceptions, False otherwise.
            - has_lost_load (bool): True if load shedding was detected, False otherwise. This is only
                                    meaningful if `has_converged` is True.
    """
    # Create action to disconnect all specified overloaded lines
    # Ensure line_id is valid before accessing name_line
    valid_overload_ids = [line_id for line_id in lines_overloaded_ids if line_id < len(obs.name_line)]
    act_deco_overloads = action_space(
        {"set_line_status": [(obs.name_line[line_id], -1) for line_id in valid_overload_ids]}
    )
    # Create action for initial contingency
    act_deco_defaut = action_space({"set_line_status": [(line, -1) for line in lines_defaut]})
    # Create action for maintenance reconnections
    act_reco_maintenance_obj = action_space({"set_line_status": [(line_reco, 1) for line_reco in lines_reco_maintenance]})

    # Simulate the combined action
    obs_simu, _, _, info = obs.simulate(act_deco_overloads + act_deco_defaut + act_reco_maintenance_obj, time_step=timestep)

    # Check for simulation failure
    if info["exception"]:
        print(f"ERROR: Simulation failed when disconnecting all specified overloads ({[obs.name_line[i] for i in valid_overload_ids]}): {info['exception']}")
        return False, False # Cannot determine load loss if simulation failed

    # Check for load shedding by comparing total load before and after
    # Add a small tolerance (e.g., 1 MW) to avoid floating point issues
    if obs_simu.load_p.sum() + 1.0 < obs_defaut.load_p.sum():
        print(f"WARNING: Load shedding occurred when simulating disconnection of all specified overloads. "
              f"Load before: {obs_defaut.load_p.sum():.2f}, Load after: {obs_simu.load_p.sum():.2f}")
        return True, True # Converged, but load was lost

    # Simulation converged without significant load loss
    return True, False