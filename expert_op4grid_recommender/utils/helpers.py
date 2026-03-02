# expert_op4grid_recommender/utils/helpers.py
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
import os
import json
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
import time


def get_theta_node(obs: Any, sub_id: int, bus: int) -> float:
    """
    Calculates the median voltage angle (theta) for a specific bus within a substation.

    It retrieves all lines connected to the specified bus at the given substation
    and calculates the median of their voltage angles at that connection point.
    Zero angles are excluded from the median calculation.

    Args:
        obs: The Grid2Op observation object, providing grid topology and state
             (e.g., `obs.get_obj_connect_to`, `obs.line_or_bus`, `obs.theta_or`).
        sub_id (int): The integer index of the target substation.
        bus (int): The bus number within the substation (e.g., 1, 2).

    Returns:
        float: The median voltage angle in degrees for the specified bus, or 0.0
               if no connected lines with non-zero angles are found.
    """
    obj_to_sub = obs.get_obj_connect_to(substation_id=sub_id)
    # Find lines connected to this bus at this substation (origin and extremity)
    lines_or = [i for i in obj_to_sub['lines_or_id'] if obs.line_or_bus[i] == bus]
    lines_ex = [i for i in obj_to_sub['lines_ex_id'] if obs.line_ex_bus[i] == bus]
    # Get the angles at the connection points
    thetas = np.append(obs.theta_or[lines_or], obs.theta_ex[lines_ex])
    # Filter out zero angles (often indicate issues or disconnected elements)
    thetas = thetas[thetas != 0]
    # Calculate median if angles exist, otherwise return 0
    return float(np.median(thetas)) if len(thetas) > 0 else 0.0


def get_delta_theta_line(obs: Any, id_line: int) -> float:
    """
    Calculates the voltage angle difference (delta-theta) across a specific power line.

    It finds the voltage angles at the origin and extremity buses of the line using
    `get_theta_node` and returns their difference (theta_origin - theta_extremity).
    Handles cases where a line end might be disconnected by assuming it connects
    to bus 1 if reconnected.

    Args:
        obs: The Grid2Op observation object, providing grid topology and state.
        id_line (int): The integer index of the power line.

    Returns:
        float: The difference between the voltage angle at the origin and the
               extremity of the line, in degrees.
    """
    # Determine the bus connection, defaulting to bus 1 if disconnected (-1)
    bus_or = obs.line_or_bus[id_line] if obs.line_or_bus[id_line] != -1 else 1
    bus_ex = obs.line_ex_bus[id_line] if obs.line_ex_bus[id_line] != -1 else 1
    # Get the substation IDs for origin and extremity
    sub_l_or = obs.line_or_to_subid[id_line]
    sub_l_ex = obs.line_ex_to_subid[id_line]
    # Calculate the angle at each end
    theta_or_l = get_theta_node(obs, sub_l_or, bus_or)
    theta_ex_l = get_theta_node(obs, sub_l_ex, bus_ex)
    # Return the difference
    return theta_or_l - theta_ex_l


def sort_actions_by_score(map_action_score: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Optional[str]], List[float]]:
    """
    Sorts a dictionary of actions based on their associated scores in descending order.

    The input dictionary is expected to map action identifiers (strings) to dictionaries
    containing at least an 'action' object and a 'score' (float). It may also contain
    'sub_impacted' or 'line_impacted' keys indicating the primary asset affected.

    Args:
        map_action_score (Dict[str, Dict[str, Any]]): A dictionary where keys are
            action IDs and values are dictionaries holding the action object,
            its score, and optionally the impacted asset name.

    Returns:
        Tuple[Dict[str, Any], List[Optional[str]], List[float]]: A tuple containing:
            - actions (Dict[str, Any]): A dictionary mapping action IDs to action objects,
              sorted by score (highest first).
            - assets (List[Optional[str]]): A list of impacted asset names (substation or line),
              in the same order as the sorted actions. Contains None if the key isn't found.
            - scores (List[float]): A list of scores, in the same sorted order.
              Returns empty structures if the input dictionary is empty.
    """
    if not map_action_score:
        return {}, [], []

    # Sort the dictionary items based on the 'score' value in descending order
    items = sorted(map_action_score.items(), key=lambda item: item[1]['score'], reverse=True)
    # Extract the sorted IDs
    sorted_ids = [item[0] for item in items]

    # Reconstruct the sorted dictionary of actions
    actions = {id: map_action_score[id]["action"] for id in sorted_ids}
    # Extract the impacted asset, preferring 'sub_impacted' then 'line_impacted'
    assets = [map_action_score[id].get("sub_impacted", map_action_score[id].get("line_impacted")) for id in sorted_ids]
    # Extract the sorted scores
    scores = [map_action_score[id]["score"] for id in sorted_ids]

    return actions, assets, scores


def add_prioritized_actions(prioritized_actions: Dict[str, Any],
                            identified_actions: Dict[str, Any],
                            n_action_max_total: int = 5,
                            n_action_max_per_type: int = 3) -> Dict[str, Any]:
    """
    Adds actions from a source dictionary to a target dictionary, respecting limits.

    This function iterates through `identified_actions` and adds them to the
    `prioritized_actions` dictionary until either the total number of actions in
    `prioritized_actions` reaches `n_action_max_total` or the number of actions added
    *from this specific call* reaches `n_action_max_per_type`.

    Args:
        prioritized_actions (Dict[str, Any]): The dictionary to add actions to (modified in place).
        identified_actions (Dict[str, Any]): The dictionary containing actions to potentially add.
        n_action_max_total (int, optional): The overall maximum number of actions allowed
            in `prioritized_actions`. Defaults to 5.
        n_action_max_per_type (int, optional): The maximum number of actions to add from
            `identified_actions` in this call. Defaults to 3.

    Returns:
        Dict[str, Any]: The updated `prioritized_actions` dictionary.
    """
    n_added_this_type = 0
    # Iterate through the actions identified for this type/category
    for action_id, action in identified_actions.items():
        if action_id in prioritized_actions:
            continue # Skip actions already added in a previous batch
        # Stop if total limit or per-type limit for this batch is reached
        if len(prioritized_actions) >= n_action_max_total or n_added_this_type >= n_action_max_per_type:
            break
        # Add the action if limits not reached
        prioritized_actions[action_id] = action
        n_added_this_type += 1
    return prioritized_actions


def get_maintenance_timestep(timestep: int,
                             lines_non_reconnectable: List[str],
                             env: Any,
                             do_reco_maintenance: bool) -> Tuple[Any, List[str]]:
    """
    Determines which lines under maintenance at simulation start are available for reconnection at the current timestep.

    It checks the maintenance schedule provided by the environment. If `do_reco_maintenance`
    is True, it identifies lines that were in maintenance at timestep 0 but are *not* in
    maintenance at the specified `timestep`, excluding any lines listed in `lines_non_reconnectable`.
    It then creates a Grid2Op action object to reconnect these eligible lines.

    Args:
        timestep (int): The current timestep index to check for maintenance release.
        lines_non_reconnectable (List[str]): A list of line names that should never be reconnected.
        env: The Grid2Op environment object, providing `env.name_line`,
             `env.chronics_handler.real_data.data.maintenance_handler.array`, and `env.action_space`.
        do_reco_maintenance (bool): If True, enables the creation of the reconnection action.
            If False, returns an empty action and list.

    Returns:
        Tuple[Any, List[str]]: A tuple containing:
            - act_reco_maintenance: The Grid2Op action object for reconnecting eligible lines
              (empty action if `do_reco_maintenance` is False or no lines are eligible).
            - maintenance_to_reco_at_t (List[str]): A list of names of the lines included in the
              reconnection action.
    """
    # Extract maintenance data (True = in maintenance)
    maintenance_df = pd.DataFrame(env.chronics_handler.real_data.data.maintenance_handler.array, columns=env.name_line)
    # Lines in maintenance at the very start (t=0)
    lines_in_maintenance_at_start = set(maintenance_df.columns[maintenance_df.iloc[0]])
    # Filter out lines that are generally non-reconnectable
    reconnectable_maintenance_lines = list(lines_in_maintenance_at_start - set(lines_non_reconnectable))

    maintenance_to_reco_at_t = []
    # Proceed only if reconnection is enabled and there are potentially reconnectable lines
    if do_reco_maintenance and reconnectable_maintenance_lines:
        # Check status at the current timestep for the initially maintained lines
        # `~` negates the boolean mask (True if NOT in maintenance)
        is_reconnectable_now_mask = ~maintenance_df.loc[timestep, reconnectable_maintenance_lines]
        # Get the names of lines that are True in the mask
        maintenance_to_reco_at_t = list(is_reconnectable_now_mask[is_reconnectable_now_mask].index)

    # Create the action object (will be empty if maintenance_to_reco_at_t is empty)
    act_reco_maintenance = env.action_space({"set_line_status": [(line, 1) for line in maintenance_to_reco_at_t]})
    return act_reco_maintenance, maintenance_to_reco_at_t


def print_filtered_out_action(n_actions_total: int, actions_to_filter: Dict[str, Dict[str, Any]]):
    """
    Prints a summary of actions that were filtered out by the expert rules.

    Outputs the description and the violated rule for each filtered action.
    Also provides a summary count and highlights how many of the filtered actions
    might have potentially reduced overloads based on simulation checks (if performed).

    Args:
        n_actions_total (int): The total number of actions considered before filtering.
        actions_to_filter (Dict[str, Dict[str, Any]]): A dictionary where keys are action IDs
            and values are dictionaries containing details like "description_unitaire",
            "broken_rule", and "is_rho_reduction".
    """
    print("\n################# ACTIONS FILTERED OUT BY EXPERT RULES #################")
    if not actions_to_filter:
        print("No actions were filtered out by the expert rules.")
        return

    # Print details for each filtered action
    for action_content in actions_to_filter.values():
        print(f"- {action_content.get('description_unitaire', 'No description')}")
        print(f"    Reason: {action_content.get('broken_rule', 'Unknown')}")

    # Calculate summary statistics
    n_filtered = len(actions_to_filter)
    n_badly_filtered = sum(1 for content in actions_to_filter.values() if content.get("is_rho_reduction"))

    # Print summary
    print(f"\nSummary: {n_filtered} out of {n_actions_total} actions were filtered by expert rules.")
    if n_badly_filtered > 0:
        print(f"Warning: {n_badly_filtered} of these filtered actions showed a potential tendency to reduce overloads according to simulation checks.")


def save_to_json(data: Dict, output_file: str):
    """
    Saves a Python dictionary to a JSON file with pretty-printing.

    Args:
        data (Dict): The dictionary to save.
        output_file (str): The full path to the output JSON file.
    """
    try:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4) # Use indent=4 for readability
    except Exception as e:
        print(f"Error saving data to JSON file {output_file}: {e}")


def save_data_for_test(env_path: str, case_name: str, df_of_g: pd.DataFrame, overflow_sim: Any, obs_simu: Any,
                       lines_non_reconnectable: List[str], non_connected_reconnectable_lines: List[str],
                       lines_overloaded_ids_kept: List[int]):
    """
    Saves relevant simulation data to disk, creating a reproducible test case.

    This function exports key data structures from a simulation run into a
    dedicated folder named `test_data_{case_name}`. This includes:
    - Simplified grid topology used by `alphaDeesp`.
    - Situational information (non-reconnectable lines, voltage levels, node mappings, etc.).
    - The DataFrame containing power flow changes (`df_of_g`).

    This allows the specific scenario to be reloaded and tested later without
    rerunning the full initial simulation.

    Args:
        env_path (str): Path to the Grid2Op environment directory (used for voltage levels).
        case_name (str): A unique name for this test case (e.g., "defaut_LINE_tSTEP").
        df_of_g (pd.DataFrame): DataFrame with flow change results from `Grid2opSimulation`.
        overflow_sim: The `alphaDeesp.Grid2opSimulation` object containing topology info.
        obs_simu: The Grid2Op observation object representing the state analyzed.
        lines_non_reconnectable (List[str]): List of non-reconnectable line names.
        non_connected_reconnectable_lines (List[str]): List of potentially reconnectable lines.
        lines_overloaded_ids_kept (List[int]): List of overloaded line indices used in the analysis.

    Side Effects:
        - Creates a directory named `test_data_{case_name}`.
        - Saves `sim_topo_{case_name}.json`, `situation_info.json`, and `df_of_g_{case_name}.csv`
          within the created directory.
        - Prints the path to the saved data folder.
    """
    save_folder = f"test_data_{case_name}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving test data for case '{case_name}' in folder: {save_folder}")

    try:
        # 1. Save simplified simulation topology
        sim_topo = {
            'edges': {'idx_or': list(np.array(overflow_sim.topo['edges']['idx_or'], dtype=float)),
                      'idx_ex': list(np.array(overflow_sim.topo['edges']['idx_ex'], dtype=float))},
            'nodes': overflow_sim.topo['nodes'] # Assuming this part is serializable
        }
        save_to_json(sim_topo, os.path.join(save_folder, f"sim_topo_{case_name}.json"))

        # 2. Save situational context information
        # Local import placed here to avoid potential circular dependency if this file is imported elsewhere
        from expert_op4grid_recommender.graph_analysis.visualization import get_zone_voltage_levels
        situation_info = {
            "lines_non_reconnectable": lines_non_reconnectable,
            "non_connected_reconnectable_lines": non_connected_reconnectable_lines,
            "ltc": lines_overloaded_ids_kept, # Lines To Cut (used in overflow sim)
            "number_nodal_dict": {name: len(set(obs_simu.sub_topology(i)) - {-1, 0}) for i, name in enumerate(obs_simu.name_sub)},
            "node_name_mapping": {i: name for i, name in enumerate(obs_simu.name_sub)},
            "voltage_levels": get_zone_voltage_levels(env_path)
        }
        save_to_json(situation_info, os.path.join(save_folder, 'situation_info.json'))

        # 3. Save the DataFrame of flow changes
        # Select relevant columns if necessary, or save all
        df_of_g.to_csv(os.path.join(save_folder, f"df_of_g_{case_name}.csv"), index=False)

        print(f"Successfully saved test data in folder: {save_folder}")

    except Exception as e:
        print(f"Error saving test data for case {case_name}: {e}")

class Timer:
    """Context manager to measure execution time of code blocks."""
    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        print(f"[Timer] {self.name} took {elapsed_time:.4f}s")