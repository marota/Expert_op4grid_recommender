#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

import sys
import os
import argparse  # Import argparse
from datetime import datetime  # Import datetime for date parsing

# This adds the parent directory (your project root) to the Python path
# so that the 'expert_op4grid_recommender' package can be found.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from expert_op4grid_recommender import config
# Import the specific defaults needed
from expert_op4grid_recommender.config import DATE as DEFAULT_DATE, TIMESTEP as DEFAULT_TIMESTEP, \
    LINES_DEFAUT as DEFAULT_LINES_DEFAUT

from expert_op4grid_recommender.environment import setup_environment_configs, switch_to_dc_load_flow, get_env_first_obs
from expert_op4grid_recommender.utils.simulation import simulate_contingency, check_simu_overloads, \
    create_default_action
from expert_op4grid_recommender.utils.helpers import get_maintenance_timestep, print_filtered_out_action, \
    save_data_for_test
from expert_op4grid_recommender.graph_analysis.builder import build_overflow_graph
from expert_op4grid_recommender.graph_analysis.processor import (
    identify_overload_lines_to_keep_overflow_graph_connected,
    get_constrained_and_dispatch_paths,
    pre_process_graph_alphadeesp
)
from expert_op4grid_recommender.graph_analysis.visualization import make_overflow_graph_visualization, \
    get_graph_file_name
from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier
from expert_op4grid_recommender.action_evaluation.rules import ActionRuleValidator
from expert_op4grid_recommender.action_evaluation.discovery import ActionDiscoverer
from expert_op4grid_recommender.utils.simulation import check_rho_reduction


def run_analysis(analysis_date, current_timestep, current_lines_defaut):
    """
    Runs the expert system analysis for a given date, timestep, and contingency.
    This function is importable and raises exceptions on failure instead of exiting.

    Args:
        analysis_date (datetime | str | None): The date to analyze.
            - If None, DEFAULT_DATE from config is used.
            - If str, it must be in "YYYY-MM-DD" format.
            - If datetime, it is used directly.
        current_timestep (int): Timestep index within the chronic.
        current_lines_defaut (list[str]): List of line names for the N-1 contingency.

    Raises:
        ValueError: If the date string is malformed.
        RuntimeError: If the initial contingency simulation fails or other critical errors occur.
    """

    # --- Argument Handling ---
    # Handle the 'date=None' case by falling back to the default
    if analysis_date is None:
        analysis_date = DEFAULT_DATE
    # Handle string-based date (e.g., from argparse)
    elif isinstance(analysis_date, str):
        try:
            analysis_date = datetime.strptime(analysis_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Error: Date format must be YYYY-MM-DD")
    # Handle if it's already a datetime object
    elif not isinstance(analysis_date, datetime):
        raise TypeError("analysis_date must be a datetime object, string, or None")
    # --- End Argument Handling ---

    print(
        f"Running analysis for Date: {analysis_date.strftime('%Y-%m-%d')}, Timestep: {current_timestep}, Contingency: {current_lines_defaut}")

    # Load setup - Pass the date argument to setup
    env, obs, path_chronic, chronic_name, custom_layout, dict_action, lines_non_reconnectable, lines_we_care_about = setup_environment_configs(
        analysis_date)  # Pass date here

    # --- Instantiate Classifier ---
    classifier = ActionClassifier(grid2op_action_space=env.action_space)

    act_reco_maintenance, maintenance_to_reco_at_t = get_maintenance_timestep(
        current_timestep, lines_non_reconnectable, env, config.DO_RECO_MAINTENANCE
    )

    # Use current_lines_defaut
    obs_simu_defaut, has_converged = simulate_contingency(env, obs, current_lines_defaut, act_reco_maintenance,
                                                          current_timestep)
    if not has_converged:
        # Raise an exception instead of calling sys.exit()
        raise RuntimeError("Initial contingency simulation failed. Cannot proceed.")

    # Find overloads and reconnectable lines
    lines_overloaded_ids = [i for i, l in enumerate(obs_simu_defaut.name_line) if
                            l in lines_we_care_about and obs_simu_defaut.rho[i] >= 1]
    non_connected_reconnectable_lines = [
        l_name for i, l_name in enumerate(env.name_line)
        if
        l_name not in current_lines_defaut and l_name not in lines_non_reconnectable and not
        obs_simu_defaut.line_status[
            i]
    ]

    # Check if graph remains connected after overload disconnection
    lines_overloaded_ids_kept, prevent_islanded_subs = identify_overload_lines_to_keep_overflow_graph_connected(
        obs_simu_defaut, lines_overloaded_ids, config.DO_FORCE_OVERLOAD_GRAPH_EVEN_IF_GRAPH_BROKEN_APART
    )
    if prevent_islanded_subs:
        print(f"Warning: Not all overloads considered, as they would island these substations: {prevent_islanded_subs}")

    if not lines_overloaded_ids_kept:
        print("Overload breaks the grid apart. No topological solution without load shedding.")
        return  # Return gracefully

    # Build the overflow graph
    has_converged, has_lost_load = check_simu_overloads(
        obs, obs_simu_defaut, env.action_space, current_timestep, current_lines_defaut, lines_overloaded_ids_kept,
        maintenance_to_reco_at_t
    )

    use_dc = config.USE_DC_LOAD_FLOW
    if not has_converged and not config.DO_FORCE_OVERLOAD_GRAPH_EVEN_IF_GRAPH_BROKEN_APART:
        use_dc = True
        # Pass the date argument
        env, obs, obs_simu_defaut = switch_to_dc_load_flow(
            env, analysis_date, current_timestep, current_lines_defaut, lines_overloaded_ids_kept,
            maintenance_to_reco_at_t
        )

    # Pass current_timestep
    df_of_g, overflow_sim, g_overflow, hubs, g_distribution_graph, node_name_mapping = build_overflow_graph(
        env, obs_simu_defaut, lines_overloaded_ids_kept, non_connected_reconnectable_lines, lines_non_reconnectable,
        current_timestep
        , do_consolidate_graph=config.DO_CONSOLIDATE_GRAPH)

    # Visualize graph - pass current args
    graph_file_name = get_graph_file_name(current_lines_defaut, chronic_name, current_timestep, use_dc)
    save_folder = config.SAVE_FOLDER_VISUALIZATION
    lines_swapped = list(df_of_g[df_of_g.new_flows_swapped].line_name)
    make_overflow_graph_visualization(
        env, overflow_sim, g_overflow, hubs, obs_simu_defaut, save_folder, graph_file_name, lines_swapped,
        custom_layout
    )

    # Save data for tests if enabled - construct case name from args
    if config.DO_SAVE_DATA_FOR_TEST:
        case_name = f"defaut_{'_'.join(current_lines_defaut)}_t{current_timestep}"
        save_data_for_test(config.ENV_PATH, case_name, df_of_g, overflow_sim, obs_simu_defaut,
                           lines_non_reconnectable, non_connected_reconnectable_lines, lines_overloaded_ids_kept)

    # Get graph paths and apply expert rules - pass current args
    lines_blue_paths, nodes_blue_path, lines_dispatch, nodes_dispatch_path = get_constrained_and_dispatch_paths(
        g_distribution_graph, obs, lines_overloaded_ids, lines_overloaded_ids_kept
    )

    # Instantiate the validator with context
    validator = ActionRuleValidator(
        obs=obs,
        action_space=env.action_space,
        classifier=classifier,
        hubs=hubs,
        paths=((lines_blue_paths, nodes_blue_path), (lines_dispatch, nodes_dispatch_path)),
        by_description=config.CHECK_WITH_ACTION_DESCRIPTION  # Get from config
    )

    # Call the categorization method, passing simulation context
    actions_to_filter, actions_unfiltered = validator.categorize_actions(
        dict_action=dict_action,
        timestep=current_timestep,
        defauts=current_lines_defaut,
        overload_ids=lines_overloaded_ids,
        lines_reco_maintenance=maintenance_to_reco_at_t,
        lines_we_care_about=lines_we_care_about
    )

    print_filtered_out_action(len(dict_action), actions_to_filter)

    # Pre-process graph for AlphaDeesp
    g_overflow_processed, g_distribution_graph_processed, simulator_data = pre_process_graph_alphadeesp(g_overflow,
                                                                                                        overflow_sim,
                                                                                                        node_name_mapping)

    if use_dc:
        print("Warning: you have used the DC load flow, so results are more approximate")
        env, obs, path_chronic = get_env_first_obs(config.ENV_FOLDER, config.ENV_NAME,
                                                   config.USE_EVALUATION_CONFIG,
                                                   analysis_date)

    # Instantiate the discoverer with all necessary context
    discoverer = ActionDiscoverer(
        env=env,
        obs=obs,
        obs_defaut=obs_simu_defaut,
        timestep=current_timestep,
        lines_defaut=current_lines_defaut,
        lines_overloaded_ids=lines_overloaded_ids,
        act_reco_maintenance=act_reco_maintenance,
        classifier=classifier,
        non_connected_reconnectable_lines=non_connected_reconnectable_lines,
        all_disconnected_lines=lines_non_reconnectable + non_connected_reconnectable_lines,
        dict_action=dict_action,
        actions_unfiltered=set(actions_unfiltered.keys()),  # Pass only the IDs
        hubs=hubs,  # Assumes hubs are names from build_overflow_graph
        g_overflow=g_overflow_processed,
        g_distribution_graph=g_distribution_graph_processed,
        simulator_data=simulator_data,
        check_action_simulation=config.CHECK_ACTION_SIMULATION,
        lines_we_care_about=lines_we_care_about
    )

    # Run the discovery process
    prioritized_actions = discoverer.discover_and_prioritize(
        n_action_max=config.N_PRIORITIZED_ACTIONS
    )

    print("\nPrioritized actions are: " + str(list(prioritized_actions.keys())))

    # reassess the prioritized actions
    act_defaut = create_default_action(env.action_space, current_lines_defaut)

    for action_id, action in prioritized_actions.items():

        print(f"{action_id}")
        if action_id in dict_action:
            action_desc = dict_action[action_id]
            if "description_unitaire" in action_desc:
                print(action_desc["description_unitaire"])

        is_rho_reduction, _ = check_rho_reduction(
            obs, current_timestep, act_defaut, action, lines_overloaded_ids,
            act_reco_maintenance, lines_we_care_about
        )
    ###################

    # Return a result if needed, e.g., the prioritized actions
    return prioritized_actions


def main():
    """
    Main function to run the expert system analysis from the command line.
    This parses arguments and calls run_analysis.
    """

    # --- Argument Parsing ---
    # Convert default date from datetime object to string for argparse
    default_date_str = DEFAULT_DATE.strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="Run ExpertOp4Grid analysis for a specific contingency.")
    parser.add_argument(
        "--date",
        default=default_date_str,  # Use default from config (as string)
        help=f"Date for the chronic in YYYY-MM-DD format (default: {default_date_str})"
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=DEFAULT_TIMESTEP,  # Use default from config
        help=f"Timestep index within the chronic (default: {DEFAULT_TIMESTEP})"
    )
    parser.add_argument(
        "--lines-defaut",
        nargs='+',
        default=DEFAULT_LINES_DEFAUT,  # Use default from config
        help=f"One or more line names for the N-1 contingency (default: {' '.join(DEFAULT_LINES_DEFAUT)})"
    )
    args = parser.parse_args()

    # --- Call the core logic function ---
    # The run_analysis function will handle date parsing from string
    try:
        run_analysis(
            analysis_date=args.date,
            current_timestep=args.timestep,
            current_lines_defaut=args.lines_defaut
        )
    except (ValueError, RuntimeError, TypeError) as e:
        # Catch errors raised by the analysis function and exit gracefully
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()