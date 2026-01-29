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
import argparse
import copy
import json
import time
from datetime import datetime
import numpy as np
import pypowsybl as pp

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

# Imports for Action Rebuilding
from expert_op4grid_recommender.utils.helpers import Timer
from expert_op4grid_recommender.utils.action_rebuilder import run_rebuild_actions
from expert_op4grid_recommender.data_loader import load_actions


def set_thermal_limits(n_grid, env, thresold_thermal_limit=0.95):
    ### Set thermal limits
    default_th_lim_value = 9999.

    thermal_limits_df = n_grid.get_operational_limits().reset_index()
    branches = thermal_limits_df.element_id.unique()

    all_branches = set(
        list(n_grid.get_lines().index) + list(n_grid.get_2_windings_transformers().index))
    dict_thermal_limits = {name: default_th_lim_value for name in all_branches}

    for branch in branches:
        thermal_limits_branch = thermal_limits_df[thermal_limits_df.element_id == branch]
        dict_thermal_limits[branch] = np.round(
            thresold_thermal_limit * thermal_limits_branch[thermal_limits_branch.name == "permanent_limit"][
                "value"].min())

    th_lim_dict_day_arr = [dict_thermal_limits[l] for l in env.name_line]
    env.set_thermal_limit(th_lim_dict_day_arr)

    return env


def run_analysis(analysis_date, current_timestep, current_lines_defaut, env_path=None, env_name=None):
    """
    Runs the expert system analysis for a given date, timestep, and contingency.
    """

    # --- Argument Handling ---
    is_bare_env = False

    if analysis_date is None:
        is_bare_env = True
        # analysis_date = DEFAULT_DATE
    elif isinstance(analysis_date, str):
        try:
            analysis_date = datetime.strptime(analysis_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Error: Date format must be YYYY-MM-DD")
    elif not isinstance(analysis_date, datetime):
        raise TypeError("analysis_date must be a datetime object, string, or None")

    date_str = analysis_date.strftime('%Y-%m-%d') if analysis_date else "None"
    print(f"Running analysis for Date: {date_str}, Timestep: {current_timestep}, Contingency: {current_lines_defaut}")

    if env_name is not None:
        config.ENV_NAME = env_name
    if env_path is not None:
        config.ENV_PATH = env_path

    # Load setup
    with Timer("Environment Setup"):
        env, obs, path_chronic, chronic_name, custom_layout, dict_action, lines_non_reconnectable, lines_we_care_about = setup_environment_configs(
            analysis_date)

        #######################
        # Temporary fix for thermal limits
        if np.mean(env.get_thermal_limit()) >= 10 ** 4:
            n_grid = env.backend._grid.network
            env = set_thermal_limits(n_grid, env, thresold_thermal_limit=0.95)
            obs = env.reset()
        #############################

    # --- Instantiate Classifier ---
    classifier = ActionClassifier(grid2op_action_space=env.action_space)

    if is_bare_env:
        act_reco_maintenance = env.action_space({})
        maintenance_to_reco_at_t = []
    else:
        act_reco_maintenance, maintenance_to_reco_at_t = get_maintenance_timestep(
            current_timestep, lines_non_reconnectable, env, config.DO_RECO_MAINTENANCE
        )

    # Simulate Contingency
    with Timer("Initial Contingency Simulation"):
        obs_simu_defaut, has_converged = simulate_contingency(env, obs, current_lines_defaut, act_reco_maintenance,
                                                              current_timestep)
    if not has_converged:
        raise RuntimeError("Initial contingency simulation failed. Cannot proceed.")

    # Find overloads and reconnectable lines
    lines_overloaded_ids = [i for i, l in enumerate(obs_simu_defaut.name_line) if
                            l in lines_we_care_about and obs_simu_defaut.rho[i] >= 1]
    non_connected_reconnectable_lines = [
        l_name for i, l_name in enumerate(env.name_line)
        if
        l_name not in current_lines_defaut and l_name not in lines_non_reconnectable and not
        obs_simu_defaut.line_status[i]
    ]
    if config.IGNORE_RECONNECTIONS:#we will not run reconnection path detection and reconnection action discovery
        non_connected_reconnectable_lines=[]

    # Check if graph remains connected
    lines_overloaded_ids_kept, prevent_islanded_subs = identify_overload_lines_to_keep_overflow_graph_connected(
        obs_simu_defaut, lines_overloaded_ids, config.DO_FORCE_OVERLOAD_GRAPH_EVEN_IF_GRAPH_BROKEN_APART
    )
    if prevent_islanded_subs:
        print(f"Warning: Not all overloads considered, as they would island these substations: {prevent_islanded_subs}")

    if not lines_overloaded_ids_kept:
        print("Overload breaks the grid apart. No topological solution without load shedding.")
        return

    # Build the overflow graph
    with Timer("Graph Building & DC Switch"):
        has_converged, has_lost_load = check_simu_overloads(
            obs, obs_simu_defaut, env.action_space, current_timestep, current_lines_defaut, lines_overloaded_ids_kept,
            maintenance_to_reco_at_t
        )

        use_dc = config.USE_DC_LOAD_FLOW
        if not has_converged and not config.DO_FORCE_OVERLOAD_GRAPH_EVEN_IF_GRAPH_BROKEN_APART:
            use_dc = True
            env, obs, obs_simu_defaut = switch_to_dc_load_flow(
                env, analysis_date, current_timestep, current_lines_defaut, lines_overloaded_ids_kept,
                maintenance_to_reco_at_t
            )

        df_of_g, overflow_sim, g_overflow, hubs, g_distribution_graph, node_name_mapping = build_overflow_graph(
            env, obs_simu_defaut, lines_overloaded_ids_kept, non_connected_reconnectable_lines, lines_non_reconnectable,
            current_timestep, do_consolidate_graph=config.DO_CONSOLIDATE_GRAPH
        )

    # Visualize graph
    with Timer("Visualization"):
        graph_file_name = get_graph_file_name(current_lines_defaut, chronic_name, current_timestep, use_dc)
        save_folder = config.SAVE_FOLDER_VISUALIZATION
        lines_swapped = list(df_of_g[df_of_g.new_flows_swapped].line_name)
        make_overflow_graph_visualization(
            env, overflow_sim, g_overflow, hubs, obs_simu_defaut, save_folder, graph_file_name, lines_swapped,
            custom_layout
        )

    # Save data for tests
    if config.DO_SAVE_DATA_FOR_TEST:
        with Timer("Saving Test Data"):
            case_name = f"defaut_{'_'.join(current_lines_defaut)}_t{current_timestep}"
            save_data_for_test(config.ENV_PATH, case_name, df_of_g, overflow_sim, obs_simu_defaut,
                               lines_non_reconnectable, non_connected_reconnectable_lines, lines_overloaded_ids_kept)

    # Get graph paths and apply expert rules
    with Timer("Path Analysis & Rule Validation"):
        lines_blue_paths, nodes_blue_path, lines_dispatch, nodes_dispatch_path = get_constrained_and_dispatch_paths(
            g_distribution_graph, obs, lines_overloaded_ids, lines_overloaded_ids_kept
        )

        validator = ActionRuleValidator(
            obs=obs,
            action_space=env.action_space,
            classifier=classifier,
            hubs=hubs,
            paths=((lines_blue_paths, nodes_blue_path), (lines_dispatch, nodes_dispatch_path)),
            by_description=config.CHECK_WITH_ACTION_DESCRIPTION
        )

        actions_to_filter, actions_unfiltered = validator.categorize_actions(
            dict_action=dict_action,
            timestep=current_timestep,
            defauts=current_lines_defaut,
            overload_ids=lines_overloaded_ids,
            lines_reco_maintenance=maintenance_to_reco_at_t,
            lines_we_care_about=lines_we_care_about,
            do_simulation_checks=config.CHECK_ACTION_SIMULATION
        )

        print_filtered_out_action(len(dict_action), actions_to_filter)

    # Pre-process graph for AlphaDeesp
    with Timer("Pre-process for AlphaDeesp"):
        g_overflow_processed, g_distribution_graph_processed, simulator_data = pre_process_graph_alphadeesp(g_overflow,
                                                                                                            overflow_sim,
                                                                                                            node_name_mapping)
    if use_dc:
        print("Warning: you have used the DC load flow, so results are more approximate")
        env, obs, path_chronic = get_env_first_obs(config.ENV_FOLDER, config.ENV_NAME,
                                                   config.USE_EVALUATION_CONFIG,
                                                   analysis_date)

    # Run the discovery process
    with Timer("Action Discovery"):
        # Instantiate the discoverer
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
            actions_unfiltered=set(actions_unfiltered.keys()),
            hubs=hubs,
            g_overflow=g_overflow_processed,
            g_distribution_graph=g_distribution_graph_processed,
            simulator_data=simulator_data,
            check_action_simulation=config.CHECK_ACTION_SIMULATION,
            lines_we_care_about=lines_we_care_about
        )

        prioritized_actions = discoverer.discover_and_prioritize(
            n_action_max=config.N_PRIORITIZED_ACTIONS
        )

    print("\nPrioritized actions are: " + str(list(prioritized_actions.keys())))

    # Reassess the prioritized actions
    with Timer("Reassessment"):
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

    return prioritized_actions


def main():
    """
    Main function to run the expert system analysis from the command line.
    """

    # --- Argument Parsing ---
    default_date_str = DEFAULT_DATE.strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="Run ExpertOp4Grid analysis for a specific contingency.")
    parser.add_argument(
        "--date",
        default=default_date_str,
        help=f"Date for the chronic in YYYY-MM-DD format (default: {default_date_str}). Pass 'None' to use the bare environment without a specific date."
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=DEFAULT_TIMESTEP,
        help=f"Timestep index within the chronic (default: {DEFAULT_TIMESTEP})"
    )
    parser.add_argument(
        "--lines-defaut",
        nargs='+',
        default=DEFAULT_LINES_DEFAUT,
        help=f"One or more line names for the N-1 contingency (default: {' '.join(DEFAULT_LINES_DEFAUT)})"
    )
    parser.add_argument(
        "--rebuild-actions",
        action='store_true',
        help="If set, rebuilds the action dictionary from REPAS files based on the current grid snapshot before analysis. Stops analysis after rebuilding."
    )
    parser.add_argument(
        "--repas-file",
        default=os.path.join("data", "action_space", "allLogics.2024.12.10.json"),
        help="Path to the REPAS actions file (default: data/action_space/allLogics.2024.12.10.json)"
    )
    parser.add_argument(
        "--grid-snapshot-file",
        default=os.path.join("data", "snapshot", "pf_20240828T0100Z_20240828T0100Z.xiidm"),
        help="Path to the snaphsot grid file in detailed topology format with switches, to rebuild action dictionnary on(default: data/snapshot/pf_20240828T0100Z_20240828T0100Z.xiidm)"
    )
    parser.add_argument(
        "--voltage-threshold",
        type=float,
        default=300.0,
        help="Voltage filter threshold for REPAS actions (default: 300)"
    )
    args = parser.parse_args()

    # --- Handle explicit "None" string for date ---
    date_arg = args.date
    if date_arg == "None":
        date_arg = None

    # --- Call the core logic function ---
    try:
        with Timer("Total Execution"):
            if args.rebuild_actions:
                grid_snapshot_file_path=args.grid_snapshot_file
                n_grid=pp.network.load(grid_snapshot_file_path)

                dict_action={}
                do_from_sratch=False
                if config.ACTION_FILE_PATH is not None and os.path.exists(config.ACTION_FILE_PATH):
                    dict_action = load_actions(config.ACTION_FILE_PATH)
                else:
                    do_from_sratch=True

                dict_action =run_rebuild_actions(n_grid, do_from_sratch, args.repas_file, dict_action_to_filter_on=dict_action,
                                    voltage_filter_threshold=args.voltage_threshold, output_file_base_name="reduced_model_actions")

                print("Action rebuilding process complete. Stopping analysis as requested.")

                return  # EXIT EARLY
            else:
                run_analysis(
                analysis_date=date_arg,
                current_timestep=args.timestep,
                current_lines_defaut=args.lines_defaut
            )
    except (ValueError, RuntimeError, TypeError) as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()