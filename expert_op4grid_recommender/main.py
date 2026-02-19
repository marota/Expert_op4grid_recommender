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
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
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

# Imports for Action Rebuilding
from expert_op4grid_recommender.utils.helpers import Timer, get_maintenance_timestep, print_filtered_out_action, \
    save_data_for_test
from expert_op4grid_recommender.utils.action_rebuilder import run_rebuild_actions
from expert_op4grid_recommender.data_loader import load_actions


class Backend(Enum):
    """Enumeration of available simulation backends."""
    GRID2OP = "grid2op"
    PYPOWSYBL = "pypowsybl"


def set_thermal_limits(n_grid, env, thresold_thermal_limit=0.95):
    """Set thermal limits from network operational limits."""
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


# =============================================================================
# GRID2OP BACKEND FUNCTIONS
# =============================================================================

def setup_environment_grid2op(analysis_date: Optional[datetime]) -> Tuple:
    """Setup environment using Grid2Op backend."""
    from expert_op4grid_recommender.environment import setup_environment_configs
    return setup_environment_configs(analysis_date)


def get_env_first_obs_grid2op(env_folder, env_name, use_evaluation_config, date, is_DC):
    """Get environment and first observation using Grid2Op."""
    from expert_op4grid_recommender.environment import get_env_first_obs
    return get_env_first_obs(env_folder, env_name, use_evaluation_config, date, is_DC)


def switch_to_dc_grid2op(env, analysis_date, current_timestep, current_lines_defaut, 
                          lines_overloaded_ids_kept, maintenance_to_reco_at_t):
    """Switch to DC load flow using Grid2Op."""
    from expert_op4grid_recommender.environment import switch_to_dc_load_flow
    return switch_to_dc_load_flow(env, analysis_date, current_timestep, current_lines_defaut,
                                   lines_overloaded_ids_kept, maintenance_to_reco_at_t)


def simulate_contingency_grid2op(env, obs, lines_defaut, act_reco_maintenance, timestep):
    """Simulate contingency using Grid2Op."""
    from expert_op4grid_recommender.utils.simulation import simulate_contingency
    return simulate_contingency(env, obs, lines_defaut, act_reco_maintenance, timestep)


def check_simu_overloads_grid2op(obs, obs_defaut, action_space, timestep, lines_defaut, 
                                  lines_overloaded_ids_kept, maintenance_to_reco_at_t):
    """Check simulation overloads using Grid2Op."""
    from expert_op4grid_recommender.utils.simulation import check_simu_overloads
    return check_simu_overloads(obs, obs_defaut, action_space, timestep, lines_defaut,
                                 lines_overloaded_ids_kept, maintenance_to_reco_at_t)


def create_default_action_grid2op(action_space, defauts):
    """Create default action using Grid2Op."""
    from expert_op4grid_recommender.utils.simulation import create_default_action
    return create_default_action(action_space, defauts)


def check_rho_reduction_grid2op(obs, timestep, act_defaut, action, overload_ids,
                                 act_reco_maintenance, lines_we_care_about):
    """Check rho reduction using Grid2Op."""
    from expert_op4grid_recommender.utils.simulation import check_rho_reduction
    return check_rho_reduction(obs, timestep, act_defaut, action, overload_ids,
                                act_reco_maintenance, lines_we_care_about)


def compute_baseline_simulation_grid2op(obs, timestep, act_defaut, act_reco_maintenance, overload_ids):
    """Compute baseline simulation using Grid2Op."""
    from expert_op4grid_recommender.utils.simulation import compute_baseline_simulation
    return compute_baseline_simulation(obs, timestep, act_defaut, act_reco_maintenance, overload_ids)


def build_overflow_graph_grid2op(env, obs_simu_defaut, lines_overloaded_ids_kept, 
                                  non_connected_reconnectable_lines, lines_non_reconnectable,
                                  timestep, do_consolidate_graph):
    """Build overflow graph using Grid2Op/alphaDeesp."""
    from expert_op4grid_recommender.graph_analysis.builder import build_overflow_graph
    return build_overflow_graph(env, obs_simu_defaut, lines_overloaded_ids_kept,
                                 non_connected_reconnectable_lines, lines_non_reconnectable,
                                 timestep, do_consolidate_graph=do_consolidate_graph)


# =============================================================================
# PYPOWSYBL BACKEND FUNCTIONS  
# =============================================================================

def setup_environment_pypowsybl(analysis_date: Optional[datetime]) -> Tuple:
    """Setup environment using pure pypowsybl backend."""
    from expert_op4grid_recommender.environment_pypowsybl import setup_environment_configs_pypowsybl
    return setup_environment_configs_pypowsybl(analysis_date)


def get_env_first_obs_pypowsybl(env_folder, env_name, use_evaluation_config, date, is_DC):
    """Get environment and first observation using pypowsybl."""
    from expert_op4grid_recommender.environment_pypowsybl import get_env_first_obs_pypowsybl
    return get_env_first_obs_pypowsybl(env_folder, env_name, is_DC=is_DC)


def switch_to_dc_pypowsybl(env, analysis_date, current_timestep, current_lines_defaut,
                            lines_overloaded_ids_kept, maintenance_to_reco_at_t):
    """Switch to DC load flow using pypowsybl."""
    from expert_op4grid_recommender.environment_pypowsybl import switch_to_dc_load_flow_pypowsybl
    return switch_to_dc_load_flow_pypowsybl(env, current_lines_defaut, lines_overloaded_ids_kept,
                                             maintenance_to_reco_at_t)


def simulate_contingency_pypowsybl(env, obs, lines_defaut, act_reco_maintenance, timestep):
    """Simulate contingency using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import simulate_contingency
    return simulate_contingency(env, obs, lines_defaut, act_reco_maintenance, timestep)


def check_simu_overloads_pypowsybl(obs, obs_defaut, action_space, timestep, lines_defaut,
                                    lines_overloaded_ids_kept, maintenance_to_reco_at_t):
    """Check simulation overloads using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import check_simu_overloads
    return check_simu_overloads(obs, obs_defaut, action_space, timestep, lines_defaut,
                                 lines_overloaded_ids_kept, maintenance_to_reco_at_t)


def create_default_action_pypowsybl(action_space, defauts):
    """Create default action using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import create_default_action
    return create_default_action(action_space, defauts)


def check_rho_reduction_pypowsybl(obs, timestep, act_defaut, action, overload_ids,
                                   act_reco_maintenance, lines_we_care_about):
    """Check rho reduction using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import check_rho_reduction
    return check_rho_reduction(obs, timestep, act_defaut, action, overload_ids,
                                act_reco_maintenance, lines_we_care_about)


def compute_baseline_simulation_pypowsybl(obs, timestep, act_defaut, act_reco_maintenance, overload_ids):
    """Compute baseline simulation using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import compute_baseline_simulation
    return compute_baseline_simulation(obs, timestep, act_defaut, act_reco_maintenance, overload_ids)


def build_overflow_graph_pypowsybl(env, obs_simu_defaut, lines_overloaded_ids_kept,
                                    non_connected_reconnectable_lines, lines_non_reconnectable,
                                    timestep, do_consolidate_graph, use_dc=False):
    """Build overflow graph using pypowsybl."""
    from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import build_overflow_graph_pypowsybl as _build
    return _build(env, obs_simu_defaut, lines_overloaded_ids_kept,
                  non_connected_reconnectable_lines, lines_non_reconnectable,
                  timestep, do_consolidate_graph=do_consolidate_graph, use_dc=use_dc)


def build_overflow_graph_pypowsybl_wrapper(env, obs_simu_defaut, lines_overloaded_ids_kept,
                                           non_connected_reconnectable_lines, lines_non_reconnectable,
                                           timestep, do_consolidate_graph):
    """Wrapper for pypowsybl overflow graph builder with correct signature."""
    from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import build_overflow_graph_pypowsybl as _build
    return _build(env, obs_simu_defaut, lines_overloaded_ids_kept,
                  non_connected_reconnectable_lines, lines_non_reconnectable,
                  timestep, do_consolidate_graph=do_consolidate_graph, 
                  use_dc=config.USE_DC_LOAD_FLOW)


# =============================================================================
# UNIFIED ANALYSIS FUNCTION
# =============================================================================

def run_analysis(analysis_date: Optional[datetime], 
                 current_timestep: int, 
                 current_lines_defaut: List[str],
                 env_path: Optional[str] = None, 
                 env_name: Optional[str] = None,
                 backend: Backend = Backend.GRID2OP) -> Dict[str, Any]:
    """
    Runs the expert system analysis for a given date, timestep, and contingency.

    Args:
        analysis_date: Date for the chronic (None for bare environment)
        current_timestep: Timestep index within the chronic
        current_lines_defaut: List of line names for N-1 contingency
        env_path: Override for environment path
        env_name: Override for environment name
        backend: Which backend to use (Backend.GRID2OP or Backend.PYPOWSYBL)

    Returns:
        Dictionary with the following structure::

            {
                "lines_overloaded_names": List[str],
                "prioritized_actions": {
                    action_id: {
                        "action": Action object,
                        "description_unitaire": str or None,
                        "rho_before": np.ndarray,  # baseline rho on overloaded lines
                        "rho_after": np.ndarray,    # rho after action on overloaded lines
                        "max_rho": float,           # new max rho across monitored lines
                        "max_rho_line": str,        # name of line with max rho
                        "is_rho_reduction": bool,   # whether action reduces all overloads
                        "observation": Observation,  # observation after action
                        # pypowsybl only: observation._variant_id contains the kept variant
                    },
                    ...
                },
                "action_scores": {
                    "line_reconnection": {action_id: float, ...},
                    "line_disconnection": {},  # placeholder, scores to be implemented
                    "open_coupling": {action_id: float, ...},
                    "close_coupling": {},      # placeholder, scores to be implemented
                }
            }
    """
    # --- Select backend functions ---
    if backend == Backend.GRID2OP:
        print(f"Using Grid2Op backend")
        setup_environment = setup_environment_grid2op
        get_env_first_obs = get_env_first_obs_grid2op
        switch_to_dc = switch_to_dc_grid2op
        simulate_contingency = simulate_contingency_grid2op
        check_simu_overloads = check_simu_overloads_grid2op
        create_default_action = create_default_action_grid2op
        check_rho_reduction = check_rho_reduction_grid2op
        compute_baseline = compute_baseline_simulation_grid2op
        build_overflow_graph = build_overflow_graph_grid2op
    elif backend == Backend.PYPOWSYBL:
        print(f"Using pure pypowsybl backend")
        setup_environment = setup_environment_pypowsybl
        get_env_first_obs = get_env_first_obs_pypowsybl
        switch_to_dc = switch_to_dc_pypowsybl
        simulate_contingency = simulate_contingency_pypowsybl
        check_simu_overloads = check_simu_overloads_pypowsybl
        create_default_action = create_default_action_pypowsybl
        check_rho_reduction = check_rho_reduction_pypowsybl
        compute_baseline = compute_baseline_simulation_pypowsybl
        build_overflow_graph = build_overflow_graph_pypowsybl_wrapper
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # --- Argument Handling ---
    is_bare_env = False

    if analysis_date is None:
        is_bare_env = True
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
        env, obs, path_chronic, chronic_name, custom_layout, dict_action, lines_non_reconnectable, lines_we_care_about = setup_environment(
            analysis_date)

        # Temporary fix for thermal limits
        if np.mean(env.get_thermal_limit()) >= 10 ** 4:
            if backend == Backend.GRID2OP:
                n_grid = env.backend._grid.network
            else:
                n_grid = env.network_manager.network
            env = set_thermal_limits(n_grid, env, thresold_thermal_limit=0.95)
            obs = env.reset() if backend == Backend.GRID2OP else env.get_obs()

    # --- Instantiate Classifier ---
    classifier = ActionClassifier(grid2op_action_space=env.action_space)

    if is_bare_env:
        act_reco_maintenance = env.action_space({})
        maintenance_to_reco_at_t = []
    else:
        # For pypowsybl backend, detect disconnected lines from network state
        # (no chronics/time-series maintenance data available)
        if backend == Backend.PYPOWSYBL:
            from expert_op4grid_recommender.utils.helpers_pypowsybl import get_maintenance_timestep_pypowsybl
            act_reco_maintenance, maintenance_to_reco_at_t = get_maintenance_timestep_pypowsybl(
                env, obs, lines_non_reconnectable, config.DO_RECO_MAINTENANCE
            )
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
    if config.IGNORE_RECONNECTIONS:
        non_connected_reconnectable_lines = []

    # Check if graph remains connected
    lines_overloaded_ids_kept, prevent_islanded_subs = identify_overload_lines_to_keep_overflow_graph_connected(
        obs_simu_defaut, lines_overloaded_ids, config.DO_FORCE_OVERLOAD_GRAPH_EVEN_IF_GRAPH_BROKEN_APART
    )
    if prevent_islanded_subs:
        print(f"Warning: Not all overloads considered, as they would island these substations: {prevent_islanded_subs}")

    lines_overloaded_names = [obs_simu_defaut.name_line[i] for i in lines_overloaded_ids]

    if not lines_overloaded_ids_kept:
        print("Overload breaks the grid apart. No topological solution without load shedding.")
        return {
            "lines_overloaded_names": lines_overloaded_names,
            "prioritized_actions": {},
            "action_scores": {
                "line_reconnection": {},
                "line_disconnection": {},
                "open_coupling": {},
                "close_coupling": {},
            },
        }

    # Build the overflow graph
    with Timer("Graph Building & DC Switch"):
        has_converged, has_lost_load = check_simu_overloads(
            obs, obs_simu_defaut, env.action_space, current_timestep, current_lines_defaut, lines_overloaded_ids_kept,
            maintenance_to_reco_at_t
        )

        use_dc = config.USE_DC_LOAD_FLOW
        if not has_converged and not config.DO_FORCE_OVERLOAD_GRAPH_EVEN_IF_GRAPH_BROKEN_APART:
            use_dc = True
            env, obs, obs_simu_defaut = switch_to_dc(
                env, analysis_date, current_timestep, current_lines_defaut, lines_overloaded_ids_kept,
                maintenance_to_reco_at_t
            )

        df_of_g, overflow_sim, g_overflow, hubs, g_distribution_graph, node_name_mapping = build_overflow_graph(
            env, obs_simu_defaut, lines_overloaded_ids_kept, non_connected_reconnectable_lines, lines_non_reconnectable,
            current_timestep, do_consolidate_graph=config.DO_CONSOLIDATE_GRAPH
        )

    # Visualize graph (only if enabled in config)
    if config.DO_VISUALIZATION:
        with Timer("Visualization"):
            graph_file_name = get_graph_file_name(current_lines_defaut, chronic_name, current_timestep, use_dc)
            save_folder = config.SAVE_FOLDER_VISUALIZATION
            lines_swapped = list(df_of_g[df_of_g.new_flows_swapped].line_name)
            make_overflow_graph_visualization(
                env, overflow_sim, g_overflow, hubs, obs_simu_defaut, save_folder, graph_file_name, lines_swapped,
                custom_layout
            )
    else:
        print("Skipping visualization (DO_VISUALIZATION=False)")

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
        g_overflow_processed, g_distribution_graph_processed, simulator_data = pre_process_graph_alphadeesp(
            g_overflow, overflow_sim, node_name_mapping
        )

    if use_dc:
        print("Warning: you have used the DC load flow, so results are more approximate")
        if backend == Backend.GRID2OP:
            env, obs, path_chronic = get_env_first_obs(config.ENV_FOLDER, config.ENV_NAME,
                                                       config.USE_EVALUATION_CONFIG, analysis_date, is_DC=False)
        else:
            env, obs, path_chronic = get_env_first_obs(config.ENV_FOLDER, config.ENV_NAME,
                                                       config.USE_EVALUATION_CONFIG, analysis_date, is_DC=False)

    # Run the discovery process
    with Timer("Action Discovery"):
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
            lines_we_care_about=lines_we_care_about,
            check_rho_reduction_func=check_rho_reduction,
            create_default_action_func=create_default_action
        )

        prioritized_actions, action_scores = discoverer.discover_and_prioritize(
            n_action_max=config.N_PRIORITIZED_ACTIONS
        )

    print("\nPrioritized actions are: " + str(list(prioritized_actions.keys())))

    # Reassess the prioritized actions and collect detailed results
    with Timer("Reassessment"):
        act_defaut = create_default_action(env.action_space, current_lines_defaut)

        # Compute baseline rho once for all actions
        baseline_rho, _ = compute_baseline(
            obs, current_timestep, act_defaut, act_reco_maintenance, lines_overloaded_ids
        )

        detailed_actions = {}
        for action_id, action in prioritized_actions.items():
            # Get description_unitaire if available
            description_unitaire = None
            if action_id in dict_action:
                action_desc = dict_action[action_id]
                description_unitaire = action_desc.get("description_unitaire")

            # Simulate action
            is_pypowsybl = backend == Backend.PYPOWSYBL
            obs_simu_action, _, _, info_action = obs.simulate(
                action + act_defaut + act_reco_maintenance,
                time_step=current_timestep,
                **({'keep_variant': True} if is_pypowsybl else {})
            )

            # Compute rho evolution and max rho
            rho_before = baseline_rho
            rho_after = None
            max_rho = 0.0
            max_rho_line = "N/A"
            is_rho_reduction = False

            if not info_action["exception"]:
                rho_after = obs_simu_action.rho[lines_overloaded_ids]

                # Determine if rho was reduced on all overloaded lines
                if rho_before is not None:
                    is_rho_reduction = bool(np.all(rho_after + 0.01 < rho_before))

                # Find max rho among lines_we_care_about (or all lines)
                if lines_we_care_about is not None and len(lines_we_care_about) > 0:
                    care_mask = np.isin(obs_simu_action.name_line, lines_we_care_about)
                    if np.any(care_mask):
                        rhos_of_interest = obs_simu_action.rho[care_mask]
                        max_rho = float(np.max(rhos_of_interest))
                        max_rho_line_idx = np.where(obs_simu_action.rho == max_rho)[0]
                        if max_rho_line_idx.size > 0 and max_rho_line_idx[0] < len(obs.name_line):
                            max_rho_line = obs.name_line[max_rho_line_idx[0]]
                else:
                    if obs_simu_action.rho.size > 0:
                        max_rho_idx = int(np.argmax(obs_simu_action.rho))
                        max_rho = float(obs_simu_action.rho[max_rho_idx])
                        if max_rho_idx < len(obs.name_line):
                            max_rho_line = obs.name_line[max_rho_idx]

            # Print summary
            print(f"{action_id}")
            if description_unitaire:
                print(f"  {description_unitaire}")
            if rho_before is not None and rho_after is not None:
                print(f"  Rho reduction from {np.round(rho_before, 2)} to {np.round(rho_after, 2)}")
                print(f"  New max rho is {max_rho:.2f} on line {max_rho_line}")

            detailed_actions[action_id] = {
                "action": action,
                "description_unitaire": description_unitaire,
                "rho_before": rho_before,
                "rho_after": rho_after,
                "max_rho": max_rho,
                "max_rho_line": max_rho_line,
                "is_rho_reduction": is_rho_reduction,
                "observation": obs_simu_action,
            }

    return {
        "lines_overloaded_names": lines_overloaded_names,
        "prioritized_actions": detailed_actions,
        "action_scores": action_scores,
    }


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
        "--backend",
        choices=["grid2op", "pypowsybl"],
        default="grid2op",
        help="Simulation backend to use (default: grid2op)"
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
        help="Path to the snapshot grid file in detailed topology format with switches, to rebuild action dictionary on (default: data/snapshot/pf_20240828T0100Z_20240828T0100Z.xiidm)"
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

    # --- Select backend ---
    backend = Backend.GRID2OP if args.backend == "grid2op" else Backend.PYPOWSYBL

    # --- Call the core logic function ---
    try:
        with Timer("Total Execution"):
            if args.rebuild_actions:
                grid_snapshot_file_path = args.grid_snapshot_file
                n_grid = pp.network.load(grid_snapshot_file_path)

                dict_action = {}
                do_from_scratch = False
                if config.ACTION_FILE_PATH is not None and os.path.exists(config.ACTION_FILE_PATH):
                    dict_action = load_actions(config.ACTION_FILE_PATH)
                else:
                    do_from_scratch = True

                dict_action = run_rebuild_actions(n_grid, do_from_scratch, args.repas_file,
                                                   dict_action_to_filter_on=dict_action,
                                                   voltage_filter_threshold=args.voltage_threshold,
                                                   output_file_base_name="reduced_model_actions")

                print("Action rebuilding process complete. Stopping analysis as requested.")
                return  # EXIT EARLY
            else:
                run_analysis(
                    analysis_date=date_arg,
                    current_timestep=args.timestep,
                    current_lines_defaut=args.lines_defaut,
                    backend=backend
                )
    except (ValueError, RuntimeError, TypeError) as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
