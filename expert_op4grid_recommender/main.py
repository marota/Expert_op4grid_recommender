#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

import os
import sys
import argparse
import copy
import json
import time
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pypowsybl as pp

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
from expert_op4grid_recommender.data_loader import load_actions, enrich_actions_lazy


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
                                  timestep, do_consolidate_graph,use_dc=False):
    """Build overflow graph using Grid2Op/alphaDeesp."""
    from expert_op4grid_recommender.graph_analysis.builder import build_overflow_graph
    return build_overflow_graph(env, obs_simu_defaut, lines_overloaded_ids_kept,
                                 non_connected_reconnectable_lines, lines_non_reconnectable,
                                 timestep, do_consolidate_graph=do_consolidate_graph,use_dc=use_dc)


# =============================================================================
# PYPOWSYBL BACKEND FUNCTIONS  
# =============================================================================

def setup_environment_pypowsybl(analysis_date: Optional[datetime], network=None,
                                 skip_initial_obs: bool = False) -> Tuple:
    """Setup environment using pure pypowsybl backend.

    `network` is an optional pre-loaded `pp.network.Network` forwarded to
    `setup_environment_configs_pypowsybl` to avoid re-parsing the .xiidm
    file when the caller already holds a Network instance.
    `skip_initial_obs=True` skips the first `env.get_obs()` call (returns
    `obs=None`) — saves ~3-5 s on large grids when the caller doesn't
    consume the initial observation.
    """
    from expert_op4grid_recommender.environment_pypowsybl import setup_environment_configs_pypowsybl
    return setup_environment_configs_pypowsybl(analysis_date, network=network, skip_initial_obs=skip_initial_obs)


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


def simulate_contingency_pypowsybl(env, obs, lines_defaut, act_reco_maintenance, timestep, fast_mode=True):
    """Simulate contingency using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import simulate_contingency
    return simulate_contingency(env, obs, lines_defaut, act_reco_maintenance, timestep, fast_mode=fast_mode)


def check_simu_overloads_pypowsybl(obs, obs_defaut, action_space, timestep, lines_defaut,
                                    lines_overloaded_ids_kept, maintenance_to_reco_at_t, fast_mode=True):
    """Check simulation overloads using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import check_simu_overloads
    return check_simu_overloads(obs, obs_defaut, action_space, timestep, lines_defaut,
                                 lines_overloaded_ids_kept, maintenance_to_reco_at_t, fast_mode=fast_mode)


def create_default_action_pypowsybl(action_space, defauts):
    """Create default action using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import create_default_action
    return create_default_action(action_space, defauts)


def check_rho_reduction_pypowsybl(obs, timestep, act_defaut, action, overload_ids,
                                   act_reco_maintenance, lines_we_care_about, fast_mode=True):
    """Check rho reduction using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import check_rho_reduction
    return check_rho_reduction(obs, timestep, act_defaut, action, overload_ids,
                                act_reco_maintenance, lines_we_care_about, fast_mode=fast_mode)


def compute_baseline_simulation_pypowsybl(obs, timestep, act_defaut, act_reco_maintenance, overload_ids, fast_mode=True):
    """Compute baseline simulation using pypowsybl."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import compute_baseline_simulation
    return compute_baseline_simulation(obs, timestep, act_defaut, act_reco_maintenance, overload_ids, fast_mode=fast_mode)


def build_overflow_graph_pypowsybl_wrapper(env, obs_simu_defaut, lines_overloaded_ids_kept,
                                           non_connected_reconnectable_lines, lines_non_reconnectable,
                                           timestep, do_consolidate_graph,use_dc=False, fast_mode=True):
    """Wrapper for pypowsybl overflow graph builder with correct signature."""
    from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import build_overflow_graph_pypowsybl as _build
    return _build(env, obs_simu_defaut, lines_overloaded_ids_kept,
                  non_connected_reconnectable_lines, lines_non_reconnectable,
                  timestep, do_consolidate_graph=do_consolidate_graph, 
                  use_dc=use_dc, param_options={"fast_mode": fast_mode})


# =============================================================================
# UNIFIED ANALYSIS FUNCTION
# =============================================================================

def run_analysis_step1(analysis_date: Optional[datetime],
                       current_timestep: int,
                       current_lines_defaut: List[str],
                       env_path: Optional[str] = None,
                       env_name: Optional[str] = None,
                       backend: Backend = Backend.GRID2OP,
                       fast_mode: Optional[bool] = None,
                       dict_action: Optional[Dict[str, Any]] = None,
                       prebuilt_env_context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    First part of the expert system analysis up to detection and selection of overloads.

    Args:
        dict_action: Pre-built enriched action dictionary (output of enrich_actions_lazy).
            When provided for the pypowsybl backend, the enrich_actions_lazy step inside
            Environment Setup is skipped, avoiding a redundant NetworkTopologyCache rebuild.
        prebuilt_env_context: Pre-built environment context dict with keys
            ``env``, ``path_chronic``, ``chronic_name``, ``custom_layout``,
            ``lines_non_reconnectable``, ``lines_we_care_about``.
            When provided for the pypowsybl backend, ``setup_environment`` is skipped
            entirely and ``env.get_obs()`` is used to read the current N-state.
            The network is reset to the base variant before reading so that stale
            variants from previous analyses do not pollute the observation.

    Returns:
        (final_result, context):
            - If final_result is not None, it means the analysis should stop here (e.g. no overloads).
            - Otherwise, context contains all information needed for Step 2.
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
    print(f"Running Step 1 analysis for Date: {date_str}, Timestep: {current_timestep}, Contingency: {current_lines_defaut}")

    if env_name is not None:
        config.ENV_NAME = env_name
    if env_path is not None:
        config.ENV_PATH = env_path

    # Load setup
    with Timer("Environment Setup"):
        if prebuilt_env_context is not None and backend == Backend.PYPOWSYBL:
            # Fast path: reuse a cached SimulationEnvironment built during update_config.
            # Reset the network to the base variant so any variants left by the previous
            # analysis are not visible in the N-state observation.
            env = prebuilt_env_context['env']
            env.network_manager.reset_to_base()
            obs = env.get_obs()
            path_chronic = prebuilt_env_context['path_chronic']
            chronic_name = prebuilt_env_context['chronic_name']
            custom_layout = prebuilt_env_context.get('custom_layout')
            lines_non_reconnectable = prebuilt_env_context['lines_non_reconnectable']
            lines_we_care_about = prebuilt_env_context['lines_we_care_about']
            n_grid = env.network_manager.network
        else:
            env, obs, path_chronic, chronic_name, custom_layout, raw_dict_action, lines_non_reconnectable, lines_we_care_about = setup_environment(
                analysis_date)

            # Get pypowsybl network reference
            if backend == Backend.GRID2OP:
                n_grid = env.backend._grid.network
            else:
                n_grid = env.network_manager.network

            # Temporary fix for thermal limits
            if np.mean(env.get_thermal_limit()) >= 10 ** 4:
                env = set_thermal_limits(n_grid, env, thresold_thermal_limit=config.MONITORING_FACTOR_THERMAL_LIMITS)
                obs = env.reset() if backend == Backend.GRID2OP else env.get_obs()

        # Wrap action dicts for lazy content computation from switches.
        # Only needed for pypowsybl backend — Grid2Op environments may use
        # node-breaker topologies where NetworkTopologyCache cannot reliably
        # compute set_bus from switches, so actions must ship with pre-computed content.
        # Skip if the caller already provided a pre-built enriched dict (avoids rebuilding
        # the NetworkTopologyCache on every run_analysis_step1 call).
        if backend == Backend.PYPOWSYBL:
            if dict_action is None:
                dict_action = enrich_actions_lazy(raw_dict_action, n_grid)
            # else: use the caller-supplied pre-built dict as-is
        else:
            dict_action = raw_dict_action

    # --- Instantiate Classifier ---
    if backend == Backend.PYPOWSYBL:
        # Build name sets once so identify_action_type can infer has_line/has_load
        # from switch IDs without triggering lazy content computation.
        _branch_names = set(n_grid.get_lines().index) | set(n_grid.get_2_windings_transformers().index)
        _load_names = set(n_grid.get_loads(attributes=[]).index)
        classifier = ActionClassifier(grid2op_action_space=env.action_space,
                                      branch_names=_branch_names, load_names=_load_names)
    else:
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
    is_pypowsybl = backend == Backend.PYPOWSYBL
    if fast_mode:
        actual_fast_mode = True
    else:
        # Resolve fast_mode if not correctly specified, falling back to config for default
        actual_fast_mode = config.PYPOWSYBL_FAST_MODE if fast_mode is None else fast_mode

    with Timer("Initial Contingency Simulation"):
        if is_pypowsybl:
            obs_simu_defaut, has_converged = simulate_contingency_pypowsybl(
                env, obs, current_lines_defaut, act_reco_maintenance, current_timestep, fast_mode=actual_fast_mode
            )
        else:
            obs_simu_defaut, has_converged = simulate_contingency(
                env, obs, current_lines_defaut, act_reco_maintenance, current_timestep
            )
    if not has_converged:
        raise RuntimeError("Initial contingency simulation failed. Cannot proceed.")

    # Find overloads caused by the contingency (not pre-existing, or significantly worsened)
    lines_overloaded_ids = []
    pre_existing_overloads = []
    pre_existing_rho = {}
    worsening_threshold = getattr(config, 'PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD', 0.02)

    for i, l in enumerate(obs_simu_defaut.name_line):
        if l in lines_we_care_about and obs_simu_defaut.rho[i] >= 1:
            if obs.rho[i] >= 1:
                if obs_simu_defaut.rho[i] > obs.rho[i] * (1 + worsening_threshold):
                    lines_overloaded_ids.append(i)
                else:
                    pre_existing_overloads.append(
                        f"{l} (rho_N={obs.rho[i]:.3f}, rho_N-1={obs_simu_defaut.rho[i]:.3f})"
                    )
                pre_existing_rho[i] = float(obs.rho[i])
            else:
                lines_overloaded_ids.append(i)
    if pre_existing_overloads:
        print(f"Ignoring {len(pre_existing_overloads)} pre-existing overload(s) "
              f"(already overloaded before contingency): {pre_existing_overloads}")
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
                "line_reconnection": {"scores": {}, "params": {}},
                "line_disconnection": {"scores": {}, "params": {}},
                "open_coupling": {"scores": {}, "params": {}},
                "close_coupling": {"scores": {}, "params": {}},
                "renewable_curtailment": {"scores": {}, "params": {}},
            },
        }, None

    # Pack everything into context for Step 2
    context = {
        "backend": backend,
        "env": env,
        "obs": obs,
        "obs_simu_defaut": obs_simu_defaut,
        "analysis_date": analysis_date,
        "current_timestep": current_timestep,
        "current_lines_defaut": current_lines_defaut,
        "lines_overloaded_ids": lines_overloaded_ids,
        "lines_overloaded_ids_kept": lines_overloaded_ids_kept,
        "maintenance_to_reco_at_t": maintenance_to_reco_at_t,
        "act_reco_maintenance": act_reco_maintenance,
        "lines_non_reconnectable": lines_non_reconnectable,
        "lines_we_care_about": lines_we_care_about,
        "classifier": classifier,
        "custom_layout": custom_layout,
        "chronic_name": chronic_name,
        "pre_existing_rho": pre_existing_rho,
        "lines_overloaded_names": lines_overloaded_names,
        "non_connected_reconnectable_lines": non_connected_reconnectable_lines,
        "dict_action": dict_action,
        "is_bare_env": is_bare_env,
        "is_pypowsybl": is_pypowsybl,
        "actual_fast_mode": actual_fast_mode,
        # Backend specific functions
        "check_simu_overloads": check_simu_overloads,
        "switch_to_dc": switch_to_dc,
        "build_overflow_graph": build_overflow_graph,
        "get_env_first_obs": get_env_first_obs,
        "simulate_contingency": simulate_contingency,
        "create_default_action": create_default_action,
        "check_rho_reduction": check_rho_reduction,
        "compute_baseline": compute_baseline,
    }

    return None, context


def run_analysis_step2_graph(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Second part of the expert system analysis, focusing on graph generation and visualization.
    """
    # Unpack context
    backend = context["backend"]
    env = context["env"]
    obs = context["obs"]
    obs_simu_defaut = context["obs_simu_defaut"]
    analysis_date = context["analysis_date"]
    current_timestep = context["current_timestep"]
    current_lines_defaut = context["current_lines_defaut"]
    lines_overloaded_ids = context["lines_overloaded_ids"]
    lines_overloaded_ids_kept = context["lines_overloaded_ids_kept"]
    maintenance_to_reco_at_t = context["maintenance_to_reco_at_t"]
    lines_non_reconnectable = context["lines_non_reconnectable"]
    lines_we_care_about = context["lines_we_care_about"]
    custom_layout = context["custom_layout"]
    chronic_name = context["chronic_name"]
    non_connected_reconnectable_lines = context["non_connected_reconnectable_lines"]
    
    is_pypowsybl = context["is_pypowsybl"]
    actual_fast_mode = context["actual_fast_mode"]
    
    check_simu_overloads = context["check_simu_overloads"]
    switch_to_dc = context["switch_to_dc"]
    build_overflow_graph = context["build_overflow_graph"]

    # Build the overflow graph
    with Timer("Graph Building & DC Switch"):
        if is_pypowsybl:
            has_converged, has_lost_load = check_simu_overloads(
                obs, obs_simu_defaut, env.action_space, current_timestep, current_lines_defaut, lines_overloaded_ids_kept,
                maintenance_to_reco_at_t, fast_mode=actual_fast_mode
            )
        else:
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

        if is_pypowsybl:
            df_of_g, overflow_sim, g_overflow, hubs, g_distribution_graph, node_name_mapping = build_overflow_graph(
                env, obs_simu_defaut, lines_overloaded_ids_kept, non_connected_reconnectable_lines, lines_non_reconnectable,
                current_timestep, do_consolidate_graph=config.DO_CONSOLIDATE_GRAPH,use_dc=use_dc, fast_mode=actual_fast_mode
            )
        else:
            df_of_g, overflow_sim, g_overflow, hubs, g_distribution_graph, node_name_mapping = build_overflow_graph(
                env, obs_simu_defaut, lines_overloaded_ids_kept, non_connected_reconnectable_lines, lines_non_reconnectable,
                current_timestep, do_consolidate_graph=config.DO_CONSOLIDATE_GRAPH,use_dc=use_dc,
            )

    # Visualize graph (only if enabled in config)
    if config.DO_VISUALIZATION:
        with Timer("Visualization"):
            graph_file_name = get_graph_file_name(current_lines_defaut, chronic_name, current_timestep, use_dc)
            save_folder = config.SAVE_FOLDER_VISUALIZATION
            lines_swapped = list(df_of_g[df_of_g.new_flows_swapped].line_name)
            make_overflow_graph_visualization(
                env, overflow_sim, g_overflow, hubs, obs_simu_defaut, save_folder, graph_file_name, lines_swapped,
                custom_layout, lines_we_care_about=lines_we_care_about,
                monitoring_factor_thermal_limits=getattr(config, 'MONITORING_FACTOR_THERMAL_LIMITS', 1.0)
            )
    else:
        print("Skipping visualization (DO_VISUALIZATION=False)")

    # Save data for tests
    if config.DO_SAVE_DATA_FOR_TEST:
        with Timer("Saving Test Data"):
            case_name = f"defaut_{'_'.join(current_lines_defaut)}_t{current_timestep}"
            save_data_for_test(config.ENV_PATH, case_name, df_of_g, overflow_sim, obs_simu_defaut,
                                lines_non_reconnectable, non_connected_reconnectable_lines, lines_overloaded_ids_kept)

    # Update context with new objects for Step 2 Discovery
    context.update({
        "env": env,
        "obs": obs,
        "obs_simu_defaut": obs_simu_defaut,
        "df_of_g": df_of_g,
        "overflow_sim": overflow_sim,
        "g_overflow": g_overflow,
        "hubs": hubs,
        "g_distribution_graph": g_distribution_graph,
        "node_name_mapping": node_name_mapping,
        "use_dc": use_dc
    })
    return context


def run_analysis_step2_discovery(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Third part of the expert system analysis focusing on path analysis and action discovery.
    """
    # Unpack context
    backend = context["backend"]
    env = context["env"]
    obs = context["obs"]
    obs_simu_defaut = context["obs_simu_defaut"]
    analysis_date = context["analysis_date"]
    current_timestep = context["current_timestep"]
    current_lines_defaut = context["current_lines_defaut"]
    lines_overloaded_ids = context["lines_overloaded_ids"]
    lines_overloaded_ids_kept = context["lines_overloaded_ids_kept"]
    maintenance_to_reco_at_t = context["maintenance_to_reco_at_t"]
    act_reco_maintenance = context["act_reco_maintenance"]
    lines_non_reconnectable = context["lines_non_reconnectable"]
    lines_we_care_about = context["lines_we_care_about"]
    classifier = context["classifier"]
    pre_existing_rho = context["pre_existing_rho"]
    lines_overloaded_names = context["lines_overloaded_names"]
    non_connected_reconnectable_lines = context["non_connected_reconnectable_lines"]
    dict_action = context["dict_action"]
    
    is_pypowsybl = context["is_pypowsybl"]
    actual_fast_mode = context["actual_fast_mode"]
    
    get_env_first_obs = context["get_env_first_obs"]
    create_default_action = context["create_default_action"]
    check_rho_reduction = context["check_rho_reduction"]
    
    g_overflow = context["g_overflow"]
    overflow_sim = context["overflow_sim"]
    hubs = context["hubs"]
    g_distribution_graph = context["g_distribution_graph"]
    node_name_mapping = context["node_name_mapping"]
    use_dc = context["use_dc"]

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
            create_default_action_func=create_default_action,
            obs_linecut=getattr(overflow_sim, 'obs_linecut', None)
        )
        
        # Monkey patch check_rho_reduction if using PyPowSybl so we can pass fast_mode through the discoverer indirectly
        if is_pypowsybl:
            original_check = discoverer._check_rho_reduction
            discoverer._check_rho_reduction = lambda *args, **kwargs: original_check(*args, fast_mode=actual_fast_mode, **kwargs)
            # Also override baseline simulation functions for optimized batch checking
            from expert_op4grid_recommender.utils.simulation_pypowsybl import (
                compute_baseline_simulation as _pypowsybl_compute_baseline,
                check_rho_reduction_with_baseline as _pypowsybl_check_with_baseline,
            )
            discoverer._compute_baseline = lambda *args, **kwargs: _pypowsybl_compute_baseline(*args, fast_mode=actual_fast_mode, **kwargs)
            discoverer._check_rho_with_baseline = lambda *args, **kwargs: _pypowsybl_check_with_baseline(*args, fast_mode=actual_fast_mode, **kwargs)

        prioritized_actions, action_scores = discoverer.discover_and_prioritize(
            n_action_max=config.N_PRIORITIZED_ACTIONS
        )

    print("\nPrioritized actions are: " + str(list(prioritized_actions.keys())))

    # Reassess the prioritized actions and collect detailed results
    with Timer("Reassessment"):
        act_defaut = create_default_action(env.action_space, current_lines_defaut)

        # Compute baseline rho once for all actions
        # recompute obs_simu_defaut if DC mode was used for overflow graph
        if is_pypowsybl and obs_simu_defaut._network_manager._default_dc:
            obs_simu_defaut, has_converged = simulate_contingency_pypowsybl(
                env, obs, current_lines_defaut, act_reco_maintenance, current_timestep, fast_mode=actual_fast_mode
            )
            obs_simu_defaut._network_manager._default_dc = False
        elif not is_pypowsybl:
            obs_simu_defaut, has_converged = simulate_contingency_grid2op(
                env, obs, current_lines_defaut, act_reco_maintenance, current_timestep
            )
        baseline_rho = obs_simu_defaut.rho[lines_overloaded_ids]

        # Performance optimization: Pre-calculate masks and baseline for large grids
        num_lines = len(obs.name_line)
        pre_existing_baseline = np.zeros(num_lines)
        is_pre_existing = np.zeros(num_lines, dtype=bool)
        for idx, rho_val in pre_existing_rho.items():
            pre_existing_baseline[idx] = rho_val
            is_pre_existing[idx] = True

        if lines_we_care_about is not None and len(lines_we_care_about) > 0:
            care_mask = np.isin(obs.name_line, list(lines_we_care_about))
        else:
            care_mask = np.ones(num_lines, dtype=bool)

        detailed_actions = {}
        for action_id, action in prioritized_actions.items():
            # Get description_unitaire if available
            description_unitaire = None
            if action_id in dict_action:
                action_desc = dict_action[action_id]
                description_unitaire = action_desc.get("description_unitaire")

            # Simulate action
            is_pypowsybl = backend == Backend.PYPOWSYBL
            if is_pypowsybl:
                # Optimized for pypowsybl: simulate starting from the N-1 converged state
                obs_simu_action, _, _, info_action = obs_simu_defaut.simulate(
                    action,
                    time_step=current_timestep,
                    keep_variant=True,
                    fast_mode=actual_fast_mode
                )
            else:
                # Standard backend behavior
                obs_simu_action, _, _, info_action = obs.simulate(
                    action + act_defaut + act_reco_maintenance,
                    time_step=current_timestep
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
                worsening_threshold = getattr(config, 'PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD', 0.02)
                action_rho = obs_simu_action.rho

                # Optimized vectorized max rho calculation
                worsened_mask = action_rho > pre_existing_baseline * (1 + worsening_threshold)
                eligible_mask = care_mask & (~is_pre_existing | worsened_mask)

                if np.any(eligible_mask):
                    masked_rho = action_rho[eligible_mask]
                    max_idx_in_masked = np.argmax(masked_rho)
                    max_rho = float(masked_rho[max_idx_in_masked])
                    max_rho_line = obs_simu_action.name_line[np.where(eligible_mask)[0][max_idx_in_masked]]

            # Capture non-convergence reason
            sim_exception = info_action.get("exception")
            non_convergence = None
            if sim_exception:
                if isinstance(sim_exception, list):
                    non_convergence = "; ".join([str(e) for e in sim_exception])
                else:
                    non_convergence = str(sim_exception)

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
                "non_convergence": non_convergence,
            }

        # Propagate non-convergence info to the score map
        for action_id, details in detailed_actions.items():
            nc = details.get("non_convergence")
            for category in action_scores:
                if action_id in action_scores[category]["scores"]:
                    # Ensure non_convergence dict exists (safety)
                    if "non_convergence" not in action_scores[category]:
                        action_scores[category]["non_convergence"] = {}
                    action_scores[category]["non_convergence"][action_id] = nc

    # Combined Action Pairs using the Superposition Theorem
    combined_actions = {}
    with Timer("Combined Action Pairs (Superposition)"):
        try:
            from expert_op4grid_recommender.utils.superposition import compute_all_pairs_superposition
            combined_actions = compute_all_pairs_superposition(
                obs_start=obs_simu_defaut,
                detailed_actions=detailed_actions,
                classifier=classifier,
                env=env,
                lines_overloaded_ids=lines_overloaded_ids,
                lines_we_care_about=lines_we_care_about,
                pre_existing_rho=pre_existing_rho,
                dict_action=dict_action,
            )
        except Exception as e:
            print(f"Warning: Failed to compute combined action pairs: {e}")
            import traceback
            traceback.print_exc()

    # Build pre-existing overloads info for the frontend
    pre_existing_info = [
        {"name": str(obs.name_line[i]), "rho_N": pre_existing_rho[i]}
        for i in sorted(pre_existing_rho.keys())
    ]

    return {
        "lines_overloaded_names": lines_overloaded_names,
        "prioritized_actions": detailed_actions,
        "action_scores": action_scores,
        "pre_existing_overloads": pre_existing_info,
        "combined_actions": combined_actions,
    }


def run_analysis_step2(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Second part of the expert system analysis after detection/selection of overloads.
    (Compatibility wrapper for the previous monolithic Step 2)
    """
    context = run_analysis_step2_graph(context)
    return run_analysis_step2_discovery(context)


def run_analysis(analysis_date: Optional[datetime], 
                 current_timestep: int, 
                 current_lines_defaut: List[str],
                 env_path: Optional[str] = None, 
                 env_name: Optional[str] = None,
                 backend: Backend = Backend.GRID2OP,
                 fast_mode: Optional[bool] = None) -> Dict[str, Any]:
    """
    Runs the expert system analysis for a given date, timestep, and contingency.
    (Now refactored to use run_analysis_step1, run_analysis_step2_graph and run_analysis_step2_discovery)
    """
    # Step 1: Detect overloads and selection of overloads to keep
    res_step1, context = run_analysis_step1(
        analysis_date=analysis_date,
        current_timestep=current_timestep,
        current_lines_defaut=current_lines_defaut,
        env_path=env_path,
        env_name=env_name,
        backend=backend,
        fast_mode=fast_mode
    )

    if res_step1 is not None:
        return res_step1

    # Step 2: Run the remaining analysis
    context = run_analysis_step2_graph(context)
    return run_analysis_step2_discovery(context)


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
    parser.add_argument(
        "--pypowsybl-format",
        action='store_true',
        help=(
            "When used with --rebuild-actions and an empty action file (from scratch), "
            "outputs the action dictionary in pypowsybl format: switch-based entries with "
            "description, description_unitaire, VoltageLevelId and switches fields "
            "(no Grid2Op set_bus content). Duplicate actions sharing identical switch "
            "states are deduplicated; removed duplicates are listed in 'other_action_ids'."
        )
    )
    parser.add_argument(
        "--ignore-lines-monitoring",
        action='store_true',
        help="If set, ignores the lignes_a_monitorer.csv file and monitors all lines."
    )
    parser.add_argument(
        "--fast-mode",
        action='store_true',
        help="If set, uses pypowsybl fast mode (no voltage control) for grid simulations."
    )
    args = parser.parse_args()

    if args.ignore_lines_monitoring:
        config.IGNORE_LINES_MONITORING = True

    sum_min_actions = (config.MIN_LINE_RECONNECTIONS +
                       config.MIN_CLOSE_COUPLING +
                       config.MIN_OPEN_COUPLING +
                       config.MIN_LINE_DISCONNECTIONS)
    
    if sum_min_actions > config.N_PRIORITIZED_ACTIONS:
        print(f"Warning: The sum of minimum actions per type ({sum_min_actions}) exceeds the "
              f"maximum number of prioritized actions overall ({config.N_PRIORITIZED_ACTIONS}). "
              f"Some minimums will not be respected.", file=sys.stderr)

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
                                                   output_file_base_name="reduced_model_actions",
                                                   pypowsybl_format=args.pypowsybl_format)

                print("Action rebuilding process complete. Stopping analysis as requested.")
                return  # EXIT EARLY
            else:
                run_analysis(
                    analysis_date=date_arg,
                    current_timestep=args.timestep,
                    current_lines_defaut=args.lines_defaut,
                    backend=backend,
                    fast_mode=args.fast_mode
                )
    except (ValueError, RuntimeError, TypeError) as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
