# expert_op4grid_recommender/environment.py
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
import json
import datetime
import sys
from expert_op4grid_recommender.utils.make_assistant_env import make_grid2op_assistant_env
from expert_op4grid_recommender.utils.make_training_env import make_grid2op_training_env
from expert_op4grid_recommender.utils.load_evaluation_data import list_all_chronics, get_first_obs_on_chronic
from expert_op4grid_recommender import config
# FIX: Import the config module relatively
from expert_op4grid_recommender.data_loader import load_interesting_lines, DELETED_LINE_NAME, load_actions
from expert_op4grid_recommender.utils.simulation import simulate_contingency, check_simu_overloads


def get_env_first_obs(env_folder, env_name, use_evaluation_config, date, is_DC=False):
    """
    Creates a Grid2Op environment and retrieves the first observation for a specific chronic.

    This function initializes either a training or an assistant Grid2Op environment based on
    the provided configuration. It can optionally configure the environment to use a DC power
    flow model. It then finds the chronic (time-series data) corresponding to the given date
    and loads the initial state (observation) for that chronic.

    Args:
        env_folder (str): The path to the folder containing the environment definition files.
        env_name (str): The specific name of the Grid2Op environment to load.
        use_evaluation_config (bool): If True, uses the assistant environment configuration;
                                      otherwise, uses the training environment configuration.
        date (datetime.datetime): The date used to identify and select the appropriate chronic file.
        is_DC (bool, optional): If True, configures the environment to use a DC power flow
                                model instead of the default AC model. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - env (grid2op.Environment): The initialized Grid2Op environment object.
            - obs (grid2op.Observation): The first observation object for the selected chronic.
            - path_chronic (str): The file path to the selected chronic data.
    """
    if use_evaluation_config:
        env = make_grid2op_assistant_env(env_folder, env_name)
        if is_DC:
            env_params = env.parameters
            env_params.ENV_DC = True
            env = make_grid2op_assistant_env(env_folder, env_name, params=env_params)
    else:
        env = make_grid2op_training_env(env_folder, env_name)
        if is_DC:
            env_params = env.parameters
            env_params.ENV_DC = True
            env = make_grid2op_training_env(env_folder, env_name, params=env_params)

    path_chronic = [path for path in env.chronics_handler.real_data.subpaths if date.strftime('%Y%m%d') in path][0]
    obs = get_first_obs_on_chronic(date, env, path_thermal_limits=path_chronic)
    return env, obs, path_chronic

def setup_environment_configs(analysis_date: datetime): # Add analysis_date argument
    """
    Sets up the Grid2Op environment and loads related configuration files based on settings in the config module.

    This function performs several setup tasks:
    1. Loads the predefined action space from a JSON file specified in the config.
    2. Initializes the Grid2Op environment using parameters from the config (folder, name, date, evaluation mode).
    3. Retrieves the first observation for the specified chronic date.
    4. Optionally loads a custom grid layout for visualization if specified in the config.
    5. Loads lists of non-reconnectable lines and lines explicitly marked for monitoring from CSV files.

    Returns:
        tuple: A tuple containing the following elements:
            - env (grid2op.Environment): The initialized Grid2Op environment object.
            - obs (grid2op.Observation): The first observation for the selected chronic.
            - path_chronic (str): The file path to the selected chronic data.
            - chronic_name (str): The name of the loaded chronic.
            - custom_layout (list or None): A list of coordinates for grid layout, or None if not used.
            - dict_action (dict): The dictionary representing the loaded action space.
            - lines_non_reconnectable (list): A list of line names that cannot be reconnected.
            - lines_we_care_about (list): A list of line names specifically designated for monitoring.

    Raises:
        FileNotFoundError: If the action space file specified in the config does not exist.
        FileNotFoundError: If the chronic for the specified date cannot be found.
    """
    dict_action = load_actions(config.ACTION_FILE_PATH)
    # Use the passed analysis_date instead of config.DATE
    env, obs, path_chronic = get_env_first_obs(config.ENV_FOLDER, config.ENV_NAME, config.USE_EVALUATION_CONFIG,
                                               analysis_date)
    chronic_name = env.chronics_handler.get_name()

    custom_layout = None
    if config.USE_GRID_LAYOUT and hasattr(env, 'grid_layout'):
        custom_layout = [env.grid_layout[sub] for sub in env.name_sub]

    lines_non_reconnectable = list(load_interesting_lines(path=path_chronic, file_name="non_reconnectable_lines.csv"))
    lines_non_reconnectable += list(DELETED_LINE_NAME)

    lines_we_care_about = load_interesting_lines(file_name=os.path.join(config.ENV_FOLDER, "lignes_a_monitorer.csv"))

    return env, obs, path_chronic, chronic_name, custom_layout, dict_action, lines_non_reconnectable, lines_we_care_about


def switch_to_dc_load_flow(env, date, timestep, lines_defaut, lines_overloaded_ids_kept, maintenance_to_reco_at_t):
    """
    Reloads the Grid2Op environment using DC power flow model and verifies simulation convergence.

    This function is typically called when AC power flow simulations fail to converge,
    often due to complex grid states resulting from multiple line disconnections.
    It attempts to re-initialize the environment with the simpler DC power flow model
    and then re-runs the initial contingency simulation and the overload disconnection check.
    If either simulation fails even in DC mode, it prints an error and exits the program.

    Args:
        env (grid2op.Environment): The original Grid2Op environment (likely configured for AC).
        date (datetime.datetime): The date for which the environment and chronic should be loaded.
        timestep (int): The specific timestep within the chronic being analyzed.
        lines_defaut (list[str]): List of line names representing the initial contingency (N-1).
        lines_overloaded_ids_kept (list[int]): List of indices of overloaded lines being considered
                                              for disconnection in the overflow graph analysis.
        maintenance_to_reco_at_t (list[str]): List of line names that were under maintenance
                                              but are scheduled for reconnection at this timestep.

    Returns:
        tuple: A tuple containing:
            - new_env (grid2op.Environment): The newly initialized environment configured for DC power flow.
            - new_obs (grid2op.Observation): The initial observation obtained from the DC environment.
            - obs_simu (grid2op.Observation): The observation resulting from simulating the initial
                                             contingency (`lines_defaut`) in the DC environment.

    Raises:
        SystemExit: If simulations fail to converge even with the DC power flow model.
    """

    print("Switching to DC load flow due to convergence issues or user setting.")
    new_env, new_obs, _ = get_env_first_obs(config.ENV_FOLDER, config.ENV_NAME, config.USE_EVALUATION_CONFIG, date,
                                            is_DC=True)

    act_reco_maintenance = new_env.action_space(
        {"set_line_status": [(line_reco, 1) for line_reco in maintenance_to_reco_at_t]}
    )

    obs_simu, has_converged_1 = simulate_contingency(new_env, new_obs, lines_defaut, act_reco_maintenance, timestep)
    has_converged_2, has_lost_load = check_simu_overloads(
        new_obs, obs_simu, new_env.action_space, timestep, lines_defaut, lines_overloaded_ids_kept,
        maintenance_to_reco_at_t
    )

    if not has_converged_1 or not has_converged_2:
        print("Simulation failed even in DC mode. Cannot build overflow graph.")
        sys.exit(0)

    return new_env, new_obs, obs_simu