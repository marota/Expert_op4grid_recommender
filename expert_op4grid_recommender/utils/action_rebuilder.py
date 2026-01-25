#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles.

"""
Action Rebuilding Module

This module contains functions for rebuilding and converting action dictionaries
from REPAS format to Grid2Op format based on network grid snapshots.
"""

import copy
import json
import os
from expert_op4grid_recommender.utils.helpers import Timer

from expert_op4grid_recommender.utils import repas
from expert_op4grid_recommender.utils.conversion_actions_repas import (
    convert_repas_actions_to_grid2op_actions,
    create_dict_disco_reco_lines_disco
)

def make_description_unitaire(switches_by_voltage_level):
    """
    Creates a unitary description string for a switch action.

    Parameters
    ----------
    switches_by_voltage_level : dict
        Dictionary mapping voltage level to switch actions.

    Returns
    -------
    str
        Human-readable description of the switch action.
    """
    # Récupérer les informations
    voltage_level = next(iter(switches_by_voltage_level.keys()))
    inner_dict = switches_by_voltage_level[voltage_level]
    switch_name = next(iter(inner_dict.keys()))
    switch_value = inner_dict[switch_name]

    # Créer la description
    action_type = "Ouverture" if switch_value else "Fermeture"
    description = f"{action_type} {switch_name} dans le poste {voltage_level}"

    return description


def make_raw_all_actions_dict(all_actions):
    """
    Creates a raw dictionary of all actions, splitting combined actions by voltage level.

    Parameters
    ----------
    all_actions : list
        List of REPAS action objects.

    Returns
    -------
    dict
        Dictionary mapping action IDs to action objects.
    """
    all_actions_dict = {}
    for action in all_actions:
        if len(action._switches_by_voltage_level) == 1:
            all_actions_dict[action._id] = action
        elif len(action._switches_by_voltage_level) >= 2:
            for voltage_level in action._switches_by_voltage_level.keys():
                action_key = action._id + "_" + voltage_level

                new_action = copy.deepcopy(action)
                filtered_switch_action = new_action._switches_by_voltage_level[voltage_level]
                new_action._switches_by_voltage_level = {voltage_level: filtered_switch_action}

                all_actions_dict[action_key] = new_action

    return all_actions_dict


def build_action_dict_for_snapshot_from_scratch(n_grid, all_actions, add_reco_disco_actions=False, filter_voltage_levels=[]):
    """
    Builds the action dictionary based on the current network grid snapshot.
    Converts all actions involving coupling breakers from REPAS format to Grid2Op format where necessary.

    Parameters
    ----------
    n_grid : pypowsybl.network.Network
        The network grid object.
    all_actions : list
        List of REPAS action objects.
    add_reco_disco_actions : bool, optional
        Whether to add reconnection/disconnection actions for disconnected lines.
    filter_voltage_levels : list, optional
        List of voltage levels to filter on for extra disco/reco actions.

    Returns
    -------
    dict
        Dictionary of converted actions.
    """
    with Timer("Rebuilding Action Dictionary"):
        # Create dictionary, first with combined actions and then for single actions
        all_actions_dict = make_raw_all_actions_dict(all_actions)

        actions_to_convert = []

        for action_key, action in all_actions_dict.items():
            switches = list(next(iter(action._switches_by_voltage_level.values())).keys())
            switches_str = str(switches)
            if "COUPL" in switches_str or "TRO." in switches_str:
                actions_to_convert.append(all_actions_dict[action_key])

        converted_actions = convert_repas_actions_to_grid2op_actions(n_grid, actions_to_convert,use_analytical=True)

        if add_reco_disco_actions:
            dict_extra_disco_reco_actions = create_dict_disco_reco_lines_disco(
                n_grid,
                filter_voltage_levels=filter_voltage_levels
            )
            converted_actions |= dict_extra_disco_reco_actions

        return converted_actions


def rebuild_action_dict_for_snapshot(n_grid, all_actions, dict_action):
    """
    Rebuilds the action dictionary based on the current network grid snapshot and already selected actions to consider.
    Converts actions from REPAS format to Grid2Op format where necessary.

    Parameters
    ----------
    n_grid : pypowsybl.network.Network
        The network grid object.
    all_actions : list
        List of REPAS action objects.
    dict_action : dict
        Dictionary of actions to filter on.

    Returns
    -------
    dict
        Dictionary of rebuilt actions.
    """
    with Timer("Rebuilding Action Dictionary"):
        # Create dictionary, first with combined actions and then for single actions
        all_actions_dict = make_raw_all_actions_dict(all_actions)

        actions_to_convert = []
        actions_to_keep_as_is = {}

        # Recover pypowsybl actions from dict action
        for action_full_id, action in dict_action.items():
            action_id_split = action_full_id.split('_')

            description = action["description"]
            if "description_unitaire" in action:
                description = action["description_unitaire"]

            if "COUPL" in description or "TRO." in description:
                action_key = action_id_split[0]
                if action_key not in all_actions_dict:
                    if "VoltageLevelId" in action.keys():
                        action_key += "_" + action["VoltageLevelId"]

                if action_key in all_actions_dict:
                    actions_to_convert.append(all_actions_dict[action_key])
                else:
                    print(f"Warning: Action {action_key} not found in REPAS actions. Skipping conversion.")
            else:
                actions_to_keep_as_is[action_full_id] = action

        converted_actions = convert_repas_actions_to_grid2op_actions(n_grid, actions_to_convert)

        # Adjust description unitaire
        for action_full_id, action in converted_actions.items():
            action_id = action_full_id.split('_')[0]
            Voltage_level = action_full_id.split('_')[1]

            possible_keys = [key for key in dict_action.keys() if action_id in key]

            if not possible_keys:
                continue

            action_key = possible_keys[0]
            if len(possible_keys) >= 2:
                filtered_keys = [key for key in possible_keys if Voltage_level in key]
                if len(filtered_keys) > 0:
                    action_key = filtered_keys[0]

            if action_key in dict_action and "description_unitaire" in dict_action[action_key]:
                action["description_unitaire"] = dict_action[action_key]["description_unitaire"]

        new_dict_actions = actions_to_keep_as_is
        new_dict_actions |= converted_actions

    return new_dict_actions


def run_rebuild_actions(n_grid, do_from_scratch, repas_file_path, dict_action_to_filter_on=None,
                        voltage_filter_threshold=300, output_file_base_name="reduced_model_actions"):
    """
    Orchestrates the action rebuilding process using the environment's grid.

    Parameters
    ----------
    n_grid : pypowsybl.network.Network
        The network grid object.
    do_from_scratch : bool
        Whether to build actions from scratch or rebuild from existing dictionary.
    repas_file_path : str
        Path to the REPAS actions JSON file.
    dict_action_to_filter_on : dict, optional
        Dictionary of actions to filter on (used when not building from scratch).
    voltage_filter_threshold : float, optional
        Voltage threshold for filtering actions (default: 300).
    output_file_base_name : str, optional
        Base name for the output file (default: "reduced_model_actions").

    Returns
    -------
    dict
        Dictionary of rebuilt actions.
    """
    if dict_action_to_filter_on is None:
        dict_action_to_filter_on = {}

    print(
        f"Rebuilding action dictionary based on current grid snapshot using REPAS file: {repas_file_path} "
        f"with voltage threshold < {voltage_filter_threshold}..."
    )

    output_file_path = os.path.join(
        "data", "action_space",
        f"{output_file_base_name}_{n_grid.case_date.strftime('%Y%m%dT%H%MZ')}.json"
    )

    try:
        with Timer("Total Rebuild Process"):
            # Load REPAS actions
            with Timer("Parse REPAS JSON"):
                all_actions = repas.parse_json(
                    repas_file_path,
                    n_grid,
                    lambda t: t[1]['nominal_v'] < voltage_filter_threshold
                )

            # Rebuild dictionary
            if do_from_scratch:
                new_dict_actions = build_action_dict_for_snapshot_from_scratch(n_grid, all_actions,add_reco_disco_actions=True)
            else:
                new_dict_actions = rebuild_action_dict_for_snapshot(n_grid, all_actions, dict_action_to_filter_on)

        # Save to file
        with open(output_file_path, "w") as json_file:
            json.dump(new_dict_actions, json_file, indent=4)

        print(f"Successfully rebuilt actions. Saved to: {output_file_path}")
        return new_dict_actions

    except Exception as e:
        print(f"Error rebuilding actions: {e}")
        return dict_action_to_filter_on  # Return original on failure