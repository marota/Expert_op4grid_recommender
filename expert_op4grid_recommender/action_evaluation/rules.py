# expert_op4grid_recommender/action_evaluation/rules.py
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
from expert_op4grid_recommender.utils.data_utils import StateInfo
from expert_op4grid_recommender.utils.load_training_data import aux_prevent_asset_reconnection
from .classifier import identify_action_type
from expert_op4grid_recommender.utils.simulation import check_rho_reduction, create_default_action
from typing import Dict, Any, List, Tuple, Optional, Callable, Set

def localize_line_action(lines, lines_constrained_path, lines_dispatch):
    """
    Determines the location category of a line action relative to critical graph paths.

    This function checks if any of the lines involved in a specific grid action
    overlap with the predefined "constrained path" or "dispatch paths", which are
    typically derived from an overflow graph analysis. The localization helps in
    applying expert rules (e.g., prohibiting disconnections on dispatch paths).

    The function prioritizes the constrained path: if any action line is on it,
    the action is categorized as "constrained_path", regardless of overlap with
    dispatch paths.

    Args:
        lines (list[str]): A list of line identifiers (e.g., names) involved
            in the action being evaluated.
        lines_constrained_path (list[str]): A list of line identifiers belonging
            to the constrained path (critical path for flow).
        lines_dispatch (list[str]): A list of line identifiers belonging to the
            dispatch paths (alternative flow routes).

    Returns:
        str: A string indicating the action's location category:
             - "constrained_path": If any action line is found in `lines_constrained_path`.
             - "dispatch_path": If no action line is in the constrained path, but at least
               one is found in `lines_dispatch`.
             - "out_of_graph": If none of the action lines are found in either the
               constrained or dispatch paths.
    """
    if set(lines).intersection(set(lines_constrained_path)):
        return "constrained_path"
    if set(lines).intersection(set(lines_dispatch)):
        return "dispatch_path"
    return "out_of_graph"


def localize_coupling_action(action_subs, hubs, nodes_constrained_path, nodes_dispatch_path):
    """
    Determines the location category of a coupling action relative to critical grid nodes/substations.

    This function checks if any of the substations involved in a coupling action
    (e.g., opening or closing a bus coupler or section switch) overlap with predefined
    sets of important locations: hubs, nodes on the constrained path, or nodes on
    dispatch paths. These locations are typically derived from an overflow graph analysis.
    The localization helps in applying expert rules (e.g., prohibiting splitting at hubs
    unless necessary).

    The function checks for overlap in a specific order of priority:
    1. Hubs
    2. Constrained Path Nodes
    3. Dispatch Path Nodes

    It returns the category corresponding to the first match found.

    Args:
        action_subs (List[str]): A list containing the name(s) of the substation(s)
            where the coupling action takes place.
        hubs (List[str]): A list of substation names identified as critical hubs.
        nodes_constrained_path (List[str]): A list of substation names belonging
            to the constrained path (critical path for flow).
        nodes_dispatch_path (List[str]): A list of substation names belonging to the
            dispatch paths (alternative flow routes).

    Returns:
        str: A string indicating the action's location category:
             - "hubs": If any action substation is found in `hubs`.
             - "constrained_path": If no action substation is in `hubs`, but at least
               one is found in `nodes_constrained_path`.
             - "dispatch_path": If no action substation is in `hubs` or `nodes_constrained_path`,
               but at least one is found in `nodes_dispatch_path`.
             - "out_of_graph": If none of the action substations are found in any of the
               specified critical location lists.
    """

    action_subs_set = set(action_subs)
    if action_subs_set.intersection(set(hubs)):
        return "hubs"
    if action_subs_set.intersection(set(nodes_constrained_path)):
        return "constrained_path"
    if action_subs_set.intersection(set(nodes_dispatch_path)):
        return "dispatch_path"
    return "out_of_graph"


def check_rules(action_type: str, localization: str, subs_topology: List[List[int]]) -> Tuple[bool, Optional[str]]:
    """
    Checks if a given grid action violates predefined expert rules based on its type and location.

    This function implements a set of heuristics, often derived from power system operator
    experience, to filter out actions that are generally considered counterproductive or
    irrelevant for resolving overloads in specific contexts (e.g., disconnecting a line
    on an alternative path when the main path is constrained).

    The rules primarily depend on the action's type (line operation, coupling operation)
    and its localization relative to critical paths (constrained path, dispatch path, or
    outside the relevant graph section) identified by overflow graph analysis.

    Actions involving loads are currently *not* filtered by this function.

    Args:
        action_type (str): The type of the action being evaluated (e.g., "open_line",
            "close_coupling", "close_line").
        localization (str): The location category of the action (e.g., "constrained_path",
            "dispatch_path", "hubs", "out_of_graph"). Determined by functions like
            `localize_line_action` or `localize_coupling_action`.
        subs_topology (List[List[int]]): A list containing the topology vectors (bus assignments)
            for each substation involved in the action, representing the state *before*
            the action. Used to check the initial bus configuration for the node splitting rule.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - do_filter_action (bool): True if the action violates one of the rules and should
              be filtered out, False otherwise.
            - broken_rule (Optional[str]): A string describing the specific rule that was
              violated if `do_filter_action` is True, otherwise None.
    """
    do_filter_action = False
    broken_rule = None

    # Rule application currently ignores actions directly manipulating loads
    if "load" not in action_type:
        # Check if the involved substation(s) were initially single-node configurations
        # (excluding disconnected buses -1 and unused 0). This is relevant for the splitting rule.
        # It checks if *all* substations involved were initially single-node.
        is_topo_subs_one_node = all(len(set(sub_topo) - {-1, 0}) == 1 for sub_topo in subs_topology)

        # Define the set of expert rules as conditions to check
        rules = {
            # Rule 1: Filter actions outside the main area of interest (overflow graph)
            "No action out of the overflow graph": (localization == "out_of_graph"),

            # Rule 2: Avoid reconnecting lines on the constrained path (likely counterproductive)
            "No line reconnection on constrained path": (
                        "line" in action_type and "close" in action_type and localization == "constrained_path"),

            # Rule 3: Avoid disconnecting lines on dispatch paths (alternative routes should be kept open)
            "No line disconnection on dispatch path": (
                        "line" in action_type and "open" in action_type and localization == "dispatch_path"),

            # Rule 4: Avoid merging nodes (closing couplings) on the constrained path
            "No node merging on constrained path": (
                        "coupling" in action_type and "close" in action_type and localization == "constrained_path"),

            # Rule 5: Avoid splitting nodes (opening couplings) on dispatch paths, *if* the sub was initially single-node
            "No node splitting on dispatch path": (
                        "coupling" in action_type and "open" in action_type and localization == "dispatch_path" and is_topo_subs_one_node),
        }

        # Check each rule; stop and return the first one that is violated
        for rule_name, is_broken in rules.items():
            if is_broken:
                do_filter_action = True
                broken_rule = rule_name
                break # Stop checking once a rule is broken

    return do_filter_action, broken_rule


def verify_action(action_desc: Dict[str, Any],
                  hubs: List[str],
                  paths: Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]],
                  subs_topology: List[List[int]],
                  by_description: bool,
                  action_space: Callable,
                  obs: Any # grid2op.Observation or similar mock
                  ) -> Tuple[bool, Optional[str]]:
    """
    Verifies if a grid action should be filtered based on its type, location, and expert rules.

    This function acts as a primary filter for individual actions. It performs several checks:
    1.  Identifies the action type (e.g., "open_line", "close_coupling") using `identify_action_type`.
    2.  If it's a line action, checks basic validity:
        - Cannot disconnect an already disconnected line.
        - Cannot reconnect an already connected line.
    3.  Determines the action's location relative to critical paths (constrained, dispatch)
        and hubs using `localize_line_action` or `localize_coupling_action`.
    4.  If not already filtered by basic checks, applies a set of expert rules defined in
        `check_rules` based on the action type and localization.

    Args:
        action_desc (Dict[str, Any]): Dictionary containing the action's description and content.
            Expected keys include "content" (with "set_bus") and potentially "description_unitaire",
            "VoltageLevelId".
        hubs (List[str]): List of substation names identified as critical hubs.
        paths (Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]]): A tuple containing two tuples:
            - The first tuple holds `(lines_constrained_path, nodes_constrained_path)`.
            - The second tuple holds `(lines_dispatch, nodes_dispatch_path)`.
            These lists contain names of lines/substations on critical paths.
        subs_topology (List[List[int]]): List of topology vectors (bus assignments) for substations
            involved in the action *before* the action is applied. Used by `check_rules`.
        by_description (bool): Flag passed to `identify_action_type` to indicate whether
            classification should use the description string or the Grid2Op object.
        action_space (Callable): The Grid2Op action space object, required by `identify_action_type`
            if `by_description` is False.
        obs (Any): The Grid2Op observation object (or a mock) representing the current grid state.
            Used to check current line statuses.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - do_filter_action (bool): True if the action violates any check or rule and should
              be filtered out, False otherwise.
            - broken_rule (Optional[str]): A string describing the reason for filtering (either
              a basic check failure or the name of the expert rule violated), or None if the
              action is not filtered.
    """
    # Unpack path information
    lines_constrained_path, nodes_constrained_path = paths[0]
    lines_dispatch, nodes_dispatch_path = paths[1]

    # 1. Identify action type
    action_type = identify_action_type(action_desc, by_description, action_space)

    do_filter_action = False
    broken_rule = None # Initialize broken_rule to None

    # 2. Basic checks for line actions
    if "line" in action_type:
        grid2op_actions_set_bus = action_desc.get("content", {}).get("set_bus", {})
        # Get lines involved in the action
        lines = list(grid2op_actions_set_bus.get("lines_or_id", {}).keys()) + \
                list(grid2op_actions_set_bus.get("lines_ex_id", {}).keys())

        if lines: # Proceed only if lines are actually involved
            # Get current status of the involved lines
            line_status_map = {name: status for name, status in zip(obs.name_line, obs.line_status)}
            current_statuses = np.array([line_status_map.get(line_name, True) for line_name in lines]) # Default to True if somehow missing

            # Check if trying to open an already open line
            if "open" in action_type and np.all(~current_statuses):
                do_filter_action, broken_rule = True, "No disconnection of a line already disconnected"
            # Check if trying to close an already closed line
            elif "close" in action_type and np.all(current_statuses):
                do_filter_action, broken_rule = True, "No reconnection of a line already connected"

        # 3a. Localize line action
        localization = localize_line_action(lines, lines_constrained_path, lines_dispatch)

    # 3b. Localize coupling action
    elif "coupling" in action_type and "VoltageLevelId" in action_desc:
        action_subs = [action_desc["VoltageLevelId"]]
        localization = localize_coupling_action(action_subs, hubs, nodes_constrained_path, nodes_dispatch_path)
    else:
        # For other action types (like load actions) or actions without clear location info
        localization = "unknown" # Or potentially "out_of_graph" depending on desired default

    # 4. Apply expert rules if not already filtered
    if not do_filter_action:
        do_filter_action, broken_rule_expert = check_rules(action_type, localization, subs_topology)
        # Only update broken_rule if an expert rule was actually broken
        if do_filter_action:
            broken_rule = broken_rule_expert

    # Return the filter decision and the reason (if any)
    return do_filter_action, broken_rule

def categorize_action_space(dict_action: Dict[str, Dict[str, Any]],
                            hubs: List[str],
                            paths: Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]],
                            obs: Any, # grid2op.Observation or similar mock
                            timestep: int,
                            defauts: List[str],
                            action_space: Callable,
                            overload_ids: List[int],
                            lines_reco_maintenance: List[str],
                            by_description: bool = True,
                            lines_we_care_about: Optional[List[str]] = None
                            ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Categorizes actions from a given dictionary into 'filtered' and 'unfiltered' sets based on expert rules.

    This function iterates through a dictionary of potential grid actions (`dict_action`).
    For each action, it uses the `verify_action` function to determine if it should be
    filtered out based on its type, location relative to critical paths (`paths`), and
    expert rules (`check_rules`).

    If an action *is* marked for filtering by `verify_action`, this function performs an
    additional (optional) step: it simulates the action using `check_rho_reduction`. This
    simulation checks if the action, despite being filtered by rules, *would have* reduced
    overloads. This information is primarily for logging and analysis to identify potentially
    useful actions that the rules might be excluding ("badly filtered").

    Actions are then placed into one of two dictionaries: `actions_to_filter` or
    `actions_unfiltered`.

    Args:
        dict_action (Dict[str, Dict[str, Any]]): A dictionary mapping action IDs (str) to
            action description dictionaries. Each description dictionary should contain
            "content" and potentially "description_unitaire", "VoltageLevelId".
        hubs (List[str]): List of substation names identified as critical hubs.
        paths (Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]]): A tuple containing
            two tuples representing critical paths: `((lines_constrained, nodes_constrained),
            (lines_dispatch, nodes_dispatch))`.
        obs (Any): The Grid2Op observation object (or a mock) representing the current grid state.
            Used by `verify_action` and potentially `check_rho_reduction`.
        timestep (int): The current simulation timestep, passed to `check_rho_reduction`.
        defauts (List[str]): List of line names representing the initial contingency (N-1),
            passed to `check_rho_reduction`.
        action_space (Callable): The Grid2Op action space object, used to create action objects
            for simulation and potentially by `verify_action`.
        overload_ids (List[int]): List of indices of the lines currently considered overloaded,
            passed to `check_rho_reduction`.
        lines_reco_maintenance (List[str]): List of line names representing maintenance reconnections,
            used to create an action passed to `check_rho_reduction`.
        by_description (bool, optional): Flag passed to `verify_action` to indicate how actions
            should be identified (by description string or Grid2Op object). Defaults to True.
        lines_we_care_about (Optional[List[str]], optional): Specific lines to monitor during the
            `check_rho_reduction` simulation. Defaults to None.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]: A tuple containing two dictionaries:
            - actions_to_filter (Dict): Maps action IDs to dictionaries containing details about
              why the action was filtered (`"broken_rule"`) and whether it would have reduced
              overloads (`"is_rho_reduction"`). Includes `"description_unitaire"`.
            - actions_unfiltered (Dict): Maps action IDs to dictionaries for actions that passed
              the filtering rules. Currently only includes `"description_unitaire"`.
    """
    actions_to_filter = {}
    actions_unfiltered = {}

    # Iterate through each candidate action
    for action_id, action_desc in dict_action.items():
        # Get the topology of the substation(s) involved (if applicable) for rule checking
        subs_topology = []
        if "VoltageLevelId" in action_desc:
            action_subs = [action_desc["VoltageLevelId"]]
            try:
                # Find substation index and get its current topology vector
                subs_topology = [obs.sub_topology(np.where(obs.name_sub == sub_name)[0][0]) for sub_name in action_subs]
            except IndexError:
                print(f"Warning: Substation {action_subs} not found in observation for action {action_id}. Skipping topology check.")
                subs_topology = [] # Cannot check topology-dependent rules

        # Verify if the action should be filtered based on type, location, and rules
        do_filter_action, broken_rule = verify_action(action_desc, hubs, paths, subs_topology, by_description,
                                                      action_space, obs)

        # Get description for logging/output
        description = action_desc.get("description_unitaire", action_desc.get("description", ""))

        # If the action violates a rule
        if do_filter_action:
            # --- Optional: Check if the filtered action would have helped ---
            # Create the default (contingency) action for simulation context
            act_defaut = create_default_action(action_space, defauts)
            # Create the action object from its description
            action = action_space(action_desc["content"])
            # Apply potential safety modifications (e.g., prevent reconnecting certain assets)
            state = StateInfo() # Assuming StateInfo is available
            action = aux_prevent_asset_reconnection(obs, state, action) # Assuming aux_prevent... is available

            # Create the maintenance reconnection action object
            act_reco_maintenance_obj = action_space(
                {"set_line_status": [(line_reco, 1) for line_reco in lines_reco_maintenance]})
            # Simulate to see if rho would have been reduced
            is_rho_reduction, _ = check_rho_reduction(obs, timestep, act_defaut, action, overload_ids,
                                                      act_reco_maintenance_obj, lines_we_care_about)

            # Log if a potentially useful action was filtered
            if is_rho_reduction:
                print(f"INFO: Action '{description}' was filtered by rule '{broken_rule}' but showed potential rho reduction.")

            # Store filter information
            actions_to_filter[action_id] = {
                "description_unitaire": description,
                "broken_rule": broken_rule,
                "is_rho_reduction": is_rho_reduction # Store simulation result
            }
        else:
            # If the action passed the rules, add it to the unfiltered set
            actions_unfiltered[action_id] = {"description_unitaire": description}

    return actions_to_filter, actions_unfiltered