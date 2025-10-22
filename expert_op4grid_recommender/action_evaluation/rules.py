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
from expert_op4grid_recommender.action_evaluation.classifier import identify_action_type
from expert_op4grid_recommender.utils.simulation import check_rho_reduction, create_default_action
from typing import Dict, Any, List, Tuple, Optional, Callable, Set

class ActionRuleValidator:
    """
    Validates and categorizes grid actions based on expert rules and grid context.

    This class holds the necessary grid context (observation, critical paths, hubs)
    and provides methods to apply filtering rules to a set of candidate actions.
    The main method is `categorize_actions`, which iterates through actions,
    verifies them against rules, and optionally simulates filtered actions to
    check for potential effectiveness.

    Attributes:
        obs (Any): The Grid2Op observation object for the current state.
        action_space (Callable): The Grid2Op action space object.
        hubs (List[str]): List of critical hub substation names.
        lines_constrained_path (List[str]): Lines on the constrained path.
        nodes_constrained_path (List[str]): Nodes on the constrained path.
        lines_dispatch (List[str]): Lines on dispatch paths.
        nodes_dispatch_path (List[str]): Nodes on dispatch paths.
        by_description (bool): Flag indicating how to identify action types.
    """

    def __init__(self,
                 obs: Any, # grid2op.Observation or similar mock
                 action_space: Callable,
                 hubs: List[str],
                 paths: Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]],
                 by_description: bool = True):
        """
        Initializes the ActionRuleValidator.

        Args:
            obs: The Grid2Op observation object representing the current grid state.
            action_space: The Grid2Op action space object.
            hubs: List of substation names identified as critical hubs.
            paths: A tuple containing two tuples representing critical paths:
                   `((lines_constrained, nodes_constrained), (lines_dispatch, nodes_dispatch))`.
            by_description: Flag passed to `identify_action_type`. Defaults to True.
        """
        self.obs = obs
        self.action_space = action_space
        self.hubs = hubs
        # Unpack path information for easier access
        self.lines_constrained_path, self.nodes_constrained_path = paths[0]
        self.lines_dispatch, self.nodes_dispatch_path = paths[1]
        self.by_description = by_description

    def localize_line_action(self, lines: List[str]) -> str:
        """
        Determines the location category of a line action relative to critical graph paths.

        Uses the path information stored in the instance.

        Args:
            lines: A list of line identifiers (e.g., names) involved in the action.

        Returns:
            "constrained_path", "dispatch_path", or "out_of_graph".
        """
        action_lines_set = set(lines)
        if action_lines_set.intersection(set(self.lines_constrained_path)):
            return "constrained_path"
        if action_lines_set.intersection(set(self.lines_dispatch)):
            return "dispatch_path"
        return "out_of_graph"

    def localize_coupling_action(self, action_subs: List[str]) -> str:
        """
        Determines the location category of a coupling action relative to critical nodes.

        Uses the hub and path information stored in the instance.

        Args:
            action_subs: A list containing the name(s) of the substation(s) involved.

        Returns:
            "hubs", "constrained_path", "dispatch_path", or "out_of_graph".
        """
        action_subs_set = set(action_subs)
        if action_subs_set.intersection(set(self.hubs)):
            return "hubs"
        if action_subs_set.intersection(set(self.nodes_constrained_path)):
            return "constrained_path"
        if action_subs_set.intersection(set(self.nodes_dispatch_path)):
            return "dispatch_path"
        return "out_of_graph"

    def check_rules(self, action_type: str, localization: str, subs_topology: List[List[int]]) -> Tuple[bool, Optional[str]]:
        """
        Checks if a given grid action violates predefined expert rules.

        (Docstring identical to the original function - see previous response)

        Args:
            action_type: The type of the action (e.g., "open_line").
            localization: The location category (e.g., "constrained_path").
            subs_topology: List of topology vectors for involved substations (before action).

        Returns:
            Tuple (do_filter_action, broken_rule).
        """
        do_filter_action = False
        broken_rule = None
        if "load" not in action_type:
            is_topo_subs_one_node = all(len(set(sub_topo) - {-1, 0}) == 1 for sub_topo in subs_topology)
            rules = {
                "No action out of the overflow graph": (localization == "out_of_graph"),
                "No line reconnection on constrained path": ("line" in action_type and "close" in action_type and localization == "constrained_path"),
                "No line disconnection on dispatch path": ("line" in action_type and "open" in action_type and localization == "dispatch_path"),
                "No node merging on constrained path": ("coupling" in action_type and "close" in action_type and localization == "constrained_path"),
                "No node splitting on dispatch path": ("coupling" in action_type and "open" in action_type and localization == "dispatch_path" and is_topo_subs_one_node),
            }
            for rule_name, is_broken in rules.items():
                if is_broken:
                    do_filter_action = True
                    broken_rule = rule_name
                    break
        return do_filter_action, broken_rule

    def verify_action(self, action_desc: Dict[str, Any], subs_topology: List[List[int]]) -> Tuple[bool, Optional[str]]:
        """
        Verifies if a single grid action should be filtered based on type, location, and rules.

        Internal method used by `categorize_actions`. Uses instance attributes for context.

        Args:
            action_desc: Dictionary containing the action's description and content.
            subs_topology: List of topology vectors before the action.

        Returns:
            Tuple (do_filter_action, broken_rule).
        """
        action_type = identify_action_type(action_desc, self.by_description, self.action_space)

        do_filter_action = False
        broken_rule = None

        if "line" in action_type:
            grid2op_actions_set_bus = action_desc.get("content", {}).get("set_bus", {})
            lines = list(grid2op_actions_set_bus.get("lines_or_id", {}).keys()) + \
                    list(grid2op_actions_set_bus.get("lines_ex_id", {}).keys())
            if lines:
                line_status_map = {name: status for name, status in zip(self.obs.name_line, self.obs.line_status)}
                current_statuses = np.array([line_status_map.get(ln, True) for ln in lines])
                if "open" in action_type and np.all(~current_statuses):
                    do_filter_action, broken_rule = True, "No disconnection of a line already disconnected"
                elif "close" in action_type and np.all(current_statuses):
                    do_filter_action, broken_rule = True, "No reconnection of a line already connected"
            localization = self.localize_line_action(lines)

        elif "coupling" in action_type and "VoltageLevelId" in action_desc:
            action_subs = [action_desc["VoltageLevelId"]]
            localization = self.localize_coupling_action(action_subs)
        else:
            localization = "unknown"

        if not do_filter_action:
            do_filter_action, broken_rule_expert = self.check_rules(action_type, localization, subs_topology)
            if do_filter_action:
                broken_rule = broken_rule_expert

        return do_filter_action, broken_rule

    def categorize_actions(self,
                            dict_action: Dict[str, Dict[str, Any]],
                            # Simulation context passed here
                            timestep: int,
                            defauts: List[str],
                            overload_ids: List[int],
                            lines_reco_maintenance: List[str],
                            lines_we_care_about: Optional[List[str]] = None
                            ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Categorizes actions from a dictionary into 'filtered' and 'unfiltered' sets.

        Iterates through `dict_action`, calls `self.verify_action` for each, and
        optionally simulates filtered actions using `check_rho_reduction` to gather
        additional information.

        Args:
            dict_action: Dictionary mapping action IDs to action descriptions.
            timestep: Current simulation timestep for simulation checks.
            defauts: List of contingency line names for simulation checks.
            overload_ids: List of overloaded line indices for simulation checks.
            lines_reco_maintenance: List of maintenance lines for simulation checks.
            lines_we_care_about: Optional list of lines to monitor in simulations.

        Returns:
            Tuple (actions_to_filter, actions_unfiltered).
        """
        actions_to_filter = {}
        actions_unfiltered = {}

        for action_id, action_desc in dict_action.items():
            subs_topology = []
            if "VoltageLevelId" in action_desc:
                action_subs = [action_desc["VoltageLevelId"]]
                try:
                    subs_topology = [self.obs.sub_topology(np.where(self.obs.name_sub == sn)[0][0]) for sn in action_subs]
                except IndexError:
                    subs_topology = []

            do_filter_action, broken_rule = self.verify_action(action_desc, subs_topology)
            description = action_desc.get("description_unitaire", action_desc.get("description", ""))

            if do_filter_action:
                act_defaut = create_default_action(self.action_space, defauts)
                try:
                    action = self.action_space(action_desc["content"])
                except Exception as e:
                    print(f"Warning: Could not create action object for {action_id}: {e}")
                    # Decide how to handle this - skip simulation?
                    is_rho_reduction = None # Mark as unknown
                else:
                    state = StateInfo()
                    action = aux_prevent_asset_reconnection(self.obs, state, action)
                    act_reco_main_obj = self.action_space({"set_line_status": [(lr, 1) for lr in lines_reco_maintenance]})
                    is_rho_reduction, _ = check_rho_reduction(
                        self.obs, timestep, act_defaut, action, overload_ids,
                        act_reco_main_obj, lines_we_care_about
                    )

                if is_rho_reduction:
                    print(f"INFO: Action '{description}' was filtered by rule '{broken_rule}' but showed potential rho reduction.")

                actions_to_filter[action_id] = {
                    "description_unitaire": description,
                    "broken_rule": broken_rule,
                    "is_rho_reduction": is_rho_reduction
                }
            else:
                actions_unfiltered[action_id] = {"description_unitaire": description}

        return actions_to_filter, actions_unfiltered