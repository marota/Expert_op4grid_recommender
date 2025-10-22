# expert_op4grid_recommender/action_evaluation/classifier.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

# expert_op4grid_recommender/action_evaluation/classifier.py
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, List


class ActionClassifier:
    """
    Classifies Grid2Op actions based on their type and effects.

    Provides methods to determine if an action involves line disconnections/reconnections,
    load disconnections, or nodal topology changes (coupling splits/merges).
    It can classify actions based on either their human-readable description or by
    analyzing a Grid2Op Action object directly.
    """

    def __init__(self, grid2op_action_space: Optional[Callable] = None):
        """
        Initializes the ActionClassifier.

        Args:
            grid2op_action_space (Optional[Callable]): A callable (like env.action_space)
                needed if classifying actions directly from Grid2Op objects
                (when using `identify_action_type` with `by_description=False`).
        """
        self._action_space = grid2op_action_space

    def _is_nodale_grid2op_action(self, act: Any) -> Tuple[bool, List[int], List[bool]]:
        """
        Checks if a Grid2Op action modifies multiple elements within the same substation(s).
        (Internal helper method)

        Args:
            act: A Grid2Op action object.

        Returns:
            Tuple: (is_nodale, concerned_subs, is_splitting_subs)
        """
        # Ensure input has the expected attributes
        if not all(hasattr(act, attr) for attr in ['_topo_vect_to_sub', '_set_topo_vect']):
            print("Warning: Action object missing expected topology attributes for nodal check.")
            return False, [], []

        try:
            elements_set_mask = (act._set_topo_vect >= 1)
            subs_with_set_elements = act._topo_vect_to_sub[elements_set_mask]
            subs, counts = np.unique(subs_with_set_elements, return_counts=True)
        except IndexError:  # Handle case where masks might be empty or indices invalid
            return False, [], []

        is_nodale_action = np.any(counts >= 2)
        concerned_subs, is_splitting_subs = [], []

        if is_nodale_action:
            concerned_subs_indices = np.where(counts >= 2)[0]
            concerned_subs = list(subs[concerned_subs_indices])
            for sub_index in concerned_subs:
                try:
                    sub_elements_mask = (act._topo_vect_to_sub == sub_index)
                    bus_targets_for_sub = act._set_topo_vect[sub_elements_mask & elements_set_mask]
                    unique_bus_targets = np.unique(bus_targets_for_sub[bus_targets_for_sub >= 1])
                    is_splitting = len(unique_bus_targets) >= 2
                    is_splitting_subs.append(is_splitting)
                except IndexError:
                    # If slicing fails, assume not splitting for this sub
                    is_splitting_subs.append(False)

        return is_nodale_action, concerned_subs, is_splitting_subs

    def _is_line_disconnection(self, grid2op_action: Any) -> bool:
        """
        Checks if a Grid2Op action disconnects at least one power line.
        (Internal helper method)

        Args:
            grid2op_action: The Grid2Op action object.

        Returns:
            True if a line disconnection is detected, False otherwise.
        """
        # Check if expected attributes exist
        if not all(hasattr(grid2op_action, attr) for attr in [
            'line_change_status', 'line_or_change_bus', 'line_ex_change_bus',
            'line_or_set_bus', 'line_ex_set_bus', 'line_set_status']):
            print("Warning: Action object missing expected line attributes for disconnection check.")
            return False

        if np.any(grid2op_action.line_change_status != 0) or \
                np.any(grid2op_action.line_or_change_bus != 0) or \
                np.any(grid2op_action.line_ex_change_bus != 0):
            print("WARNING: Using change_* attributes is not fully supported in _is_line_disconnection.")

        return np.any(grid2op_action.line_or_set_bus == -1) or \
            np.any(grid2op_action.line_ex_set_bus == -1) or \
            np.any(grid2op_action.line_set_status == -1)

    def _is_line_reconnection(self, grid2op_action: Any) -> bool:
        """
        Checks if a Grid2Op action reconnects at least one power line.
        (Internal helper method)

        Args:
            grid2op_action: The Grid2Op action object.

        Returns:
            True if a line reconnection is detected, False otherwise.
        """
        if not all(hasattr(grid2op_action, attr) for attr in [
            'line_change_status', 'line_or_change_bus', 'line_ex_change_bus',
            'line_or_set_bus', 'line_ex_set_bus', 'line_set_status']):
            print("Warning: Action object missing expected line attributes for reconnection check.")
            return False

        if np.any(grid2op_action.line_change_status != 0) or \
                np.any(grid2op_action.line_or_change_bus != 0) or \
                np.any(grid2op_action.line_ex_change_bus != 0):
            print("WARNING: Using change_* attributes is not fully supported in _is_line_reconnection.")

        # Check explicit set conditions
        return np.any(grid2op_action.line_or_set_bus * grid2op_action.line_ex_set_bus == 1) or \
            np.any(grid2op_action.line_set_status == 1)

    def _is_load_disconnection(self, grid2op_action: Any) -> bool:
        """
        Checks if a Grid2Op action disconnects at least one load.
        (Internal helper method)

        Args:
            grid2op_action: The Grid2Op action object.

        Returns:
            True if a load disconnection is detected, False otherwise.
        """
        if not all(hasattr(grid2op_action, attr) for attr in ['load_change_bus', 'load_set_bus']):
            print("Warning: Action object missing expected load attributes for disconnection check.")
            return False

        if np.any(grid2op_action.load_change_bus != 0):
            print("WARNING: load_change_bus is not supported in _is_load_disconnection.")

        return np.any(grid2op_action.load_set_bus == -1)

    def identify_grid2op_action_type(self, grid2op_action: Any) -> str:
        """
        Identifies the primary type of a Grid2Op action object based on its effects.

        (Docstring similar to the original function - see previous responses)

        Args:
            grid2op_action: The Grid2Op action object to classify.

        Returns:
            A string representing the action type (e.g., "open_line", "close_coupling").
        """
        is_nodale, _, is_splitting = self._is_nodale_grid2op_action(grid2op_action)
        if is_nodale:
            return "open_coupling" if any(is_splitting) else "close_coupling"

        is_line_disco = self._is_line_disconnection(grid2op_action)
        is_load_disco = self._is_load_disconnection(grid2op_action)
        if is_line_disco:
            return "open_line_load" if is_load_disco else "open_line"
        if self._is_line_reconnection(grid2op_action):
            return "close_line"
        if is_load_disco:
            return "open_load"
        return "unknown"

    def identify_action_type(self, actions_desc: Dict[str, Any], by_description: bool = True) -> str:
        """
        Identifies the type of a grid action based on its description or Grid2Op representation.

        (Docstring similar to the original function - see previous responses)

        Args:
            actions_desc: Dictionary containing action details ("content", "description_unitaire").
            by_description: If True, uses description string; otherwise, uses Grid2Op object.

        Returns:
            A string representing the action type.

        Raises:
            TypeError: If `by_description` is False and the classifier was not initialized
                       with a valid `grid2op_action_space`.
            KeyError: If `by_description` is True and `actions_desc` lacks the expected
                      `content` or `set_bus` structure.
        """
        action_type = "unknown"

        if by_description:
            try:
                content = actions_desc.get("content", {})
                dict_action = content.get("set_bus", {})
                has_load = "loads_id" in dict_action and len(dict_action["loads_id"]) != 0
                has_line = (("lines_or_id" in dict_action and len(dict_action["lines_or_id"]) != 0) or
                            ("lines_ex_id" in dict_action and len(dict_action["lines_ex_id"]) != 0))
                description = actions_desc.get("description_unitaire", actions_desc.get("description", ""))

                if ("COUPL" in description or "TRO." in description):
                    action_type = "open_coupling" if "Ouverture" in description else "close_coupling"
                elif "Ouverture" in description or "deconnection" in description:
                    if has_load and has_line:
                        action_type = "open_line_load"
                    elif has_line:
                        action_type = "open_line"
                    elif has_load:
                        action_type = "open_load"
                elif "Fermeture" in description or "reconnection" in description:
                    if has_load and has_line:
                        action_type = "close_line_load"
                    elif has_line:
                        action_type = "close_line"
                    elif has_load:
                        action_type = "close_load"

            except KeyError as e:
                print(
                    f"Warning: Missing expected key for description-based classification: {e}. Action: {actions_desc.get('description_unitaire', 'N/A')}")
                action_type = "unknown"  # Fallback if structure is wrong
            except AttributeError as e:  # Handle cases where content['set_bus'] might not be a dict
                print(
                    f"Warning: Invalid structure in action description content: {e}. Action: {actions_desc.get('description_unitaire', 'N/A')}")
                action_type = "unknown"

        else:
            if self._action_space is None or not callable(self._action_space):
                raise TypeError(
                    "`grid2op_action_space` must be provided during initialization when by_description=False")
            try:
                grid2op_action = self._action_space(actions_desc["content"])
                action_type = self.identify_grid2op_action_type(grid2op_action)
            except Exception as e:
                print(f"Warning: Could not create or classify Grid2Op action object: {e}")
                action_type = "unknown"  # Fallback on error

        return action_type