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

    def __init__(self, grid2op_action_space: Optional[Callable] = None,
                 branch_names=None, load_names=None):
        """
        Initializes the ActionClassifier.

        Args:
            grid2op_action_space (Optional[Callable]): A callable (like env.action_space)
                needed if classifying actions directly from Grid2Op objects
                (when using `identify_action_type` with `by_description=False`).
            branch_names: Collection of branch (line/transformer) IDs in the network.
                When provided alongside a pypowsybl action dict (with a ``switches``
                field), asset names extracted from switch IDs are matched against this
                set to infer ``has_line`` without triggering lazy content computation.
            load_names: Collection of load IDs in the network, used analogously to
                ``branch_names`` for inferring ``has_load``.
        """
        self._action_space = grid2op_action_space
        self._branch_names: Optional[frozenset] = frozenset(branch_names) if branch_names is not None else None
        self._load_names: Optional[frozenset] = frozenset(load_names) if load_names is not None else None

    def _infer_has_line_load(self, actions_desc: Dict[str, Any]) -> Tuple[bool, bool]:
        """Return (has_line, has_load) without triggering lazy content computation.

        For pypowsybl-format actions (those with a ``switches`` field) and when
        ``branch_names``/``load_names`` were supplied at construction time, asset
        names are extracted from switch IDs using the naming convention
        ``"{VoltageLevel}_{AssetName} {SuffixXX}_OC"`` and matched against the
        pre-built frozensets.  This avoids lazy-loading ``content.set_bus`` for
        every action in the dictionary.

        Falls back to reading ``content.set_bus`` (the grid2op path, which may
        trigger lazy computation) when switches or name sets are unavailable.
        """
        switches = actions_desc.get("switches")
        if switches is not None:
            # If it has switches, it's definitely a network action (line or coupling).
            # For pypowsybl backend, we try matching, but if it fails, we assume it's a line/transformer action
            # since those are the most common in REPAS.
            has_line, has_load = False, False
            if self._branch_names is not None:
                for switch_id in switches:
                    # Try patterns for AssetName
                    parts = switch_id.split("_")
                    if len(parts) >= 2:
                        remainder = "_".join(parts[1:])
                        # Candidates: with and without suffix space
                        candidates = [remainder.replace("_OC", ""), remainder.replace("_OC", "").rsplit(" ", 1)[0]]
                        for cand in candidates:
                            if not has_line and self._branch_names and cand in self._branch_names:
                                has_line = True
                                break
                            if not has_load and self._load_names and cand in self._load_names:
                                has_load = True
                                break
                    if has_line and has_load: break
            
            # If extraction failed but it has switches, assume it's at least a line/transformer action
            # unless we find "charge" or "load" in the description.
            if not has_line and not has_load:
                description = actions_desc.get("description_unitaire", actions_desc.get("description", "")).lower()
                if "charge" in description or "load" in description:
                    has_load = True
                else:
                    has_line = True # Default for REPAS actions with switches
                    
            return has_line, has_load

        # Grid2Op path or fallback: use content.set_bus BUT check if it's already available
        # to avoid triggering lazy loading for nothing during categorization.
        if "content" in actions_desc:
            # We check if it's a LazyActionDict and if it's already computed
            if hasattr(actions_desc, "_content_computed") and not actions_desc._content_computed:
                # IT IS LAZY AND NOT COMPUTED. Try to guess from description instead of triggering it.
                description = actions_desc.get("description_unitaire", actions_desc.get("description", "")).lower()
                has_line = ("ligne" in description or "line" in description or "branch" in description)
                has_load = ("charge" in description or "load" in description)
                return has_line, has_load
            
            # Already computed or not lazy:
            content = actions_desc.get("content", {})
            set_bus = content.get("set_bus", {})
            has_load = "loads_id" in set_bus and len(set_bus["loads_id"]) != 0
            has_line = (("lines_or_id" in set_bus and len(set_bus["lines_or_id"]) != 0) or
                        ("lines_ex_id" in set_bus and len(set_bus["lines_ex_id"]) != 0))
            return has_line, has_load
        
        return False, False

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

    def _is_gen_disconnection(self, grid2op_action: Any) -> bool:
        """
        Checks if a Grid2Op action disconnects at least one generator.
        (Internal helper method)

        Args:
            grid2op_action: The Grid2Op action object.

        Returns:
            True if a generator disconnection is detected, False otherwise.
        """
        if not all(hasattr(grid2op_action, attr) for attr in ['gen_change_bus', 'gen_set_bus']):
            return False
        return np.any(grid2op_action.gen_set_bus == -1)

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
        if self._is_gen_disconnection(grid2op_action):
            return "open_gen"
        if hasattr(grid2op_action, "pst_tap") and grid2op_action.pst_tap:
            return "pst_tap"
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
                description = actions_desc.get("description_unitaire", actions_desc.get("description", ""))

                if "COUPL" in description or "TRO." in description:
                    action_type = "open_coupling" if "Ouverture" in description else "close_coupling"
                elif "Variation de slot" in description or "tap" in description.lower():
                    action_type = "pst_tap"
                elif "Ouverture" in description or "deconnection" in description:
                    has_line, has_load = self._infer_has_line_load(actions_desc)
                    if has_load and has_line:
                        action_type = "open_line_load"
                    elif has_line:
                        action_type = "open_line"
                    elif has_load:
                        action_type = "open_load"
                elif "Fermeture" in description or "reconnection" in description:
                    has_line, has_load = self._infer_has_line_load(actions_desc)
                    if has_load and has_line:
                        action_type = "close_line_load"
                    elif has_line:
                        action_type = "close_line"
                    elif has_load:
                        action_type = "close_load"

            except (KeyError, AttributeError) as e:
                print(
                    f"Warning: Could not classify action by description: {e}. Action: {actions_desc.get('description_unitaire', 'N/A')}")
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