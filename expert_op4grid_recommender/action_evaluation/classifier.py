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

import numpy as np


def is_nodale_grid2op_action(act):
    """
    Checks if a Grid2Op action modifies multiple elements within the same substation(s).

    This function analyzes the topology vectors within a Grid2Op action object
    (`_set_topo_vect` and `_topo_vect_to_sub`) to determine if the action
    explicitly sets the bus connection for two or more grid elements
    (lines, loads, generators) belonging to the same substation. It also checks
    if these modifications result in splitting the elements across different buses
    within that substation.

    Args:
        act (grid2op.Action): A Grid2Op action object containing internal topology
            vectors: `_set_topo_vect` (bus assignments, >=1 means explicitly set)
            and `_topo_vect_to_sub` (mapping elements to substation indices).

    Returns:
        tuple[bool, list[int], list[bool]]: A tuple containing:
            - is_nodale_action (bool): True if at least one substation has multiple
              elements being explicitly assigned to buses by the action, False otherwise.
            - concerned_subs (list[int]): A list of substation indices where multiple
              elements are explicitly assigned bus connections. Empty if `is_nodale_action`
              is False.
            - is_splitting_subs (list[bool]): A list parallel to `concerned_subs`. Each
              boolean indicates whether the corresponding substation's elements are being
              assigned to at least two *different* buses (i.e., a bus split occurs).
              Empty if `is_nodale_action` is False.
    """

    subs, counts = np.unique(act._topo_vect_to_sub[(act._set_topo_vect >= 1)], return_counts=True)
    is_nodale_action = np.any(counts >= 2)
    concerned_subs, is_splitting_subs = [], []
    if is_nodale_action:
        concerned_subs = [sub for i, sub in enumerate(subs) if counts[i] >= 2]
        for sub in concerned_subs:
            bus_for_set = np.unique(act._set_topo_vect[act._topo_vect_to_sub == sub])
            is_splitting = len(bus_for_set[bus_for_set >= 1]) >= 2
            is_splitting_subs.append(is_splitting)
    return is_nodale_action, concerned_subs, is_splitting_subs


def is_line_disconnection(grid2op_action):
    """
    Checks if a Grid2Op action represents the disconnection of at least one power line.

    This function determines if an action disconnects a line by checking if
    either the origin bus (`line_or_set_bus`), the extremity bus (`line_ex_set_bus`),
    or the line status (`line_set_status`) is explicitly set to -1 for any line.

    It also prints a warning if the action uses `line_change_status`,
    `line_or_change_bus`, or `line_ex_change_bus`, as these methods are not
    fully supported by this specific check function.

    Args:
        grid2op_action (grid2op.Action): The Grid2Op action object to inspect.

    Returns:
        bool: True if the action explicitly sets any line's origin bus,
              extremity bus, or status to -1, indicating a disconnection.
              False otherwise.
    """
    if np.any(grid2op_action.line_change_status != 0) or np.any(grid2op_action.line_or_change_bus != 0) or np.any(
            grid2op_action.line_ex_change_bus != 0):
        print("WARNING: line_change_status is not supported in this is_line_disconnection function ")
    return np.any(grid2op_action.line_or_set_bus == -1) or np.any(grid2op_action.line_ex_set_bus == -1) or np.any(
        grid2op_action.line_set_status == -1)


def is_line_reconnection(grid2op_action):
    """
    Checks if a Grid2Op action represents the reconnection of at least one power line.

    This function determines if an action reconnects a line by checking for two conditions:
    1. Both the origin bus (`line_or_set_bus`) AND the extremity bus (`line_ex_set_bus`)
       are explicitly set to bus 1 for the same line.
    2. The line status (`line_set_status`) is explicitly set to 1 (connected) for any line.

    It also prints a warning if the action uses `line_change_status`,
    `line_or_change_bus`, or `line_ex_change_bus`, as these methods are not
    fully supported by this specific check function for determining reconnection.

    Args:
        grid2op_action (grid2op.Action): The Grid2Op action object to inspect.

    Returns:
        bool: True if the action explicitly sets both buses to 1 for any line OR
              sets the status to 1 for any line, indicating a reconnection.
              False otherwise.
    """
    if np.any(grid2op_action.line_change_status != 0) or np.any(grid2op_action.line_or_change_bus != 0) or np.any(
            grid2op_action.line_ex_change_bus != 0):
        print("WARNING: line_change_status is not supported in this is_line_reconnection function")
    return np.any(grid2op_action.line_or_set_bus * grid2op_action.line_ex_set_bus == 1) or np.any(
        grid2op_action.line_set_status == 1)


def is_load_disconnection(grid2op_action):
    """
    Checks if a Grid2Op action represents the disconnection of at least one load.

    This function determines if an action disconnects a load by checking if
    the `load_set_bus` attribute is explicitly set to -1 for any load.

    It also prints a warning if the action uses `load_change_bus`, as this
    method is not supported by this specific check function.

    Args:
        grid2op_action (grid2op.Action): The Grid2Op action object to inspect.

    Returns:
        bool: True if the action explicitly sets any load's bus connection to -1,
              indicating a disconnection. False otherwise.
    """
    if np.any(grid2op_action.load_change_bus != 0):
        print("WARNING: load_change_bus is not supported in this is_load_disconnection function ")
    return np.any(grid2op_action.load_set_bus == -1)


def identify_grid2op_action_type(grid2op_action):
    """
    Identifies the primary type of a Grid2Op action based on its effects.

    This function analyzes a Grid2Op action object to classify it into one of
    several predefined categories, such as line operations, load operations, or
    substation topology changes (coupling). It relies on helper functions
    (`is_nodale_grid2op_action`, `is_line_disconnection`, etc.) to determine the
    specific components affected by the action.

    The classification logic prioritizes nodal actions first, then line disconnections
    (potentially combined with load disconnections), line reconnections, and finally
    standalone load disconnections.

    Args:
        grid2op_action (grid2op.Action): The Grid2Op action object to classify.

    Returns:
        str: A string representing the identified type of action. Possible values include:
             - "open_coupling": A nodal action that splits buses within a substation.
             - "close_coupling": A nodal action that merges buses or connects elements
                                   within a substation without splitting.
             - "open_line": Disconnects one or more power lines.
             - "open_line_load": Disconnects both power line(s) and load(s).
             - "close_line": Reconnects one or more power lines.
             - "open_load": Disconnects one or more loads (without line disconnections).
             - "unknown": If the action does not match any of the known types based on
                          the checks performed.
    """
    is_nodale, _, is_splitting = is_nodale_grid2op_action(grid2op_action)
    if is_nodale:
        return "open_coupling" if any(is_splitting) else "close_coupling"

    is_line_disco = is_line_disconnection(grid2op_action)
    is_load_disco = is_load_disconnection(grid2op_action)
    if is_line_disco:
        return "open_line_load" if is_load_disco else "open_line"
    if is_line_reconnection(grid2op_action):
        return "close_line"
    if is_load_disco:
        return "open_load"
    return "unknown"


def identify_action_type(actions_desc, by_description=True, grid2op_action_space=None):
    """
    Identifies the type of a grid action based on its description or its Grid2Op representation.

    This function classifies an action into categories like line operations, load operations,
    or substation topology changes (coupling). It can use two methods:

    1.  **By Description (`by_description=True`)**: Parses the human-readable description
        string (e.g., "Ouverture Ligne X", "Fermeture COUPL Y") and checks the action's
        content (presence of lines/loads) to infer the type. It looks for keywords like
        "Ouverture" (Open), "Fermeture" (Close), "COUPL" (Coupler), "TRO." (Section).

    2.  **By Grid2Op Object (`by_description=False`)**: Converts the action's content
        into a formal Grid2Op action object using the provided `grid2op_action_space`
        and then uses the `identify_grid2op_action_type` helper function for classification
        based on the object's attributes.

    Args:
        actions_desc (Dict[str, Any]): A dictionary containing the action's details.
            Expected keys include:
            - "content": A dictionary, often with a "set_bus" sub-dictionary detailing
              which lines, loads, etc., are affected.
            - "description_unitaire" or "description": A human-readable string describing
              the action.
        by_description (bool, optional): If True, uses the description string for
            classification. If False, uses the Grid2Op object method. Defaults to True.
        grid2op_action_space (Optional[Callable], optional): A callable (like the
            `env.action_space` object from Grid2Op) that can convert an action
            dictionary into a Grid2Op action object. Required if `by_description` is False.
            Defaults to None.

    Returns:
        str: A string representing the identified type of action. Possible values include:
             - "open_coupling": Opening a coupler or section switch.
             - "close_coupling": Closing a coupler or section switch.
             - "open_line": Disconnecting a power line.
             - "open_load": Disconnecting a load.
             - "open_line_load": Disconnecting both a line and a load.
             - "close_line": Reconnecting a power line.
             - "close_load": Reconnecting a load.
             - "close_line_load": Reconnecting both a line and a load.
             - "unknown": If the type cannot be determined by the chosen method.

    Raises:
        TypeError: If `by_description` is False and `grid2op_action_space` is not provided
                   or is not callable.
    """
    if by_description:
        dict_action = actions_desc["content"]["set_bus"]
        has_load = "loads_id" in dict_action and len(dict_action["loads_id"]) != 0
        has_line = ("lines_or_id" in dict_action and len(dict_action["lines_or_id"]) != 0) or \
                   ("lines_ex_id" in dict_action and len(dict_action["lines_ex_id"]) != 0)

        description = actions_desc.get("description_unitaire", actions_desc.get("description", ""))

        if ("COUPL" in description or "TRO." in description):
            return "open_coupling" if "Ouverture" in description else "close_coupling"
        if "Ouverture" in description or "deconnection" in description:
            if has_load and has_line: return "open_line_load"
            if has_line: return "open_line"
            return "open_load"
        if "Fermeture" in description or "reconnection" in description:
            if has_load and has_line: return "close_line_load"
            if has_line: return "close_line"
            return "close_load"
    else:
        grid2op_action = grid2op_action_space(actions_desc["content"])
        return identify_grid2op_action_type(grid2op_action)
    return "unknown"