# expert_op4grid_recommender/action_evaluation/discovery.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.
__author__ = "marota"

import numpy as np
from alphaDeesp.core.alphadeesp import AlphaDeesp_warmStart
from expert_op4grid_recommender.utils.simulation import check_rho_reduction, create_default_action
from expert_op4grid_recommender.utils.helpers import get_delta_theta_line, sort_actions_by_score, add_prioritized_actions
from .classifier import identify_action_type


def _is_sublist(small, large):
    n = len(small)
    return any(small == large[i:i + n] for i in range(len(large) - n + 1))


def _get_line_substations(obs, line_name):
    line_id = np.where(obs.name_line == line_name)[0][0]
    sub_or_name = obs.name_sub[obs.line_or_to_subid[line_id]]
    sub_ex_name = obs.name_sub[obs.line_ex_to_subid[line_id]]
    return sub_or_name, sub_ex_name


def _find_paths_for_line(line_subs, red_loop_paths):
    sub_or, sub_ex = line_subs
    return [path for path in red_loop_paths if
            _is_sublist([sub_or, sub_ex], path) or _is_sublist([sub_ex, sub_or], path)]


def _get_active_edges_between(g_overflow, node_a, node_b):
    """
    Retrieve active edges (non-dashed/dotted) between two substations.
    Returns list of active edge names.
    """
    active_edges = []

    # A → B
    in_edges = g_overflow.g.get_edge_data(node_a, node_b)
    if in_edges is not None:
        active_edges += [
            e_dict["name"]
            for e_dict in in_edges.values()
            if "style" not in e_dict or e_dict["style"] not in ["dashed", "dotted"]
        ]

    # B → A
    out_edges = g_overflow.g.get_edge_data(node_b, node_a)
    if out_edges is not None:
        active_edges += [
            e_dict["name"]
            for e_dict in out_edges.values()
            if "style" not in e_dict or e_dict["style"] not in ["dashed", "dotted"]
        ]

    return active_edges



def _has_blocking_disconnected_line(obs, found_path, line_reco, all_disconnected_lines, g_overflow):
    for line in all_disconnected_lines:
        if line == line_reco:
            continue
        sub_or, sub_ex = _get_line_substations(obs, line)
        if not (_is_sublist([sub_or, sub_ex], found_path) or _is_sublist([sub_ex, sub_or], found_path)):
            continue
        if not _get_active_edges_between(g_overflow, sub_or, sub_ex):
            return True, line
    return False, None


def check_other_reconnectable_line_on_path(obs, line_reco, all_disconnected_lines, red_loop_paths, g_overflow):
    """
    Checks if reconnecting a specific line (`line_reco`) might be blocked by another disconnected line on the same path.

    This function determines if a candidate line reconnection is potentially effective.
    It identifies network paths (usually loops from the overflow graph analysis) that
    contain the candidate line. For each such path, it checks if any *other* line
    that is currently disconnected (`all_disconnected_lines`) also lies on that path.

    A path is considered **blocked** if such another disconnected line exists *and*
    there is no other active power line running in parallel between the substations
    connected by that blocking line (checked using `_has_blocking_disconnected_line`).

    The function returns `True` as soon as it finds *at least one* path containing
    `line_reco` that is *not* blocked. If all found paths are blocked, it returns
    `False` and the name of the first blocking line encountered.

    Args:
        obs (grid2op.Observation): The Grid2Op observation object, used to map line
            names to their connected substations.
        line_reco (str): The name of the candidate line being considered for reconnection.
        all_disconnected_lines (list[str]): A list of names of all lines currently
            disconnected in the grid state being analyzed (including potentially `line_reco`).
        red_loop_paths (list[list[str]]): A list of relevant network paths, where each
            path is represented as a list of substation names. These usually correspond
            to dispatch paths (red loops) identified in the overflow graph.
        g_overflow (alphaDeesp.OverFlowGraph): The overflow graph object, used to check
            for the presence and status (active/inactive) of parallel lines between
            substations via its internal graph (`g_overflow.g`).

    Returns:
        tuple[bool, str | None]: A tuple containing:
            - has_effective_path (bool): True if at least one path containing `line_reco`
              was found that is not blocked by another disconnected line. False otherwise.
            - blocking_line (str | None): If `has_effective_path` is False and at least
              one path was found, this is the name of the first disconnected line
              identified as blocking a path. If no paths were found for `line_reco` or
              if an effective path was found, this is None.
    """

    line_subs = _get_line_substations(obs, line_reco)
    found_paths = _find_paths_for_line(line_subs, red_loop_paths)
    if not found_paths:
        return False, None

    for path in found_paths:
        is_blocked, blocker = _has_blocking_disconnected_line(obs, path, line_reco, all_disconnected_lines, g_overflow)
        if not is_blocked:
            return True, None
    return False, blocker


def verify_relevant_reconnections(obs, obs_defaut, timestep, action_space, defauts, overload_ids, act_reco_maintenance,
                                  g_overflow, lines_to_reconnect, red_loop_paths, all_disconnected_lines,
                                  check_action_simulation=True, lines_we_care_about=None):
    """
    Evaluates potential line reconnections for effectiveness in reducing grid overloads.

    This function assesses a list of candidate disconnected lines (`lines_to_reconnect`)
    to determine if reconnecting them could help alleviate existing overloads.
    The process involves several steps:

    1.  **Path Filtering**: For each candidate line, it checks if its reconnection path
        is blocked by another disconnected line using `check_other_reconnectable_line_on_path`.
        Lines on blocked paths are considered irrelevant and skipped.
    2.  **Scoring**: Relevant lines (those on unblocked paths) are scored based on the absolute
        voltage angle difference (delta-theta) across their terminals, calculated using
        `get_delta_theta_line`. A smaller delta-theta suggests a potentially less disruptive
        reconnection.
    3.  **Sorting**: The scored reconnection actions are sorted, typically from lowest to highest
        delta-theta (though `sort_actions_by_score` might implement a different order).
    4.  **Simulation Check (Optional)**: If `check_action_simulation` is True, each sorted
        action is simulated using `check_rho_reduction` to verify if it actually reduces
        the loading (`rho`) on the specified `overload_ids` by a certain tolerance.
    5.  **Categorization**: Based on the simulation results (or just the path filtering if
        simulation is skipped), the candidate lines are categorized into 'effective'
        (reduce rho) and 'ineffective' (do not reduce rho). All actions created for
        relevant lines are returned in the 'identified' dictionary.

    Args:
        obs (grid2op.Observation): The current Grid2Op observation representing the state
            *before* applying any corrective actions (used for simulation checks).
        obs_defaut (grid2op.Observation): The Grid2Op observation representing the state
            after the initial contingency but *before* reconnection (used for path checking
            and delta-theta calculation).
        timestep (int): The current simulation timestep.
        action_space (callable): The Grid2Op action space object, used to create reconnection actions.
        defauts (list[str]): List of line names representing the initial contingency (N-1),
            used to create a default action for simulation checks.
        overload_ids (list[int]): List of indices of the lines currently considered overloaded,
            used as the target for `check_rho_reduction`.
        act_reco_maintenance (grid2op.Action): An action object representing any maintenance
            reconnections that should be applied alongside the candidate action during simulation.
        g_overflow (alphaDeesp.OverFlowGraph): The overflow graph object, used by
            `check_other_reconnectable_line_on_path` to check for parallel active lines.
        lines_to_reconnect (list[str]): A list of names of the disconnected lines to evaluate
            for potential reconnection.
        red_loop_paths (list[list[str]]): A list of relevant network paths (substation names),
            used by `check_other_reconnectable_line_on_path`.
        all_disconnected_lines (list[str]): A list of names of *all* currently disconnected lines,
            used by `check_other_reconnectable_line_on_path`.
        check_action_simulation (bool, optional): If True, performs simulation checks using
            `check_rho_reduction` to categorize lines as effective/ineffective. If False,
            all lines passing the path filter are considered potentially effective, and the
            effective/ineffective lists might be empty or based solely on path filtering.
            Defaults to True.
        lines_we_care_about (list[str], optional): Specific lines to monitor during the
            `check_rho_reduction` simulation. Defaults to None.

    Returns:
        tuple[list[str], list[str], dict[str, grid2op.Action]]: A tuple containing:
            - effective (list[str]): A list of line names whose reconnection was found
              to reduce overloads (only populated if `check_action_simulation` is True).
            - ineffective (list[str]): A list of line names whose reconnection did not
              reduce overloads (only populated if `check_action_simulation` is True).
            - identified (dict[str, grid2op.Action]): A dictionary where keys are action IDs
              (e.g., "reco_LINE_NAME") and values are the corresponding Grid2Op reconnection
              action objects created for all lines that passed the initial path filtering step.
    """
    map_action_score = {}
    red_loop_paths_sorted = sorted(red_loop_paths, key=len)

    for line_reco in lines_to_reconnect:
        has_found_effective_path, other_line = check_other_reconnectable_line_on_path(obs_defaut, line_reco,
                                                                                      all_disconnected_lines,
                                                                                      red_loop_paths_sorted, g_overflow)
        if has_found_effective_path:
            line_id = np.where(obs.name_line == line_reco)[0][0]
            delta_theta = abs(get_delta_theta_line(obs_defaut, line_id))
            map_action_score["reco_" + line_reco] = {
                "action": action_space({"set_line_status": [(line_reco, 1)]}),
                "score": delta_theta,
                "line_impacted": line_reco
            }
        else:
            print(
                f"Line reconnection {line_reco} might not be relevant as another disconnected line {other_line} is on the path.")

    actions, lines_impacted, scores = sort_actions_by_score(map_action_score)
    effective, ineffective, identified = [], [], {}

    for action_id, line_reco, score in zip(actions.keys(), lines_impacted, scores):
        action = actions[action_id]
        identified[action_id] = action
        if check_action_simulation:
            is_rho_reduction, _ = check_rho_reduction(obs, timestep, create_default_action(action_space, defauts),
                                                      action, overload_ids, act_reco_maintenance, lines_we_care_about)
            if is_rho_reduction:
                print(f"Line reconnection {line_reco} reduces overloads by at least 2% line loading")
                effective.append(line_reco)
            else:
                print(f"Line reconnection {line_reco} is not effective")
                ineffective.append(line_reco)
    return effective, ineffective, identified


def find_relevant_disconnections(actions_unfiltered, dict_actions, lines_constrained_path, obs, timestep, action_space,
                                 defauts, overload_ids, act_reco_maintenance, lines_we_care_about=None):
    """
    Identifies and evaluates potential line disconnection actions for effectiveness in reducing overloads.

    This function iterates through a dictionary of potential actions (`dict_actions`),
    filters them to find actions classified as "open_line" (line disconnections),
    and further filters these based on whether the disconnected line is part of the
    critical "constrained path" identified in the overflow graph analysis.

    For the remaining relevant disconnection actions, it performs a simulation check
    using `check_rho_reduction` to determine if applying the action actually reduces
    the loading (`rho`) on the specified `overload_ids`.

    The function categorizes the action IDs based on this evaluation.

    Args:
        actions_unfiltered (set | list | dict_keys): An iterable containing the action IDs
            to be considered from `dict_actions`.
        dict_actions (dict): A dictionary mapping action IDs (str) to action description
            dictionaries. Each description dictionary should contain at least "content"
            (with "set_bus" details) and potentially a "description_unitaire".
        lines_constrained_path (list[str]): A list of line names that belong to the
            constrained path (blue path) identified in the overflow graph. Actions
            disconnecting lines *not* on this path are typically ignored.
        obs (grid2op.Observation): The current Grid2Op observation representing the state
            *before* applying any corrective actions (used for simulation checks).
        timestep (int): The current simulation timestep.
        action_space (callable): The Grid2Op action space object, used to convert action
            description content into executable Grid2Op action objects.
        defauts (list[str]): List of line names representing the initial contingency (N-1),
            used to create a default action for simulation checks.
        overload_ids (list[int]): List of indices of the lines currently considered overloaded,
            used as the target for `check_rho_reduction`.
        act_reco_maintenance (grid2op.Action): An action object representing any maintenance
            reconnections that should be applied alongside the candidate action during simulation.
        lines_we_care_about (list[str], optional): Specific lines to monitor during the
            `check_rho_reduction` simulation. Defaults to None.

    Returns:
        tuple[dict, list, list, list]: A tuple containing:
            - identified (dict[str, grid2op.Action]): A dictionary mapping the IDs of relevant
              "open_line" actions on the constrained path to their corresponding Grid2Op
              action objects.
            - effective (list[str]): A list of action IDs from `identified` whose simulation
              resulted in a reduction of overload (`rho`).
            - ineffective (list[str]): A list of action IDs from `identified` whose simulation
              did *not* result in a reduction of overload.
            - ignored (list[str]): A list of action IDs from `actions_unfiltered` that were
              skipped, either because they were not "open_line" type or because the line
              was not on the `lines_constrained_path`.
    """
    identified, effective, ineffective, ignored = {}, [], [], []
    act_defaut = create_default_action(action_space, defauts)

    for action_id in actions_unfiltered:
        action_desc = dict_actions[action_id]
        action_type = identify_action_type(action_desc, by_description=True, grid2op_action_space=action_space)

        if "open_line" in action_type:
            lines_in_action = set(list(action_desc["content"]["set_bus"].get('lines_ex_id', {}).keys()) + list(
                action_desc["content"]["set_bus"].get('lines_or_id', {}).keys()))
            if lines_in_action.intersection(set(lines_constrained_path)):
                action = action_space(action_desc["content"])
                identified[action_id] = action
                is_rho_reduction, _ = check_rho_reduction(obs, timestep, act_defaut, action, overload_ids,
                                                          act_reco_maintenance, lines_we_care_about)

                # --- Step 5: Print result for this action ---
                if is_rho_reduction:
                    print(f"{action_id} reduces overloads by at least 2% line loading")
                    effective.append(action_id)
                else:
                    print(f"{action_id} is not effective")
                    ineffective.append(action_id)
            else:
                print(f"{action_id} is not effective")
                ignored.append(action_id)
        else:
            ignored.append(action_id)
    return identified, effective, ineffective, ignored


def compute_node_splitting_action_score(action, sub_impacted_id, obs_defaut, alphaDeesp_ranker, g):
    """
    Computes a score for a node-splitting action using the AlphaDeesp ranker.

    This function evaluates how "good" a potential node-splitting action is
    without running a full power flow simulation. It first determines if the
    target substation initially has only one active bus. It then simulates the
    effect of the action on the substation's topology vector using the Grid2Op
    observation addition operator (`obs + action`). Finally, it calls the
    AlphaDeesp ranker's `rank_current_topo_at_node_x` method with the resulting
    topology to get a heuristic score.

    Args:
        action (grid2op.Action): The node-splitting action object to be scored.
        sub_impacted_id (int): The integer index of the substation affected by the action.
        obs_defaut (grid2op.Observation): The Grid2Op observation representing the grid state
            *before* the node-splitting action is applied (typically after the initial contingency).
        alphaDeesp_ranker (AlphaDeesp_warmStart): An initialized instance of the AlphaDeesp
            ranker object, used to calculate the score based on graph properties and topology.
        g (networkx.Graph): The network graph (usually the processed overflow graph with integer nodes)
            required by the `alphaDeesp_ranker`.

    Returns:
        float: The score assigned by the AlphaDeesp ranker to the resulting topology
               after applying the `action`. Higher scores generally indicate a potentially
               more beneficial split according to AlphaDeesp's heuristics.
    """
    topo_vect_init = obs_defaut.sub_topology(sub_id=sub_impacted_id)
    is_single_node = len(set(topo_vect_init) - {0, -1}) == 1

    impact_obs = obs_defaut + action
    sub_info = obs_defaut.sub_info
    start = int(np.sum(sub_info[:sub_impacted_id]))
    length = int(sub_info[sub_impacted_id])
    action_topo_vect = impact_obs.topo_vect[start:start + length] - 1

    return alphaDeesp_ranker.rank_current_topo_at_node_x(g, sub_impacted_id, isSingleNode=is_single_node,
                                                         topo_vect=action_topo_vect, is_score_specific_substation=False)


def identify_and_score_node_splitting_actions(dict_action, hubs, nodes_blue_path, obs_defaut, action_space,
                                              alphaDeesp_ranker, g):
    """
    Identifies relevant node-splitting ("open_coupling") actions and computes their heuristic scores.

    This function iterates through a dictionary of action descriptions. It performs the following steps for each action:
    1. Classifies the action type using `identify_action_type`.
    2. If the action is an "open_coupling" type (representing node splitting):
       a. Converts the description into a Grid2Op action object.
       b. Determines the substation impacted by the action.
       c. Checks if the impacted substation is considered relevant (i.e., it's listed in `hubs` or `nodes_blue_path`).
       d. If relevant, computes a heuristic score for the action using `compute_node_splitting_action_score`
          which leverages the `alphaDeesp_ranker`.
       e. Stores the action object, its score, and the impacted substation name in a dictionary.
    3. Actions that are not "open_coupling" or impact non-relevant substations are added to an 'ignored' list.

    Args:
        dict_action (dict): A dictionary mapping action IDs (str) to action description
            dictionaries. Each description should contain "content" and potentially
            "description_unitaire".
        hubs (list[str]): A list of substation names identified as critical hubs in the network.
        nodes_blue_path (list[str]): A list of substation names belonging to the constrained
            (blue) path identified in the overflow graph analysis.
        obs_defaut (grid2op.Observation): The Grid2Op observation representing the grid state
            *before* the node-splitting action would be applied (used for context like substation names
            and initial topology).
        action_space (callable): The Grid2Op action space object, used to convert action
            description content into executable Grid2Op action objects.
        alphaDeesp_ranker (AlphaDeesp_warmStart): An initialized instance of the AlphaDeesp
            ranker object, used to calculate the heuristic score.
        g (networkx.Graph): The network graph (usually the processed overflow graph with integer nodes)
            required by the `alphaDeesp_ranker`.

    Returns:
        tuple[dict, list]: A tuple containing:
            - map_action_score (dict): A dictionary where keys are the action IDs of the relevant
              node-splitting actions. Values are dictionaries containing the Grid2Op `action` object,
              the computed `score` (float), and the `sub_impacted` substation name (str).
            - ignored_actions (list): A list containing the action description dictionaries
              for actions that were skipped (either not "open_coupling" type or impacting
              a non-relevant substation).
    """
    map_action_score, ignored_actions = {}, []
    for action_id, action_desc in dict_action.items():
        if "open_coupling" in identify_action_type(action_desc, by_description=True,grid2op_action_space=action_space):
            action = action_space(action_desc["content"])
            _, subs_impacted_bool = action.get_topological_impact()
            sub_impacted_id = np.where(subs_impacted_bool)[0][0]
            sub_impacted_name = obs_defaut.name_sub[sub_impacted_id]

            if sub_impacted_name in hubs + nodes_blue_path:
                score = compute_node_splitting_action_score(action, sub_impacted_id, obs_defaut, alphaDeesp_ranker, g)
                map_action_score[action_id] = {"action": action, "score": score, "sub_impacted": sub_impacted_name}
            else:
                ignored_actions.append(action_desc)
    return map_action_score, ignored_actions


# --- ALSO UPDATE THE find_relevant_node_splitting function call logic ---
def find_relevant_node_splitting(actions_unfiltered, dict_action, hubs, nodes_blue_path, g, g_distribution_graph,
                                 simulator_data, obs_t0, obs_defaut, timestep, action_space, defauts, overload_ids,
                                 act_reco_maintenance, check_action_simulation=True, lines_we_care_about=None):
    """
    Identifies, scores, sorts, and optionally simulates node-splitting actions to find effective ones.

    This function orchestrates the process of finding beneficial node-splitting ("open_coupling")
    actions among a set of candidates. It performs the following steps:

    1.  **Initialization**: Creates an AlphaDeesp ranker for heuristic scoring.
    2.  **Identification & Scoring**: Uses `identify_and_score_node_splitting_actions` to filter
        for "open_coupling" actions affecting relevant substations (hubs or blue path) and
        calculates a heuristic score for each using the AlphaDeesp ranker.
    3.  **Sorting**: Sorts the identified and scored actions based on their scores, typically
        from highest to lowest (best potential according to heuristics).
    4.  **Simulation Check (Optional)**: If `check_action_simulation` is True, it simulates each
        sorted action using `check_rho_reduction` to verify if it actually reduces grid overloads.
    5.  **Categorization**: Classifies the actions into 'effective' (reduces overload) and
        'ineffective' based on the simulation results.

    Args:
        actions_unfiltered (set | list | dict_keys): An iterable containing the action IDs
            to be considered from `dict_action` (often ignored if `dict_action` is the primary source).
        dict_action (dict): A dictionary mapping action IDs (str) to action description
            dictionaries. Used by `identify_and_score_node_splitting_actions`.
        hubs (list[str]): A list of substation names identified as critical hubs.
        nodes_blue_path (list[str]): A list of substation names belonging to the constrained
            (blue) path.
        g (networkx.Graph): The network graph (usually the processed overflow graph with integer nodes)
            required by the `alphaDeesp_ranker`.
        g_distribution_graph (Structured_Overload_Distribution_Graph): The alphaDeesp distribution
            graph object, required by the `alphaDeesp_ranker`.
        simulator_data (dict): Metadata extracted from the `Grid2opSimulation`, required by
            the `alphaDeesp_ranker`.
        obs_t0 (grid2op.Observation): The Grid2Op observation representing the state *before*
            applying any corrective actions (used for simulation checks).
        obs_defaut (grid2op.Observation): The Grid2Op observation representing the state after
            the initial contingency (used for context in scoring).
        timestep (int): The current simulation timestep.
        action_space (callable): The Grid2Op action space object, used to create action objects.
        defauts (list[str]): List of line names representing the initial contingency (N-1),
            used for simulation checks.
        overload_ids (list[int]): List of indices of the lines currently considered overloaded,
            used as the target for `check_rho_reduction`.
        act_reco_maintenance (grid2op.Action): Action object representing maintenance reconnections
            to apply during simulation checks.
        check_action_simulation (bool, optional): If True, performs simulation checks to categorize
            actions as effective/ineffective. Defaults to True.
        lines_we_care_about (list[str], optional): Specific lines to monitor during the
            `check_rho_reduction` simulation. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - identified (dict[str, grid2op.Action]): A dictionary mapping action IDs to Grid2Op
              action objects for all relevant node-splitting actions, sorted by score.
            - effective (list[grid2op.Action]): A list of Grid2Op action objects found to be
              effective in reducing overloads (populated only if `check_action_simulation` is True).
            - ineffective (list[grid2op.Action]): A list of Grid2Op action objects found to be
              ineffective (populated only if `check_action_simulation` is True).
            - ignored (list[dict]): A list of action description dictionaries for actions that
              were skipped during the identification phase.
            - scores (list[float]): A list of heuristic scores corresponding to the `identified`
              actions, in the same sorted order.
    """
    # 1. Initialize AlphaDeesp Ranker
    alphaDeesp_ranker = AlphaDeesp_warmStart(g, g_distribution_graph, simulator_data)

    # 2. Identify relevant "open_coupling" actions and score them heuristically
    # Pass substation names to the identification function
    map_action_score, ignored = identify_and_score_node_splitting_actions(
        dict_action, hubs, nodes_blue_path, obs_defaut, action_space, alphaDeesp_ranker, g
    )

    # 3. Sort the identified actions by their heuristic scores
    actions, subs_impacted, scores = sort_actions_by_score(map_action_score)

    # 4. If simulation checks are disabled, return early
    if not check_action_simulation:
        # Return identified actions, empty effective/ineffective lists, ignored actions, and scores
        return actions, [], [], ignored, scores

    # 5. Perform simulation checks for effectiveness
    effective, ineffective = [], []
    # Create the default action (contingency) for simulation context
    act_defaut = create_default_action(action_space, defauts)

    # Simulate each identified action (in sorted order)
    for action_id, sub_impacted in zip(actions.keys(), subs_impacted):
        action = actions[action_id]
        # Check if the action reduces rho on overloaded lines
        is_rho_reduction, _ = check_rho_reduction(
            obs_t0, timestep, act_defaut, action, overload_ids,
            act_reco_maintenance, lines_we_care_about
        )
        # 6. Categorize based on simulation result
        is_hub = (sub_impacted in hubs)
        if is_rho_reduction:
            effective.append(action)
            print(
                f"Node splitting at sub {sub_impacted} (hub: {is_hub}) reduces overloads by at least 2% line loading.")
            print(f"  Effective node split found: {action_id} at {sub_impacted}")
        else:
            ineffective.append(action)
            print(f"Node splitting at sub {sub_impacted} (hub: {is_hub}) is not effective.")

    return actions, effective, ineffective, ignored, scores


def find_relevant_node_merging(nodes_dispatch_path, obs, timestep, action_space, defauts, overload_ids,
                               act_reco_maintenance, check_action_simulation=True, lines_we_care_about=None):
    """
    Identifies and evaluates potential node-merging actions within specified substations.

    Node merging involves reconfiguring a substation's topology to connect elements
    currently on different buses (typically bus 2 or higher) back to the main bus (bus 1).
    This function iterates through a list of candidate substations (`nodes_dispatch_path`):

    1.  **Eligibility Check**: For each substation, it checks if there are currently at least
        two distinct connected buses (excluding disconnected elements bus -1 or unused bus 0).
        If not, merging is irrelevant for that substation.
    2.  **Action Creation**: If eligible, it creates a Grid2Op action that sets the topology
        for the substation, mapping all elements currently on bus >= 2 to bus 1.
    3.  **Simulation Check (Optional)**: If `check_action_simulation` is True, it simulates
        the created merging action using `check_rho_reduction` to verify if it reduces
        grid overloads on the specified `overload_ids`.
    4.  **Categorization**: Based on the simulation results (if performed), the created
        action object is classified as 'effective' (reduces overload) or 'ineffective'.
        All created actions (regardless of effectiveness) are stored.

    Args:
        nodes_dispatch_path (list[str]): A list of substation names where node merging actions
            should be considered. Typically, these are substations on the dispatch paths
            identified by the overflow graph analysis.
        obs (grid2op.Observation): The current Grid2Op observation representing the state
            *before* applying any corrective actions (used for eligibility checks and simulations).
        timestep (int): The current simulation timestep.
        action_space (callable): The Grid2Op action space object, used to create the node-merging
            action objects.
        defauts (list[str]): List of line names representing the initial contingency (N-1),
            used to create a default action for simulation checks.
        overload_ids (list[int]): List of indices of the lines currently considered overloaded,
            used as the target for `check_rho_reduction`.
        act_reco_maintenance (grid2op.Action): Action object representing maintenance reconnections
            to apply during simulation checks.
        check_action_simulation (bool, optional): If True, performs simulation checks to categorize
            actions as effective/ineffective. Defaults to True.
        lines_we_care_about (list[str], optional): Specific lines to monitor during the
            `check_rho_reduction` simulation. Defaults to None.

    Returns:
        tuple[dict, list, list]: A tuple containing:
            - identified (dict[str, grid2op.Action]): A dictionary mapping generated action IDs
              (e.g., "node_merging_SUBNAME") to the corresponding Grid2Op action objects for
              all eligible substations.
            - effective (list[grid2op.Action]): A list of Grid2Op action objects found to be
              effective in reducing overloads (populated only if `check_action_simulation` is True).
            - ineffective (list[grid2op.Action]): A list of Grid2Op action objects found to be
              ineffective (populated only if `check_action_simulation` is True).
    """
    identified, effective, ineffective = {}, [], []

    # Iterate through the candidate substations provided
    for sub_name in nodes_dispatch_path:
        # Find the index of the substation
        sub_id_array = np.where(sub_name == obs.name_sub)[0]
        if sub_id_array.size == 0:
            print(f"Warning: Substation '{sub_name}' not found in observation. Skipping node merge check.")
            continue
        sub_id = sub_id_array[0]

        # Get the current bus assignments for elements in this substation
        current_sub_topo = obs.sub_topology(sub_id=sub_id)

        # 1. Eligibility Check: Are there at least 2 distinct connected buses (>= 1)?
        connected_buses = set(current_sub_topo) - {-1, 0}
        if len(connected_buses) >= 2:
            # 2. Action Creation: Define the target topology (merge >=2 to bus 1)
            topo_target = [1 if bus_id >= 2 else bus_id for bus_id in current_sub_topo]

            # Create the action object using the action space
            action = action_space({"set_bus": {"substations_id": [(sub_id, topo_target)]}})
            action_id = f"node_merging_{sub_name}"
            identified[action_id] = action

            # 3. Simulation Check (Optional)
            if check_action_simulation:
                # Create the default action (contingency) for simulation context
                act_defaut = create_default_action(action_space, defauts)
                # Check if the merging action reduces overload
                is_rho_reduction, _ = check_rho_reduction(
                    obs, timestep, act_defaut, action, overload_ids,
                    act_reco_maintenance, lines_we_care_about
                )

                # 4. Categorization
                if is_rho_reduction:
                    effective.append(action)
                    print(f"Node merging at sub {sub_name} reduces overloads by at least 2% line loading")
                else:
                    ineffective.append(action)
                    print(f"Node merging at sub {sub_name} is not effective")

    return identified, effective, ineffective


def find_relevant_actions(env, non_connected_reconnectable_lines, actions_unfiltered, dict_action, hubs, g_overflow,
                          g_distribution_graph, simulator_data, obs, obs_defaut, timestep, lines_defaut,
                          lines_overloaded_ids, act_reco_maintenance, n_action_max=5, check_action_simulation=True,
                          lines_we_care_about=None, all_disconected_lines=[]):
    """
    Orchestrates the identification, evaluation, and prioritization of various corrective grid actions.

    This function serves as a high-level coordinator that calls specialized functions to find
    and evaluate different types of actions aimed at resolving grid overloads:
    1.  **Line Reconnections**: Looks for disconnected lines on dispatch paths that could be reconnected.
    2.  **Node Merging**: Looks for opportunities to merge buses in substations on dispatch paths.
    3.  **Node Splitting**: Looks for opportunities to split buses in important substations (hubs or on constrained paths).
    4.  **Line Disconnections**: Looks for lines on the constrained path whose disconnection might help.

    Each action type is evaluated (optionally via simulation) by its respective function
    (`verify_relevant_reconnections`, `find_relevant_node_merging`, etc.). The identified
    actions from each category are then added to a final prioritized list, respecting overall
    and per-type limits (`n_action_max`, `n_action_max_per_type`).

    Args:
        env (grid2op.Environment): The Grid2Op environment instance.
        non_connected_reconnectable_lines (list[str]): List of names of lines that are currently
            disconnected but are allowed to be reconnected.
        actions_unfiltered (set | list | dict_keys): An iterable containing the action IDs
            from `dict_action` that should be considered initially.
        dict_action (dict): A dictionary mapping action IDs (str) to action description dictionaries.
        hubs (list[str]): List of substation names identified as critical hubs.
            (Note: Assumes names, ensure consistency with graph generation).
        g_overflow (OverFlowGraph): The processed `OverFlowGraph` object, expected to have
            integer node IDs suitable for AlphaDeesp.
        g_distribution_graph (Structured_Overload_Distribution_Graph): The processed distribution graph
            object derived from `g_overflow`, also with integer node IDs.
        simulator_data (dict): Metadata extracted from `Grid2opSimulation`, required by the
            AlphaDeesp ranker used in node splitting.
        obs (grid2op.Observation): The observation object representing the grid state *before*
            any corrective action (used for simulation checks).
        obs_defaut (grid2op.Observation): The observation object representing the grid state *after*
            the initial contingency (used for context like calculating delta-theta).
        timestep (int): The current simulation timestep index.
        lines_defaut (list[str]): List of line names representing the initial contingency (N-1).
        lines_overloaded_ids (list[int]): List of integer indices of the lines currently
            considered overloaded.
        act_reco_maintenance (grid2op.Action): Action object representing maintenance reconnections
            to apply during simulation checks.
        n_action_max (int, optional): The maximum total number of prioritized actions to return.
            Defaults to 5.
        check_action_simulation (bool, optional): If True, enables simulation checks within
            the specialized functions to verify action effectiveness. Defaults to True.
        lines_we_care_about (list[str], optional): Specific lines to monitor during simulation checks.
            Defaults to None.
        all_disconected_lines (list[str], optional): A comprehensive list of all line names
            currently disconnected (including non-reconnectable ones). Used for path checking.
            Defaults to an empty list.

    Returns:
        dict[str, grid2op.Action]: A dictionary mapping action IDs to Grid2Op action objects.
            This dictionary contains the prioritized actions, limited by `n_action_max` and
            per-type constraints within `add_prioritized_actions`.
    """

    prioritized_actions = {}

    # 1. Line Reconnections
    print("\n--- Verifying relevant line reconnections ---")
    lines_dispatch, _ = g_distribution_graph.get_dispatch_edges_nodes(only_loop_paths=False)
    # Since the graph nodes are now indices, lines_dispatch contains indices. We need names.
    # However, check_other_reconnectable_line_on_path uses obs to map names to subs, so it expects names.
    # Let's get the names from the obs object.
    lines_dispatch_names = lines_dispatch#[obs.name_line[line_idx] for line_idx in lines_dispatch]
    interesting_lines_to_reconnect = set(lines_dispatch_names).intersection(set(non_connected_reconnectable_lines))

    red_loops_df = g_distribution_graph.red_loops
    indices_unique = list(red_loops_df["Path"].astype(str).drop_duplicates().index)
    red_loop_paths_indices = red_loops_df["Path"].iloc[indices_unique]
    # Convert red loop paths from indices to names
    red_loop_paths_names = [[obs.name_sub[node_idx] for node_idx in path] for path in red_loop_paths_indices]

    _, _, identified_recos = verify_relevant_reconnections(obs, obs_defaut, timestep, env.action_space, lines_defaut,
                                                           lines_overloaded_ids, act_reco_maintenance, g_overflow,
                                                           interesting_lines_to_reconnect, red_loop_paths_names,
                                                           all_disconected_lines, check_action_simulation,
                                                           lines_we_care_about)
    prioritized_actions = add_prioritized_actions(prioritized_actions, identified_recos, n_action_max,
                                                  n_action_max_per_type=2)

    # 2. Node Merging
    print("\n--- Verifying relevant node merging ---")
    # FIX IS HERE: Convert node indices from the processed graph back to substation names
    _, nodes_dispatch_path_indices = g_distribution_graph.get_dispatch_edges_nodes(only_loop_paths=True)
    nodes_dispatch_path_names = [obs.name_sub[i] for i in nodes_dispatch_path_indices]

    identified_merges, _, _ = find_relevant_node_merging(
        nodes_dispatch_path_names,  # Pass the list of names, not indices
        obs, timestep, env.action_space, lines_defaut, lines_overloaded_ids, act_reco_maintenance,
        check_action_simulation, lines_we_care_about
    )
    prioritized_actions = add_prioritized_actions(prioritized_actions, identified_merges, n_action_max)

    # 3. Node Splitting
    print("\n--- Verifying relevant node splitting ---")
    # This function is designed to work with indices, so it is correct.
    _, nodes_constrained_path_indices, _, other_blue_nodes_indices = g_distribution_graph.get_constrained_edges_nodes()
    nodes_blue_path_indices = nodes_constrained_path_indices + other_blue_nodes_indices
    # Convert indices to names for the function
    hubs_names = hubs#[obs.name_sub[i] for i in hubs]
    nodes_blue_path_names = [obs.name_sub[i] for i in nodes_blue_path_indices]

    identified_splits, _, _, _, _ = find_relevant_node_splitting(
        actions_unfiltered, dict_action, hubs_names, nodes_blue_path_names, g_overflow.g,
        g_distribution_graph, simulator_data, obs, obs_defaut, timestep, env.action_space,
        lines_defaut, lines_overloaded_ids, act_reco_maintenance, check_action_simulation, lines_we_care_about
    )
    prioritized_actions = add_prioritized_actions(prioritized_actions, identified_splits, n_action_max,
                                                  n_action_max_per_type=3)

    # 4. Line Disconnections
    print("\n--- Verifying relevant line disconnections ---")
    # This function also expects names, so we convert indices back to names
    lines_constrained_path_names, _, _, _ = g_distribution_graph.get_constrained_edges_nodes()
    #lines_constrained_path_names = [obs.name_line[i] for i in lines_constrained_path_indices]

    identified_discos, _, _, _ = find_relevant_disconnections(
        actions_unfiltered, dict_action, lines_constrained_path_names, obs, timestep,
        env.action_space, lines_defaut, lines_overloaded_ids, act_reco_maintenance, lines_we_care_about
    )
    prioritized_actions = add_prioritized_actions(prioritized_actions, identified_discos, n_action_max)

    return prioritized_actions
