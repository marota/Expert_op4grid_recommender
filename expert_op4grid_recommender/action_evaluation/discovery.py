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
from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier
from typing import Dict, Any, List, Tuple, Optional, Callable, Set
from alphaDeesp.core.graphsAndPaths import OverFlowGraph, Structured_Overload_Distribution_Graph # For type hinting
import networkx as nx

class ActionDiscoverer:
    """
    Discovers, evaluates, and prioritizes corrective actions for grid overloads.

    This class encapsulates the logic for identifying potential actions (line reconnections,
    disconnections, node merging, node splitting), scoring them heuristically, optionally
    simulating their effectiveness, and producing a prioritized list.

    It is initialized with the grid state, environment details, graph analysis results,
    and configuration parameters. The main method `discover_and_prioritize` orchestrates
    the analysis, calling specific methods for each action type. The results for each
    type are stored as attributes.

    Attributes:
        env: The Grid2Op environment instance.
        obs (Any): Observation before corrective action (for simulation).
        obs_defaut (Any): Observation after initial contingency (for context).
        action_space (Callable): Grid2Op action space function.
        timestep (int): Current simulation timestep.
        lines_defaut (List[str]): Lines defining the initial contingency.
        lines_overloaded_ids (List[int]): Indices of overloaded lines.
        act_reco_maintenance (Any): Action for maintenance reconnections.
        non_connected_reconnectable_lines (List[str]): Allowed lines for reconnection.
        all_disconnected_lines (List[str]): All currently disconnected lines.
        dict_action (Dict): Dictionary of candidate action descriptions.
        actions_unfiltered (Set[str]): Set of action IDs to consider.
        hubs (List[str]): List of hub substation names.
        g_overflow (OverFlowGraph): Processed overflow graph (integer nodes).
        g_distribution_graph (Structured_OverLoad_Distribution_Graph): Processed distribution graph.
        simulator_data (Dict): Data required by AlphaDeesp ranker.
        check_action_simulation (bool): Flag to enable/disable simulation checks.
        lines_we_care_about (Optional[List[str]]): Specific lines to monitor.

        # Results attributes (populated after running discover_and_prioritize)
        identified_reconnections (Dict): Identified reconnection actions.
        effective_reconnections (List): Effective reconnection line names.
        ineffective_reconnections (List): Ineffective reconnection line names.
        identified_merges (Dict): Identified node merging actions.
        effective_merges (List): Effective node merging actions.
        ineffective_merges (List): Ineffective node merging actions.
        identified_splits (Dict): Identified node splitting actions and scores.
        effective_splits (List): Effective node splitting actions.
        ineffective_splits (List): Ineffective node splitting actions.
        ignored_splits (List): Ignored node splitting action descriptions.
        scores_splits (List): Scores for identified node splitting actions.
        identified_disconnections (Dict): Identified disconnection actions.
        effective_disconnections (List): Effective disconnection action IDs.
        ineffective_disconnections (List): Ineffective disconnection action IDs.
        ignored_disconnections (List): Ignored disconnection action IDs.
        prioritized_actions (Dict): Final dictionary of prioritized actions.
    """

    def __init__(self,
                 env: Any,
                 obs: Any,
                 obs_defaut: Any,
                 timestep: int,
                 lines_defaut: List[str],
                 lines_overloaded_ids: List[int],
                 act_reco_maintenance: Any,
                 classifier: ActionClassifier,  # Accept classifier instance
                 non_connected_reconnectable_lines: List[str],
                 all_disconnected_lines: List[str],
                 dict_action: Dict,
                 actions_unfiltered: Set[str],
                 hubs: List[str],
                 g_overflow: OverFlowGraph,
                 g_distribution_graph: Structured_Overload_Distribution_Graph,
                 simulator_data: Dict,
                 check_action_simulation: bool = True,
                 lines_we_care_about: Optional[List[str]] = None):
        """
        Initializes the ActionDiscoverer with the necessary context and parameters.

        Args:
            env: The Grid2Op environment instance.
            obs: Observation before corrective action.
            obs_defaut: Observation after initial contingency.
            timestep: Current simulation timestep.
            lines_defaut: Lines defining the initial contingency.
            lines_overloaded_ids: Indices of overloaded lines.
            act_reco_maintenance: Action for maintenance reconnections.
            classifier: An initialized ActionClassifier instance. # Added classifier
            non_connected_reconnectable_lines: Allowed lines for reconnection.
            all_disconnected_lines: All currently disconnected lines.
            dict_action: Dictionary of candidate action descriptions.
            actions_unfiltered: Set of action IDs to consider initially.
            hubs: List of hub substation names.
            g_overflow: Processed overflow graph (integer nodes).
            g_distribution_graph: Processed distribution graph.
            simulator_data: Data required by AlphaDeesp ranker.
            check_action_simulation: Flag to enable/disable simulation checks.
            lines_we_care_about: Specific lines to monitor.
        """
        self.env = env
        self.obs = obs
        self.obs_defaut = obs_defaut
        self.action_space = env.action_space
        self.timestep = timestep
        self.lines_defaut = lines_defaut
        self.lines_overloaded_ids = lines_overloaded_ids
        self.act_reco_maintenance = act_reco_maintenance
        self.classifier = classifier  # Store the classifier instance
        self.non_connected_reconnectable_lines = non_connected_reconnectable_lines
        self.all_disconnected_lines = all_disconnected_lines
        self.dict_action = dict_action
        self.actions_unfiltered = actions_unfiltered
        self.hubs = hubs
        self.g_overflow = g_overflow
        self.g_distribution_graph = g_distribution_graph
        self.simulator_data = simulator_data
        self.check_action_simulation = check_action_simulation
        self.lines_we_care_about = lines_we_care_about

        # Initialize results holders (remain the same)
        self.identified_reconnections = {}
        self.effective_reconnections = []
        self.ineffective_reconnections = []
        self.identified_merges = {}
        self.effective_merges = []
        self.ineffective_merges = []
        self.identified_splits = {}
        self.effective_splits = []
        self.ineffective_splits = []
        self.ignored_splits = []
        self.scores_splits = []
        self.identified_disconnections = {}
        self.effective_disconnections = []
        self.ineffective_disconnections = []
        self.ignored_disconnections = []
        self.prioritized_actions = {}

    # --- Helper Methods (Internal logic, kept private) ---

    def _is_sublist(self, small: List, large: List) -> bool:
        """Checks if 'small' is a contiguous sublist of 'large'."""
        n = len(small)
        return any(small == large[i:i + n] for i in range(len(large) - n + 1))

    def _get_line_substations(self, line_name: str) -> Tuple[str, str]:
        """Gets substation names for a given line name."""
        line_id_array = np.where(self.obs.name_line == line_name)[0]
        if line_id_array.size == 0:
            raise ValueError(f"Line name '{line_name}' not found in observation.")
        line_id = line_id_array[0]
        sub_or_name = self.obs.name_sub[self.obs.line_or_to_subid[line_id]]
        sub_ex_name = self.obs.name_sub[self.obs.line_ex_to_subid[line_id]]
        return sub_or_name, sub_ex_name

    def _find_paths_for_line(self, line_subs: Tuple[str, str], red_loop_paths: List[List[str]]) -> List[List[str]]:
        """Finds paths containing the given line's substations."""
        sub_or, sub_ex = line_subs
        return [path for path in red_loop_paths if
                self._is_sublist([sub_or, sub_ex], path) or self._is_sublist([sub_ex, sub_or], path)]

    def _get_active_edges_between(self, node_a: str, node_b: str) -> List[str]:
        """Retrieves active edges between two nodes (names) in the graph."""
        active_edges = []
        graph_to_check = self.g_overflow.g # Use the processed graph (integer nodes)

        try:
            # Map names back to indices for graph lookup
            node_a_idx = np.where(self.obs.name_sub == node_a)[0][0]
            node_b_idx = np.where(self.obs.name_sub == node_b)[0][0]
        except IndexError:
             print(f"Warning: Could not find index for substations {node_a} or {node_b}.")
             return [] # Cannot check edges if nodes not found

        for u, v in [(node_a_idx, node_b_idx), (node_b_idx, node_a_idx)]:
             if graph_to_check.has_edge(u, v):
                  edge_data_dict = graph_to_check.get_edge_data(u, v)
                  if edge_data_dict:
                      for e_dict in edge_data_dict.values():
                          if "style" not in e_dict or e_dict["style"] not in ["dashed", "dotted"]:
                               if "name" in e_dict:
                                    active_edges.append(e_dict["name"])
        return active_edges

    def _has_blocking_disconnected_line(self, found_path: List[str], line_reco: str) -> Tuple[bool, Optional[str]]:
        """Checks if a path is blocked by other disconnected lines."""
        for line in self.all_disconnected_lines:
            if line == line_reco: continue
            try:
                sub_or, sub_ex = self._get_line_substations(line)
            except ValueError:
                 continue # Skip if line not found
            if not (self._is_sublist([sub_or, sub_ex], found_path) or self._is_sublist([sub_ex, sub_or], found_path)):
                continue
            if not self._get_active_edges_between(sub_or, sub_ex): # Pass names
                return True, line
        return False, None

    def _check_other_reconnectable_line_on_path(self, line_reco: str, red_loop_paths: List[List[str]]) -> Tuple[bool, Optional[str]]:
        """Checks path blockage for a specific reconnection candidate."""
        try:
            line_subs = self._get_line_substations(line_reco)
        except ValueError:
            return False, None # Line not found

        found_paths = self._find_paths_for_line(line_subs, red_loop_paths)
        if not found_paths:
            return False, None

        blocker = None
        for path in found_paths:
            is_blocked, current_blocker = self._has_blocking_disconnected_line(path, line_reco)
            if not is_blocked:
                return True, None
            if blocker is None:
                 blocker = current_blocker
        return False, blocker

    # --- Action Discovery Methods (Public) ---

    def verify_relevant_reconnections(self, lines_to_reconnect: Set[str], red_loop_paths: List[List[str]]):
        """
        Finds and evaluates relevant line reconnections based on path checks and scoring.

        Populates `self.identified_reconnections`, `self.effective_reconnections`,
        and `self.ineffective_reconnections`.

        Args:
            lines_to_reconnect (Set[str]): Set of candidate line names for reconnection.
            red_loop_paths (List[List[str]]): List of relevant network paths (substation names).
        """
        map_action_score = {}
        red_loop_paths_sorted = sorted(red_loop_paths, key=len)

        print(f"Evaluating {len(lines_to_reconnect)} potential reconnections...")
        for line_reco in lines_to_reconnect:
            has_found_effective_path, other_line = self._check_other_reconnectable_line_on_path(
                line_reco, red_loop_paths_sorted
            )
            if has_found_effective_path:
                try:
                    line_id = np.where(self.obs.name_line == line_reco)[0][0]
                    #TODO: could make this stronger by considering the direction of delta theta, not only the absolute value, as it would give and indidation if flow will make it in or out
                    delta_theta = abs(get_delta_theta_line(self.obs_defaut, line_id))
                    map_action_score["reco_" + line_reco] = {
                        "action": self.action_space({"set_bus":{"lines_or_id": {line_reco: 1},"lines_ex_id": {line_reco: 1}}}),#self.action_space({"set_line_status": [(line_reco, 1)]}),
                        "score": delta_theta,
                        "line_impacted": line_reco
                    }
                except (IndexError, ValueError) as e:
                     print(f"Warning: Could not process line {line_reco}: {e}")
            else:
                print(f"  Skipping {line_reco}: Path potentially blocked by {other_line}.")

        # Sort by score (lower delta-theta is better for reconnections, so ascending sort)
        actions, lines_impacted, scores = sort_actions_by_score(map_action_score) # Note: sort_actions_by_score sorts DESCENDING! Need to adjust or reverse.
        # Let's assume sort_actions_by_score is fixed to sort descending, reverse lists if needed
        # actions_rev = dict(reversed(list(actions.items())))
        # lines_impacted_rev = list(reversed(lines_impacted))
        # scores_rev = list(reversed(scores))

        effective, ineffective, identified = [], [], actions # identified holds all passing path filter

        if self.check_action_simulation:
            print("  Simulating effectiveness...")
            act_defaut = create_default_action(self.action_space, self.lines_defaut)
            # Iterate using original descending score order from sort_actions_by_score
            for action_id, line_reco, score in zip(actions.keys(), lines_impacted, scores):
                action = actions[action_id]
                is_rho_reduction, obs_simu_action = check_rho_reduction(
                    self.obs, self.timestep, act_defaut, action, self.lines_overloaded_ids,
                    self.act_reco_maintenance, self.lines_we_care_about
                )
                if is_rho_reduction:
                    print(f"    Effective: {line_reco} (Score: {score:.2f})")
                    effective.append(line_reco)
                else:
                    print(f"    Ineffective: {line_reco} (Score: {score:.2f})")
                    ineffective.append(line_reco)

        self.identified_reconnections = identified
        self.effective_reconnections = effective
        self.ineffective_reconnections = ineffective


    def find_relevant_disconnections(self, lines_constrained_path_names: List[str]):
        """
        Finds and evaluates relevant line disconnections on the constrained path.

        Populates `self.identified_disconnections`, `self.effective_disconnections`,
        `self.ineffective_disconnections`, and `self.ignored_disconnections`.

        Args:
            lines_constrained_path_names (List[str]): List of line names on the constrained path.
        """
        identified, effective, ineffective, ignored = {}, [], [], []
        act_defaut = create_default_action(self.action_space, self.lines_defaut)

        print(f"Evaluating {len(self.actions_unfiltered)} potential disconnections...")
        for action_id in sorted(list(self.actions_unfiltered)):#as order in a set is no fixed, and since the order will matter in the subset of actions selected, fix the order for full reproducibility
            action_desc = self.dict_action[action_id]
            action_type = self.classifier.identify_action_type(action_desc, by_description=True)

            if "open_line" in action_type:
                content = action_desc.get("content", {}).get("set_bus", {})
                lines_in_action = set(list(content.get('lines_ex_id', {}).keys()) +
                                      list(content.get('lines_or_id', {}).keys()))

                if lines_in_action.intersection(set(lines_constrained_path_names)):
                    action = self.action_space(action_desc["content"])
                    identified[action_id] = action
                    if self.check_action_simulation:
                        is_rho_reduction, _ = check_rho_reduction(
                            self.obs, self.timestep, act_defaut, action, self.lines_overloaded_ids,
                            self.act_reco_maintenance, self.lines_we_care_about
                        )
                        if is_rho_reduction:
                            print(f"{action_id} reduces overloads by at least 2% line loading")
                            effective.append(action_id)
                        else:
                            print(f"{action_id} is not effective")
                            ineffective.append(action_id)
                else:
                    ignored.append(action_id)
            else:
                ignored.append(action_id)

        print(f"  Found {len(effective)} effective, {len(ineffective)} ineffective disconnections.")
        self.identified_disconnections = identified
        self.effective_disconnections = effective
        self.ineffective_disconnections = ineffective
        self.ignored_disconnections = ignored


    def compute_node_splitting_action_score(self, action: Any, sub_impacted_id: int, alphaDeesp_ranker: Any) -> float:
        """
        Computes the heuristic score for a single node splitting action.

        Args:
            action: The Grid2Op node splitting action object.
            sub_impacted_id: The integer index of the affected substation.
            alphaDeesp_ranker: The initialized AlphaDeesp ranker.

        Returns:
            The heuristic score as a float.
        """
        topo_vect_init = self.obs_defaut.sub_topology(sub_id=sub_impacted_id)
        is_single_node = len(set(topo_vect_init) - {0, -1}) == 1
        impact_obs = self.obs_defaut + action
        sub_info = self.obs_defaut.sub_info
        start = int(np.sum(sub_info[:sub_impacted_id]))
        length = int(sub_info[sub_impacted_id])
        action_topo_vect = impact_obs.topo_vect[start:start + length] - 1
        return alphaDeesp_ranker.rank_current_topo_at_node_x(
            self.g_overflow.g, sub_impacted_id, isSingleNode=is_single_node,
            topo_vect=action_topo_vect, is_score_specific_substation=False
        )


    def identify_and_score_node_splitting_actions(self, hubs_names: List[str], nodes_blue_path_names: List[str], alphaDeesp_ranker: Any) -> Tuple[Dict, List]:
        """
        Identifies relevant node splitting actions and calculates their scores.

        Args:
            hubs_names: List of hub substation names.
            nodes_blue_path_names: List of substation names on the blue path.
            alphaDeesp_ranker: The initialized AlphaDeesp ranker.

        Returns:
            A tuple: (map_action_score, ignored_actions)
        """
        map_action_score, ignored_actions = {}, []
        for action_id, action_desc in self.dict_action.items():
            if action_id not in self.actions_unfiltered:
                 ignored_actions.append(action_desc)
                 continue

            action_type = self.classifier.identify_action_type(action_desc, by_description=True)

            if "open_coupling" in action_type:
                action = self.action_space(action_desc["content"])
                #_, subs_impacted_bool = action.get_topological_impact()
                #sub_impacted_id_array = np.where(subs_impacted_bool)[0]
                #if sub_impacted_id_array.size == 0:
                #     ignored_actions.append(action_desc)
                #     continue
                #sub_impacted_id = sub_impacted_id_array[0]

                #WARNING: Other approach if some issues are observed above, like returning always the first substation
                topology_impact = action.impact_on_objects()["topology"]
                subs_impacted = list(set([assignment['substation'] for assignment in topology_impact["assigned_bus"]]))
                sub_impacted_id = subs_impacted[0]
                sub_impacted_name = self.obs_defaut.name_sub[sub_impacted_id]

                if sub_impacted_name in hubs_names or sub_impacted_name in nodes_blue_path_names:
                    score = self.compute_node_splitting_action_score(action, sub_impacted_id, alphaDeesp_ranker)
                    map_action_score[action_id] = {"action": action, "score": score, "sub_impacted": sub_impacted_name}
                else:
                    ignored_actions.append(action_desc)
            else:
                 ignored_actions.append(action_desc)
        return map_action_score, ignored_actions


    def find_relevant_node_splitting(self, hubs_names: List[str], nodes_blue_path_names: List[str]):
        """
        Finds, scores, sorts, and evaluates node splitting actions.

        Populates `self.identified_splits`, `self.effective_splits`, `self.ineffective_splits`,
        `self.ignored_splits`, and `self.scores_splits`.

        Args:
            hubs_names: List of hub substation names.
            nodes_blue_path_names: List of substation names on the blue path.
        """
        alphaDeesp_ranker = AlphaDeesp_warmStart(self.g_overflow.g, self.g_distribution_graph, self.simulator_data)

        map_action_score, ignored = self.identify_and_score_node_splitting_actions(
            hubs_names, nodes_blue_path_names, alphaDeesp_ranker
        )
        actions, subs_impacted, scores = sort_actions_by_score(map_action_score) # Higher score first

        effective, ineffective = [], []
        if self.check_action_simulation and actions:
            print("  Simulating effectiveness...")
            act_defaut = create_default_action(self.action_space, self.lines_defaut)
            for action_id, sub_impacted in zip(actions.keys(), subs_impacted):
                action = actions[action_id]
                is_rho_reduction, _ = check_rho_reduction(
                    self.obs, self.timestep, act_defaut, action, self.lines_overloaded_ids,
                    self.act_reco_maintenance, self.lines_we_care_about
                )
                is_hub = (sub_impacted in hubs_names)
                if is_rho_reduction:
                    effective.append(action)
                    print(f"    Effective node split: {action_id} at {sub_impacted} (hub: {is_hub})")
                else:
                    ineffective.append(action)
                    print(f"    Ineffective node split: {action_id} at {sub_impacted} (hub: {is_hub})")

        self.identified_splits = actions
        self.effective_splits = effective
        self.ineffective_splits = ineffective
        self.ignored_splits = ignored
        self.scores_splits = scores


    def find_relevant_node_merging(self, nodes_dispatch_path_names: List[str], percentage_threshold_min_dispatch_flow=0.1):
        """
        Finds and evaluates relevant node merging actions on dispatch paths.

        Populates `self.identified_merges`, `self.effective_merges`, and `self.ineffective_merges`.

        Args:
            nodes_dispatch_path_names: List of substation names on dispatch paths.
            percentage_threshold_min_dispatch_flow: float between 0 and 1. threshold to filter out unsignificant node merging actions given not significant enough dispatch flow
        """
        identified, effective, ineffective = {}, [], []
        capacity_dict = nx.get_edge_attributes(self.g_overflow.g, "capacity")
        max_dispatch_flow=max([abs(val) for val in capacity_dict.values()])

        print(f"Evaluating node merging for {len(nodes_dispatch_path_names)} substations...")
        for sub_name in nodes_dispatch_path_names:
            sub_id_array = np.where(sub_name == self.obs.name_sub)[0]
            if sub_id_array.size == 0: continue
            sub_id = sub_id_array[0]
            current_sub_topo = self.obs.sub_topology(sub_id=sub_id)
            connected_buses = set(current_sub_topo) - {-1, 0}

            if len(connected_buses) >= 2:
                #check if significant enough, that is with some minimal redispatch flow
                sub_edges=list(self.g_overflow.g.out_edges(sub_id, keys=True))+list(self.g_overflow.g.in_edges(sub_id, keys=True))

                max_dispatch_flow_node=max([abs(capacity_dict[edge]) for edge in sub_edges])

                if max_dispatch_flow_node>=max_dispatch_flow*percentage_threshold_min_dispatch_flow:
                    topo_target = [1 if bus_id >= 2 else bus_id for bus_id in current_sub_topo]
                    action = self.action_space({"set_bus": {"substations_id": [(sub_id, topo_target)]}})
                    action_id = f"node_merging_{sub_name}"
                    identified[action_id] = action

                    if self.check_action_simulation:
                        act_defaut = create_default_action(self.action_space, self.lines_defaut)
                        is_rho_reduction, _ = check_rho_reduction(
                            self.obs, self.timestep, act_defaut, action, self.lines_overloaded_ids,
                            self.act_reco_maintenance, self.lines_we_care_about
                        )
                        (effective if is_rho_reduction else ineffective).append(action)
                        print(f"  {'Effective' if is_rho_reduction else 'Ineffective'} node merge: {action_id}")

        self.identified_merges = identified
        self.effective_merges = effective
        self.ineffective_merges = ineffective


    # --- Main Orchestration Method ---

    def discover_and_prioritize(self, n_action_max: int = 5, n_reco_max: int = 2, n_split_max: int = 3) -> Dict[str, Any]:
        """
        Runs the full discovery and prioritization pipeline for all action types.

        This method coordinates the analysis by:
        1. Extracting necessary path information (dispatch lines/nodes, red loops, constrained lines/nodes)
           from the processed distribution graph and converting indices to names.
        2. Calling the specific evaluation method for each action type:
           - `verify_relevant_reconnections`
           - `find_relevant_node_merging`
           - `find_relevant_node_splitting`
           - `find_relevant_disconnections`
        3. Adding the identified actions from each step to the `prioritized_actions` dictionary,
           respecting the overall (`n_action_max`) and per-type (`n_reco_max`, `n_split_max`) limits.

        Args:
            n_action_max (int): Max total actions in the final prioritized list. Defaults to 5.
            n_reco_max (int): Max number of reconnection actions to prioritize. Defaults to 2.
            n_split_max (int): Max number of node splitting actions to prioritize. Defaults to 3.

        Returns:
            Dict[str, Any]: The final dictionary of prioritized actions (Action ID -> Action Object).
                            The results for each category are also stored in instance attributes
                            (e.g., `self.effective_splits`).
        """
        self.prioritized_actions = {}

        # --- Extract Path Information (Convert Indices to Names) ---
        lines_dispatch, _ = self.g_distribution_graph.get_dispatch_edges_nodes(only_loop_paths=False)
        # Since the graph nodes are now indices, lines_dispatch contains indices. We need names.
        # However, check_other_reconnectable_line_on_path uses obs to map names to subs, so it expects names.
        # Let's get the names from the obs object.
        lines_dispatch_names = lines_dispatch  # [obs.name_line[line_idx] for line_idx in lines_dispatch]

        if hasattr(self.g_distribution_graph, 'red_loops') and self.g_distribution_graph.red_loops is not None and not self.g_distribution_graph.red_loops.empty:
            df = self.g_distribution_graph.red_loops
            try: # Robust check for Path column existence
                if "Path" in df.columns:
                    indices = list(df["Path"].astype(str).drop_duplicates().index)
                    paths_indices = df["Path"].iloc[indices]
                    red_loop_paths_names = [[self.obs.name_sub[idx] for idx in p if idx < len(self.obs.name_sub)] for p in paths_indices]
                else: red_loop_paths_names = []
            except Exception as e:
                print(f"Warning: Error processing red_loops DataFrame: {e}")
                red_loop_paths_names = []
        else:
            red_loop_paths_names = []

        _, nodes_dispatch_loop_indices = self.g_distribution_graph.get_dispatch_edges_nodes(only_loop_paths=True)
        nodes_dispatch_loop_names = [self.obs.name_sub[idx] for idx in nodes_dispatch_loop_indices if idx < len(self.obs.name_sub)]

        lines_constrained_names, nodes_constrained_indices, _, other_blue_nodes_indices = self.g_distribution_graph.get_constrained_edges_nodes()
        nodes_blue_path_indices = nodes_constrained_indices + other_blue_nodes_indices
        nodes_blue_path_names = [self.obs.name_sub[idx] for idx in nodes_blue_path_indices if idx < len(self.obs.name_sub)]
        hubs_names = self.hubs # Assume hubs passed during init were already names

        # --- Call Discovery Methods ---
        print("\n--- Verifying relevant line reconnections ---")
        interesting_lines_to_reconnect = sorted(list(set(lines_dispatch_names).intersection(set(self.non_connected_reconnectable_lines))))
        print(interesting_lines_to_reconnect)
        self.verify_relevant_reconnections(interesting_lines_to_reconnect, red_loop_paths_names)
        self.prioritized_actions = add_prioritized_actions(
            self.prioritized_actions, self.identified_reconnections, n_action_max, n_action_max_per_type=n_reco_max
        )

        print("\n--- Verifying relevant node merging ---")
        self.find_relevant_node_merging(nodes_dispatch_loop_names)
        self.prioritized_actions = add_prioritized_actions(
            self.prioritized_actions, self.identified_merges, n_action_max
        )

        print("\n--- Verifying relevant node splitting ---")
        self.find_relevant_node_splitting(hubs_names, nodes_blue_path_names)
        self.prioritized_actions = add_prioritized_actions(
            self.prioritized_actions, self.identified_splits, n_action_max, n_action_max_per_type=n_split_max
        )

        print("\n--- Verifying relevant line disconnections ---")
        self.find_relevant_disconnections(lines_constrained_names)
        self.prioritized_actions = add_prioritized_actions(
            self.prioritized_actions, self.identified_disconnections, n_action_max
        )

        print(f"\nDiscovery complete. Total prioritized actions: {len(self.prioritized_actions)}")
        return self.prioritized_actions

