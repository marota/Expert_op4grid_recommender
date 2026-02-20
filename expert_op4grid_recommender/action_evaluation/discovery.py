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
# Default imports for grid2op backend - can be overridden via constructor
from expert_op4grid_recommender.utils.simulation import check_rho_reduction as _default_check_rho_reduction
from expert_op4grid_recommender.utils.simulation import create_default_action as _default_create_default_action
from expert_op4grid_recommender.utils.helpers import get_delta_theta_line, get_theta_node, sort_actions_by_score, add_prioritized_actions
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
                 lines_we_care_about: Optional[List[str]] = None,
                 check_rho_reduction_func: Optional[Callable] = None,
                 create_default_action_func: Optional[Callable] = None):
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
            check_rho_reduction_func: Function to check if an action reduces line loading.
                                      Defaults to grid2op version if not provided.
            create_default_action_func: Function to create a default (contingency) action.
                                        Defaults to grid2op version if not provided.
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

        # Store backend-specific simulation functions (use defaults for grid2op if not provided)
        self._check_rho_reduction = check_rho_reduction_func or _default_check_rho_reduction
        self._create_default_action = create_default_action_func or _default_create_default_action

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
        self.scores_reconnections = {}
        self.scores_splits_dict = {}
        self.scores_disconnections = {}
        self.scores_merges = {}
        self.params_reconnections = {}
        self.params_splits_dict = {}
        self.params_disconnections = {}
        self.params_merges = {}
        self.prioritized_actions = {}

    # --- Helper Methods (Internal logic, kept private) ---

    def _build_lookup_caches(self):
        """Pre-computes name-to-index lookup dictionaries to avoid repeated np.where calls."""
        if not hasattr(self, '_line_name_to_id'):
            self._line_name_to_id = {name: idx for idx, name in enumerate(self.obs.name_line)}
            self._sub_name_to_id = {name: idx for idx, name in enumerate(self.obs.name_sub)}
            # Pre-compute line -> (sub_or_name, sub_ex_name) mapping
            self._line_to_subs = {}
            for name, line_id in self._line_name_to_id.items():
                sub_or_name = self.obs.name_sub[self.obs.line_or_to_subid[line_id]]
                sub_ex_name = self.obs.name_sub[self.obs.line_ex_to_subid[line_id]]
                self._line_to_subs[name] = (sub_or_name, sub_ex_name)

    def _build_active_edges_cache(self):
        """Pre-computes which node pairs have active (non-dashed/dotted) edges."""
        if hasattr(self, '_active_edges_cache'):
            return
        self._active_edges_cache = {}
        graph = self.g_overflow.g
        for u, v, key, data in graph.edges(keys=True, data=True):
            style = data.get("style", "")
            if style not in ("dashed", "dotted") and "name" in data:
                pair = (min(u, v), max(u, v))
                if pair not in self._active_edges_cache:
                    self._active_edges_cache[pair] = []
                self._active_edges_cache[pair].append(data["name"])

    def _get_active_edges_between_cached(self, sub_or_name: str, sub_ex_name: str) -> List[str]:
        """Fast lookup of active edges between two substations using pre-computed cache."""
        node_a = self._sub_name_to_id.get(sub_or_name)
        node_b = self._sub_name_to_id.get(sub_ex_name)
        if node_a is None or node_b is None:
            return []
        pair = (min(node_a, node_b), max(node_a, node_b))
        return self._active_edges_cache.get(pair, [])

    def _is_sublist(self, small: List, large: List) -> bool:
        """Checks if 'small' is a contiguous sublist of 'large'."""
        n = len(small)
        return any(small == large[i:i + n] for i in range(len(large) - n + 1))

    def _get_line_substations(self, line_name: str) -> Tuple[str, str]:
        """Gets substation names for a given line name."""
        self._build_lookup_caches()
        if line_name not in self._line_to_subs:
            raise ValueError(f"Line name '{line_name}' not found in observation.")
        return self._line_to_subs[line_name]

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

    def _build_path_consecutive_pairs(self, paths: List[List[str]]) -> List[Set[Tuple[str, str]]]:
        """Pre-computes set of consecutive substation pairs for each path for fast membership checks."""
        path_pairs = []
        for path in paths:
            pairs = set()
            for i in range(len(path) - 1):
                pairs.add((path[i], path[i + 1]))
            path_pairs.append(pairs)
        return path_pairs

    def _check_other_reconnectable_line_on_path(self, line_reco: str, red_loop_paths: List[List[str]]) -> Tuple[bool, Optional[str]]:
        """Checks path blockage for a specific reconnection candidate.

        Uses pre-computed caches when available (set by verify_relevant_reconnections):
        - _reco_pair_to_paths: reverse index from substation pairs to path indices (O(1) lookup)
        - _reco_path_blockers: pre-computed blocking disconnected lines per path (O(1) check)
        """
        # Fast path: use pre-computed caches if available
        if hasattr(self, '_reco_pair_to_paths') and hasattr(self, '_reco_path_blockers'):
            line_subs = self._line_to_subs.get(line_reco)
            if line_subs is None:
                return False, None, None

            sub_or, sub_ex = line_subs
            # O(1) lookup: which paths contain this line?
            found_path_indices = self._reco_pair_to_paths.get((sub_or, sub_ex), set()) | \
                                 self._reco_pair_to_paths.get((sub_ex, sub_or), set())

            if not found_path_indices:
                return False, None, None

            blocker = None
            for path_idx in sorted(found_path_indices):
                path = red_loop_paths[path_idx]
                # O(1) lookup: which lines block this path?
                blocking_lines = self._reco_path_blockers[path_idx]
                # The only blocker that doesn't count is the line we're reconnecting
                actual_blockers = blocking_lines - {line_reco}

                if not actual_blockers:
                    return True, path, None
                if blocker is None:
                    blocker = next(iter(actual_blockers))

            return False, None, blocker

        # Slow fallback path (used when caches are not set, e.g. in unit tests)
        try:
            line_subs = self._get_line_substations(line_reco)
        except ValueError:
            return False, None, None

        found_paths = self._find_paths_for_line(line_subs, red_loop_paths)
        if not found_paths:
            return False, None, None

        blocker = None
        for path in found_paths:
            is_blocked, current_blocker = self._has_blocking_disconnected_line(path, line_reco)
            if not is_blocked:
                return True, path, None
            if blocker is None:
                 blocker = current_blocker
        return False, None, blocker

    # --- Line Disconnection Scoring ---

    @staticmethod
    def _asymmetric_bell_score(observed_flow: float, min_flow: float, max_flow: float,
                               alpha: float = 3.0, beta: float = 1.5,
                               tail_scale: float = 2.0) -> float:
        """
        Computes an asymmetric bell-curve score for a line disconnection action.

        The score is based on the Beta(alpha, beta) kernel inside [min_flow, max_flow],
        which peaks closer to max_flow (since alpha > beta). Outside this range the score
        becomes negative via quadratic tails.

        Args:
            observed_flow: The redispatch flow of the line being evaluated.
            min_flow: Lower bound (minimum redispatch to relieve worst overload).
            max_flow: Upper bound (maximum redispatch before creating new overloads).
            alpha: Beta distribution first shape parameter (controls right skew). Default 3.0.
            beta: Beta distribution second shape parameter. Default 1.5.
            tail_scale: Multiplier for the negative quadratic tails. Default 2.0.

        Returns:
            float: Score in [-inf, 1]. Positive inside the acceptable range,
                   peaking closer to max_flow; zero at boundaries; negative outside.
        """
        if max_flow <= min_flow:
            return 0.0

        # Normalize to [0, 1] where 0=min_flow, 1=max_flow
        x = (observed_flow - min_flow) / (max_flow - min_flow)

        if 0.0 <= x <= 1.0:
            # Beta kernel: x^(alpha-1) * (1-x)^(beta-1), zero at boundaries
            score = (x ** (alpha - 1)) * ((1.0 - x) ** (beta - 1))
            # Normalize so peak value = 1
            x_peak = (alpha - 1) / (alpha + beta - 2)
            peak_val = (x_peak ** (alpha - 1)) * ((1.0 - x_peak) ** (beta - 1))
            score = score / peak_val if peak_val > 0 else 0.0
        else:
            # Negative quadratic tails
            if x < 0.0:
                score = -tail_scale * (x ** 2)
            else:
                score = -tail_scale * ((x - 1.0) ** 2)

        return score

    def _build_line_capacity_map(self) -> Dict[str, float]:
        """
        Builds a mapping from line name to its maximum absolute capacity on the overflow graph.

        Returns:
            Dict[str, float]: line_name -> max absolute capacity (redispatch flow in MW).
        """
        edge_names = nx.get_edge_attributes(self.g_overflow.g, "name")
        capacity_dict = nx.get_edge_attributes(self.g_overflow.g, "capacity")
        name_to_capacity: Dict[str, float] = {}
        for edge, name in edge_names.items():
            cap = abs(float(capacity_dict.get(edge, 0.0)))
            if name not in name_to_capacity or cap > name_to_capacity[name]:
                name_to_capacity[name] = cap
        return name_to_capacity

    def _compute_disconnection_flow_bounds(self) -> Tuple[float, float, float]:
        """
        Computes the min/max acceptable redispatch flow bounds for scoring disconnection actions.

        The bounds define the window of "useful" redispatch:
        - min_redispatch: the minimum flow needed to bring the worst overload below 100%.
          ``(max_rho_overloaded - 1) * max_overload_flow``
        - max_redispatch: the maximum flow the system can absorb without creating new overloads.
          For each line with increased loading:
          ``capacity_l * (1 - rho_before) / (rho_after - rho_before)``
          The binding constraint (minimum across all such lines) gives max_redispatch.

        Returns:
            Tuple[float, float, float]:
                - max_overload_flow: the maximum absolute redispatch flow on any edge.
                - min_redispatch: the minimum useful redispatch flow (MW).
                - max_redispatch: the maximum safe redispatch flow (MW).
        """
        name_to_capacity = self._build_line_capacity_map()
        if not name_to_capacity:
            return 0.0, 0.0, 0.0

        max_overload_flow = max(name_to_capacity.values())

        # --- min_redispatch: excess loading on worst overloaded line ---
        rho_overloaded = self.obs_defaut.rho[self.lines_overloaded_ids]
        if len(rho_overloaded) > 0:
            max_rho_overloaded = float(np.max(rho_overloaded))
            min_redispatch = (max_rho_overloaded - 1.0) * max_overload_flow
        else:
            min_redispatch = 0.0

        # --- max_redispatch: binding flow margin before any line hits 100% ---
        self._build_lookup_caches()
        max_redispatch = float('inf')
        for line_name, capacity_l in name_to_capacity.items():
            line_id = self._line_name_to_id.get(line_name)
            if line_id is None or capacity_l < 1e-6:
                continue
            rho_before = float(self.obs.rho[line_id])
            rho_after = float(self.obs_defaut.rho[line_id])
            delta_rho = rho_after - rho_before
            if delta_rho > 0.01:
                ratio = capacity_l * (1.0 - rho_before) / delta_rho
                if ratio > 0:
                    max_redispatch = min(max_redispatch, ratio)

        # Fallback: if no line provided a binding constraint, max_redispatch
        # stays at inf — this signals the "unconstrained" regime where all
        # disconnections are safe and scoring should use a linear ramp.

        return max_overload_flow, min_redispatch, max_redispatch

    def compute_disconnection_score(self, lines_in_action: set) -> float:
        """
        Computes a heuristic score for a line disconnection action based on its redispatch flow.

        Two scoring regimes exist:

        **Constrained** (``max_redispatch < inf``):
        An asymmetric bell curve between ``min_redispatch`` and ``max_redispatch``,
        peaking closer to ``max_redispatch``.  Negative outside.

        **Unconstrained** (``max_redispatch == inf``):
        No line in the overflow graph gets overloaded from the redispatch, so
        disconnections are inherently safe.  Score = 1 at ``max_overload_flow``
        (the strongest action), linearly decreasing to 0 at ``min_redispatch``,
        and negative below.

        Args:
            lines_in_action: Set of line names being disconnected by this action.

        Returns:
            float: The heuristic score for this disconnection action.
        """
        # Lazy-compute and cache the bounds and capacity map
        if not hasattr(self, '_disco_bounds'):
            self._disco_bounds = self._compute_disconnection_flow_bounds()
            self._disco_capacity_map = self._build_line_capacity_map()

        max_overload_flow, min_redispatch, max_redispatch = self._disco_bounds

        if max_overload_flow < 1e-6:
            return 0.0

        # Sum capacities of lines being disconnected (observed redispatch flow)
        observed_flow = sum(
            self._disco_capacity_map.get(line, 0.0) for line in lines_in_action
        )

        if max_redispatch == float('inf'):
            # Unconstrained regime: linear ramp from 0 (at min_redispatch)
            # to 1 (at max_overload_flow), negative below min_redispatch
            return self._unconstrained_linear_score(
                observed_flow, min_redispatch, max_overload_flow
            )
        else:
            # Constrained regime: bell curve between min and max
            return self._asymmetric_bell_score(observed_flow, min_redispatch, max_redispatch)

    @staticmethod
    def _unconstrained_linear_score(observed_flow: float, min_flow: float,
                                    max_flow: float, tail_scale: float = 2.0) -> float:
        """
        Linear score for the unconstrained disconnection regime.

        All disconnections above ``min_flow`` are useful (no upper overload risk).
        Score = 1 at ``max_flow``, linearly decreasing to 0 at ``min_flow``.
        Below ``min_flow`` a negative quadratic tail penalises insufficient flow.

        Args:
            observed_flow: The redispatch flow of the action being evaluated.
            min_flow: Minimum useful redispatch (score = 0 boundary).
            max_flow: Maximum redispatch flow on the graph (score = 1).
            tail_scale: Multiplier for the negative quadratic tail. Default 2.0.

        Returns:
            float: Score in [-inf, 1].
        """
        if max_flow <= min_flow:
            return 0.0

        x = (observed_flow - min_flow) / (max_flow - min_flow)

        if x >= 0.0:
            # Linear ramp: 0 at min_flow, 1 at max_flow, >1 beyond (capped at 1)
            return min(x, 1.0)
        else:
            # Negative quadratic tail below min_flow (same convention as bell)
            return -tail_scale * (x ** 2)

    # --- Action Discovery Methods (Public) ---

    def verify_relevant_reconnections(self, lines_to_reconnect: Set[str], red_loop_paths: List[List[str]], percentage_threshold_min_dispatch_flow=0.1):
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
        capacity_dict = nx.get_edge_attributes(self.g_overflow.g, "capacity")
        max_dispatch_flow = max(abs(val) for val in capacity_dict.values())

        # --- Pre-compute data structures for O(1) lookups ---
        self._build_lookup_caches()
        self._build_active_edges_cache()

        # Pre-compute consecutive pair sets for each path
        path_pairs_list = self._build_path_consecutive_pairs(red_loop_paths_sorted)

        # Build reverse index: substation pair -> set of path indices that contain it
        # This replaces the O(n_paths) scan per candidate with an O(1) dict lookup
        self._reco_pair_to_paths = {}
        for path_idx, pairs_set in enumerate(path_pairs_list):
            for pair in pairs_set:
                if pair not in self._reco_pair_to_paths:
                    self._reco_pair_to_paths[pair] = set()
                self._reco_pair_to_paths[pair].add(path_idx)

        # Pre-compute blocking disconnected lines per path
        # This replaces the O(n_disconnected) inner loop per candidate×path with an O(1) set lookup
        self._reco_path_blockers = {}
        for path_idx, pairs_set in enumerate(path_pairs_list):
            blockers = set()
            for disc_line in self.all_disconnected_lines:
                disc_subs = self._line_to_subs.get(disc_line)
                if disc_subs is None:
                    continue
                d_sub_or, d_sub_ex = disc_subs
                if (d_sub_or, d_sub_ex) not in pairs_set and (d_sub_ex, d_sub_or) not in pairs_set:
                    continue
                # This disconnected line is on this path — check if active edges bypass it
                if not self._get_active_edges_between_cached(d_sub_or, d_sub_ex):
                    blockers.add(disc_line)
            self._reco_path_blockers[path_idx] = blockers

        # Pre-compute max dispatch flow per node to avoid repeated edge iteration
        max_dispatch_flow_per_node = {}
        for node in self.g_overflow.g.nodes():
            sub_edges = list(self.g_overflow.g.out_edges(node, keys=True)) + list(
                self.g_overflow.g.in_edges(node, keys=True))
            if sub_edges:
                max_dispatch_flow_per_node[node] = max(abs(capacity_dict[edge]) for edge in sub_edges)
            else:
                max_dispatch_flow_per_node[node] = 0.0

        dispatch_flow_threshold = max_dispatch_flow * percentage_threshold_min_dispatch_flow

        print(f"Evaluating {len(lines_to_reconnect)} potential reconnections...")
        for line_reco in lines_to_reconnect:
            has_found_effective_path, path, other_line = self._check_other_reconnectable_line_on_path(
                line_reco, red_loop_paths_sorted
            )

            if has_found_effective_path:
                line_id = self._line_name_to_id[line_reco]
                # check dispatch flow at substation extremities are significant enough
                start_path_sub_id = self._sub_name_to_id[path[0]]
                end_path_sub_id = self._sub_name_to_id[path[-1]]
                max_dispatch_flow_line_nodes = max(
                    max_dispatch_flow_per_node.get(start_path_sub_id, 0.0),
                    max_dispatch_flow_per_node.get(end_path_sub_id, 0.0)
                )

                if max_dispatch_flow_line_nodes <= dispatch_flow_threshold:
                    print(f"  Skipping {line_reco}: Path not significant enough with max redispatch flow of only blocked by {max_dispatch_flow_line_nodes} MW.")
                else:
                    try:

                        #TODO: could make this stronger by considering the direction of delta theta, not only the absolute value, as it would give and indidation if flow will make it in or out
                        #Example on contingency P.SAOLRONCI 20240828T0100Z, reco_line 'CREYSL71G.ILE', Theta CRESEY:-0.012, Theta G.ILE: -0.10.
                        #Reconnection will create a flow from CRESEY to G.ILE whie the other direction would have brought more flow on CRESEY loop path
                        #which would have been benefitial. So this action should have been filtered out.
                        #Also most influential reconnections are the ones connecting constrained path and loop path, directly in parallel to the constrained path
                        #TODO: actually rather than looking at phases, see if both extreities of path containing the line to reconnect have some influential dispatch flows.
                        #If yes this is a very good sign that it could be influential
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
            act_defaut = self._create_default_action(self.action_space, self.lines_defaut)
            # Iterate using original descending score order from sort_actions_by_score
            for action_id, line_reco, score in zip(actions.keys(), lines_impacted, scores):
                action = actions[action_id]
                is_rho_reduction, obs_simu_action = self._check_rho_reduction(
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
        self.scores_reconnections = {action_id: map_action_score[action_id]["score"]
                                     for action_id in map_action_score}
        self.params_reconnections = {
            "percentage_threshold_min_dispatch_flow": percentage_threshold_min_dispatch_flow,
            "max_dispatch_flow": max_dispatch_flow,
        }

        # Clean up temporary caches specific to this call
        if hasattr(self, '_reco_pair_to_paths'):
            del self._reco_pair_to_paths
        if hasattr(self, '_reco_path_blockers'):
            del self._reco_path_blockers


    def find_relevant_disconnections(self, lines_constrained_path_names: List[str]):
        """
        Finds and evaluates relevant line disconnections on the constrained path.

        Each identified disconnection is scored using an asymmetric bell curve based on
        the line's redispatch flow relative to min/max acceptable bounds.

        Populates ``self.identified_disconnections``, ``self.effective_disconnections``,
        ``self.ineffective_disconnections``, ``self.ignored_disconnections``,
        and ``self.scores_disconnections``.

        Args:
            lines_constrained_path_names (List[str]): List of line names on the constrained path.
        """
        identified, effective, ineffective, ignored = {}, [], [], []
        scores_map: Dict[str, float] = {}
        act_defaut = self._create_default_action(self.action_space, self.lines_defaut)

        # Invalidate cached disconnection bounds so they are recomputed for this call
        if hasattr(self, '_disco_bounds'):
            del self._disco_bounds
            del self._disco_capacity_map

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

                    # Compute heuristic score
                    score = self.compute_disconnection_score(lines_in_action)
                    scores_map[action_id] = score

                    if self.check_action_simulation:
                        is_rho_reduction, _ = self._check_rho_reduction(
                            self.obs, self.timestep, act_defaut, action, self.lines_overloaded_ids,
                            self.act_reco_maintenance, self.lines_we_care_about
                        )
                        if is_rho_reduction:
                            print(f"{action_id} reduces overloads (score: {score:.3f})")
                            effective.append(action_id)
                        else:
                            print(f"{action_id} is not effective (score: {score:.3f})")
                            ineffective.append(action_id)
                    else:
                        print(f"  {action_id} identified (score: {score:.3f})")
                else:
                    ignored.append(action_id)
            else:
                ignored.append(action_id)

        print(f"  Found {len(effective)} effective, {len(ineffective)} ineffective disconnections.")
        self.identified_disconnections = identified
        self.effective_disconnections = effective
        self.ineffective_disconnections = ineffective
        self.ignored_disconnections = ignored
        self.scores_disconnections = scores_map

        # Capture computed bounds before cleanup
        if hasattr(self, '_disco_bounds'):
            max_overload_flow, min_redispatch, max_redispatch = self._disco_bounds
            if max_redispatch == float('inf'):
                # Unconstrained regime: linear ramp, peak at max_overload_flow
                self.params_disconnections = {
                    "regime": "unconstrained",
                    "min_redispatch": min_redispatch,
                    "max_overload_flow": max_overload_flow,
                }
            else:
                # Constrained regime: bell curve
                # Peak redispatch: x_peak = (alpha-1)/(alpha+beta-2) = 2/2.5 = 0.8
                peak_redispatch = min_redispatch + 0.8 * (max_redispatch - min_redispatch)
                self.params_disconnections = {
                    "regime": "constrained",
                    "min_redispatch": min_redispatch,
                    "max_redispatch": max_redispatch,
                    "peak_redispatch": peak_redispatch,
                }
        else:
            self.params_disconnections = {}

        # Clean up cached bounds
        if hasattr(self, '_disco_bounds'):
            del self._disco_bounds
            del self._disco_capacity_map


    def identify_bus_of_interest_in_node_splitting_(self, node_type, buses, buses_negative_inflow,
                                                    buses_negative_out_flow):
        """
        Identifies the specific bus within a split node that is most critical for relieving the constraint.

        The logic depends on whether the node is upstream (amont) or downstream (aval) of the constraint:
        - **Amont (Upstream):** We look for the bus with the most negative dispatch outflow (pushing flow away).
        - **Aval (Downstream):** We look for the bus with the most negative dispatch inflow.

        Args:
            node_type (str): The classification of the node ('amont', 'aval', or other).
            buses (list): List of bus identifiers involved in the split (e.g., [1, 2]).
            buses_negative_inflow (list or np.array): Magnitude of negative inflow for each bus.
            buses_negative_out_flow (list or np.array): Magnitude of negative outflow for each bus.

        Returns:
            int: The identifier of the bus of interest (e.g., 1 or 2).
        """
        bus_of_interest = 1
        buses_negative_inflow = np.array(buses_negative_inflow)
        buses_negative_out_flow = np.array(buses_negative_out_flow)

        # Check if there is any negative flow to analyze
        if np.sum(buses_negative_inflow) != 0 or np.sum(buses_negative_out_flow) != 0:
            if node_type == "amont":
                # a) is_Amont: At least one out_edge belongs to the constrained path.
                # Strategy: Identify the bus with the most negative dispatch outflow.
                bus_of_interest = buses[np.argmax(buses_negative_out_flow - buses_negative_inflow)]
            elif node_type == "aval":
                # b) is_Aval: At least one in_edge belongs to the constrained path.
                # Strategy: Identify the bus with the most negative dispatch inflow.
                bus_of_interest = buses[np.argmax(buses_negative_inflow - buses_negative_out_flow)]
            else:
                # Fallback for loop nodes or unclassified nodes: find max absolute difference
                id_of_interest = np.argmax(abs(buses_negative_inflow - buses_negative_out_flow))
                bus_of_interest = buses[id_of_interest]
        else:
            print("Warning: no negative flow detected")

        return bus_of_interest

    def computing_buses_values_of_interest(self, graph, node, dict_edge_names_buses):
        """
        Aggregates flow attributes for all buses within a specific node (substation).

        Calculates total positive/negative inflows and outflows for every bus configuration
        in the provided graph.

        Args:
            graph (nx.Graph): The network graph containing edge attributes (flow values).
            node (int): The ID of the node (substation) being analyzed.
            dict_edge_names_buses (dict): Mapping of line names to their assigned bus (1 or 2).

        Returns:
            tuple:
                - buses (list): Unique bus IDs (excluding 0 or -1 which represent disconnection).
                - buses_negative_inflow (list): Sum of negative inflows per bus.
                - buses_negative_out_flow (list): Sum of negative outflows per bus.
                - buses_positive_inflow (list): Sum of positive inflows per bus.
                - buses_positive_out_flow (list): Sum of positive outflows per bus.
        """
        all_edges_value_attributes = nx.get_edge_attributes(graph, "label")  # dict[edge_tuple] -> flow value
        all_edges_names = nx.get_edge_attributes(graph, "name")

        # Filter for valid buses (Grid2Op buses usually start at 1; 0/-1 are disconnected)
        buses = list(set(dict_edge_names_buses.values()) - set([0, -1]))

        # Helper function to sum flows based on direction and sign
        # Note: 'keys=True' is used because this is likely a MultiGraph
        buses_negative_inflow = [
            np.sum([abs(float(all_edges_value_attributes[edge]))
                    for edge in graph.in_edges(node, keys=True)
                    if dict_edge_names_buses[all_edges_names[edge]] == bus
                    and float(all_edges_value_attributes[edge]) < 0])
            for bus in buses
        ]

        buses_negative_out_flow = [
            np.sum([abs(float(all_edges_value_attributes[edge]))
                    for edge in graph.out_edges(node, keys=True)
                    if dict_edge_names_buses[all_edges_names[edge]] == bus
                    and float(all_edges_value_attributes[edge]) < 0])
            for bus in buses
        ]

        buses_positive_inflow = [
            np.sum([abs(float(all_edges_value_attributes[edge]))
                    for edge in graph.in_edges(node, keys=True)
                    if dict_edge_names_buses[all_edges_names[edge]] == bus
                    and float(all_edges_value_attributes[edge]) >= 0])
            for bus in buses
        ]

        buses_positive_out_flow = [
            np.sum([abs(float(all_edges_value_attributes[edge]))
                    for edge in graph.out_edges(node, keys=True)
                    if dict_edge_names_buses[all_edges_names[edge]] == bus
                    and float(all_edges_value_attributes[edge]) >= 0])
            for bus in buses
        ]

        return buses, buses_negative_inflow, buses_negative_out_flow, buses_positive_inflow, buses_positive_out_flow

    def identify_node_splitting_type(self, node, g_distribution_graph):
        """
        Classifies a node based on its position relative to the constrained path.

        Args:
            node (int): The node ID.
            g_distribution_graph (object): An object containing the constrained path and loop data.

        Returns:
            str: Node type ('amont', 'aval', 'loop', or None).
        """
        constrained_path = g_distribution_graph.get_constrained_path()
        red_loops = g_distribution_graph.get_loops()
        red_loops_nodes = set([x for loop in range(len(red_loops.Path)) for x in red_loops.Path[loop]])

        node_type = None
        if node in constrained_path.n_amont():
            node_type = "amont"  # Upstream
        elif node in constrained_path.n_aval():
            node_type = "aval"  # Downstream
        elif node in red_loops_nodes:
            node_type = "loop"  # Part of a topological loop

        return node_type

    def compute_node_splitting_action_bus_score(self, node_type, bus_of_interest, buses,
                                                buses_negative_inflow, buses_negative_out_flow,
                                                buses_positive_inflow, buses_positive_out_flow):
        """
        Calculates a heuristic score for a splitting action on a specific bus.

        The score represents how effectively the split separates "helpful" flows (negative dispatch)
        from "harmful" flows. It uses a combination of a weight factor and a repulsion factor.

        Score Formula:
            $$Score = WeightFactor \times Repulsion$$

        Where for **Amont** (Upstream):
            - $WeightFactor = \frac{NegOut - (NegIn + PosIn + PosOut)}{TotalFlow}$
            - $Repulsion = NegOut - PosOut$

        Args:
            node_type (str): 'amont', 'aval', or implied by flow direction.
            bus_of_interest (int): The bus ID being scored.
            buses (list): List of all bus IDs.
            buses_negative_inflow (list): Negative inflow values per bus.
            buses_negative_out_flow (list): Negative outflow values per bus.
            buses_positive_inflow (list): Positive inflow values per bus.
            buses_positive_out_flow (list): Positive outflow values per bus.

        Returns:
            float: The calculated score. Higher is better.
        """
        idx_bus_of_interest = [i for i, bus in enumerate(buses) if bus == bus_of_interest][0]

        # Extract flows for the specific bus
        bus_negative_inflow = buses_negative_inflow[idx_bus_of_interest]
        bus_negative_out_flow = buses_negative_out_flow[idx_bus_of_interest]
        bus_positive_inflow = buses_positive_inflow[idx_bus_of_interest]
        bus_positive_out_flow = buses_positive_out_flow[idx_bus_of_interest]

        TotalInOutDispatchFlow = bus_negative_inflow + bus_negative_out_flow + bus_positive_inflow + bus_positive_out_flow

        harmonized_node_type = node_type

        # If not explicitly Amont or Aval, infer type based on dominant negative flow direction
        if node_type not in ["amont", "aval"]:
            # TODO: interesting cases to test for
            # 1) GEN.PP6 on Full France for SAOL31RONCI contingency on case 20240828T0100Z => should not be effective
            # 2) FRON5L31LOUHA_chronic_20240828_0000_timestep_36 and subs CREYSP7, MAGNYP3, GEN.PP6, FLEYRP6
            # 3) P.SAOL31RONCI_chronic_20240828_0000_timestep_1 and sub MAGNYP6
            # 4) a case where VOUGLP6 is in the middle of loop paths but with negative dispatch flows. But which case was this ? There is P.SAOL31RONCI so should probably be BEON L31CPVAN contingency
            if bus_negative_out_flow >= bus_negative_inflow:
                harmonized_node_type = "amont"  # Treat as upstream (negative out edge is dominant)
            else:
                harmonized_node_type = "aval"

        weight_factor = 0
        repulsion = 0

        if harmonized_node_type == "amont":
            # Strategy: Separate negative outflow from all other flows.
            # Repulsion: Contrast negative outflow against positive outflow (separation of paths).
            repulsion = bus_negative_out_flow - bus_positive_out_flow
            # Weight: Ratio of the desired flow vs everything else.
            weight_factor = (bus_negative_out_flow - (
                        bus_negative_inflow + bus_positive_inflow + bus_positive_out_flow)) / TotalInOutDispatchFlow

        elif harmonized_node_type == "aval":
            # Strategy: Separate negative inflow from all other flows.
            repulsion = bus_negative_inflow - bus_positive_inflow
            weight_factor = (bus_negative_inflow - (
                        bus_negative_out_flow + bus_positive_inflow + bus_positive_out_flow)) / TotalInOutDispatchFlow

        score = weight_factor * repulsion

        # Penalize configurations where the split results in opposing indicators (both negative)
        if weight_factor < 0 and repulsion < 0:
            score = -score

        return score

    def compute_node_splitting_action_score_value(self, overflow_graph, g_distribution_graph, node: int,
                                                  dict_edge_names_buses=None):
        """
        Main orchestration function to score a specific node splitting topology.

        Steps:
        1. Identify the node type (Upstream, Downstream, Loop).
        2. Compute flow values for all buses in the proposed split.
        3. Identify the 'bus of interest' (the one carrying the relieving flow).
        4. Compute the score based on flow separation logic.

        Args:
            overflow_graph (nx.Graph): Graph representing flows and overflows.
            g_distribution_graph (object): Object containing path constraints and loops.
            node (int): The ID of the substation being split.
            topo_vect_buses (list): The topology vector representing the split (e.g., [1, 1, 2, 2]).
            dict_edge_names_buses (dict): Mapping of edge names to bus IDs.

        Returns:
            Tuple[float, Dict]: A tuple of (score, details) where details contains:
                - node_type: str ("amont", "aval", or other)
                - bus_of_interest: int (bus number used for scoring)
                - in_negative_flows, out_negative_flows, in_positive_flows, out_positive_flows: floats
        """
        # 1) Identify node type
        node_type = self.identify_node_splitting_type(node, g_distribution_graph)

        # 2) Compute buses values of interest (aggregate flows)
        buses, buses_negative_inflow, buses_negative_out_flow, buses_positive_inflow, buses_positive_out_flow = \
            self.computing_buses_values_of_interest(overflow_graph, node, dict_edge_names_buses)

        # Handle edge case: no valid buses found (all disconnected or empty)
        if not buses:
            print(f"Warning: No valid buses found for node {node}, returning score 0")
            return 0.0, {}

        # 3) Detect bus of interest
        bus_of_interest = self.identify_bus_of_interest_in_node_splitting_(
            node_type, buses, buses_negative_inflow, buses_negative_out_flow
        )

        # Handle edge case: bus_of_interest not in buses list
        # This can happen when identify_bus_of_interest returns default value 1
        # but the actual buses are different (e.g., [2] or [2, 3])
        if bus_of_interest not in buses:
            # Fall back to the first available bus
            bus_of_interest = buses[0]
            print(f"Warning: Default bus_of_interest not in buses list, using {bus_of_interest}")

        # 4) Compute score
        bus_of_interest_score = self.compute_node_splitting_action_bus_score(
            node_type, bus_of_interest, buses,
            buses_negative_inflow, buses_negative_out_flow,
            buses_positive_inflow, buses_positive_out_flow
        )

        # 5) Build per-action details for the bus of interest
        bus_idx = buses.index(bus_of_interest)
        details = {
            "node_type": node_type,
            "bus_of_interest": bus_of_interest,
            "in_negative_flows": float(buses_negative_inflow[bus_idx]),
            "out_negative_flows": float(buses_negative_out_flow[bus_idx]),
            "in_positive_flows": float(buses_positive_inflow[bus_idx]),
            "out_positive_flows": float(buses_positive_out_flow[bus_idx]),
        }

        return bus_of_interest_score, details

    def _get_action_topo_vect(self, sub_impacted_id, action):
        """
        Retrieves the topology vector for a specific substation after an action is applied.

        This handles the Grid2Op topology vector format, merging the action's changes
        with the default state.

        Args:
            sub_impacted_id (int): The ID of the impacted substation.
            action (Action): The Grid2Op action being evaluated.

        Returns:
            tuple:
                - action_topo_vect (np.array): The new topology vector for the substation.
                - is_single_node (bool): True if the substation remains a single bus (no split).
        """
        topo_vect_init = self.obs_defaut.sub_topology(sub_id=sub_impacted_id)
        is_single_node = len(set(topo_vect_init) - {0, -1}) == 1

        impact_obs = self.obs_defaut + action
        sub_info = self.obs_defaut.sub_info

        # Calculate start and end indices in the global topology vector
        start = int(np.sum(sub_info[:sub_impacted_id]))
        length = int(sub_info[sub_impacted_id])

        action_topo_vect = impact_obs.topo_vect[start:start + length]

        # Preserve disconnected status from initial state if applicable
        action_topo_vect[topo_vect_init <= 0] = topo_vect_init[topo_vect_init <= 0]

        return action_topo_vect, is_single_node

    def _get_assets_on_bus_for_sub(self, sub_id, bus, bus_assignments=None):
        """Get asset names connected to a specific bus at a substation.

        Uses the element ordering from ``get_obj_connect_to`` / ``sub_topology``
        (loads, generators, lines_or, lines_ex) to match each element to its bus.

        Args:
            sub_id: Substation index.
            bus: Target bus number (1 or 2).
            bus_assignments: Optional array of bus assignments for all elements at
                the substation (e.g. from an action's topology vector).
                If *None*, the current observation's ``sub_topology`` is used.

        Returns:
            dict with keys ``"lines"``, ``"loads"``, ``"generators"``,
            each containing a list of element name strings.
        """
        obs = self.obs_defaut
        obj = obs.get_obj_connect_to(substation_id=sub_id)

        if bus_assignments is None:
            bus_assignments = obs.sub_topology(sub_id=sub_id)

        # sub_topology order: loads, gens, lines_or, lines_ex
        load_ids = obj.get('loads_id', [])
        gen_ids = obj.get('generators_id', [])
        line_or_ids = obj.get('lines_or_id', [])
        line_ex_ids = obj.get('lines_ex_id', [])

        assets = {"lines": [], "loads": [], "generators": []}
        pos = 0

        for load_idx in load_ids:
            if pos < len(bus_assignments) and int(bus_assignments[pos]) == bus:
                if hasattr(obs, 'name_load') and load_idx < len(obs.name_load):
                    assets["loads"].append(str(obs.name_load[load_idx]))
            pos += 1

        for gen_idx in gen_ids:
            if pos < len(bus_assignments) and int(bus_assignments[pos]) == bus:
                if hasattr(obs, 'name_gen') and gen_idx < len(obs.name_gen):
                    assets["generators"].append(str(obs.name_gen[gen_idx]))
            pos += 1

        for line_idx in line_or_ids:
            if pos < len(bus_assignments) and int(bus_assignments[pos]) == bus:
                line_name = str(obs.name_line[line_idx])
                if line_name not in assets["lines"]:
                    assets["lines"].append(line_name)
            pos += 1

        for line_idx in line_ex_ids:
            if pos < len(bus_assignments) and int(bus_assignments[pos]) == bus:
                line_name = str(obs.name_line[line_idx])
                if line_name not in assets["lines"]:
                    assets["lines"].append(line_name)
            pos += 1

        return assets

    def _edge_names_buses_dict(self, obs, action_topo_vect, sub_impacted_id):
        """
        Creates a mapping between line names and their bus assignments.

        Args:
            obs (Observation): The Grid2Op observation object.
            action_topo_vect (np.array): The topology vector of the substation.
            sub_impacted_id (int): The ID of the substation.

        Returns:
            dict: Keys are line names, Values are bus IDs (e.g., {'Line_A': 1, 'Line_B': 2}).
        """
        dict_edge_names_buses = {}

        length = len(action_topo_vect)
        sub_info = obs.sub_info
        start = int(np.sum(sub_info[:sub_impacted_id]))

        for i in range(length):
            topo_vect_pos = start + i
            element = obs.topo_vect_element(topo_vect_pos)

            # Check if the element corresponds to a line (not a load or gen)
            if 'line_id' in element:
                if 'line_or_id' in element:
                    line_id = element['line_or_id']  # Origin
                else:
                    line_id = element['line_ex_id']  # Extremity

                line_name = obs.name_line[line_id]
                dict_edge_names_buses[line_name] = action_topo_vect[i]

        return dict_edge_names_buses

    def _edge_names_buses_dict_new(self, action_dict):
        """
        Creates a mapping between line names and their bus assignments.

        Args:
            obs (Observation): The Grid2Op observation object.
            action_topo_vect (np.array): The topology vector of the substation.
            sub_impacted_id (int): The ID of the substation.

        Returns:
            dict: Keys are line names, Values are bus IDs (e.g., {'Line_A': 1, 'Line_B': 2}).
        """
        dict_edge_names_buses = {}
        dict_edge_names_buses.update(action_dict["set_bus"]['lines_or_id'])
        dict_edge_names_buses.update(action_dict["set_bus"]['lines_ex_id'])

        return dict_edge_names_buses

        #length = len(action_topo_vect)
        #sub_info = obs.sub_info
        #start = int(np.sum(sub_info[:sub_impacted_id]))
#
        #for i in range(length):
        #    topo_vect_pos = start + i
        #    element = obs.topo_vect_element(topo_vect_pos)
#
        #    # Check if the element corresponds to a line (not a load or gen)
        #    if 'line_id' in element:
        #        if 'line_or_id' in element:
        #            line_id = element['line_or_id']  # Origin
        #        else:
        #            line_id = element['line_ex_id']  # Extremity
#
        #        line_name = obs.name_line[line_id]
        #        dict_edge_names_buses[line_name] = action_topo_vect[i]
#
        #return dict_edge_names_buses

    def compute_node_splitting_action_score(self, action_dict: Any, sub_impacted_id: int, alphaDeesp_ranker: Any) -> Tuple[float, Dict]:
        """
        Computes the heuristic score for a single node splitting action.

        Args:
            action_dict: The Grid2Op node splitting action dictionary.
            sub_impacted_id: The integer index of the affected substation.
            alphaDeesp_ranker: The initialized AlphaDeesp ranker.

        Returns:
            Tuple[float, Dict]: The heuristic score and per-action details dict.
        """
        # Extract bus assignments directly from action dictionary (backend-agnostic)
        # This avoids relying on topology vector operations which may not be available
        # in all backends (e.g., pypowsybl)

        action = self.action_space(action_dict)
        action_topo_vect,is_single_node=self._get_action_topo_vect(sub_impacted_id,action)
        action_topo_vect_alphadeesp=action_topo_vect-1

        #########"
        dict_edge_names_buses=self._edge_names_buses_dict(self.obs_defaut, action_topo_vect, sub_impacted_id)
        #dict_edge_names_buses=self._edge_names_buses_dict_new(action_dict)#self._edge_names_buses_dict(self.obs_defaut,action_topo_vect,sub_impacted_id)

        result = self.compute_node_splitting_action_score_value(
            self.g_overflow.g, self.g_distribution_graph,
            node=sub_impacted_id,
            dict_edge_names_buses=dict_edge_names_buses
        )

        # Handle both old (float) and new (tuple) return formats for backward compatibility
        if isinstance(result, tuple):
            score, details = result
        else:
            score, details = result, {}

        # Enrich details with the list of assets on the bus of interest
        if details and "bus_of_interest" in details:
            details["assets"] = self._get_assets_on_bus_for_sub(
                sub_impacted_id, details["bus_of_interest"],
                bus_assignments=action_topo_vect
            )

        return score, details


    def _get_subs_impacted_from_action_desc(self, action_desc: Dict) -> List[int]:
        """
        Extract impacted substation IDs from an action description.
        
        This is a backend-agnostic way to find which substations are affected
        by a topology action, without relying on grid2op's impact_on_objects().
        
        Args:
            action_desc: Action description dictionary with 'content' key
            
        Returns:
            List of substation IDs that are impacted by this action
        """
        subs_impacted = set()
        content = action_desc.get("content", {})
        set_bus = content.get("set_bus", {})
        
        # Check for substations_id (direct substation topology changes)
        if "substations_id" in set_bus:
            for sub_id, _ in set_bus["substations_id"]:
                subs_impacted.add(sub_id)
        
        # Check for lines_or_id (line origin bus changes)
        if "lines_or_id" in set_bus:
            for line_name, bus in set_bus["lines_or_id"].items():
                # Find substation for this line's origin
                line_id_array = np.where(self.obs_defaut.name_line == line_name)[0]
                if line_id_array.size > 0:
                    line_id = line_id_array[0]
                    sub_id = self.obs_defaut.line_or_to_subid[line_id]
                    subs_impacted.add(sub_id)
        
        # Check for lines_ex_id (line extremity bus changes)
        if "lines_ex_id" in set_bus:
            for line_name, bus in set_bus["lines_ex_id"].items():
                # Find substation for this line's extremity
                line_id_array = np.where(self.obs_defaut.name_line == line_name)[0]
                if line_id_array.size > 0:
                    line_id = line_id_array[0]
                    sub_id = self.obs_defaut.line_ex_to_subid[line_id]
                    subs_impacted.add(sub_id)
        
        # Check for loads_id (load bus changes)
        if "loads_id" in set_bus:
            for load_name, bus in set_bus["loads_id"].items():
                # Find substation for this load
                load_id_array = np.where(self.obs_defaut.name_load == load_name)[0]
                if load_id_array.size > 0 and hasattr(self.obs_defaut, 'load_to_subid'):
                    load_id = load_id_array[0]
                    sub_id = self.obs_defaut.load_to_subid[load_id]
                    subs_impacted.add(sub_id)
        
        # Check for generators_id (generator bus changes)
        if "generators_id" in set_bus:
            for gen_name, bus in set_bus["generators_id"].items():
                # Find substation for this generator
                gen_id_array = np.where(self.obs_defaut.name_gen == gen_name)[0]
                if gen_id_array.size > 0 and hasattr(self.obs_defaut, 'gen_to_subid'):
                    gen_id = gen_id_array[0]
                    sub_id = self.obs_defaut.gen_to_subid[gen_id]
                    subs_impacted.add(sub_id)
        
        return list(subs_impacted)

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
                
                # Get impacted substations from action description (backend-agnostic)
                #topology_impact = action.impact_on_objects()["topology"] if grid2op
                #subs_impacted = list(set([assignment['substation'] for assignment in topology_impact["assigned_bus"]]))
                #sub_impacted_id = subs_impacted[0]
                subs_impacted = self._get_subs_impacted_from_action_desc(action_desc)
                if not subs_impacted:
                    ignored_actions.append(action_desc)
                    continue
                
                sub_impacted_id = subs_impacted[0]
                sub_impacted_name = self.obs_defaut.name_sub[sub_impacted_id]

                if sub_impacted_name in hubs_names or sub_impacted_name in nodes_blue_path_names:
                    score, details = self.compute_node_splitting_action_score(action_desc["content"], sub_impacted_id, alphaDeesp_ranker)
                    map_action_score[action_id] = {"action": action, "score": score, "sub_impacted": sub_impacted_name, "details": details}
                    #print(action_desc["content"]["set_bus"])
                    #print(action_id+": "+str(score))
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
            act_defaut = self._create_default_action(self.action_space, self.lines_defaut)
            for action_id, sub_impacted in zip(actions.keys(), subs_impacted):
                action = actions[action_id]
                is_rho_reduction, _ = self._check_rho_reduction(
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
        self.scores_splits_dict = {action_id: map_action_score[action_id]["score"]
                                   for action_id in map_action_score}
        self.params_splits_dict = {action_id: map_action_score[action_id].get("details", {})
                                   for action_id in map_action_score}


    def compute_node_merging_score(self, sub_id: int, connected_buses: list) -> Tuple[float, Dict]:
        """
        Computes a heuristic score for a node merging action based on the voltage angle
        difference (delta phase) between the two buses being merged.

        The bus connected to the red loop (carrying positive dispatch flow) is identified
        as the one with more positive capacity on its overflow graph edges.
        Its phase is theta1. The other bus has phase theta2.

        Score = theta2 - theta1. A positive score means flows would naturally go from
        the higher-phase bus (theta2) towards the lower-phase red loop bus (theta1),
        which is the desired direction to relieve overloads.

        Args:
            sub_id: The integer index of the substation being merged.
            connected_buses: List of connected bus IDs (e.g., [1, 2]).

        Returns:
            Tuple[float, Dict]: The delta phase score and a details dict containing
                ``red_loop_bus`` and ``assets`` on that bus.
        """
        buses = sorted(connected_buses)
        if len(buses) < 2:
            return 0.0, {}

        # Determine which bus carries the red loop (positive dispatch) flow
        # by summing the positive capacity on overflow graph edges per bus
        capacity_dict = nx.get_edge_attributes(self.g_overflow.g, "capacity")
        edge_names = nx.get_edge_attributes(self.g_overflow.g, "name")

        # Build line-to-bus mapping for this substation using line_or/ex_to_subid and bus arrays
        self._build_lookup_caches()
        obs = self.obs_defaut

        line_to_bus = {}
        for line_id, line_name in enumerate(obs.name_line):
            if obs.line_or_to_subid[line_id] == sub_id:
                line_to_bus[line_name] = int(obs.line_or_bus[line_id])
            elif obs.line_ex_to_subid[line_id] == sub_id:
                line_to_bus[line_name] = int(obs.line_ex_bus[line_id])

        # Sum positive capacity per bus from overflow graph edges
        positive_flow_per_bus = {bus: 0.0 for bus in buses}
        all_edges = list(self.g_overflow.g.out_edges(sub_id, keys=True)) + \
                    list(self.g_overflow.g.in_edges(sub_id, keys=True))

        for edge in all_edges:
            cap = capacity_dict.get(edge, 0.0)
            if cap > 0:
                ename = edge_names.get(edge, "")
                bus = line_to_bus.get(ename)
                if bus in positive_flow_per_bus:
                    positive_flow_per_bus[bus] += cap

        # The red loop bus is the one with more positive dispatch flow
        red_loop_bus = max(buses, key=lambda b: positive_flow_per_bus.get(b, 0.0))
        other_bus = [b for b in buses if b != red_loop_bus][0]

        # Compute theta for each bus
        theta1 = get_theta_node(self.obs_defaut, sub_id, red_loop_bus)
        theta2 = get_theta_node(self.obs_defaut, sub_id, other_bus)

        score = theta2 - theta1

        # Build details with assets on the red loop bus (bus of interest)
        assets = self._get_assets_on_bus_for_sub(sub_id, red_loop_bus)
        details = {
            "red_loop_bus": red_loop_bus,
            "assets": assets,
        }

        return score, details

    def find_relevant_node_merging(self, nodes_dispatch_path_names: List[str], percentage_threshold_min_dispatch_flow=0.1):
        """
        Finds and evaluates relevant node merging actions on dispatch paths.

        Only substations that lie on loop dispatch paths and have 2+ connected buses
        are candidates. They are further filtered by requiring a minimum dispatch flow
        (at least ``percentage_threshold_min_dispatch_flow`` of the global max).

        Populates `self.identified_merges`, `self.effective_merges`, `self.ineffective_merges`,
        and `self.scores_merges`.

        Args:
            nodes_dispatch_path_names: List of substation names on dispatch paths.
            percentage_threshold_min_dispatch_flow: float between 0 and 1. threshold to filter out unsignificant node merging actions given not significant enough dispatch flow
        """
        identified, effective, ineffective = {}, [], []
        scores_map = {}
        details_map = {}
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

                    # Compute delta phase score and per-action details (including assets)
                    try:
                        score, details = self.compute_node_merging_score(sub_id, list(connected_buses))
                        scores_map[action_id] = score
                        details_map[action_id] = details
                        print(f"  Scored node merge {action_id}: delta_phase={score:.4f}")
                    except Exception as e:
                        print(f"  Warning: Could not score {action_id}: {e}")
                        scores_map[action_id] = 0.0
                        details_map[action_id] = {}

                    if self.check_action_simulation:
                        act_defaut = self._create_default_action(self.action_space, self.lines_defaut)
                        is_rho_reduction, _ = self._check_rho_reduction(
                            self.obs, self.timestep, act_defaut, action, self.lines_overloaded_ids,
                            self.act_reco_maintenance, self.lines_we_care_about
                        )
                        (effective if is_rho_reduction else ineffective).append(action)
                        print(f"  {'Effective' if is_rho_reduction else 'Ineffective'} node merge: {action_id}")

        self.identified_merges = identified
        self.effective_merges = effective
        self.ineffective_merges = ineffective
        self.scores_merges = scores_map
        self.params_merges = details_map


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
            Tuple[Dict[str, Any], Dict[str, Dict]]:
                - prioritized_actions: The final dictionary of prioritized actions (Action ID -> Action Object).
                  The results for each category are also stored in instance attributes
                  (e.g., `self.effective_splits`).
                - action_scores: A dictionary per action type with keys:
                  ``"line_reconnection"``, ``"line_disconnection"``, ``"open_coupling"``, ``"close_coupling"``.
                  Each value is a dict with two fields:
                    - ``"scores"``: {action_id: float, ...} sorted by descending score.
                    - ``"params"``: underlying hypotheses/parameters used for scoring.
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

        # Build global action scores dictionary per action type, sorted by descending score
        # Each type contains "scores" (sorted dict) and "params" (underlying hypotheses)
        # All float values are rounded to 2 decimals for readability
        def _round_scores(d):
            return {k: round(v, 2) for k, v in d.items()}

        def _round_params(d):
            """Round float values in a params dict (handles flat dicts and per-action nested dicts)."""
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
                elif isinstance(v, float):
                    out[k] = round(v, 2)
                else:
                    out[k] = v
            return out

        self.action_scores = {
            "line_reconnection": {
                "scores": _round_scores(dict(sorted(self.scores_reconnections.items(), key=lambda x: x[1], reverse=True))),
                "params": _round_params(self.params_reconnections),
            },
            "line_disconnection": {
                "scores": _round_scores(dict(sorted(self.scores_disconnections.items(), key=lambda x: x[1], reverse=True))),
                "params": _round_params(self.params_disconnections),
            },
            "open_coupling": {
                "scores": _round_scores(dict(sorted(self.scores_splits_dict.items(), key=lambda x: x[1], reverse=True))),
                "params": _round_params(self.params_splits_dict),
            },
            "close_coupling": {
                "scores": _round_scores(dict(sorted(self.scores_merges.items(), key=lambda x: x[1], reverse=True))),
                "params": _round_params(self.params_merges),
            },
        }

        print(f"\nDiscovery complete. Total prioritized actions: {len(self.prioritized_actions)}")
        return self.prioritized_actions, self.action_scores

