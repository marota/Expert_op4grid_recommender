# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Top-level orchestration for action discovery."""
import numpy as np
from typing import Any, Dict

from expert_op4grid_recommender import config
from expert_op4grid_recommender.utils.helpers import Timer, add_prioritized_actions

class OrchestratorMixin:
    """Top-level orchestration for action discovery."""

    def discover_and_prioritize(
        self,
        n_action_max: int = 5,
        n_reco_max: int = 2,
        n_split_max: int = 3,
        n_pst_max: int = 2,
    ) -> Dict[str, Any]:
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
            n_pst_max (int): Max number of PST actions to prioritize. Defaults to 2.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Dict]]:
                - prioritized_actions: The final dictionary of prioritized actions (Action ID -> Action Object).
                  The results for each category are also stored in instance attributes
                  (e.g., `self.effective_splits`).
                - action_scores: A dictionary per action type with keys:
                  ``"line_reconnection"``, ``"line_disconnection"``, ``"open_coupling"``, ``"close_coupling"``, ``"pst_tap"``.
                  Each value is a dict with two fields:
                    - ``"scores"``: {action_id: float, ...} sorted by descending score.
                    - ``"params"``: underlying hypotheses/parameters used for scoring.
        """
        self.prioritized_actions = {}
        # Use n_pst_max as the per-type limit for PST actions in the remaining fill phase

        with Timer("Priorization Preparation"):
            name_sub_arr = np.array(self.obs.name_sub)
            n_subs = len(name_sub_arr)

            # --- Extract Path Information (Convert Indices to Names) ---
            lines_dispatch, _ = self.g_distribution_graph.get_dispatch_edges_nodes(
                only_loop_paths=False
            )
            # Since the graph nodes are now indices, lines_dispatch contains indices. We need names.
            # However, check_other_reconnectable_line_on_path uses obs to map names to subs, so it expects names.
            # Let's get the names from the obs object.
            lines_dispatch_names = lines_dispatch  # [obs.name_line[line_idx] for line_idx in lines_dispatch]

            if (
                hasattr(self.g_distribution_graph, "red_loops")
                and self.g_distribution_graph.red_loops is not None
                and not self.g_distribution_graph.red_loops.empty
            ):
                df = self.g_distribution_graph.red_loops
                try:
                    if "Path" in df.columns:
                        # Use tuples instead of string conversion for faster deduplication
                        unique_paths = df["Path"].map(tuple).unique()
                        # Use NumPy fancy indexing for much faster name lookups
                        red_loop_paths_names = [
                            list(name_sub_arr[[idx for idx in p if idx < n_subs]])
                            for p in unique_paths
                        ]
                    else:
                        red_loop_paths_names = []
                except Exception as e:
                    print(f"Warning: Error processing red_loops DataFrame: {e}")
                    red_loop_paths_names = []
            else:
                red_loop_paths_names = []

            _, nodes_dispatch_loop_indices = (
                self.g_distribution_graph.get_dispatch_edges_nodes(only_loop_paths=True)
            )
            nodes_dispatch_loop_names = list(
                name_sub_arr[
                    [idx for idx in nodes_dispatch_loop_indices if idx < n_subs]
                ]
            )

            (
                lines_constrained_names,
                nodes_constrained_indices,
                _,
                other_blue_nodes_indices,
            ) = self.g_distribution_graph.get_constrained_edges_nodes()
            nodes_blue_path_indices = (
                nodes_constrained_indices + other_blue_nodes_indices
            )
            nodes_blue_path_names = list(
                name_sub_arr[[idx for idx in nodes_blue_path_indices if idx < n_subs]]
            )
            hubs_names = self.hubs  # Assume hubs passed during init were already names

        # --- Call Discovery Methods ---
        interesting_lines_to_reconnect = sorted(
            list(
                set(lines_dispatch_names).intersection(
                    set(self.non_connected_reconnectable_lines)
                )
            )
        )
        if interesting_lines_to_reconnect:
            with Timer("Verifying relevant line reconnections"):
                print("\n--- Verifying relevant line reconnections ---")
                print(interesting_lines_to_reconnect)
                self.verify_relevant_reconnections(
                    interesting_lines_to_reconnect, red_loop_paths_names
                )

        if nodes_dispatch_loop_names:
            with Timer("Verifying relevant node merging"):
                print("\n--- Verifying relevant node merging ---")
                self.find_relevant_node_merging(nodes_dispatch_loop_names)

        if hubs_names or nodes_blue_path_names:
            with Timer("Verifying relevant node splitting"):
                print("\n--- Verifying relevant node splitting ---")
                self.find_relevant_node_splitting(hubs_names, nodes_blue_path_names)

        if lines_constrained_names:
            with Timer("Verifying relevant line disconnections"):
                print("\n--- Verifying relevant line disconnections ---")
                self.find_relevant_disconnections(lines_constrained_names)

        if nodes_blue_path_names or red_loop_paths_names:
            with Timer("Verifying relevant PST actions"):
                print("\n--- Verifying relevant PST actions ---")
                self.find_relevant_pst_actions(
                    nodes_blue_path_names, red_loop_paths_names
                )

        # Load shedding: target downstream (aval) nodes on the constrained path
        constrained_path = self.g_distribution_graph.get_constrained_path()
        nodes_aval_indices = (
            list(constrained_path.n_aval()) if constrained_path is not None else []
        )
        if nodes_aval_indices:
            with Timer("Verifying relevant load shedding"):
                print("\n--- Verifying relevant load shedding ---")
                self.find_relevant_load_shedding(nodes_aval_indices)
        # Renewable curtailment: target nodes in the constrained path (excluding aval) or red loop nodes
        nodes_rc_indices = list(
            (set(nodes_blue_path_indices) - set(nodes_aval_indices))
            | set(nodes_dispatch_loop_indices)
        )
        if nodes_rc_indices:
            with Timer("Verifying relevant renewable curtailment"):
                print("\n--- Verifying relevant renewable curtailment ---")
                all_nodes = [i for i in range(n_subs)]
                self.find_relevant_renewable_curtailment(
                    all_nodes, nodes_dispatch_loop_names
                )  # (nodes_rc_indices)

        with Timer("Finalizing   Priorization"):
            # 1. Add minimum required actions using a high per-type limit exactly equal to the min required
            from expert_op4grid_recommender import config

            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_reconnections,
                n_action_max,
                n_action_max_per_type=config.MIN_LINE_RECONNECTIONS,
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_merges,
                n_action_max,
                n_action_max_per_type=config.MIN_CLOSE_COUPLING,
            )
            # Step 1.3: Add minimum required PST actions
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                getattr(self, "identified_pst_actions", {}),
                n_action_max,
                n_action_max_per_type=config.MIN_PST,
            )

            # Step 1.4: Add minimum required node splitting and line disconnections
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_splits,
                n_action_max,
                n_action_max_per_type=config.MIN_OPEN_COUPLING,
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_disconnections,
                n_action_max,
                n_action_max_per_type=config.MIN_LINE_DISCONNECTIONS,
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_renewable_curtailment,
                n_action_max,
                n_action_max_per_type=config.MIN_RENEWABLE_CURTAILMENT,
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_load_shedding,
                n_action_max,
                n_action_max_per_type=config.MIN_LOAD_SHEDDING,
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_renewable_curtailment,
                n_action_max,
                n_action_max_per_type=getattr(config, "MIN_RENEWABLE_CURTAILMENT", 0),
            )

            # 2. Fill the remaining slots sequentially using the original priority logic and limits
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_reconnections,
                n_action_max,
                n_action_max_per_type=n_reco_max,
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions, self.identified_merges, n_action_max
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_splits,
                n_action_max,
                n_action_max_per_type=n_split_max,
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions, self.identified_disconnections, n_action_max
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                getattr(self, "identified_pst_actions", {}),
                n_action_max,
                n_action_max_per_type=n_pst_max,
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_renewable_curtailment,
                n_action_max,
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions, self.identified_load_shedding, n_action_max
            )
            self.prioritized_actions = add_prioritized_actions(
                self.prioritized_actions,
                self.identified_renewable_curtailment,
                n_action_max,
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
                    out[k] = {
                        kk: round(vv, 2) if isinstance(vv, float) else vv
                        for kk, vv in v.items()
                    }
                elif isinstance(v, float):
                    out[k] = round(v, 2)
                else:
                    out[k] = v
            return out

        self.action_scores = {
            "line_reconnection": {
                "scores": _round_scores(
                    dict(
                        sorted(
                            self.scores_reconnections.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                ),
                "params": _round_params(self.params_reconnections),
                "non_convergence": {},
            },
            "line_disconnection": {
                "scores": _round_scores(
                    dict(
                        sorted(
                            self.scores_disconnections.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                ),
                "params": _round_params(self.params_disconnections),
                "non_convergence": {},
            },
            "open_coupling": {
                "scores": _round_scores(
                    dict(
                        sorted(
                            self.scores_splits_dict.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                ),
                "params": _round_params(self.params_splits_dict),
                "non_convergence": {},
            },
            "close_coupling": {
                "scores": _round_scores(
                    dict(
                        sorted(
                            self.scores_merges.items(), key=lambda x: x[1], reverse=True
                        )
                    )
                ),
                "params": _round_params(self.params_merges),
                "non_convergence": {},
            },
            "pst_tap": {
                "scores": _round_scores(
                    dict(
                        sorted(
                            getattr(self, "scores_pst_actions", {}).items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                ),
                "params": _round_params(getattr(self, "params_pst_actions", {})),
                "non_convergence": {},
            },
            "load_shedding": {
                "scores": _round_scores(
                    dict(
                        sorted(
                            self.scores_load_shedding.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                ),
                "params": _round_params(self.params_load_shedding),
                "non_convergence": {},
            },
            "renewable_curtailment": {
                "scores": _round_scores(
                    dict(
                        sorted(
                            self.scores_renewable_curtailment.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                ),
                "params": _round_params(self.params_renewable_curtailment),
                "non_convergence": {},
            },
        }

        print(
            f"\nDiscovery complete. Total prioritized actions: {len(self.prioritized_actions)}"
        )
        return self.prioritized_actions, self.action_scores
