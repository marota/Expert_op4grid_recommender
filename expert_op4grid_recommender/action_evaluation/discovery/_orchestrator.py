# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Top-level orchestration for action discovery."""
import numpy as np
from typing import Any, Dict

from expert_op4grid_recommender.action_evaluation.discovery._results import (
    ACTION_SCORES_ORDER,
    FAMILY_MIN_CONFIG_ATTR,
    FAMILY_SPECS,
    FILL_PHASE_ORDER,
    MIN_PHASE_ORDER,
)
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

        # Optional recommender restriction: when ``config.ALLOWED_ACTION_TYPES``
        # is non-empty, ONLY the listed action families are discovered (others
        # are skipped entirely — saving time and keeping the result focused).
        from expert_op4grid_recommender import config as _cfg
        _allowed_types = set(getattr(_cfg, "ALLOWED_ACTION_TYPES", []) or [])

        # Antenna mode: the overflow graph is a synthetic downstream graph of an
        # islanded radial pocket — only injection actions make sense there, so
        # topological families are filtered out regardless of ALLOWED_ACTION_TYPES.
        antenna_mode = getattr(self, "antenna_mode", False)
        if antenna_mode:
            _allowed_types = {"ls", "rc", "redispatch"}

        def _type_allowed(token: str) -> bool:
            return not _allowed_types or token in _allowed_types

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

            if antenna_mode:
                # A radial pocket is a pure tree: no parallel red dispatch loops.
                # Skip get_dispatch_edges_nodes(only_loop_paths=True), which
                # additionally raises on an empty red_loops DataFrame.
                nodes_dispatch_loop_indices = []
            else:
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
        if _type_allowed("reco") and interesting_lines_to_reconnect:
            with Timer("Verifying relevant line reconnections"):
                print("\n--- Verifying relevant line reconnections ---")
                print(interesting_lines_to_reconnect)
                self.verify_relevant_reconnections(
                    interesting_lines_to_reconnect, red_loop_paths_names
                )

        if _type_allowed("close") and nodes_dispatch_loop_names:
            with Timer("Verifying relevant node merging"):
                print("\n--- Verifying relevant node merging ---")
                self.find_relevant_node_merging(nodes_dispatch_loop_names)

        if _type_allowed("open") and (hubs_names or nodes_blue_path_names):
            with Timer("Verifying relevant node splitting"):
                print("\n--- Verifying relevant node splitting ---")
                self.find_relevant_node_splitting(hubs_names, nodes_blue_path_names)

        if _type_allowed("disco") and lines_constrained_names:
            with Timer("Verifying relevant line disconnections"):
                print("\n--- Verifying relevant line disconnections ---")
                self.find_relevant_disconnections(lines_constrained_names)

        if _type_allowed("pst") and (nodes_blue_path_names or red_loop_paths_names):
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
        if antenna_mode and getattr(self, "antenna_meta", None):
            # The islanded pocket can sit on EITHER side of the constraint: a
            # consumer pocket is downstream (aval), a producer pocket — feeding
            # the rest of the grid up through the overload — is upstream (amont).
            # Injection actions always target the pocket itself, so address it
            # directly by its substation ids rather than the amont/aval split
            # (which now correctly reflects the real flow direction). The
            # per-action simulation check keeps only the ones that help.
            pocket_ids = [int(s) for s in self.antenna_meta.get("antenna_sub_ids", [])]
            if pocket_ids:
                nodes_aval_indices = pocket_ids
        if _type_allowed("ls") and nodes_aval_indices:
            with Timer("Verifying relevant load shedding"):
                print("\n--- Verifying relevant load shedding ---")
                self.find_relevant_load_shedding(nodes_aval_indices)
        # Renewable curtailment: target nodes in the constrained path (excluding aval) or red loop nodes
        nodes_rc_indices = list(
            (set(nodes_blue_path_indices) - set(nodes_aval_indices))
            | set(nodes_dispatch_loop_indices)
        )
        if antenna_mode:
            # A radial pocket is a pure tree with no dispatch loops: every node
            # is downstream (aval). Curtailment must still be able to target the
            # pocket's renewable generators (relevant for a net-producer pocket).
            nodes_rc_indices = list(nodes_aval_indices)
        if _type_allowed("rc") and nodes_rc_indices:
            with Timer("Verifying relevant renewable curtailment"):
                print("\n--- Verifying relevant renewable curtailment ---")
                all_nodes = [i for i in range(n_subs)]
                self.find_relevant_renewable_curtailment(
                    all_nodes, nodes_dispatch_loop_names
                )  # (nodes_rc_indices)

        # Redispatching: raise dispatchable production downstream (aval) of the
        # constrained path or on the parallel red dispatch loops; lower it
        # upstream (amont). Amont is approximated as the blue path minus aval.
        nodes_up_indices = list(
            set(nodes_aval_indices) | set(nodes_dispatch_loop_indices)
        )
        nodes_down_indices = list(
            set(nodes_blue_path_indices) - set(nodes_aval_indices)
        )
        if antenna_mode:
            # In a radial pocket, raising production (net-consumer pocket) and
            # lowering production (net-producer pocket) both target the pocket
            # itself. Offer both; the per-action simulation check keeps only the
            # ones that actually reduce the overload.
            nodes_down_indices = list(set(nodes_down_indices) | set(nodes_aval_indices))
        if _type_allowed("redispatch") and (nodes_up_indices or nodes_down_indices):
            with Timer("Verifying relevant redispatching"):
                print("\n--- Verifying relevant redispatching ---")
                self.find_relevant_redispatch(
                    nodes_up_indices, nodes_down_indices, nodes_dispatch_loop_names
                )

        with Timer("Finalizing   Priorization"):
            # 1. Add minimum required actions using a high per-type limit exactly equal to the min required
            from expert_op4grid_recommender import config

            # The per-type MIN_* counts are GUARANTEED floors: at least this
            # many actions of each type must be returned when candidates exist.
            # Their sum can exceed ``n_action_max`` (the fill target). If we
            # capped the minimum-enforcement phase at ``n_action_max``, the
            # types added last (load shedding, redispatch) would be starved
            # once the earlier types already filled the budget — silently
            # dropping a requested floor. So the minimum phase uses a cap that
            # admits every floor; only the fill phase (step 2 below) honours
            # ``n_action_max`` as the overall target.
            min_phase_cap = max(
                n_action_max,
                config.MIN_LINE_RECONNECTIONS
                + config.MIN_CLOSE_COUPLING
                + config.MIN_PST
                + config.MIN_OPEN_COUPLING
                + config.MIN_LINE_DISCONNECTIONS
                + config.MIN_RENEWABLE_CURTAILMENT
                + config.MIN_LOAD_SHEDDING
                + getattr(config, "MIN_REDISPATCH", 0),
            )

            # Two data-driven passes over ordered ``(family, per_type_cap)``
            # tables reading from the FamilyResult store — replacing the 18
            # hand-written add_prioritized_actions calls (and the PST
            # ``getattr``). One entry per (family, phase) makes a slipped-in
            # duplicate structurally impossible. The min phase enforces the
            # per-type MIN_* floors under ``min_phase_cap``; the fill phase tops
            # up to ``n_action_max`` with the per-family fill caps (default 3
            # where the historical call passed none — reco/split/pst carry their
            # explicit n_*_max). The interleave differs between phases (PST is
            # 3rd in the floor pass but 5th in the fill pass) and is preserved.
            for _family_key in MIN_PHASE_ORDER:
                _per_type = getattr(config, FAMILY_MIN_CONFIG_ATTR[_family_key], 0)
                self.prioritized_actions = add_prioritized_actions(
                    self.prioritized_actions,
                    self.results[_family_key].identified,
                    min_phase_cap,
                    n_action_max_per_type=_per_type,
                )

            # 2. Fill the remaining slots sequentially using the original
            #    priority logic and limits. Only reco / split / pst carry an
            #    explicit per-family cap; the rest use add_prioritized_actions'
            #    default of 3.
            _fill_caps = {
                "reconnections": n_reco_max,
                "splits": n_split_max,
                "pst": n_pst_max,
            }
            for _family_key in FILL_PHASE_ORDER:
                self.prioritized_actions = add_prioritized_actions(
                    self.prioritized_actions,
                    self.results[_family_key].identified,
                    n_action_max,
                    n_action_max_per_type=_fill_caps.get(_family_key, 3),
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

        # Data-driven over the FAMILY_SPECS registry in the fixed
        # ACTION_SCORES_ORDER, reading straight from the FamilyResult store —
        # replacing the 8× hand-written literal (and the PST ``getattr``
        # special-casing, now that every family, PST included, is always present
        # in ``self.results``). Each entry: descending-sorted ``scores`` +
        # rounded ``params`` + an empty ``non_convergence`` filled later by
        # reassessment.
        self.action_scores = {}
        for _family_key in ACTION_SCORES_ORDER:
            _spec = FAMILY_SPECS[_family_key]
            _result = self.results[_family_key]
            self.action_scores[_spec.scores_key] = {
                "scores": _round_scores(
                    dict(
                        sorted(
                            _result.scores.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                ),
                "params": _round_params(_result.params),
                "non_convergence": {},
            }

        print(
            f"\nDiscovery complete. Total prioritized actions: {len(self.prioritized_actions)}"
        )
        return self.prioritized_actions, self.action_scores
