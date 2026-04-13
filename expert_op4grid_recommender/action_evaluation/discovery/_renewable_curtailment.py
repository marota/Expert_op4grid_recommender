# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Renewable curtailment discovery mixin."""
import numpy as np
from typing import List

from expert_op4grid_recommender import config

class RenewableCurtailmentMixin:
    """Renewable curtailment discovery mixin."""

    def find_relevant_renewable_curtailment(
        self, nodes_indices: List[int], nodes_dispatch_loop_names: List[str] = []
    ):
        """
        Discovers renewable curtailment candidates on upstream (amont) nodes or loop nodes.
        Mirroring load shedding logic but for generators (WIND/SOLAR) on the opposite side of the flow.

        Optimized for large networks (>500 nodes, >1000 lines) with:
        - Pre-computed substation-to-renewable-generator mapping (avoids per-node get_obj_connect_to)
        - Cached single-pass edge attribute extraction
        - Cached blue edge names and node flow calculations
        - Pre-computed baseline simulation for batch action checking
        """
        self._build_lookup_caches()
        # Ensure edge data cache is populated (single pass over all edges)
        self._get_edge_data_cache()
        obs = self.obs_defaut

        margin = getattr(config, "RENEWABLE_CURTAILMENT_MARGIN", 0.05)
        min_mw = getattr(config, "RENEWABLE_CURTAILMENT_MIN_MW", 1.0)

        # Compute overload excess in MW (uses cached capacity map)
        name_to_capacity = self._build_line_capacity_map()
        if not name_to_capacity:
            return

        overloaded_line_names = {obs.name_line[i] for i in self.lines_overloaded_ids}
        overloaded_caps = [
            name_to_capacity[n] for n in overloaded_line_names if n in name_to_capacity
        ]
        max_overload_flow = (
            max(overloaded_caps) if overloaded_caps else max(name_to_capacity.values())
        )

        if len(self.lines_overloaded_ids) == 0:
            return
        rho_max = float(np.max(obs.rho[self.lines_overloaded_ids]))
        if rho_max <= 1.0:
            return
        P_overload_excess = (rho_max - 1.0) * max_overload_flow

        # Use cached blue edge names (fixes double call to get_constrained_edges_nodes)
        blue_edge_names_set = self._get_blue_edge_names_set()

        # Pre-compute dispatch loop names as set for O(1) lookup
        dispatch_loop_set = (
            set(nodes_dispatch_loop_names) if nodes_dispatch_loop_names else None
        )

        # Use cached node flow computation (single pass over edges)
        node_flow_cache = self._build_node_flow_cache(blue_edge_names_set, dispatch_loop_set)

        # Pre-filter: only iterate substations that have renewable generators (huge win on 500+ node networks)
        subs_with_renewable = self._get_subs_with_renewable_gens()
        nodes_set = set(nodes_indices)
        # Only consider substations in the requested nodes AND that have renewable generators AND non-zero flow
        relevant_subs = [
            sub_id for sub_id in subs_with_renewable
            if sub_id in nodes_set and sub_id in node_flow_cache
        ]

        identified, effective, ineffective = {}, [], []
        scores_map, params_map = {}, {}

        for node_idx in relevant_subs:
            sub_name = str(obs.name_sub[node_idx])

            # Use pre-computed renewable generator list (avoids get_obj_connect_to + filtering)
            renewable_gen_ids = subs_with_renewable[node_idx]

            # Get pre-computed influence flows (O(1) lookup)
            node_flows = node_flow_cache[node_idx]
            total_neg_in = node_flows["neg_in"]
            total_neg_out = node_flows["neg_out"]
            total_pos_in = node_flows["pos_in"]
            total_pos_out = node_flows["pos_out"]

            influence_flow = max(
                total_neg_in, total_neg_out, total_pos_in, total_pos_out
            )
            if influence_flow <= 0:
                continue

            influence_factor = (
                min(1.0, influence_flow / max_overload_flow)
                if max_overload_flow > 0
                else 0.0
            )
            if influence_factor <= 0:
                continue

            # Pre-compute common values for all generators at this node
            mw_required = (
                (P_overload_excess * (1 + margin)) / influence_factor
                if influence_factor > 0
                else P_overload_excess * (1 + margin)
            )

            for gen_id in renewable_gen_ids:
                try:
                    gen_name = obs.name_gen[gen_id]
                except (IndexError, KeyError):
                    continue

                gen_p_array = getattr(obs, "gen_p", getattr(obs, "prod_p", None))
                if gen_p_array is None or gen_id >= len(gen_p_array):
                    continue

                gen_p = float(gen_p_array[gen_id])
                if gen_p > 0:
                    continue
                gen_p = abs(gen_p)
                if gen_p < min_mw:
                    continue

                coverage_ratio = (
                    min(1.0, gen_p / mw_required) if mw_required > 0 else 1.0
                )
                score = influence_factor * coverage_ratio

                action_id = f"curtail_{gen_name}"
                # Reduce active power to 0 MW instead of disconnecting
                identified[action_id] = self.action_space(
                    {"set_gen_p": {gen_name: 0.0}}
                )
                scores_map[action_id] = round(score, 2)
                params_map[action_id] = {
                    "substation": sub_name,
                    "node_type": "aval",
                    "gen_name": gen_name,
                    "action_mode": "power_reduction",
                    "target_p_MW": 0.0,
                    "reduction_MW": round(gen_p, 2),
                    "influence_factor": round(influence_factor, 2),
                    "mw_required": round(mw_required, 2),
                    "gen_p": round(gen_p, 2),
                    "coverage_ratio": round(coverage_ratio, 2),
                }

        if self.check_action_simulation and identified:
            try:
                act_defaut = self._create_default_action(
                    self.action_space, self.lines_defaut
                )
                # Pre-compute baseline simulation once for all actions
                baseline_rho, _ = self._compute_baseline(
                    self.obs, self.timestep, act_defaut,
                    self.act_reco_maintenance, self.lines_overloaded_ids
                )
                if baseline_rho is not None:
                    for action_id, action in identified.items():
                        is_reduced, _ = self._check_rho_with_baseline(
                            self.obs,
                            self.timestep,
                            act_defaut,
                            action,
                            self.lines_overloaded_ids,
                            self.act_reco_maintenance,
                            baseline_rho,
                            self.lines_we_care_about,
                        )
                        if is_reduced:
                            effective.append(action_id)
                        else:
                            ineffective.append(action_id)
            except Exception as e:
                print(
                    f"Warning: Simulation check failed for renewable curtailment: {e}"
                )

        self.identified_renewable_curtailment = identified
        self.effective_renewable_curtailment = effective
        self.ineffective_renewable_curtailment = ineffective
        self.scores_renewable_curtailment = scores_map
        self.params_renewable_curtailment = params_map
