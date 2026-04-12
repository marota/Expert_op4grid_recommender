# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Load shedding discovery mixin."""
import numpy as np
from typing import List

from expert_op4grid_recommender import config

class LoadSheddingMixin:
    """Load shedding discovery mixin."""

    def find_relevant_load_shedding(self, nodes_aval_indices: List[int]):
        """
        Discovers load shedding candidates on downstream (aval) nodes of the constrained path.

        Optimized for large networks (>500 nodes, >1000 lines) with:
        - Cached single-pass edge attribute extraction
        - Pre-computed substation-to-load mapping (avoids per-node get_obj_connect_to)
        - Cached node flow calculations
        - Hoisted default action and baseline simulation
        """
        self._build_lookup_caches()
        # Ensure edge data cache is populated (single pass over all edges)
        self._get_edge_data_cache()
        obs = self.obs_defaut

        margin = getattr(config, "LOAD_SHEDDING_MARGIN", 0.05)
        min_mw = getattr(config, "LOAD_SHEDDING_MIN_MW", 1.0)

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

        rho_overloaded = obs.rho[self.lines_overloaded_ids]
        if len(rho_overloaded) == 0:
            return
        rho_max = float(np.max(rho_overloaded))
        if rho_max <= 1.0:
            return
        P_overload_excess = (rho_max - 1.0) * max_overload_flow

        # Use cached blue edge set and node flow cache
        blue_edge_names_set = self._get_blue_edge_names_set()
        node_influence_flows = self._build_node_flow_cache(blue_edge_names_set)

        # Pre-filter: only consider aval nodes that have loads with positive power
        subs_with_loads = self._get_subs_with_loads()
        aval_set = set(nodes_aval_indices)
        relevant_nodes = [n for n in nodes_aval_indices if n in subs_with_loads and n in node_influence_flows]

        identified = {}
        scores_map = {}
        details_map = {}
        effective = []
        ineffective = []

        # Hoist default action creation outside the loop
        act_defaut = None
        baseline_rho = None
        if self.check_action_simulation:
            act_defaut = self._create_default_action(
                self.action_space, self.lines_defaut
            )
            # Pre-compute baseline simulation once for all actions
            baseline_rho, _ = self._compute_baseline(
                self.obs, self.timestep, act_defaut,
                self.act_reco_maintenance, self.lines_overloaded_ids
            )

        for node_idx in relevant_nodes:
            sub_name = str(obs.name_sub[node_idx])

            # Use pre-computed load list (avoids get_obj_connect_to)
            load_ids = subs_with_loads[node_idx]

            # Load powers already filtered to positive in _get_subs_with_loads
            load_powers = [(lid, float(obs.load_p[lid])) for lid in load_ids]
            available_load = sum(p for _, p in load_powers)

            # Get pre-computed influence flow (O(1) lookup)
            node_flows = node_influence_flows[node_idx]
            total_neg_in = node_flows["neg_in"]
            total_neg_out = node_flows["neg_out"]
            influence_flow = max(total_neg_in, total_neg_out)

            if influence_flow <= 0:
                continue

            influence_factor = (
                min(1.0, influence_flow / max_overload_flow)
                if max_overload_flow > 0
                else 0.0
            )
            if influence_factor <= 0:
                continue

            # Compute shedding volume
            P_shedding_min = (
                P_overload_excess / influence_factor * (1.0 + margin)
                if influence_factor > 0
                else P_overload_excess * (1.0 + margin)
            )
            P_shedding = max(P_shedding_min, min_mw)
            P_shedding = min(P_shedding, available_load)

            # Sort loads by power descending
            load_powers.sort(key=lambda x: x[1], reverse=True)

            assets = self._get_assets_on_bus_for_sub(node_idx, 1)

            for lid, load_power in load_powers:
                load_name = (
                    str(obs.name_load[lid]) if lid < len(obs.name_load) else str(lid)
                )
                action_id = f"load_shedding_{load_name}"

                load_coverage = (
                    min(1.0, load_power / P_shedding_min) if P_shedding_min > 0 else 1.0
                )
                load_score = influence_factor * load_coverage

                try:
                    action = self.action_space({"set_load_p": {load_name: 0.0}})
                except Exception as e:
                    print(
                        f"Warning: Could not create load shedding action for {load_name}: {e}"
                    )
                    continue

                identified[action_id] = action
                scores_map[action_id] = round(load_score, 2)

                details_map[action_id] = {
                    "substation": sub_name,
                    "node_type": "aval",
                    "load_name": load_name,
                    "action_mode": "power_reduction",
                    "target_p_MW": 0.0,
                    "reduction_MW": round(load_power, 2),
                    "influence_factor": round(influence_factor, 2),
                    "in_negative_flows": round(total_neg_in, 2),
                    "out_negative_flows": round(total_neg_out, 2),
                    "P_shedding_MW": round(min(load_power, P_shedding), 2),
                    "P_overload_excess_MW": round(P_overload_excess, 2),
                    "available_load_MW": round(load_power, 2),
                    "coverage_ratio": round(load_coverage, 2),
                    "loads_shed": [load_name],
                    "assets": assets,
                }

                if self.check_action_simulation and baseline_rho is not None:
                    try:
                        is_reduction, obs_after = self._check_rho_with_baseline(
                            self.obs,
                            self.timestep,
                            act_defaut,
                            action,
                            self.lines_overloaded_ids,
                            self.act_reco_maintenance,
                            baseline_rho,
                            self.lines_we_care_about,
                        )
                        if is_reduction:
                            effective.append(action_id)
                        else:
                            ineffective.append(action_id)
                    except Exception as e:
                        print(f"Warning: Simulation check failed for {action_id}: {e}")
                        ineffective.append(action_id)

        self.identified_load_shedding = dict(
            sorted(
                identified.items(), key=lambda x: scores_map.get(x[0], 0), reverse=True
            )
        )
        self.effective_load_shedding = effective
        self.ineffective_load_shedding = ineffective
        self.scores_load_shedding = scores_map
        self.params_load_shedding = details_map
