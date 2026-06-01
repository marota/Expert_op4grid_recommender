# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Line disconnection discovery and scoring mixin."""
from typing import Dict, List

from expert_op4grid_recommender.utils.helpers import sort_actions_by_score

class LineDisconnectionMixin:
    """Line disconnection discovery and scoring mixin."""

    def compute_disconnection_score(self, lines_in_action: set) -> float:
        """
        Computes a heuristic score for a line disconnection action based on its redispatch flow.

        Two scoring regimes exist:

        **Constrained** (``max_redispatch < inf``):
        An asymmetric bell curve between ``min_redispatch`` and ``max_redispatch``,
        peaking closer to ``max_redispatch``.  Negative outside.

        **Unconstrained** (``max_redispatch == inf``):
        No line in the overflow graph gets overloaded from the redispatch, so
        disconnections are inherently safe.  Score equals the ratio of the
        action's redispatch flow to the overloaded line's flow:
        ``observed_flow / max_overload_flow``, capped at 1.0.
        Disconnecting the overloaded line itself scores 1.0.

        **Overload disconnection priority bonus:**
        When the action directly disconnects one of the overloaded lines (the most
        straightforward corrective action) AND we are in the unconstrained regime
        (no new overloads are created), the score is boosted by adding 1.0.  This
        places such actions in the [1.0, 2.0] range, above all other disconnections
        which remain in the [-inf, 1.0] range.

        Args:
            lines_in_action: Set of line names being disconnected by this action.

        Returns:
            float: The heuristic score for this disconnection action.
        """
        # Lazy-compute and cache the bounds and capacity map
        if not hasattr(self, "_disco_bounds"):
            self._disco_bounds = self._compute_disconnection_flow_bounds()
            self._disco_capacity_map = self._build_line_capacity_map()

        max_overload_flow, min_redispatch, max_redispatch = self._disco_bounds

        if max_overload_flow < 1e-6:
            return 0.0

        # Sum capacities of lines being disconnected (observed redispatch flow)
        observed_flow = sum(
            self._disco_capacity_map.get(line, 0.0) for line in lines_in_action
        )

        if max_redispatch == float("inf"):
            # Unconstrained regime: simple ratio of observed flow to overloaded
            # line's flow.  All disconnections are safe (no new overloads), so
            # we don't penalise partial relief — score is just how much of the
            # overloaded flow this action redirects: 0 at 0 MW, 1 at full flow.
            score = self._unconstrained_linear_score(
                observed_flow, 0.0, max_overload_flow
            )
            # Priority bonus: directly disconnecting an overloaded line is the most
            # straightforward corrective action and should rank above all other
            # disconnections.  Boost the score by 1.0 so these actions occupy the
            # [1.0, 2.0] range while others stay in [0.0, 1.0].
            overloaded_line_names = {
                self.obs_defaut.name_line[i] for i in self.lines_overloaded_ids
            }
            if lines_in_action.intersection(overloaded_line_names):
                score += 1.0
            return score
        else:
            # Constrained regime: bell curve between min and max
            return self._asymmetric_bell_score(
                observed_flow, min_redispatch, max_redispatch
            )

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
        if hasattr(self, "_disco_bounds"):
            del self._disco_bounds
            del self._disco_capacity_map

        overloaded_line_names = {
            self.obs_defaut.name_line[i] for i in self.lines_overloaded_ids
        }
        print(f"Evaluating {len(self.actions_unfiltered)} potential disconnections...")
        for action_id in sorted(
            list(self.actions_unfiltered)
        ):  # as order in a set is no fixed, and since the order will matter in the subset of actions selected, fix the order for full reproducibility
            action_desc = self.dict_action[action_id]
            action_type = self.classifier.identify_action_type(
                action_desc, by_description=True
            )

            if "open_line" in action_type:
                content = action_desc.get("content", {}).get("set_bus", {})
                lines_in_action = set(
                    list(content.get("lines_ex_id", {}).keys())
                    + list(content.get("lines_or_id", {}).keys())
                )

                # Include actions on the constrained path OR that directly disconnect an overloaded
                # line (the most straightforward corrective action deserves consideration even if
                # not strictly on the alphaDeesp-computed constrained path).
                is_on_constrained_path = bool(
                    lines_in_action.intersection(set(lines_constrained_path_names))
                )
                is_overload_disconnection = bool(
                    lines_in_action.intersection(overloaded_line_names)
                )
                if is_on_constrained_path or is_overload_disconnection:
                    action = self.action_space(action_desc["content"])
                    identified[action_id] = action

                    # Compute heuristic score
                    score = self.compute_disconnection_score(lines_in_action)
                    scores_map[action_id] = score

                    if self.check_action_simulation:
                        is_rho_reduction, _ = self._check_rho_reduction(
                            self.obs,
                            self.timestep,
                            act_defaut,
                            action,
                            self.lines_overloaded_ids,
                            self.act_reco_maintenance,
                            self.lines_we_care_about,
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

        print(
            f"  Found {len(effective)} effective, {len(ineffective)} ineffective disconnections."
        )

        # Sort identified disconnections by score descending (higher score = better candidate)
        map_action_score_disco = {
            action_id: {"action": action, "score": scores_map[action_id]}
            for action_id, action in identified.items()
        }
        sorted_actions, _, _ = sort_actions_by_score(map_action_score_disco)
        self.identified_disconnections = sorted_actions
        self.effective_disconnections = effective
        self.ineffective_disconnections = ineffective
        self.ignored_disconnections = ignored
        self.scores_disconnections = scores_map

        # Capture computed bounds before cleanup
        if hasattr(self, "_disco_bounds"):
            max_overload_flow, min_redispatch, max_redispatch = self._disco_bounds
            if max_redispatch == float("inf"):
                # Unconstrained regime: linear ramp, peak at max_overload_flow
                self.params_disconnections = {
                    "regime": "unconstrained",
                    "min_redispatch": min_redispatch,
                    "max_overload_flow": max_overload_flow,
                }
            else:
                # Constrained regime: bell curve
                # Peak redispatch: x_peak = (alpha-1)/(alpha+beta-2) = 2/2.5 = 0.8
                peak_redispatch = min_redispatch + 0.8 * (
                    max_redispatch - min_redispatch
                )
                self.params_disconnections = {
                    "regime": "constrained",
                    "min_redispatch": min_redispatch,
                    "max_redispatch": max_redispatch,
                    "peak_redispatch": peak_redispatch,
                }
        else:
            self.params_disconnections = {}

        # Clean up cached bounds
        if hasattr(self, "_disco_bounds"):
            del self._disco_bounds
            del self._disco_capacity_map
