# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Shared base for the three injection discovery families (revision R5).

Load shedding, renewable curtailment and redispatch each opened their
``find_relevant_*`` method with the same ~15-line overload preamble (warm the
caches, read the family margin / minimum-MW knobs, compute the reference
``max_overload_flow`` and the ``P_overload_excess`` in MW) and the same
saturating influence-factor formula — duplicated three times and already
drifting. :class:`InjectionDiscoveryBase` factors out exactly those two shared,
byte-identical pieces.

What is DELIBERATELY left in each family, because the divergence is
load-bearing (documented per family):

* which node-flow cache is built — blue-only (load shedding) vs. blue + red
  dispatch loops (curtailment / redispatch). This is coupled to the number of
  flow components each family's ``_influence_of`` reads (load shedding's
  blue-only cache has structurally-zero positive flows, so its ``max(neg_in,
  neg_out)`` equals the 4-component max; redispatch reads 2 or 4 by direction).
* the candidate loop (node/load-centric vs. generator-centric),
* the per-candidate simulation-check mode (load shedding: inline, uncapped;
  the generator families: capped, post-loop),
* the family-specific ``params`` / ``details`` schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from expert_op4grid_recommender import config


@dataclass(frozen=True)
class InjectionOverloadContext:
    """Result of the shared injection preamble.

    ``P_overload_excess`` is the MW the worst overload exceeds its limit by,
    scaled by ``max_overload_flow`` (the reference flow scoring 1.0).
    """

    obs: Any
    margin: float
    min_mw: float
    max_overload_flow: float
    P_overload_excess: float


class InjectionDiscoveryBase:
    """Mixin providing the shared overload preamble + influence factor.

    Plain mixin (like the family mixins): its methods call ``DiscovererBase``
    helpers through ``self`` on the composed :class:`ActionDiscoverer`. Each
    family sets :data:`MARGIN_KEY` / :data:`MIN_MW_KEY` to its config knobs.
    """

    #: config attribute names for the family's margin / minimum-MW knobs
    #: (overridden per family; the defaults mirror load shedding).
    MARGIN_KEY: str = "LOAD_SHEDDING_MARGIN"
    MIN_MW_KEY: str = "LOAD_SHEDDING_MIN_MW"

    def _injection_overload_context(self) -> Optional[InjectionOverloadContext]:
        """Warm caches, read the family knobs, compute the overload reference.

        Returns ``None`` to signal an early return (no line-capacity map, or no
        active overload) — the caller then returns without recording candidates,
        exactly as each family's inline preamble did.
        """
        self._build_lookup_caches()
        # Ensure the edge-data cache is populated (single pass over all edges).
        self._get_edge_data_cache()
        obs = self.obs_defaut

        margin = getattr(config, self.MARGIN_KEY, 0.05)
        min_mw = getattr(config, self.MIN_MW_KEY, 1.0)

        # Overload excess in MW (uses the cached per-line capacity map).
        name_to_capacity = self._build_line_capacity_map()
        if not name_to_capacity:
            return None

        name_line_arr = obs.name_line
        overloaded_line_names = {name_line_arr[i] for i in self.lines_overloaded_ids}
        overloaded_caps = [
            name_to_capacity[n] for n in overloaded_line_names if n in name_to_capacity
        ]
        max_overload_flow = (
            max(overloaded_caps) if overloaded_caps else max(name_to_capacity.values())
        )

        if len(self.lines_overloaded_ids) == 0:
            return None
        rho_max = float(np.max(obs.rho[self.lines_overloaded_ids]))
        if rho_max <= 1.0:
            return None
        P_overload_excess = (rho_max - 1.0) * max_overload_flow

        return InjectionOverloadContext(
            obs=obs,
            margin=margin,
            min_mw=min_mw,
            max_overload_flow=max_overload_flow,
            P_overload_excess=P_overload_excess,
        )

    @staticmethod
    def _injection_influence_factor(
        influence_flow: float, max_overload_flow: float
    ) -> float:
        """Saturating influence factor shared by all three injection families."""
        return (
            min(1.0, influence_flow / max_overload_flow)
            if max_overload_flow > 0
            else 0.0
        )
