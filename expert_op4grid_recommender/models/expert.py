# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Built-in expert recommender exposed through the pluggable interface.

Wraps the existing :class:`ActionDiscoverer` so the rest of the
pipeline can call it via the same :class:`RecommenderModel` protocol
used by external models (random, ML, ...).
"""
from __future__ import annotations

import logging
from typing import List

from expert_op4grid_recommender import config
from expert_op4grid_recommender.models.base import (
    ParamSpec,
    RecommenderInputs,
    RecommenderModel,
    RecommenderOutput,
)

logger = logging.getLogger(__name__)


class ExpertRecommender(RecommenderModel):
    """Rule-based expert discovery, exposed as a :class:`RecommenderModel`.

    All scoring knobs live in :mod:`expert_op4grid_recommender.config`;
    :meth:`params_spec` mirrors them so the UI renders the same controls
    as before.
    """

    name = "expert"
    label = "Expert system"
    requires_overflow_graph = True

    @classmethod
    def params_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("n_prioritized_actions", "N Prioritized Actions", "int",
                      default=config.N_PRIORITIZED_ACTIONS, min=1, max=200,
                      description="Total number of actions returned"),
            ParamSpec("min_line_reconnections", "Min Line Reconnections", "int",
                      default=config.MIN_LINE_RECONNECTIONS, min=0, max=20),
            ParamSpec("min_close_coupling", "Min Close Coupling", "int",
                      default=config.MIN_CLOSE_COUPLING, min=0, max=20),
            ParamSpec("min_open_coupling", "Min Open Coupling", "int",
                      default=config.MIN_OPEN_COUPLING, min=0, max=20),
            ParamSpec("min_line_disconnections", "Min Line Disconnections", "int",
                      default=config.MIN_LINE_DISCONNECTIONS, min=0, max=20),
            ParamSpec("min_pst", "Min PST Actions", "int",
                      default=config.MIN_PST, min=0, max=20),
            ParamSpec("min_load_shedding", "Min Load Shedding", "int",
                      default=config.MIN_LOAD_SHEDDING, min=0, max=20),
            ParamSpec("min_renewable_curtailment_actions",
                      "Min Renewable Curtailment", "int",
                      default=config.MIN_RENEWABLE_CURTAILMENT, min=0, max=20),
            ParamSpec("ignore_reconnections", "Ignore Reconnections", "bool",
                      default=config.IGNORE_RECONNECTIONS),
        ]

    def recommend(self, inputs: RecommenderInputs, params: dict) -> RecommenderOutput:
        """Run the existing expert pipeline on the overflow-graph context.

        Implementation note: delegates to the legacy helper
        :func:`_run_expert_discovery` in
        :mod:`expert_op4grid_recommender.main` because the expert
        discovery is tightly coupled to internal context (validator,
        graph pre-processing, simulation helpers). External models
        rebuild a clean path via ``inputs`` instead — the ``_context``
        escape hatch is private to this class.
        """
        if inputs._context is None:
            raise RuntimeError(
                "ExpertRecommender requires the full analysis context. "
                "It is meant to be called from run_analysis_step2_discovery, "
                "not directly by external code."
            )
        from expert_op4grid_recommender.main import _run_expert_discovery
        n_action_max = int(params.get("n_prioritized_actions",
                                      config.N_PRIORITIZED_ACTIONS))
        prioritized_actions, action_scores = _run_expert_discovery(
            inputs._context, n_action_max=n_action_max,
        )
        return RecommenderOutput(
            prioritized_actions=prioritized_actions,
            action_scores=action_scores,
        )
