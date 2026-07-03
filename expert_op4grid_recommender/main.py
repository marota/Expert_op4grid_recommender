#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.
"""Backward-compatible facade over the split pipeline / CLI / backends.

``main.py`` used to be the entry point, the pipeline core AND the import hub
all at once (the source of the three import cycles the review flagged). The
pipeline now lives in :mod:`expert_op4grid_recommender.pipeline`, the CLI in
:mod:`expert_op4grid_recommender.cli`, and the per-backend physics in
:mod:`expert_op4grid_recommender.backends`.

This module is kept as a thin re-export facade so every historical
``from expert_op4grid_recommender.main import ...`` keeps resolving — including
the two ``simulate_contingency_*`` helpers a couple of external callers/tests
still import directly.
"""
from typing import Any, List

# Backends (protocol + implementations + enum)
from expert_op4grid_recommender.backends import (
    Backend,
    Grid2opBackend,
    PypowsyblBackend,
    SimulationBackend,
    make_backend,
)

# Typed pipeline spine + the analysis functions
from expert_op4grid_recommender.pipeline import (
    AnalysisContext,
    AnalysisResult,
    run_analysis,
    run_analysis_step1,
    run_analysis_step2,
    run_analysis_step2_discovery,
    run_analysis_step2_graph,
    set_thermal_limits,
)

# Expert discovery helpers (moved under models/) — re-exported for any
# external caller that still references ``main._run_expert_discovery``.
from expert_op4grid_recommender.models._expert_discovery import (
    _run_expert_action_filter,
    _run_expert_discovery,
)

# CLI entry point
from expert_op4grid_recommender.cli import main


def simulate_contingency_grid2op(env, obs, lines_defaut, act_reco_maintenance, timestep):
    """Backward-compatible grid2op contingency-simulation shim.

    Kept for the handful of external callers that import this name from
    ``main`` directly; the pipeline itself now goes through
    :class:`~expert_op4grid_recommender.backends.Grid2opBackend`.
    """
    from expert_op4grid_recommender.utils.simulation import simulate_contingency
    return simulate_contingency(env, obs, lines_defaut, act_reco_maintenance, timestep)


def simulate_contingency_pypowsybl(env, obs, lines_defaut, act_reco_maintenance,
                                   timestep, fast_mode: bool = True):
    """Backward-compatible pypowsybl contingency-simulation shim."""
    from expert_op4grid_recommender.utils.simulation_pypowsybl import simulate_contingency
    return simulate_contingency(env, obs, lines_defaut, act_reco_maintenance,
                                timestep, fast_mode=fast_mode)


__all__: List[Any] = [
    "Backend",
    "SimulationBackend",
    "Grid2opBackend",
    "PypowsyblBackend",
    "make_backend",
    "AnalysisContext",
    "AnalysisResult",
    "set_thermal_limits",
    "run_analysis",
    "run_analysis_step1",
    "run_analysis_step2",
    "run_analysis_step2_graph",
    "run_analysis_step2_discovery",
    "simulate_contingency_grid2op",
    "simulate_contingency_pypowsybl",
    "_run_expert_action_filter",
    "_run_expert_discovery",
    "main",
]


if __name__ == "__main__":
    main()
