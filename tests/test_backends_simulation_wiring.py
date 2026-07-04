# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""R4 — SimulationBackend wiring onto the unified utils.simulation module.

Verifies that each backend forwards the *right* backend contract to the single
backend-agnostic simulation module: the pypowsybl backend threads
``keep_variant`` / ``fast_mode`` via ``simulate_kwargs`` and branches candidates
from the kept baseline (``reapply_contingency=False``), while the grid2op backend
uses the healthy-N-state defaults. Because the backend methods import the module
functions lazily, we patch them on ``expert_op4grid_recommender.utils.simulation``.
"""
from __future__ import annotations

from unittest.mock import patch

from expert_op4grid_recommender.backends import (
    Backend,
    Grid2opBackend,
    PypowsyblBackend,
    make_backend,
)


# ---------------------------------------------------------------------------
# pypowsybl simulate_kwargs shape
# ---------------------------------------------------------------------------

def test_pypowsybl_simulate_kwargs_helpers_thread_fast_mode():
    b = PypowsyblBackend(fast_mode=True)
    assert b._sk_kept == {"keep_variant": True, "fast_mode": True}
    assert b._sk == {"fast_mode": True}
    b2 = PypowsyblBackend(fast_mode=False)
    assert b2._sk_kept == {"keep_variant": True, "fast_mode": False}
    assert b2._sk == {"fast_mode": False}


def test_pypowsybl_simulate_contingency_passes_keep_variant():
    b = make_backend(Backend.PYPOWSYBL, fast_mode=True)
    with patch("expert_op4grid_recommender.utils.simulation.simulate_contingency") as m:
        m.return_value = ("OBS", True)
        b.simulate_contingency("env", "obs", ["L0"], "M", 0)
    assert m.call_args.kwargs["simulate_kwargs"] == {"keep_variant": True, "fast_mode": True}


def test_pypowsybl_check_simu_overloads_passes_fast_mode_only():
    b = make_backend(Backend.PYPOWSYBL, fast_mode=False)
    with patch("expert_op4grid_recommender.utils.simulation.check_simu_overloads") as m:
        m.return_value = (True, False)
        b.check_simu_overloads("obs", "obs_d", "asp", 0, ["L0"], [0], [])
    assert m.call_args.kwargs["simulate_kwargs"] == {"fast_mode": False}


def test_pypowsybl_compute_baseline_keeps_variant():
    b = make_backend(Backend.PYPOWSYBL, fast_mode=True)
    with patch("expert_op4grid_recommender.utils.simulation.compute_baseline_simulation") as m:
        m.return_value = (None, None)
        b.compute_baseline("obs", 0, "D", "M", [0])
    assert m.call_args.kwargs["simulate_kwargs"] == {"keep_variant": True, "fast_mode": True}


def test_pypowsybl_check_rho_with_baseline_does_not_reapply_contingency():
    b = make_backend(Backend.PYPOWSYBL, fast_mode=True)
    with patch(
        "expert_op4grid_recommender.utils.simulation.check_rho_reduction_with_baseline"
    ) as m:
        m.return_value = (True, "OBS")
        b.check_rho_with_baseline("obs", 0, "D", "A", [0], "M", "RHO")
    assert m.call_args.kwargs["reapply_contingency"] is False
    assert m.call_args.kwargs["simulate_kwargs"] == {"fast_mode": True}


def test_pypowsybl_check_rho_reduction_branches_from_baseline():
    b = make_backend(Backend.PYPOWSYBL, fast_mode=True)
    with patch("expert_op4grid_recommender.utils.simulation.check_rho_reduction") as m:
        m.return_value = (False, None)
        b.check_rho_reduction("obs", 0, "D", "A", [0], "M")
    kw = m.call_args.kwargs
    assert kw["reapply_contingency"] is False
    assert kw["baseline_simulate_kwargs"] == {"keep_variant": True, "fast_mode": True}
    assert kw["candidate_simulate_kwargs"] == {"fast_mode": True}


# ---------------------------------------------------------------------------
# grid2op uses the healthy-N-state defaults (no pypowsybl kwargs)
# ---------------------------------------------------------------------------

def test_grid2op_flags_use_n_state_defaults():
    b = Grid2opBackend(fast_mode=True)
    assert b.is_pypowsybl is False
    assert b.branch_candidates_from_baseline is False
    assert b.use_shared_baseline_for_topological is False


def test_grid2op_check_rho_reduction_uses_defaults():
    b = make_backend(Backend.GRID2OP)
    with patch("expert_op4grid_recommender.utils.simulation.check_rho_reduction") as m:
        m.return_value = (False, None)
        b.check_rho_reduction("obs", 0, "D", "A", [0], "M", "care")
    # grid2op passes only the positional args; the reapply/branch defaults
    # (reapply_contingency=True, empty simulate_kwargs) are the module defaults.
    assert "reapply_contingency" not in m.call_args.kwargs
    assert m.call_args.args == ("obs", 0, "D", "A", [0], "M", "care")
