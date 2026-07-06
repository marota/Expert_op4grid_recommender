# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for the shared injection base + memoised disconnection bounds (R5)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from expert_op4grid_recommender.action_evaluation.discovery._base import DiscovererBase
from expert_op4grid_recommender.action_evaluation.discovery._injection_base import (
    InjectionDiscoveryBase,
    InjectionOverloadContext,
)
from expert_op4grid_recommender.action_evaluation.discovery._results import (
    DisconnectionBounds,
)


# ---------------------------------------------------------------------------
# _injection_influence_factor (pure)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flow,ref,expected", [
    (50.0, 100.0, 0.5),      # partial
    (100.0, 100.0, 1.0),     # exactly saturating
    (150.0, 100.0, 1.0),     # clamped to 1.0
    (0.0, 100.0, 0.0),       # no influence
    (50.0, 0.0, 0.0),        # zero reference -> 0.0 (no div-by-zero)
    (50.0, -1.0, 0.0),       # non-positive reference -> 0.0
])
def test_injection_influence_factor(flow, ref, expected):
    assert InjectionDiscoveryBase._injection_influence_factor(flow, ref) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _injection_overload_context
# ---------------------------------------------------------------------------

class _StubInjection(InjectionDiscoveryBase):
    """Minimal host exposing exactly what the shared preamble reads."""

    MARGIN_KEY = "LOAD_SHEDDING_MARGIN"
    MIN_MW_KEY = "LOAD_SHEDDING_MIN_MW"

    def __init__(self, capacity_map, rho, name_line, overloaded_ids):
        self._cap = capacity_map
        self.obs_defaut = SimpleNamespace(
            name_line=list(name_line), rho=np.array(rho, dtype=float)
        )
        self.lines_overloaded_ids = list(overloaded_ids)

    def _build_lookup_caches(self):
        pass

    def _get_edge_data_cache(self):
        pass

    def _build_line_capacity_map(self):
        return self._cap


def test_context_none_when_no_capacity_map():
    stub = _StubInjection({}, [1.5], ["L0"], [0])
    assert stub._injection_overload_context() is None


def test_context_none_when_not_overloaded():
    # rho_max <= 1.0 -> no active overload -> None.
    stub = _StubInjection({"L0": 100.0}, [0.9, 1.0], ["L0", "L1"], [0, 1])
    assert stub._injection_overload_context() is None


def test_context_happy_path_uses_overloaded_line_capacity():
    # Overloaded line L1 has capacity 200; rho_max = 1.3 -> excess = 0.3 * 200 = 60.
    stub = _StubInjection(
        capacity_map={"L0": 500.0, "L1": 200.0},
        rho=[0.5, 1.3],
        name_line=["L0", "L1"],
        overloaded_ids=[1],
    )
    ctx = stub._injection_overload_context()
    assert isinstance(ctx, InjectionOverloadContext)
    assert ctx.max_overload_flow == pytest.approx(200.0)
    assert ctx.P_overload_excess == pytest.approx(0.3 * 200.0)
    assert ctx.margin == pytest.approx(0.05)   # LOAD_SHEDDING_MARGIN default
    assert ctx.min_mw == pytest.approx(1.0)     # LOAD_SHEDDING_MIN_MW default
    assert ctx.obs is stub.obs_defaut


def test_context_falls_back_to_max_capacity_when_overloaded_line_absent():
    # Overloaded id maps to a line not in the capacity map -> fall back to the
    # global max capacity (500).
    stub = _StubInjection(
        capacity_map={"L0": 500.0, "L1": 200.0},
        rho=[1.2, 0.4],
        name_line=["MISSING", "L1"],
        overloaded_ids=[0],
    )
    ctx = stub._injection_overload_context()
    assert ctx.max_overload_flow == pytest.approx(500.0)
    assert ctx.P_overload_excess == pytest.approx(0.2 * 500.0)


# ---------------------------------------------------------------------------
# _get_disconnection_bounds — memoised once per run (A5 PST-coupling fix)
# ---------------------------------------------------------------------------

class _StubDisco:
    """Bare host for DiscovererBase._get_disconnection_bounds."""

    def __init__(self):
        self.compute_calls = 0
        self.capacity_calls = 0

    def _compute_disconnection_flow_bounds(self):
        self.compute_calls += 1
        return (100.0, 10.0, 50.0)

    def _build_line_capacity_map(self):
        self.capacity_calls += 1
        return {"L1": 40.0}


def test_get_disconnection_bounds_returns_frozen_holder():
    stub = _StubDisco()
    bounds = DiscovererBase._get_disconnection_bounds(stub)
    assert isinstance(bounds, DisconnectionBounds)
    assert bounds.max_overload_flow == 100.0
    assert bounds.min_redispatch == 10.0
    assert bounds.max_redispatch == 50.0
    assert bounds.capacity_map == {"L1": 40.0}


def test_get_disconnection_bounds_is_memoised():
    """Both disconnection and PST scoring call this; it must compute once."""
    stub = _StubDisco()
    b1 = DiscovererBase._get_disconnection_bounds(stub)
    b2 = DiscovererBase._get_disconnection_bounds(stub)
    assert b1 is b2                      # same cached object
    assert stub.compute_calls == 1       # not recomputed
    assert stub.capacity_calls == 1
