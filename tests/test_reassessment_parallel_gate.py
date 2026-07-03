# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Tests for the reassessment parallel-vs-serial gate (0.2.7.post1).

The parallel per-action reassessment clones the whole network per worker, a
fixed overhead only amortized above ~3-4 cores. Below the threshold the serial
path is faster (measured: ~42 s vs ~9 s for the same 15-action pan-European
case on a 2-vCPU Space vs a many-core Mac). These tests pin the gate so a
2-vCPU host stays serial while ≥4-core hosts keep the speed-up, plus the env
override.
"""
from __future__ import annotations

import pytest

from expert_op4grid_recommender.utils.reassessment import (
    _DEFAULT_MIN_PARALLEL_WORKERS,
    _MIN_PARALLEL_WORKERS_ENV,
    _min_parallel_workers,
    _should_parallelize_reassessment,
)


class TestMinParallelWorkers:
    def test_default_is_four(self, monkeypatch):
        monkeypatch.delenv(_MIN_PARALLEL_WORKERS_ENV, raising=False)
        assert _min_parallel_workers() == _DEFAULT_MIN_PARALLEL_WORKERS == 4

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(_MIN_PARALLEL_WORKERS_ENV, "2")
        assert _min_parallel_workers() == 2
        monkeypatch.setenv(_MIN_PARALLEL_WORKERS_ENV, "99")
        assert _min_parallel_workers() == 99

    def test_env_floored_at_two(self, monkeypatch):
        # A single worker is strictly worse than serial → never below 2.
        monkeypatch.setenv(_MIN_PARALLEL_WORKERS_ENV, "1")
        assert _min_parallel_workers() == 2
        monkeypatch.setenv(_MIN_PARALLEL_WORKERS_ENV, "0")
        assert _min_parallel_workers() == 2

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(_MIN_PARALLEL_WORKERS_ENV, "not-a-number")
        assert _min_parallel_workers() == _DEFAULT_MIN_PARALLEL_WORKERS


class TestShouldParallelize:
    def test_two_cores_stays_serial_by_default(self, monkeypatch):
        # HuggingFace 2-vCPU Space: workers=2 → serial (the fix).
        monkeypatch.delenv(_MIN_PARALLEL_WORKERS_ENV, raising=False)
        assert _should_parallelize_reassessment(True, workers=2, n_actions=15) is False

    def test_four_cores_parallelizes(self, monkeypatch):
        # The v0.2.6-measured positive case (+1.43x on a 4-core host).
        monkeypatch.delenv(_MIN_PARALLEL_WORKERS_ENV, raising=False)
        assert _should_parallelize_reassessment(True, workers=4, n_actions=15) is True

    def test_many_cores_parallelizes(self, monkeypatch):
        monkeypatch.delenv(_MIN_PARALLEL_WORKERS_ENV, raising=False)
        assert _should_parallelize_reassessment(True, workers=9, n_actions=15) is True

    def test_grid2op_never_parallelizes(self, monkeypatch):
        monkeypatch.delenv(_MIN_PARALLEL_WORKERS_ENV, raising=False)
        assert _should_parallelize_reassessment(False, workers=16, n_actions=15) is False

    def test_single_action_stays_serial(self, monkeypatch):
        monkeypatch.delenv(_MIN_PARALLEL_WORKERS_ENV, raising=False)
        assert _should_parallelize_reassessment(True, workers=8, n_actions=1) is False

    def test_env_can_force_serial_on_any_host(self, monkeypatch):
        monkeypatch.setenv(_MIN_PARALLEL_WORKERS_ENV, "99")
        assert _should_parallelize_reassessment(True, workers=16, n_actions=64) is False

    def test_env_can_restore_aggressive_behaviour(self, monkeypatch):
        # Set to 2 → the pre-0.2.7.post1 behaviour (parallel from 2 workers).
        monkeypatch.setenv(_MIN_PARALLEL_WORKERS_ENV, "2")
        assert _should_parallelize_reassessment(True, workers=2, n_actions=15) is True


@pytest.mark.parametrize("workers,expected", [(1, False), (2, False), (3, False),
                                              (4, True), (8, True)])
def test_default_threshold_boundary(monkeypatch, workers, expected):
    monkeypatch.delenv(_MIN_PARALLEL_WORKERS_ENV, raising=False)
    assert _should_parallelize_reassessment(True, workers=workers, n_actions=15) is expected
