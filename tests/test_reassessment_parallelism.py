# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""Container-aware CPU detection + reassessment parallelism gating.

Guards the fix for the 2-vCPU HuggingFace deployment: ``os.cpu_count()``
reports the *host* core count (e.g. 16) inside a CPU-limited container, so the
old worker-count math spawned ~10 threads that over-subscribed 2 vCPUs and made
the reassessment SLOWER than serial. These tests pin the detection and the
auto/force gating so that regression can't silently return.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from expert_op4grid_recommender.utils import reassessment as R


# ---------------------------------------------------------------------
# _read_cgroup_cpu_quota
# ---------------------------------------------------------------------

def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="ascii")
    return str(p)


def test_cgroup_v2_quota_two_cpus(tmp_path):
    cpu_max = _write(tmp_path, "cpu.max", "200000 100000")
    with patch.object(R, "_CGROUP_V2_CPU_MAX", cpu_max), \
         patch.object(R, "_CGROUP_V1_CPU_QUOTA", "/nonexistent"):
        assert R._read_cgroup_cpu_quota() == pytest.approx(2.0)


def test_cgroup_v2_unlimited_returns_none(tmp_path):
    cpu_max = _write(tmp_path, "cpu.max", "max 100000")
    with patch.object(R, "_CGROUP_V2_CPU_MAX", cpu_max), \
         patch.object(R, "_CGROUP_V1_CPU_QUOTA", "/nonexistent"):
        assert R._read_cgroup_cpu_quota() is None


def test_cgroup_v1_quota_two_cpus(tmp_path):
    quota = _write(tmp_path, "quota", "200000")
    period = _write(tmp_path, "period", "100000")
    with patch.object(R, "_CGROUP_V2_CPU_MAX", "/nonexistent"), \
         patch.object(R, "_CGROUP_V1_CPU_QUOTA", quota), \
         patch.object(R, "_CGROUP_V1_CPU_PERIOD", period):
        assert R._read_cgroup_cpu_quota() == pytest.approx(2.0)


def test_cgroup_v1_unlimited_returns_none(tmp_path):
    quota = _write(tmp_path, "quota", "-1")
    period = _write(tmp_path, "period", "100000")
    with patch.object(R, "_CGROUP_V2_CPU_MAX", "/nonexistent"), \
         patch.object(R, "_CGROUP_V1_CPU_QUOTA", quota), \
         patch.object(R, "_CGROUP_V1_CPU_PERIOD", period):
        assert R._read_cgroup_cpu_quota() is None


def test_cgroup_absent_returns_none():
    with patch.object(R, "_CGROUP_V2_CPU_MAX", "/nonexistent/cpu.max"), \
         patch.object(R, "_CGROUP_V1_CPU_QUOTA", "/nonexistent/quota"):
        assert R._read_cgroup_cpu_quota() is None


# ---------------------------------------------------------------------
# _effective_cpu_count
# ---------------------------------------------------------------------

def test_effective_cpu_uses_cgroup_quota_over_host_count():
    """The 2-vCPU-on-a-16-core-host scenario: quota wins."""
    with patch("os.cpu_count", return_value=16), \
         patch.object(R.os, "sched_getaffinity", return_value=set(range(16)), create=True), \
         patch.object(R, "_read_cgroup_cpu_quota", return_value=2.0):
        assert R._effective_cpu_count() == 2


def test_effective_cpu_uses_affinity_mask():
    """A cpuset-pinned container (affinity < host) is respected too."""
    with patch("os.cpu_count", return_value=16), \
         patch.object(R.os, "sched_getaffinity", return_value={0, 1, 2}, create=True), \
         patch.object(R, "_read_cgroup_cpu_quota", return_value=None):
        assert R._effective_cpu_count() == 3


def test_effective_cpu_falls_back_to_host_count():
    with patch("os.cpu_count", return_value=8), \
         patch.object(R.os, "sched_getaffinity", return_value=set(range(8)), create=True), \
         patch.object(R, "_read_cgroup_cpu_quota", return_value=None):
        assert R._effective_cpu_count() == 8


def test_effective_cpu_floored_at_one():
    with patch("os.cpu_count", return_value=1), \
         patch.object(R.os, "sched_getaffinity", return_value=set(), create=True), \
         patch.object(R, "_read_cgroup_cpu_quota", return_value=None):
        assert R._effective_cpu_count() == 1


# ---------------------------------------------------------------------
# _reassessment_worker_count — raw, container-aware pool size
# (whether to actually use >1 worker is decided by the gate below)
# ---------------------------------------------------------------------

def test_worker_count_capped_at_ten():
    with patch.object(R, "_effective_cpu_count", return_value=16):
        cores, workers = R._reassessment_worker_count(15)
    assert cores == 16
    assert workers == R._MAX_REASSESSMENT_WORKERS  # 10


def test_worker_count_never_exceeds_actions():
    with patch.object(R, "_effective_cpu_count", return_value=8):
        assert R._reassessment_worker_count(3) == (8, 3)


def test_worker_count_two_vcpu_is_two():
    with patch.object(R, "_effective_cpu_count", return_value=2):
        assert R._reassessment_worker_count(15) == (2, 2)


def test_worker_count_single_core_is_one():
    with patch.object(R, "_effective_cpu_count", return_value=1):
        assert R._reassessment_worker_count(15) == (1, 1)


# ---------------------------------------------------------------------
# _should_parallelize_reassessment — the single gate (force knob + threshold)
# ---------------------------------------------------------------------

def test_gate_auto_two_vcpu_host_goes_serial():
    """THE HuggingFace 2-vCPU case: on a 2-core box the pool is 2 workers, and
    auto mode keeps it serial because 2 < the min-parallel threshold (4)."""
    with patch.object(R.config, "REASSESSMENT_PARALLEL", None), \
         patch.object(R, "_min_parallel_workers", return_value=4):
        assert R._should_parallelize_reassessment(True, workers=2, n_actions=15) is False


def test_gate_auto_many_cores_parallelises():
    with patch.object(R.config, "REASSESSMENT_PARALLEL", None), \
         patch.object(R, "_min_parallel_workers", return_value=4):
        assert R._should_parallelize_reassessment(True, workers=10, n_actions=15) is True


def test_gate_force_serial_overrides_plentiful_cores():
    with patch.object(R.config, "REASSESSMENT_PARALLEL", False):
        assert R._should_parallelize_reassessment(True, workers=10, n_actions=20) is False


def test_gate_force_parallel_overrides_low_cores():
    """Force=True lets a 3-worker pool parallelise despite being below the
    auto-mode minimum."""
    with patch.object(R.config, "REASSESSMENT_PARALLEL", True):
        assert R._should_parallelize_reassessment(True, workers=3, n_actions=15) is True


def test_gate_single_worker_is_serial_even_when_forced():
    with patch.object(R.config, "REASSESSMENT_PARALLEL", True):
        assert R._should_parallelize_reassessment(True, workers=1, n_actions=15) is False


def test_gate_non_pypowsybl_is_serial():
    with patch.object(R.config, "REASSESSMENT_PARALLEL", True):
        assert R._should_parallelize_reassessment(False, workers=10, n_actions=15) is False


def test_gate_needs_at_least_two_actions():
    with patch.object(R.config, "REASSESSMENT_PARALLEL", None), \
         patch.object(R, "_min_parallel_workers", return_value=2):
        assert R._should_parallelize_reassessment(True, workers=8, n_actions=1) is False
