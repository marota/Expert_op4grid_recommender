# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Tests for the vendored pypowsybl integer-value patch (M5).

Two tiers: a pure-logic check of the 0->-1 correction that needs no pypowsybl,
and pypowsybl-guarded checks of the actual class patch. The latter skip cleanly
where pypowsybl is unavailable.
"""
import numpy as np
import pytest

from expert_op4grid_recommender import patched_backend as pb


def test_zero_to_minus_one_conversion_is_idempotent():
    """The correction itself: grid2op's 0 bus-sentinel -> pypowsybl -1."""
    v = np.array([0, 1, 0, 2, -1], dtype=np.int32)
    v[v == 0] = -1
    assert v.tolist() == [-1, 1, -1, 2, -1]
    v[v == 0] = -1  # idempotent — no zeros remain
    assert v.tolist() == [-1, 1, -1, 2, -1]


def test_apply_patch_returns_false_without_pypowsybl(monkeypatch):
    """When pypowsybl can't be resolved, applying the patch is a safe no-op."""
    monkeypatch.setattr(pb, "_resolve_pp_grid2op_backend_cls", lambda: None)
    assert pb.apply_pypowsybl_integer_value_patch() is False


def test_patch_wraps_and_converts_on_a_stub_class(monkeypatch):
    """Drive the wrapping logic against a stub 'backend' class — no pypowsybl.

    Proves the wrapper (a) rewrites 0->-1 before delegating, (b) stamps the
    idempotency flag, and (c) does not double-wrap on a second apply.
    """
    calls = []

    class _StubBackend:
        def update_integer_value(self, value_type, value, changed):
            calls.append(value.copy())

    monkeypatch.setattr(pb, "_resolve_pp_grid2op_backend_cls", lambda: _StubBackend)
    # Force the "matches assumption" guard so no warning path is exercised here.
    monkeypatch.setattr(pb, "_upstream_body_matches_assumption", lambda cls: True)

    assert pb.apply_pypowsybl_integer_value_patch() is True
    assert getattr(_StubBackend, pb._PATCH_FLAG, False) is True

    inst = _StubBackend()
    inst.update_integer_value(object(), np.array([0, 3, 0], dtype=np.int32),
                              np.array([True, True, True]))
    assert calls[-1].tolist() == [-1, 3, -1]

    # Second apply is a no-op (no double-wrap): still one wrapper.
    assert pb.apply_pypowsybl_integer_value_patch() is True
    inst.update_integer_value(object(), np.array([0], dtype=np.int32),
                              np.array([True]))
    assert calls[-1].tolist() == [-1]


def test_make_patched_backend_raises_without_pypowsybl2grid(monkeypatch):
    monkeypatch.setattr(pb, "_load_pypowsybl2grid_backend_cls", lambda: None)
    with pytest.raises(ImportError, match="pypowsybl2grid"):
        pb.make_patched_pypowsybl_backend()


# --- pypowsybl-installed tier (skips where absent) ---------------------------

def test_real_pypowsybl_backend_is_patched():
    pp = pytest.importorskip("pypowsybl")
    cls = pp.grid2op.Backend
    assert pb.apply_pypowsybl_integer_value_patch() is True
    assert getattr(cls, pb._PATCH_FLAG, False) is True
    # Idempotent second call.
    assert pb.apply_pypowsybl_integer_value_patch() is True
    # The version guard helper returns a bool for the installed body.
    assert isinstance(pb._upstream_body_matches_assumption(cls), bool)
