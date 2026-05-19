# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for the ``_run_ac_with_init_fallback`` DC_VALUES retry on
*non-converged status* (in addition to synchronous PowsyblException).

Background
----------
On large grids with active transformer / shunt voltage control, an AC
LF seeded with ``PREVIOUS_VALUES`` after a topology mutation (e.g.
node-merging that closes a coupler with non-trivial pre-merge angle
gap between the two buses) can return a ``ComponentResult`` whose
``status`` is ``FAILED`` ("Unrealistic state" reached by
OpenLoadFlow's voltage-control consistency check) or
``MAX_ITERATION_REACHED`` — *without* raising a Python exception.

The original fallback (commit 22e8a39e) only kicked in on synchronous
``PowsyblException``, so these silent failures were propagated upstream
as ``non_convergence`` flags on the action card, even though a
``DC_VALUES`` seed would have converged in a handful of outer-loop
iterations.

This module guards three scenarios:

1. ``FAILED`` from PREVIOUS_VALUES → retry with DC_VALUES.
2. ``MAX_ITERATION_REACHED`` from PREVIOUS_VALUES → retry with DC_VALUES.
3. ``CONVERGED`` from PREVIOUS_VALUES → no retry (regression guard
   for the warm-start path).

It also guards the bumped ``maxOuterLoopIterations`` default (20 → 100)
so the post-merge ``IncrementalTransformerVoltageControl`` outer loop
has room to settle once seeded with DC_VALUES.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pypowsybl = pytest.importorskip("pypowsybl")
import pypowsybl.loadflow as lf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(status):
    r = MagicMock()
    r.status = status
    return r


@pytest.fixture
def nm():
    from expert_op4grid_recommender.pypowsybl_backend import NetworkManager
    network = pypowsybl.network.create_ieee9()
    return NetworkManager(network=network)


# ---------------------------------------------------------------------------
# Default LF parameters: maxOuterLoopIterations bump
# ---------------------------------------------------------------------------

class TestDefaultLfParametersOuterLoopCap:
    """Regression guard: the default ``maxOuterLoopIterations`` must be
    raised from OpenLoadFlow's stock 20 so the post-merge tap-changer
    outer loop has room to converge under a DC_VALUES seed.
    """

    def test_max_outer_loop_iterations_is_at_least_40(self, nm):
        pp = nm.lf_parameters.provider_parameters
        raw = pp.get("maxOuterLoopIterations")
        assert raw is not None, (
            "Default lf_parameters.provider_parameters must declare "
            "maxOuterLoopIterations explicitly; stock OpenLoadFlow default "
            "of 20 is too low for post-node-merging AC convergence."
        )
        assert int(raw) >= 40, (
            f"maxOuterLoopIterations is {raw}; empirically slow + DC_VALUES "
            "needs >= 40 outer iterations on PYMONP3 / P.SAOL31RONCI."
        )


# ---------------------------------------------------------------------------
# _run_ac_with_init_fallback: retry on non-converged status
# ---------------------------------------------------------------------------

class TestInitFallbackOnNonConvergedStatus:
    """``_run_ac_with_init_fallback`` must retry with DC_VALUES whenever
    the first attempt (PREVIOUS_VALUES) returns a non-CONVERGED status.
    """

    def test_failed_status_triggers_dc_values_retry(self, nm):
        """status=FAILED from PREVIOUS_VALUES → second call with DC_VALUES."""
        seq = [
            [_make_result(lf.ComponentStatus.FAILED)],          # 1st: PREVIOUS
            [_make_result(lf.ComponentStatus.CONVERGED)],       # 2nd: DC_VALUES
        ]
        captured_modes = []

        def fake_run_ac(network, parameters=None):
            captured_modes.append(parameters.voltage_init_mode)
            return seq.pop(0)

        with patch(
            "expert_op4grid_recommender.pypowsybl_backend.network_manager.lf.run_ac",
            side_effect=fake_run_ac,
        ) as mock_run_ac:
            out = nm._run_ac_with_init_fallback(nm.lf_parameters)

        assert mock_run_ac.call_count == 2, (
            f"Expected 2 run_ac calls (PREVIOUS → DC_VALUES retry), "
            f"got {mock_run_ac.call_count}"
        )
        assert captured_modes[0] == lf.VoltageInitMode.PREVIOUS_VALUES
        assert captured_modes[1] == lf.VoltageInitMode.DC_VALUES
        assert out[0].status == lf.ComponentStatus.CONVERGED

    def test_max_iteration_status_triggers_dc_values_retry(self, nm):
        """status=MAX_ITERATION_REACHED from PREVIOUS_VALUES → DC_VALUES retry."""
        seq = [
            [_make_result(lf.ComponentStatus.MAX_ITERATION_REACHED)],
            [_make_result(lf.ComponentStatus.CONVERGED)],
        ]
        captured_modes = []

        def fake_run_ac(network, parameters=None):
            captured_modes.append(parameters.voltage_init_mode)
            return seq.pop(0)

        with patch(
            "expert_op4grid_recommender.pypowsybl_backend.network_manager.lf.run_ac",
            side_effect=fake_run_ac,
        ) as mock_run_ac:
            out = nm._run_ac_with_init_fallback(nm.lf_parameters)

        assert mock_run_ac.call_count == 2
        assert captured_modes == [
            lf.VoltageInitMode.PREVIOUS_VALUES,
            lf.VoltageInitMode.DC_VALUES,
        ]
        assert out[0].status == lf.ComponentStatus.CONVERGED

    def test_converged_first_try_does_not_trigger_retry(self, nm):
        """The warm-start path must NOT pay the DC_VALUES retry cost when
        PREVIOUS_VALUES already converged.
        """
        seq = [[_make_result(lf.ComponentStatus.CONVERGED)]]

        def fake_run_ac(network, parameters=None):
            return seq.pop(0)

        with patch(
            "expert_op4grid_recommender.pypowsybl_backend.network_manager.lf.run_ac",
            side_effect=fake_run_ac,
        ) as mock_run_ac:
            out = nm._run_ac_with_init_fallback(nm.lf_parameters)

        assert mock_run_ac.call_count == 1, (
            "CONVERGED on the first try must not trigger a DC_VALUES retry"
        )
        assert out[0].status == lf.ComponentStatus.CONVERGED

    def test_dc_values_init_with_failed_status_does_not_retry(self, nm):
        """If we are already on DC_VALUES, we have nowhere left to retry —
        the bad status must propagate as-is rather than loop forever.
        """
        params = lf.Parameters.from_json(nm.lf_parameters.to_json())
        params.voltage_init_mode = lf.VoltageInitMode.DC_VALUES

        seq = [[_make_result(lf.ComponentStatus.FAILED)]]

        def fake_run_ac(network, parameters=None):
            return seq.pop(0)

        with patch(
            "expert_op4grid_recommender.pypowsybl_backend.network_manager.lf.run_ac",
            side_effect=fake_run_ac,
        ) as mock_run_ac:
            out = nm._run_ac_with_init_fallback(params)

        assert mock_run_ac.call_count == 1, (
            "Already-DC_VALUES init must not trigger another DC_VALUES retry "
            "even on FAILED status (no second seed to try)"
        )
        assert out[0].status == lf.ComponentStatus.FAILED

    def test_exception_path_still_triggers_dc_values_retry(self, nm):
        """Regression guard: the pre-existing exception-based retry
        (commit 22e8a39e, v0.2.0) must remain intact.
        """
        captured_modes = []

        def fake_run_ac(network, parameters=None):
            captured_modes.append(parameters.voltage_init_mode)
            if len(captured_modes) == 1:
                raise Exception("Voltage magnitude is undefined for bus 'X_0'")
            return [_make_result(lf.ComponentStatus.CONVERGED)]

        with patch(
            "expert_op4grid_recommender.pypowsybl_backend.network_manager.lf.run_ac",
            side_effect=fake_run_ac,
        ) as mock_run_ac:
            out = nm._run_ac_with_init_fallback(nm.lf_parameters)

        assert mock_run_ac.call_count == 2
        assert captured_modes == [
            lf.VoltageInitMode.PREVIOUS_VALUES,
            lf.VoltageInitMode.DC_VALUES,
        ]
        assert out[0].status == lf.ComponentStatus.CONVERGED

    def test_dc_values_init_exception_propagates(self, nm):
        """Already-DC_VALUES init + exception must re-raise (no second seed
        to fall back to).
        """
        params = lf.Parameters.from_json(nm.lf_parameters.to_json())
        params.voltage_init_mode = lf.VoltageInitMode.DC_VALUES

        def fake_run_ac(network, parameters=None):
            raise Exception("boom")

        with patch(
            "expert_op4grid_recommender.pypowsybl_backend.network_manager.lf.run_ac",
            side_effect=fake_run_ac,
        ):
            with pytest.raises(Exception, match="boom"):
                nm._run_ac_with_init_fallback(params)

    def test_retry_does_not_mutate_shared_lf_parameters(self, nm):
        """The DC_VALUES retry must copy params, not mutate the singleton."""
        seq = [
            [_make_result(lf.ComponentStatus.FAILED)],
            [_make_result(lf.ComponentStatus.CONVERGED)],
        ]

        def fake_run_ac(network, parameters=None):
            return seq.pop(0)

        with patch(
            "expert_op4grid_recommender.pypowsybl_backend.network_manager.lf.run_ac",
            side_effect=fake_run_ac,
        ):
            nm._run_ac_with_init_fallback(nm.lf_parameters)

        # The shared lf_parameters object must remain on PREVIOUS_VALUES.
        assert (
            nm.lf_parameters.voltage_init_mode
            == lf.VoltageInitMode.PREVIOUS_VALUES
        ), (
            "DC_VALUES retry leaked into the shared NetworkManager.lf_parameters; "
            "every subsequent LF would start cold instead of warm-starting."
        )
