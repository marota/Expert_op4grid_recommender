# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for the ``voltage_init_mode`` wiring on the pypowsybl backend.

Guards:

1. :meth:`NetworkManager.run_load_flow` accepts a ``voltage_init_mode``
   kwarg and applies it to the LF parameters passed to
   ``pypowsybl.loadflow.run_ac``.
2. :meth:`SimulationEnvironment._ensure_valid_state` passes
   ``voltage_init_mode=DC_VALUES`` explicitly.  This is necessary because
   the LF called at construction/reset time has no previous voltage
   magnitudes to read from — without the override, pypowsybl throws
   "Voltage magnitude is undefined for bus ..." and retries internally
   with DC_VALUES, wasting ~70 ms and polluting the logs.

Both guards correspond to upstream commit ``a377a968`` (0.2.0.post7).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pypowsybl = pytest.importorskip("pypowsybl")

import pypowsybl.loadflow as lf


class TestRunLoadFlowVoltageInitMode:
    """``NetworkManager.run_load_flow(voltage_init_mode=...)`` wiring."""

    @pytest.fixture
    def nm(self):
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager
        network = pypowsybl.network.create_ieee9()
        return NetworkManager(network=network)

    def test_signature_accepts_voltage_init_mode_kwarg(self, nm):
        """The kwarg must exist — older versions of the API didn't have it."""
        import inspect
        sig = inspect.signature(nm.run_load_flow)
        assert "voltage_init_mode" in sig.parameters, (
            "NetworkManager.run_load_flow must expose voltage_init_mode kwarg "
            "so SimulationEnvironment._ensure_valid_state can request DC_VALUES."
        )

    def test_default_voltage_init_mode_is_previous_values(self, nm):
        """Without the kwarg, lf_parameters must stay on PREVIOUS_VALUES.

        PREVIOUS_VALUES is the right default for warm-started LFs after
        a mutation. Only the initial LF needs the DC_VALUES override.
        """
        assert (
            nm.lf_parameters.voltage_init_mode
            == lf.VoltageInitMode.PREVIOUS_VALUES
        )

    def test_voltage_init_mode_override_is_applied_to_params(self, nm):
        """When caller passes voltage_init_mode=DC_VALUES, run_ac must see
        a Parameters object whose voltage_init_mode is DC_VALUES.
        """
        fake_result = MagicMock()
        fake_result.status = lf.ComponentStatus.CONVERGED

        with patch.object(
            lf, "run_ac", return_value=[fake_result]
        ) as mock_run_ac:
            nm.run_load_flow(
                voltage_init_mode=lf.VoltageInitMode.DC_VALUES,
            )

        assert mock_run_ac.call_count >= 1
        call_params = mock_run_ac.call_args.kwargs.get("parameters")
        assert call_params is not None, (
            "pypowsybl.loadflow.run_ac must be called with parameters=..."
        )
        assert call_params.voltage_init_mode == lf.VoltageInitMode.DC_VALUES, (
            "voltage_init_mode override was not forwarded to run_ac parameters"
        )

    def test_voltage_init_mode_override_does_not_mutate_shared_parameters(
        self, nm,
    ):
        """The override must copy the shared lf_parameters, not mutate it —
        otherwise subsequent LFs would inherit DC_VALUES and never
        benefit from the PREVIOUS_VALUES warm-start.
        """
        fake_result = MagicMock()
        fake_result.status = lf.ComponentStatus.CONVERGED

        with patch.object(lf, "run_ac", return_value=[fake_result]):
            nm.run_load_flow(
                voltage_init_mode=lf.VoltageInitMode.DC_VALUES,
            )

        # The singleton lf_parameters on the NetworkManager must remain
        # on PREVIOUS_VALUES.
        assert (
            nm.lf_parameters.voltage_init_mode
            == lf.VoltageInitMode.PREVIOUS_VALUES
        ), (
            "run_load_flow(voltage_init_mode=...) leaked a mutation into "
            "the shared NetworkManager.lf_parameters — it must copy first."
        )

    def test_voltage_init_mode_ignored_in_dc_mode(self, nm):
        """DC load flow has its own params stack and shouldn't try to
        override voltage_init_mode on the AC parameters.  Just verify
        the call doesn't explode.
        """
        # run_dc will run on the real tiny IEEE9 network — cheap.
        result = nm.run_load_flow(
            dc=True,
            voltage_init_mode=lf.VoltageInitMode.DC_VALUES,
        )
        # IEEE9 converges in DC trivially.
        assert result is not None


class TestEnsureValidStateUsesDcValues:
    """:meth:`SimulationEnvironment._ensure_valid_state` must pass
    ``voltage_init_mode=DC_VALUES`` to ``run_load_flow``.
    """

    def test_ensure_valid_state_requests_dc_values_init(self):
        """Construction triggers _ensure_valid_state — capture the call
        and verify the kwarg.
        """
        from expert_op4grid_recommender.pypowsybl_backend import (
            SimulationEnvironment,
        )

        network = pypowsybl.network.create_ieee9()

        captured = {}

        real_run_load_flow = None

        def capturing_run_load_flow(self, *args, **kwargs):
            captured.setdefault("calls", []).append(kwargs)
            # Return a converged-looking result so the warning branch doesn't fire.
            result = MagicMock()
            result.status = lf.ComponentStatus.CONVERGED
            return result

        from expert_op4grid_recommender.pypowsybl_backend.network_manager import (
            NetworkManager,
        )

        with patch.object(
            NetworkManager, "run_load_flow", autospec=True,
            side_effect=capturing_run_load_flow,
        ):
            env = SimulationEnvironment(network=network)
            # reset() also calls _ensure_valid_state
            env.reset()

        assert captured.get("calls"), (
            "SimulationEnvironment did not invoke network_manager.run_load_flow"
        )
        # Every call made by _ensure_valid_state must request DC_VALUES.
        # (We don't know if other call sites are exercised here; but both
        # __init__ and reset() go through _ensure_valid_state.)
        for kwargs in captured["calls"]:
            assert (
                kwargs.get("voltage_init_mode") == lf.VoltageInitMode.DC_VALUES
            ), (
                "_ensure_valid_state must pass voltage_init_mode=DC_VALUES. "
                f"Got kwargs={kwargs!r}. Without the override, pypowsybl "
                "throws 'Voltage magnitude is undefined' on the initial LF "
                "and retries internally — wasted work + spurious warning."
            )
