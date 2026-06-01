# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for :func:`get_maintenance_timestep_pypowsybl`.

Covers the fast-exit short-circuit added when ``do_reco_maintenance``
is False: the function must skip the disconnected-line scan + the
formatted ``print`` and return an empty action immediately.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from expert_op4grid_recommender.utils.helpers_pypowsybl import (
    get_maintenance_timestep_pypowsybl,
)


def _make_env(captured=None):
    """Build a minimal env stub whose ``action_space`` records the call."""
    env = MagicMock(name="env")

    def action_space(payload):
        if captured is not None:
            captured.append(payload)
        return MagicMock(name=f"action({payload!r})")

    env.action_space.side_effect = action_space
    return env


class TestMaintenanceFastExit:
    """When ``do_reco_maintenance`` is False, the function must not
    iterate over the disconnected-line table at all — its return value
    would be informational only."""

    def test_returns_empty_action_when_do_reco_is_false(self):
        env = _make_env()
        obs = MagicMock(name="obs")

        with patch(
            "expert_op4grid_recommender.utils.helpers_pypowsybl.get_disconnected_lines_pypowsybl"
        ) as get_disco:
            act, lines = get_maintenance_timestep_pypowsybl(
                env, obs, lines_non_reconnectable=["DUMMY"], do_reco_maintenance=False,
            )

        # The expensive disconnected-line scan is skipped entirely.
        get_disco.assert_not_called()
        # The empty action object was requested (one call, payload empty).
        assert lines == []
        assert act is not None

    def test_empty_action_payload_has_no_reconnections(self):
        """The fast-exit path requests ``set_line_status: []``."""
        captured: list = []
        env = _make_env(captured)
        obs = MagicMock(name="obs")

        with patch(
            "expert_op4grid_recommender.utils.helpers_pypowsybl.get_disconnected_lines_pypowsybl"
        ):
            get_maintenance_timestep_pypowsybl(
                env, obs, lines_non_reconnectable=[], do_reco_maintenance=False,
            )

        assert captured == [{"set_line_status": []}]

    def test_does_not_print_when_do_reco_is_false(self, capsys):
        """The fast-exit path also suppresses the
        ``Detected N disconnected lines …`` chatter that on large grids
        formats hundreds of np.str_ values into a single line."""
        env = _make_env()
        obs = MagicMock(name="obs")

        with patch(
            "expert_op4grid_recommender.utils.helpers_pypowsybl.get_disconnected_lines_pypowsybl"
        ) as get_disco:
            # Even if get_disco WAS somehow called, it would return
            # disconnected lines — but the fast-exit path skips it,
            # so the print should never fire.
            get_disco.return_value = ["LINE_X", "LINE_Y"]
            get_maintenance_timestep_pypowsybl(
                env, obs, lines_non_reconnectable=[], do_reco_maintenance=False,
            )

        captured = capsys.readouterr()
        assert "Detected" not in captured.out
        assert "Will attempt to reconnect" not in captured.out


class TestMaintenanceFullPath:
    """When ``do_reco_maintenance`` is True the function keeps its
    original behaviour: scan, filter, optionally print, return action."""

    def test_runs_scan_when_do_reco_is_true(self):
        env = _make_env()
        obs = MagicMock(name="obs")

        with patch(
            "expert_op4grid_recommender.utils.helpers_pypowsybl.get_disconnected_lines_pypowsybl"
        ) as get_disco:
            get_disco.return_value = ["LINE_A", "LINE_B", "BAD_LINE"]
            act, lines = get_maintenance_timestep_pypowsybl(
                env, obs,
                lines_non_reconnectable=["BAD_LINE"],
                do_reco_maintenance=True,
            )

        get_disco.assert_called_once_with(env, obs)
        # Non-reconnectable filtered out, the rest become reconnection targets.
        assert sorted(lines) == ["LINE_A", "LINE_B"]
        assert act is not None

    def test_returns_empty_when_no_disconnected_lines_even_with_flag(self):
        env = _make_env()
        obs = MagicMock(name="obs")

        with patch(
            "expert_op4grid_recommender.utils.helpers_pypowsybl.get_disconnected_lines_pypowsybl"
        ) as get_disco:
            get_disco.return_value = []
            act, lines = get_maintenance_timestep_pypowsybl(
                env, obs, lines_non_reconnectable=[], do_reco_maintenance=True,
            )

        assert lines == []
        assert act is not None
