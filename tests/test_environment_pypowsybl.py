# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for :mod:`expert_op4grid_recommender.environment_pypowsybl`.

These tests exercise the pure-Python setup-layer logic (file discovery,
thermal-limit rescaling, compatibility wrappers) without standing up a real
pypowsybl simulation environment. The heavy :class:`SimulationEnvironment`
dependency is patched out at the module level.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import expert_op4grid_recommender.environment_pypowsybl as env_pp


# ---------------------------------------------------------------------------
# get_env_first_obs_pypowsybl — file discovery logic
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sim_env(monkeypatch):
    """Patches :class:`SimulationEnvironment` used inside the module under test.

    Returns the mock class so tests can inspect the constructor call.
    """
    fake_cls = MagicMock(name="SimulationEnvironment")
    fake_instance = MagicMock(name="sim_env_instance")
    fake_instance.get_obs.return_value = MagicMock(name="obs")
    fake_instance.network_manager = MagicMock()
    fake_cls.return_value = fake_instance
    monkeypatch.setattr(env_pp, "SimulationEnvironment", fake_cls)
    return fake_cls, fake_instance


def test_get_env_first_obs_finds_xiidm_file_in_subdir(tmp_path, fake_sim_env):
    fake_cls, _ = fake_sim_env
    env_folder = tmp_path
    env_name = "my_env"
    subdir = env_folder / env_name
    subdir.mkdir()
    network_file = subdir / "grid.xiidm"
    network_file.write_text("<network/>")

    env, obs, path = env_pp.get_env_first_obs_pypowsybl(
        env_folder=str(env_folder),
        env_name=env_name,
    )

    fake_cls.assert_called_once()
    call_kwargs = fake_cls.call_args.kwargs
    assert call_kwargs["network_path"] == network_file
    assert call_kwargs["thermal_limits_path"] is None
    assert path == str(subdir)


def test_get_env_first_obs_uses_direct_xiidm_file(tmp_path, fake_sim_env):
    """If the env_name points directly at a .xiidm file, use it as-is."""
    fake_cls, _ = fake_sim_env
    network_file = tmp_path / "direct.xiidm"
    network_file.write_text("<network/>")

    env, obs, path = env_pp.get_env_first_obs_pypowsybl(
        env_folder=str(tmp_path),
        env_name="direct.xiidm",
    )

    call_kwargs = fake_cls.call_args.kwargs
    assert call_kwargs["network_path"] == network_file
    # Env path becomes the parent directory for auxiliary files lookups
    assert path == str(tmp_path)


def test_get_env_first_obs_finds_grid_subfolder_file(tmp_path, fake_sim_env):
    """Falls back to looking in <env_name>/grid/ for a network file."""
    fake_cls, _ = fake_sim_env
    env_name = "case42"
    grid_dir = tmp_path / env_name / "grid"
    grid_dir.mkdir(parents=True)
    network_file = grid_dir / "case.iidm"
    network_file.write_text("<network/>")

    env_pp.get_env_first_obs_pypowsybl(
        env_folder=str(tmp_path), env_name=env_name
    )

    call_kwargs = fake_cls.call_args.kwargs
    assert call_kwargs["network_path"] == network_file


def test_get_env_first_obs_raises_when_no_network_file(tmp_path, fake_sim_env):
    (tmp_path / "empty_env").mkdir()
    with pytest.raises(FileNotFoundError):
        env_pp.get_env_first_obs_pypowsybl(
            env_folder=str(tmp_path), env_name="empty_env"
        )


def test_get_env_first_obs_picks_up_thermal_limits_json(tmp_path, fake_sim_env):
    """If ``thermal_limits.json`` is next to the network file it is
    auto-detected even when ``thermal_limits_file`` is not provided."""
    fake_cls, _ = fake_sim_env
    env_name = "env_x"
    subdir = tmp_path / env_name
    subdir.mkdir()
    (subdir / "grid.xiidm").write_text("<network/>")
    tl = subdir / "thermal_limits.json"
    tl.write_text("{}")

    env_pp.get_env_first_obs_pypowsybl(
        env_folder=str(tmp_path), env_name=env_name
    )
    call_kwargs = fake_cls.call_args.kwargs
    assert call_kwargs["thermal_limits_path"] == tl


def test_get_env_first_obs_dc_mode_sets_network_manager_flag(tmp_path, fake_sim_env):
    fake_cls, fake_instance = fake_sim_env
    env_name = "env_dc"
    subdir = tmp_path / env_name
    subdir.mkdir()
    (subdir / "grid.xiidm").write_text("<network/>")

    env_pp.get_env_first_obs_pypowsybl(
        env_folder=str(tmp_path), env_name=env_name, is_DC=True
    )

    assert fake_instance._use_dc is True
    assert fake_instance.network_manager._default_dc is True


# ---------------------------------------------------------------------------
# set_thermal_limits_from_network
# ---------------------------------------------------------------------------

def test_set_thermal_limits_applies_threshold_and_default():
    """Thermal limits are scaled by the threshold, missing ones fall back
    to the sentinel ``9999.0``."""
    env = MagicMock()
    env.network_manager.get_thermal_limits.return_value = {
        "L1": 1000.0,
        "L2": 200.0,
    }
    env.name_line = np.array(["L1", "L2", "L3"])  # L3 missing → sentinel

    returned = env_pp.set_thermal_limits_from_network(env, threshold=0.9)

    env.set_thermal_limit.assert_called_once()
    called_arr = env.set_thermal_limit.call_args.args[0]
    np.testing.assert_allclose(called_arr, np.array([900.0, 180.0, 9999.0 * 0.9]))
    assert returned is env


# ---------------------------------------------------------------------------
# Compatibility wrappers
# ---------------------------------------------------------------------------

def test_get_env_first_obs_delegates_to_pypowsybl_wrapper():
    """The compatibility ``get_env_first_obs`` strips the ``date`` parameter
    and forwards everything else to ``get_env_first_obs_pypowsybl``."""
    with patch.object(env_pp, "get_env_first_obs_pypowsybl") as mocked:
        mocked.return_value = ("env", "obs", "/tmp/x")
        result = env_pp.get_env_first_obs(
            env_folder="/tmp",
            env_name="case",
            use_evaluation_config=True,
            date="2024-01-01",   # silently dropped
            is_DC=True,
        )

    mocked.assert_called_once_with(
        env_folder="/tmp", env_name="case", is_DC=True
    )
    assert result == ("env", "obs", "/tmp/x")


def test_setup_environment_configs_delegates_to_pypowsybl_wrapper():
    with patch.object(env_pp, "setup_environment_configs_pypowsybl") as mocked:
        mocked.return_value = "sentinel"
        result = env_pp.setup_environment_configs("2024-01-01")

    # `network=None` is threaded through the compatibility wrapper so
    # external callers that pass a pre-loaded Network survive the
    # compat layer too.
    mocked.assert_called_once_with("2024-01-01", network=None)
    assert result == "sentinel"


def test_setup_environment_configs_forwards_injected_network():
    """`network=` on the compat wrapper reaches the inner function."""
    with patch.object(env_pp, "setup_environment_configs_pypowsybl") as mocked:
        mocked.return_value = "sentinel"
        fake_net = MagicMock(name="preloaded_network")
        env_pp.setup_environment_configs("2024-01-01", network=fake_net)

    mocked.assert_called_once_with("2024-01-01", network=fake_net)


# ---------------------------------------------------------------------------
# Pre-loaded Network injection (load-deduplication path)
# ---------------------------------------------------------------------------

def test_get_env_first_obs_skips_file_load_when_network_is_injected(tmp_path, fake_sim_env):
    """When a pre-loaded `pp.network.Network` is passed, the .xiidm file
    discovery is skipped entirely. `SimulationEnvironment` is constructed
    with `network=<injected>` and `network_path=None`. This is the
    load-deduplication fast path — no disk I/O, no pypowsybl re-parse."""
    fake_cls, _ = fake_sim_env
    fake_net = MagicMock(name="preloaded_network")

    # Deliberately point at a directory with NO .xiidm file — the
    # injection path must not attempt a discovery.
    env_folder = tmp_path
    (env_folder / "empty_env").mkdir()

    env_pp.get_env_first_obs_pypowsybl(
        env_folder=str(env_folder),
        env_name="empty_env",
        network=fake_net,
    )

    call_kwargs = fake_cls.call_args.kwargs
    assert call_kwargs["network"] is fake_net
    assert call_kwargs["network_path"] is None


def test_get_env_first_obs_injected_network_still_looks_up_companion_files(tmp_path, fake_sim_env):
    """File discovery is skipped for the network itself, but companion
    files (thermal_limits.json in particular) are still looked up next to
    the env folder — the injection only short-circuits the network load."""
    fake_cls, _ = fake_sim_env
    fake_net = MagicMock(name="preloaded_network")
    env_name = "env_tl"
    subdir = tmp_path / env_name
    subdir.mkdir()
    tl = subdir / "thermal_limits.json"
    tl.write_text("{}")

    env_pp.get_env_first_obs_pypowsybl(
        env_folder=str(tmp_path),
        env_name=env_name,
        network=fake_net,
    )

    call_kwargs = fake_cls.call_args.kwargs
    assert call_kwargs["network"] is fake_net
    assert call_kwargs["network_path"] is None
    assert call_kwargs["thermal_limits_path"] == tl


def test_get_env_first_obs_injected_network_handles_direct_xiidm_env_name(tmp_path, fake_sim_env):
    """When `env_name` points at a file path (e.g. `/data/grid.xiidm`)
    AND a pre-loaded network is injected, the env_path resolves to the
    file's parent so companion files still work."""
    fake_cls, _ = fake_sim_env
    fake_net = MagicMock(name="preloaded_network")
    # File doesn't have to exist on disk — the injection path skips
    # the .exists() discovery branch. We only need the suffix to be
    # recognised so env_path falls back to the parent directory.
    network_file = tmp_path / "direct.xiidm"
    network_file.write_text("<network/>")

    _, _, path = env_pp.get_env_first_obs_pypowsybl(
        env_folder=str(tmp_path),
        env_name="direct.xiidm",
        network=fake_net,
    )

    assert path == str(tmp_path)  # parent of the direct file
    call_kwargs = fake_cls.call_args.kwargs
    assert call_kwargs["network"] is fake_net
    assert call_kwargs["network_path"] is None


# ---------------------------------------------------------------------------
# skip_non_reconnectable_detection (deferred-detection path)
# ---------------------------------------------------------------------------

def test_setup_environment_configs_runs_detection_by_default(tmp_path):
    """By default (no flag), the setup function must run the topology
    detection of non-reconnectable lines — preserves backward compat."""
    from unittest.mock import MagicMock, patch

    fake_env = MagicMock(name="env")
    fake_env.network_manager.detect_non_reconnectable_lines.return_value = ["line_topo"]
    fake_env.network_manager.network.get_operational_limits.return_value = MagicMock()
    fake_env.name_line = []

    with patch.object(env_pp, "load_actions", return_value={}), \
         patch.object(env_pp, "get_env_first_obs_pypowsybl",
                      return_value=(fake_env, MagicMock(), str(tmp_path))), \
         patch.object(env_pp, "load_interesting_lines", side_effect=FileNotFoundError()), \
         patch.object(env_pp.config, "ENV_FOLDER", str(tmp_path)), \
         patch.object(env_pp.config, "ENV_NAME", "case"), \
         patch.object(env_pp.config, "ACTION_FILE_PATH", "/tmp/actions.json"), \
         patch.object(env_pp.config, "IGNORE_LINES_MONITORING", True):
        result = env_pp.setup_environment_configs_pypowsybl()

    fake_env.network_manager.detect_non_reconnectable_lines.assert_called_once()
    lines_non_reco = result[6]
    assert "line_topo" in lines_non_reco


def test_setup_environment_configs_skips_detection_when_flag_set(tmp_path):
    """When skip_non_reconnectable_detection=True, the slow topology walk
    MUST be skipped. The caller is then responsible for computing it
    in the background. See the deferred-detection pattern in
    Co-Study4Grid's recommender_service."""
    from unittest.mock import MagicMock, patch

    fake_env = MagicMock(name="env")
    fake_env.network_manager.detect_non_reconnectable_lines.return_value = ["should_not_appear"]
    fake_env.network_manager.network.get_operational_limits.return_value = MagicMock()
    fake_env.name_line = []

    with patch.object(env_pp, "load_actions", return_value={}), \
         patch.object(env_pp, "get_env_first_obs_pypowsybl",
                      return_value=(fake_env, MagicMock(), str(tmp_path))), \
         patch.object(env_pp, "load_interesting_lines", side_effect=FileNotFoundError()), \
         patch.object(env_pp.config, "ENV_FOLDER", str(tmp_path)), \
         patch.object(env_pp.config, "ENV_NAME", "case"), \
         patch.object(env_pp.config, "ACTION_FILE_PATH", "/tmp/actions.json"), \
         patch.object(env_pp.config, "IGNORE_LINES_MONITORING", True):
        result = env_pp.setup_environment_configs_pypowsybl(
            skip_non_reconnectable_detection=True,
        )

    # The expensive call MUST NOT have been made.
    fake_env.network_manager.detect_non_reconnectable_lines.assert_not_called()
    # And its would-be result must NOT leak into the returned list.
    lines_non_reco = result[6]
    assert "should_not_appear" not in lines_non_reco
