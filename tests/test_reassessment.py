# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""Tests for :mod:`expert_op4grid_recommender.utils.reassessment`.

These tests focus on the pure logic that does not require a live
pypowsybl / grid2op environment: input DTO mapping, network extraction,
non-convergence propagation, and graceful failure of the superposition
helper.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from expert_op4grid_recommender.utils.reassessment import (
    _extract_pypowsybl_network,
    build_recommender_inputs,
    compute_combined_pairs,
    propagate_non_convergence_to_scores,
)


# ---------------------------------------------------------------------
# propagate_non_convergence_to_scores
# ---------------------------------------------------------------------

def test_propagate_writes_reason_into_score_map():
    detailed = {
        "a1": {"non_convergence": "divergence"},
        "a2": {"non_convergence": None},
    }
    scores = {
        "line_reconnection": {"scores": {"a1": 0.5}, "params": {}},
        "line_disconnection": {"scores": {"a2": 0.3}, "params": {}},
    }
    out = propagate_non_convergence_to_scores(detailed, scores)
    assert out["line_reconnection"]["non_convergence"]["a1"] == "divergence"
    assert out["line_disconnection"]["non_convergence"]["a2"] is None


def test_propagate_unknown_id_does_not_create_entry():
    detailed = {"orphan": {"non_convergence": "x"}}
    scores = {"line_reconnection": {"scores": {}, "params": {}}}
    out = propagate_non_convergence_to_scores(detailed, scores)
    assert out["line_reconnection"].get("non_convergence", {}) == {}


def test_propagate_creates_non_convergence_dict_when_absent():
    detailed = {"a1": {"non_convergence": "X"}}
    scores = {"line_reconnection": {"scores": {"a1": 0.1}, "params": {}}}
    out = propagate_non_convergence_to_scores(detailed, scores)
    assert out["line_reconnection"]["non_convergence"] == {"a1": "X"}


# ---------------------------------------------------------------------
# _extract_pypowsybl_network
# ---------------------------------------------------------------------

def test_extract_network_from_pypowsybl_env():
    """pypowsybl backend exposes the Network via ``env.network_manager.network``."""
    fake_net = object()
    env = SimpleNamespace(network_manager=SimpleNamespace(network=fake_net))
    assert _extract_pypowsybl_network(env) is fake_net


def test_extract_network_from_grid2op_env():
    """grid2op backend exposes the Network via ``env.backend._grid.network``."""
    fake_net = object()
    env = SimpleNamespace(backend=SimpleNamespace(_grid=SimpleNamespace(network=fake_net)))
    assert _extract_pypowsybl_network(env) is fake_net


def test_extract_network_returns_none_when_unavailable():
    """A bare env with no recognised attribute path returns ``None``."""
    assert _extract_pypowsybl_network(SimpleNamespace()) is None


def test_extract_network_returns_none_for_none_env():
    assert _extract_pypowsybl_network(None) is None


def test_extract_network_prefers_pypowsybl_path_over_grid2op():
    """When both attribute paths exist the pypowsybl one wins."""
    primary = object()
    fallback = object()
    env = SimpleNamespace(
        network_manager=SimpleNamespace(network=primary),
        backend=SimpleNamespace(_grid=SimpleNamespace(network=fallback)),
    )
    assert _extract_pypowsybl_network(env) is primary


# ---------------------------------------------------------------------
# build_recommender_inputs
# ---------------------------------------------------------------------

def _full_context(env=None, n_grid=None):
    return {
        "obs": "obs",
        "obs_simu_defaut": "obs_d",
        "current_lines_defaut": ["L1"],
        "lines_overloaded_names": ["L2"],
        "lines_overloaded_ids": [2],
        "dict_action": {"a": {}},
        "env": env if env is not None else "env",
        "classifier": "cls",
        "current_timestep": 7,
        "g_overflow": "g",
        "g_distribution_graph": "gd",
        "overflow_sim": "sim",
        "hubs": ["h1"],
        "node_name_mapping": {1: "a"},
        "non_connected_reconnectable_lines": ["L3"],
        "lines_non_reconnectable": ["L4"],
        "lines_we_care_about": ["L5"],
        "maintenance_to_reco_at_t": [],
        "act_reco_maintenance": "act",
        "use_dc": False,
        "is_pypowsybl": True,
        "actual_fast_mode": True,
        **({"n_grid": n_grid} if n_grid is not None else {}),
    }


def test_build_recommender_inputs_maps_full_context():
    ctx = _full_context()
    inputs = build_recommender_inputs(ctx)
    assert inputs.obs == "obs"
    assert inputs.obs_defaut == "obs_d"
    assert inputs.timestep == 7
    assert inputs.lines_defaut == ["L1"]
    assert inputs.lines_overloaded_names == ["L2"]
    assert inputs.lines_overloaded_ids == [2]
    assert inputs.overflow_graph == "g"
    assert inputs.distribution_graph == "gd"
    assert inputs.hubs == ["h1"]
    assert inputs.non_connected_reconnectable_lines == ["L3"]
    assert inputs.use_dc is False
    assert inputs.is_pypowsybl is True
    assert inputs.fast_mode is True


def test_build_recommender_inputs_exposes_context_as_escape_hatch():
    ctx = _full_context()
    inputs = build_recommender_inputs(ctx)
    # The expert model relies on this escape hatch to reach internals.
    assert inputs._context is ctx


def test_build_recommender_inputs_copies_list_inputs():
    ctx = _full_context()
    inputs = build_recommender_inputs(ctx)
    inputs.lines_defaut.append("new")
    # Mutation should not leak back into context.
    assert ctx["current_lines_defaut"] == ["L1"]


def test_build_recommender_inputs_tolerates_missing_optional_keys():
    minimal_ctx = {
        "obs": "o",
        "obs_simu_defaut": "od",
        "current_lines_defaut": [],
        "lines_overloaded_names": [],
        "lines_overloaded_ids": [],
        "dict_action": {},
        "env": "e",
        "classifier": "c",
        "current_timestep": 0,
    }
    inputs = build_recommender_inputs(minimal_ctx)
    assert inputs.overflow_graph is None
    assert inputs.distribution_graph is None
    assert inputs.hubs is None
    assert inputs.use_dc is False
    # Defaults from `get(..., default)` calls take over.
    assert inputs.is_pypowsybl is True
    assert inputs.fast_mode is False


# ---------------------------------------------------------------------
# build_recommender_inputs: network handle
# ---------------------------------------------------------------------

def test_build_recommender_inputs_uses_explicit_n_grid_when_present():
    fake_net = object()
    ctx = _full_context(n_grid=fake_net)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network is fake_net


def test_build_recommender_inputs_falls_back_to_env_pypowsybl_path():
    fake_net = object()
    env = SimpleNamespace(network_manager=SimpleNamespace(network=fake_net))
    ctx = _full_context(env=env)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network is fake_net


def test_build_recommender_inputs_falls_back_to_env_grid2op_path():
    fake_net = object()
    env = SimpleNamespace(backend=SimpleNamespace(_grid=SimpleNamespace(network=fake_net)))
    ctx = _full_context(env=env)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network is fake_net


def test_build_recommender_inputs_network_is_none_when_env_does_not_expose_one():
    ctx = _full_context(env=SimpleNamespace())  # neither path available
    inputs = build_recommender_inputs(ctx)
    assert inputs.network is None


def test_build_recommender_inputs_explicit_n_grid_wins_over_env():
    """context['n_grid'] is the explicit override and takes priority."""
    primary = object()
    fallback = object()
    env = SimpleNamespace(network_manager=SimpleNamespace(network=fallback))
    ctx = _full_context(env=env, n_grid=primary)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network is primary


# ---------------------------------------------------------------------
# compute_combined_pairs
# ---------------------------------------------------------------------

def test_compute_combined_pairs_returns_empty_on_exception():
    ctx = {
        "obs_simu_defaut": None,
        "classifier": None,
        "env": None,
        "lines_overloaded_ids": [],
        "lines_we_care_about": [],
        "pre_existing_rho": {},
        "dict_action": {},
    }
    with patch(
        "expert_op4grid_recommender.utils.superposition.compute_all_pairs_superposition",
        side_effect=RuntimeError("boom"),
    ):
        # Must not propagate the error — superposition is decorative metadata.
        result = compute_combined_pairs({}, ctx)
    assert result == {}


def test_compute_combined_pairs_forwards_arguments():
    ctx = {
        "obs_simu_defaut": "obs_d",
        "classifier": "cls",
        "env": "env",
        "lines_overloaded_ids": [2],
        "lines_we_care_about": ["L5"],
        "pre_existing_rho": {0: 0.9},
        "dict_action": {"a": {}},
    }
    captured = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        return {"a+b": {"betas": [0.5, 0.5]}}

    with patch(
        "expert_op4grid_recommender.utils.superposition.compute_all_pairs_superposition",
        side_effect=_stub,
    ):
        result = compute_combined_pairs({"a": {"action": object()}}, ctx)

    assert result == {"a+b": {"betas": [0.5, 0.5]}}
    assert captured["obs_start"] == "obs_d"
    assert captured["classifier"] == "cls"
    assert captured["env"] == "env"
    assert captured["lines_overloaded_ids"] == [2]
    assert captured["dict_action"] == {"a": {}}
