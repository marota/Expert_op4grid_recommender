# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""Tests for :mod:`expert_op4grid_recommender.utils.reassessment`.

These tests focus on the pure logic that does not require a live
pypowsybl / grid2op environment: input DTO mapping, network extraction
(N + N-K), non-convergence propagation, and graceful failure of the
superposition helper.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from expert_op4grid_recommender.utils.reassessment import (
    _extract_pypowsybl_network,
    _extract_pypowsybl_network_from_obs,
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
# _extract_pypowsybl_network (env)
# ---------------------------------------------------------------------

def test_extract_network_from_pypowsybl_env():
    fake_net = object()
    env = SimpleNamespace(network_manager=SimpleNamespace(network=fake_net))
    assert _extract_pypowsybl_network(env) is fake_net


def test_extract_network_from_grid2op_env():
    fake_net = object()
    env = SimpleNamespace(backend=SimpleNamespace(_grid=SimpleNamespace(network=fake_net)))
    assert _extract_pypowsybl_network(env) is fake_net


def test_extract_network_returns_none_when_unavailable():
    assert _extract_pypowsybl_network(SimpleNamespace()) is None


def test_extract_network_returns_none_for_none_env():
    assert _extract_pypowsybl_network(None) is None


def test_extract_network_prefers_pypowsybl_path_over_grid2op():
    primary = object()
    fallback = object()
    env = SimpleNamespace(
        network_manager=SimpleNamespace(network=primary),
        backend=SimpleNamespace(_grid=SimpleNamespace(network=fallback)),
    )
    assert _extract_pypowsybl_network(env) is primary


# ---------------------------------------------------------------------
# _extract_pypowsybl_network_from_obs (N-K observation)
# ---------------------------------------------------------------------

def test_extract_network_from_obs_pypowsybl_backend():
    """pypowsybl obs exposes the Network via ``obs._network_manager.network``."""
    fake_net = object()
    obs = SimpleNamespace(_network_manager=SimpleNamespace(network=fake_net))
    assert _extract_pypowsybl_network_from_obs(obs) is fake_net


def test_extract_network_from_obs_none_when_no_manager():
    """Grid2Op observations don't have ``_network_manager`` — returns None."""
    obs = SimpleNamespace()  # no _network_manager
    assert _extract_pypowsybl_network_from_obs(obs) is None


def test_extract_network_from_obs_none_when_obs_is_none():
    assert _extract_pypowsybl_network_from_obs(None) is None


def test_extract_network_from_obs_none_when_manager_has_no_network():
    obs = SimpleNamespace(_network_manager=SimpleNamespace())
    assert _extract_pypowsybl_network_from_obs(obs) is None


# ---------------------------------------------------------------------
# build_recommender_inputs
# ---------------------------------------------------------------------

def _full_context(env=None, obs_simu_defaut="obs_d", n_grid=None, n_grid_defaut=None):
    ctx = {
        "obs": "obs",
        "obs_simu_defaut": obs_simu_defaut,
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
    }
    if n_grid is not None:
        ctx["n_grid"] = n_grid
    if n_grid_defaut is not None:
        ctx["n_grid_defaut"] = n_grid_defaut
    return ctx


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
    assert inputs._context is ctx


def test_build_recommender_inputs_copies_list_inputs():
    ctx = _full_context()
    inputs = build_recommender_inputs(ctx)
    inputs.lines_defaut.append("new")
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
    assert inputs.is_pypowsybl is True
    assert inputs.fast_mode is False


# ---------------------------------------------------------------------
# build_recommender_inputs: ``network`` (N-state, paired with obs)
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
    ctx = _full_context(env=SimpleNamespace(), obs_simu_defaut=SimpleNamespace())
    inputs = build_recommender_inputs(ctx)
    assert inputs.network is None


def test_build_recommender_inputs_explicit_n_grid_wins_over_env():
    primary = object()
    fallback = object()
    env = SimpleNamespace(network_manager=SimpleNamespace(network=fallback))
    ctx = _full_context(env=env, n_grid=primary)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network is primary


# ---------------------------------------------------------------------
# build_recommender_inputs: ``network_defaut`` (N-K, paired with obs_defaut)
# ---------------------------------------------------------------------

def test_build_recommender_inputs_network_defaut_from_obs_simu():
    """On the pypowsybl backend, the post-fault network comes from
    ``obs_simu_defaut._network_manager.network`` (the contingency-variant
    NetworkManager)."""
    fake_net = object()
    obs_simu = SimpleNamespace(_network_manager=SimpleNamespace(network=fake_net))
    ctx = _full_context(obs_simu_defaut=obs_simu)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network_defaut is fake_net


def test_build_recommender_inputs_uses_explicit_n_grid_defaut_when_present():
    """context['n_grid_defaut'] takes priority over obs / env introspection."""
    primary = object()
    fallback_obs_net = object()
    fallback_env_net = object()
    obs_simu = SimpleNamespace(_network_manager=SimpleNamespace(network=fallback_obs_net))
    env = SimpleNamespace(network_manager=SimpleNamespace(network=fallback_env_net))
    ctx = _full_context(
        env=env, obs_simu_defaut=obs_simu, n_grid_defaut=primary,
    )
    inputs = build_recommender_inputs(ctx)
    assert inputs.network_defaut is primary


def test_build_recommender_inputs_network_defaut_falls_back_to_env():
    """When obs has no ``_network_manager`` (e.g. grid2op) and no
    explicit ``n_grid_defaut`` is set, fall back to env introspection."""
    fake_net = object()
    env = SimpleNamespace(backend=SimpleNamespace(_grid=SimpleNamespace(network=fake_net)))
    ctx = _full_context(env=env, obs_simu_defaut=SimpleNamespace())
    inputs = build_recommender_inputs(ctx)
    assert inputs.network_defaut is fake_net


def test_build_recommender_inputs_network_defaut_is_none_when_unavailable():
    """No env path, no obs network handle, no explicit override — None."""
    ctx = _full_context(env=SimpleNamespace(), obs_simu_defaut=SimpleNamespace())
    inputs = build_recommender_inputs(ctx)
    assert inputs.network_defaut is None


def test_build_recommender_inputs_network_and_network_defaut_are_independent():
    """The two handles can point to different objects when explicit overrides
    are provided (real backends typically share the same instance with different
    variants active)."""
    n_state = object()
    nk_state = object()
    ctx = _full_context(n_grid=n_state, n_grid_defaut=nk_state)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network is n_state
    assert inputs.network_defaut is nk_state
    assert inputs.network is not inputs.network_defaut


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
