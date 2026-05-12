# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""Tests for :mod:`expert_op4grid_recommender.utils.reassessment`."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from expert_op4grid_recommender.utils.reassessment import (
    _extract_overloaded_rho,
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
    fake_net = object()
    obs = SimpleNamespace(_network_manager=SimpleNamespace(network=fake_net))
    assert _extract_pypowsybl_network_from_obs(obs) is fake_net


def test_extract_network_from_obs_none_when_no_manager():
    obs = SimpleNamespace()
    assert _extract_pypowsybl_network_from_obs(obs) is None


def test_extract_network_from_obs_none_when_obs_is_none():
    assert _extract_pypowsybl_network_from_obs(None) is None


# ---------------------------------------------------------------------
# _extract_overloaded_rho
# ---------------------------------------------------------------------

def test_extract_overloaded_rho_from_numpy_array():
    """Real obs.rho is a numpy array — result should be a list of plain floats."""
    obs = SimpleNamespace(rho=np.array([0.5, 1.2, 0.9, 1.8, 0.3]))
    out = _extract_overloaded_rho(obs, [1, 3])
    assert out == [1.2, 1.8]
    assert all(isinstance(v, float) for v in out)


def test_extract_overloaded_rho_from_list():
    obs = SimpleNamespace(rho=[0.4, 0.8, 1.1])
    assert _extract_overloaded_rho(obs, [0, 2]) == [0.4, 1.1]


def test_extract_overloaded_rho_none_when_no_rho_attr():
    obs = SimpleNamespace()  # no rho
    assert _extract_overloaded_rho(obs, [0, 1]) is None


def test_extract_overloaded_rho_none_for_empty_ids():
    obs = SimpleNamespace(rho=np.array([0.5, 1.2]))
    assert _extract_overloaded_rho(obs, []) is None


def test_extract_overloaded_rho_none_for_none_obs():
    assert _extract_overloaded_rho(None, [0]) is None


def test_extract_overloaded_rho_none_on_index_error():
    obs = SimpleNamespace(rho=[0.1, 0.2])
    # index 5 is out of range — must be caught, not propagated.
    assert _extract_overloaded_rho(obs, [5]) is None


# ---------------------------------------------------------------------
# build_recommender_inputs
# ---------------------------------------------------------------------

def _full_context(env=None, obs_simu_defaut=None,
                  n_grid=None, n_grid_defaut=None,
                  lines_overloaded_ids_kept=None, pre_existing_rho=None):
    obs_d = obs_simu_defaut
    if obs_d is None:
        obs_d = SimpleNamespace(rho=np.array([0.0, 1.2, 0.5, 1.5]))
    ctx = {
        "obs": "obs",
        "obs_simu_defaut": obs_d,
        "current_lines_defaut": ["L1"],
        "lines_overloaded_names": ["L2", "L4"],
        "lines_overloaded_ids": [1, 3],
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
        "lines_overloaded_ids_kept": lines_overloaded_ids_kept,
        "pre_existing_rho": pre_existing_rho if pre_existing_rho is not None else {},
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
    assert inputs.timestep == 7
    assert inputs.lines_defaut == ["L1"]
    assert inputs.lines_overloaded_names == ["L2", "L4"]
    assert inputs.lines_overloaded_ids == [1, 3]
    assert inputs.overflow_graph == "g"
    assert inputs.hubs == ["h1"]


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
        "obs_simu_defaut": SimpleNamespace(),
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
    assert inputs.use_dc is False
    assert inputs.is_pypowsybl is True
    # Pre-computed step-1 outcome is None when missing from context.
    assert inputs.lines_overloaded_rho is None
    assert inputs.lines_overloaded_ids_kept is None
    assert inputs.pre_existing_rho is None


# ---------------------------------------------------------------------
# build_recommender_inputs: network handles
# ---------------------------------------------------------------------

def test_build_recommender_inputs_uses_explicit_n_grid_when_present():
    fake_net = object()
    ctx = _full_context(n_grid=fake_net)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network is fake_net


def test_build_recommender_inputs_network_defaut_from_obs_simu():
    fake_net = object()
    obs_simu = SimpleNamespace(
        rho=np.array([0.5, 1.2, 0.9, 1.8]),
        _network_manager=SimpleNamespace(network=fake_net),
    )
    ctx = _full_context(obs_simu_defaut=obs_simu)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network_defaut is fake_net


def test_build_recommender_inputs_uses_explicit_n_grid_defaut_when_present():
    primary = object()
    ctx = _full_context(n_grid_defaut=primary)
    inputs = build_recommender_inputs(ctx)
    assert inputs.network_defaut is primary


# ---------------------------------------------------------------------
# build_recommender_inputs: pre-computed step-1 outcome
# ---------------------------------------------------------------------

def test_build_recommender_inputs_pre_extracts_overloaded_rho():
    """obs.rho[ids] is computed once and exposed on the DTO so models
    don't repeat the indexing."""
    obs_simu = SimpleNamespace(rho=np.array([0.5, 1.2, 0.5, 1.8]))
    ctx = _full_context(obs_simu_defaut=obs_simu)
    inputs = build_recommender_inputs(ctx)
    assert inputs.lines_overloaded_rho == [1.2, 1.8]
    assert all(isinstance(v, float) for v in inputs.lines_overloaded_rho)


def test_build_recommender_inputs_overloaded_rho_aligned_with_names():
    """The rho list is in the same order as ``lines_overloaded_names``
    (and ``lines_overloaded_ids``)."""
    obs_simu = SimpleNamespace(rho=np.array([0.0, 1.4, 0.5, 1.7]))
    ctx = _full_context(obs_simu_defaut=obs_simu)
    inputs = build_recommender_inputs(ctx)
    assert len(inputs.lines_overloaded_rho) == len(inputs.lines_overloaded_names)
    # Pair-up: names[i] <-> ids[i] <-> rho[i]
    paired = dict(zip(inputs.lines_overloaded_names, inputs.lines_overloaded_rho))
    assert paired == {"L2": 1.4, "L4": 1.7}


def test_build_recommender_inputs_overloaded_rho_none_when_obs_lacks_rho():
    ctx = _full_context(obs_simu_defaut=SimpleNamespace())  # no .rho
    inputs = build_recommender_inputs(ctx)
    assert inputs.lines_overloaded_rho is None


def test_build_recommender_inputs_passes_through_ids_kept():
    ctx = _full_context(lines_overloaded_ids_kept=[1])
    inputs = build_recommender_inputs(ctx)
    assert inputs.lines_overloaded_ids_kept == [1]


def test_build_recommender_inputs_ids_kept_is_subset_of_ids():
    """Sanity check: the kept list is by construction a subset of
    ``lines_overloaded_ids`` (the pipeline drops overloads that would
    island substations)."""
    ctx = _full_context(lines_overloaded_ids_kept=[1])
    inputs = build_recommender_inputs(ctx)
    assert set(inputs.lines_overloaded_ids_kept).issubset(set(inputs.lines_overloaded_ids))


def test_build_recommender_inputs_ids_kept_independent_from_context():
    """Mutating the list on the DTO doesn't leak back into context."""
    kept = [1]
    ctx = _full_context(lines_overloaded_ids_kept=kept)
    inputs = build_recommender_inputs(ctx)
    inputs.lines_overloaded_ids_kept.append(99)
    assert kept == [1]


def test_build_recommender_inputs_passes_through_pre_existing_rho():
    ctx = _full_context(pre_existing_rho={0: 1.05, 2: 1.01})
    inputs = build_recommender_inputs(ctx)
    assert inputs.pre_existing_rho == {0: 1.05, 2: 1.01}


def test_build_recommender_inputs_pre_existing_rho_independent_from_context():
    pre = {0: 1.1}
    ctx = _full_context(pre_existing_rho=pre)
    inputs = build_recommender_inputs(ctx)
    inputs.pre_existing_rho[99] = 0.0
    # Original dict in context is untouched.
    assert 99 not in pre


def test_build_recommender_inputs_empty_pre_existing_is_preserved_as_empty():
    """Empty dict is meaningful (= no pre-existing overload) and must NOT
    be coerced to None."""
    ctx = _full_context(pre_existing_rho={})
    inputs = build_recommender_inputs(ctx)
    assert inputs.pre_existing_rho == {}


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
    assert captured["lines_overloaded_ids"] == [2]
    assert captured["dict_action"] == {"a": {}}
