# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""Tests for :mod:`expert_op4grid_recommender.utils.reassessment`.

These tests focus on the pure logic that does not require a live
pypowsybl / grid2op environment: input DTO mapping, non-convergence
propagation, and graceful failure of the superposition helper.
"""
from __future__ import annotations

from unittest.mock import patch

from expert_op4grid_recommender.utils.reassessment import (
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
    # No entry created when the action_id isn't part of any category.
    assert out["line_reconnection"].get("non_convergence", {}) == {}


def test_propagate_creates_non_convergence_dict_when_absent():
    detailed = {"a1": {"non_convergence": "X"}}
    scores = {"line_reconnection": {"scores": {"a1": 0.1}, "params": {}}}
    out = propagate_non_convergence_to_scores(detailed, scores)
    assert out["line_reconnection"]["non_convergence"] == {"a1": "X"}


# ---------------------------------------------------------------------
# build_recommender_inputs
# ---------------------------------------------------------------------

def _full_context():
    return {
        "obs": "obs",
        "obs_simu_defaut": "obs_d",
        "current_lines_defaut": ["L1"],
        "lines_overloaded_names": ["L2"],
        "lines_overloaded_ids": [2],
        "dict_action": {"a": {}},
        "env": "env",
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
    # The expert model relies on this escape hatch to reach internal helpers.
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
