# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""Tests for the pluggable :mod:`expert_op4grid_recommender.models.base` contract."""
from __future__ import annotations

import pytest

from expert_op4grid_recommender.models.base import (
    ParamSpec,
    RecommenderInputs,
    RecommenderModel,
    RecommenderOutput,
    SimulatedAction,
)


# ---------------------------------------------------------------------
# RecommenderModel ABC contract
# ---------------------------------------------------------------------

def test_recommender_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        RecommenderModel()  # type: ignore[abstract]


def test_subclass_missing_recommend_is_still_abstract():
    class _Half(RecommenderModel):
        name = "half"
        label = "Half"

        @classmethod
        def params_spec(cls):
            return []

    with pytest.raises(TypeError):
        _Half()  # type: ignore[abstract]


def test_subclass_missing_params_spec_is_still_abstract():
    class _Half(RecommenderModel):
        name = "half"
        label = "Half"

        def recommend(self, inputs, params):
            return RecommenderOutput(prioritized_actions={})

    with pytest.raises(TypeError):
        _Half()  # type: ignore[abstract]


def test_complete_subclass_instantiates_with_defaults():
    class _Complete(RecommenderModel):
        name = "complete"
        label = "Complete"

        @classmethod
        def params_spec(cls):
            return [ParamSpec("k", "K", "int", default=1)]

        def recommend(self, inputs, params):
            return RecommenderOutput(prioritized_actions={})

    instance = _Complete()
    assert instance.name == "complete"
    assert instance.label == "Complete"
    # Default for the capability flag.
    assert instance.requires_overflow_graph is False


# ---------------------------------------------------------------------
# RecommenderInputs DTO
# ---------------------------------------------------------------------

def _minimal_inputs(**overrides):
    base = dict(
        obs="obs",
        obs_defaut="obs_d",
        lines_defaut=["L1"],
        lines_overloaded_names=["L2"],
        lines_overloaded_ids=[2],
        dict_action={"a": {}},
        env="env",
        classifier="cls",
    )
    base.update(overrides)
    return RecommenderInputs(**base)


def test_recommender_inputs_default_optional_fields_are_none_or_false():
    inputs = _minimal_inputs()
    assert inputs.timestep == 0
    assert inputs.overflow_graph is None
    assert inputs.distribution_graph is None
    assert inputs.overflow_sim is None
    assert inputs.hubs is None
    assert inputs.node_name_mapping is None
    assert inputs.non_connected_reconnectable_lines is None
    assert inputs.lines_non_reconnectable is None
    assert inputs.lines_we_care_about is None
    assert inputs.maintenance_to_reco_at_t is None
    assert inputs.act_reco_maintenance is None
    assert inputs.use_dc is False
    assert inputs.filtered_candidate_actions is None
    assert inputs.is_pypowsybl is True
    assert inputs.fast_mode is False
    assert inputs._context is None


def test_recommender_inputs_accepts_overflow_graph_artifacts():
    inputs = _minimal_inputs(
        overflow_graph="g", distribution_graph="gd",
        hubs=["h1", "h2"], filtered_candidate_actions=["a1"],
    )
    assert inputs.overflow_graph == "g"
    assert inputs.distribution_graph == "gd"
    assert inputs.hubs == ["h1", "h2"]
    assert inputs.filtered_candidate_actions == ["a1"]


# ---------------------------------------------------------------------
# RecommenderOutput DTO
# ---------------------------------------------------------------------

def test_recommender_output_default_scores_is_empty_dict():
    out = RecommenderOutput(prioritized_actions={"a": object()})
    assert out.action_scores == {}
    # Default factory — each instance gets its own dict.
    other = RecommenderOutput(prioritized_actions={})
    other.action_scores["x"] = 1
    assert out.action_scores == {}


# ---------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------

def test_param_spec_minimal():
    p = ParamSpec("foo", "Foo", "int", default=5)
    assert p.min is None
    assert p.max is None
    assert p.description is None
    assert p.group is None


def test_param_spec_full():
    p = ParamSpec(
        "bar", "Bar", "float",
        default=0.1, min=0, max=1,
        description="d", group="g",
    )
    assert p.min == 0
    assert p.max == 1
    assert p.description == "d"
    assert p.group == "g"


# ---------------------------------------------------------------------
# SimulatedAction
# ---------------------------------------------------------------------

def test_simulated_action_default_non_convergence_is_none():
    sa = SimulatedAction(
        action="a", description_unitaire="d",
        rho_before=[0.5], rho_after=[0.4],
        max_rho=0.4, max_rho_line="L1",
        is_rho_reduction=True, observation="o",
    )
    assert sa.non_convergence is None
