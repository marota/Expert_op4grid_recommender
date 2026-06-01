# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""Tests for :class:`ExpertRecommender`."""
from __future__ import annotations

import pytest

from expert_op4grid_recommender.models.base import RecommenderInputs
from expert_op4grid_recommender.models.expert import ExpertRecommender


def test_expert_metadata_is_canonical():
    assert ExpertRecommender.name == "expert"
    assert ExpertRecommender.label == "Expert system"
    assert ExpertRecommender.requires_overflow_graph is True


def test_expert_params_spec_includes_all_legacy_knobs():
    names = {p.name for p in ExpertRecommender.params_spec()}
    for required in (
        "n_prioritized_actions",
        "min_line_reconnections",
        "min_close_coupling",
        "min_open_coupling",
        "min_line_disconnections",
        "min_pst",
        "min_load_shedding",
        "min_renewable_curtailment_actions",
        "ignore_reconnections",
    ):
        assert required in names, f"missing param {required!r}"


def test_expert_n_prioritized_is_an_int_with_min_one():
    n = next(
        p for p in ExpertRecommender.params_spec()
        if p.name == "n_prioritized_actions"
    )
    assert n.kind == "int"
    assert n.min == 1
    assert n.max == 200


def test_expert_ignore_reconnections_is_bool():
    ir = next(
        p for p in ExpertRecommender.params_spec()
        if p.name == "ignore_reconnections"
    )
    assert ir.kind == "bool"


def test_expert_recommend_without_context_raises():
    """The expert model needs the internal pipeline context.

    Calling :meth:`recommend` directly from external code is a misuse —
    it should error out clearly so external callers either supply the
    context (private path) or use a model designed to consume the DTO.
    """
    rec = ExpertRecommender()
    inputs = RecommenderInputs(
        obs="o", obs_defaut="od",
        lines_defaut=[], lines_overloaded_names=[],
        lines_overloaded_ids=[], dict_action={},
        env="e", classifier="c",
    )
    with pytest.raises(RuntimeError, match="full analysis context"):
        rec.recommend(inputs, params={"n_prioritized_actions": 5})
