# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
"""Tests for ``filtered_candidate_actions`` propagation through the DTO.

Regression coverage for the silent bug where
:func:`_run_expert_action_filter` populated
``context['filtered_candidate_actions']`` correctly but
:func:`build_recommender_inputs` never forwarded it to
:class:`RecommenderInputs`. The DTO field defaulted to ``None``, the
recommender thought the filter had been skipped, and the fallback
sampled from the full dictionary.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from expert_op4grid_recommender.utils.reassessment import build_recommender_inputs


def _full_context(**overrides):
    base = {
        "obs": "obs",
        "obs_simu_defaut": SimpleNamespace(rho=np.array([0.0, 1.2])),
        "current_lines_defaut": ["L1"],
        "lines_overloaded_names": ["L2"],
        "lines_overloaded_ids": [1],
        "dict_action": {"a": {}},
        "env": "env",
        "classifier": "cls",
        "current_timestep": 0,
        "lines_overloaded_ids_kept": [1],
        "pre_existing_rho": {},
        "is_pypowsybl": True,
        "actual_fast_mode": False,
    }
    base.update(overrides)
    return base


def test_forwards_filtered_candidate_actions_from_context():
    """Regression: the DTO must carry the rule-validator-filtered list.

    The log we hit was:
        Summary: 23974 out of 24347 actions were filtered by expert rules.
        ...
        RandomOverflowRecommender: filtered_candidate_actions is None
    The filter ran fine, but the DTO had a stale default. This test
    pins the wiring.
    """
    ctx = _full_context()
    ctx["filtered_candidate_actions"] = ["a1", "a2", "a3"]
    inputs = build_recommender_inputs(ctx)
    assert inputs.filtered_candidate_actions == ["a1", "a2", "a3"]


def test_is_none_when_filter_did_not_run():
    """Missing key → None on the DTO. Models distinguish this case
    ("filter didn't run, fall back") from the empty-list case below."""
    ctx = _full_context()
    ctx.pop("filtered_candidate_actions", None)
    inputs = build_recommender_inputs(ctx)
    assert inputs.filtered_candidate_actions is None


def test_empty_list_preserved_not_coerced_to_none():
    """Filter ran but nothing passed → the DTO must keep [] (not None).

    Downstream models use this exact distinction: [] means "the filter
    ran and returned no candidates, do not fall back to dict_action".
    Coercing it to None would re-trigger the silent fallback the
    sampling models are explicitly trying to avoid.
    """
    ctx = _full_context()
    ctx["filtered_candidate_actions"] = []
    inputs = build_recommender_inputs(ctx)
    assert inputs.filtered_candidate_actions == []
    # Strict identity-distinction from None.
    assert inputs.filtered_candidate_actions is not None


def test_defensive_copy_dto_mutation_does_not_leak_to_context():
    original = ["a1", "a2"]
    ctx = _full_context()
    ctx["filtered_candidate_actions"] = original
    inputs = build_recommender_inputs(ctx)
    inputs.filtered_candidate_actions.append("a3")
    assert original == ["a1", "a2"], (
        "Mutating the DTO must not leak back into the shared context"
    )


def test_context_mutation_after_build_does_not_leak_to_dto():
    """Symmetric: mutating the context list later doesn't change the DTO."""
    ctx = _full_context()
    ctx["filtered_candidate_actions"] = ["a1"]
    inputs = build_recommender_inputs(ctx)
    ctx["filtered_candidate_actions"].append("a99")
    assert inputs.filtered_candidate_actions == ["a1"]


def test_large_filtered_list_propagates_intact():
    """Pin the wiring for the realistic case from the log
    (373 actions left after the rule validator ran on a 24k dict)."""
    ctx = _full_context()
    ctx["filtered_candidate_actions"] = [f"action_{i}" for i in range(373)]
    inputs = build_recommender_inputs(ctx)
    assert len(inputs.filtered_candidate_actions) == 373
    assert inputs.filtered_candidate_actions[0] == "action_0"
    assert inputs.filtered_candidate_actions[-1] == "action_372"
