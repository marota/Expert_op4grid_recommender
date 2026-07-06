# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Tests for the discovery ``_results`` data model (revision R5).

Locks in the FamilyResult store, the declarative FAMILY_SPECS registry, the
legacy-attribute property bridge, and — crucially — the invariant that each
prioritization phase lists every family **exactly once**, which is what makes
the historical twice-slipped ``renewable_curtailment`` add structurally
impossible.
"""
from __future__ import annotations

import pytest

from expert_op4grid_recommender.action_evaluation.discovery._results import (
    ACTION_SCORES_ORDER,
    FAMILY_MIN_CONFIG_ATTR,
    FAMILY_SPECS,
    FILL_PHASE_ORDER,
    FamilyResult,
    MIN_PHASE_ORDER,
    install_family_result_properties,
    new_results,
)

FAMILY_KEYS = set(FAMILY_SPECS)


# --- FamilyResult -----------------------------------------------------------

def test_family_result_defaults_are_independent_empties():
    a = FamilyResult()
    b = FamilyResult()
    assert a.identified == {} and a.scores == {} and a.params == {}
    assert a.effective == [] and a.ineffective == [] and a.non_convergence == {}
    a.identified["x"] = 1
    a.effective.append("y")
    # Default factories must not share mutable state across instances.
    assert b.identified == {} and b.effective == []


def test_new_results_has_all_eight_families():
    r = new_results()
    assert set(r) == FAMILY_KEYS
    assert len(r) == 8
    assert all(isinstance(v, FamilyResult) for v in r.values())


# --- Registry ---------------------------------------------------------------

def test_registry_has_eight_families_with_unique_scores_keys():
    assert len(FAMILY_SPECS) == 8
    scores_keys = [s.scores_key for s in FAMILY_SPECS.values()]
    assert len(scores_keys) == len(set(scores_keys))
    assert set(scores_keys) == {
        "line_reconnection", "line_disconnection", "open_coupling",
        "close_coupling", "pst_tap", "load_shedding",
        "renewable_curtailment", "redispatch",
    }


def test_legacy_attr_names_are_unique_across_families():
    """The bridged legacy names must not collide (they become class properties)."""
    names = []
    for spec in FAMILY_SPECS.values():
        names += [spec.identified_attr, spec.effective_attr,
                  spec.ineffective_attr, spec.scores_attr, spec.params_attr]
    assert len(names) == len(set(names)) == 40


def test_irregular_legacy_stems_are_preserved():
    """Splits expose scores as ``scores_splits_dict``; PST as ``scores_pst_actions``
    — the byte-compatible historical names."""
    assert FAMILY_SPECS["splits"].scores_attr == "scores_splits_dict"
    assert FAMILY_SPECS["splits"].params_attr == "params_splits_dict"
    assert FAMILY_SPECS["pst"].identified_attr == "identified_pst_actions"
    assert FAMILY_SPECS["pst"].scores_attr == "scores_pst_actions"


# --- Ordered tables (the R5 no-duplicate guarantee) -------------------------

@pytest.mark.parametrize("order", [MIN_PHASE_ORDER, FILL_PHASE_ORDER, ACTION_SCORES_ORDER])
def test_order_lists_each_family_exactly_once(order):
    assert len(order) == 8
    assert len(set(order)) == 8, f"a family appears more than once: {order}"
    assert set(order) == FAMILY_KEYS


def test_min_phase_config_attr_covers_all_families():
    assert set(FAMILY_MIN_CONFIG_ATTR) == FAMILY_KEYS


def test_min_and_fill_orders_differ():
    """The two phases interleave differently on purpose (PST 3rd vs 5th)."""
    assert MIN_PHASE_ORDER != FILL_PHASE_ORDER
    assert MIN_PHASE_ORDER.index("pst") == 2
    assert FILL_PHASE_ORDER.index("pst") == 4


# --- Property bridge --------------------------------------------------------

def test_property_bridge_round_trips_via_results():
    class Dummy:
        def __init__(self):
            self.results = new_results()

    install_family_result_properties(Dummy)
    d = Dummy()
    # Write through a legacy name -> lands in the FamilyResult store.
    d.identified_reconnections = {"a": 1}
    assert d.results["reconnections"].identified == {"a": 1}
    # Write through the store -> visible on the legacy name.
    d.results["splits"].scores = {"s": 0.5}
    assert d.scores_splits_dict == {"s": 0.5}
    # PST bridged like any other family (no getattr special-casing).
    assert d.scores_pst_actions == {}
    d.scores_pst_actions = {"p": 0.9}
    assert d.results["pst"].scores == {"p": 0.9}
