# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Tests for the typed ActionType vocabulary (revision R5)."""
from __future__ import annotations

import pytest

from expert_op4grid_recommender.action_evaluation.action_types import (
    ActionType,
    classify_by_description,
    coerce,
)


def test_enum_values_are_the_historical_strings():
    assert ActionType.OPEN_LINE.value == "open_line"
    assert ActionType.CLOSE_COUPLING.value == "close_coupling"
    assert ActionType.PST_TAP.value == "pst_tap"
    assert ActionType.GEN_REDISPATCH.value == "gen_redispatch"
    assert {t.value for t in ActionType} == {
        "open_line", "open_line_load", "open_load", "open_gen",
        "close_line", "close_line_load", "close_load",
        "open_coupling", "close_coupling", "pst_tap",
        "load_power_reduction", "gen_power_reduction", "gen_redispatch",
        "unknown",
    }


@pytest.mark.parametrize("t,expected", [
    (ActionType.OPEN_LINE, dict(line=True, load=False, gen=False, coup=False, op=True, cl=False, topo=True)),
    (ActionType.OPEN_LINE_LOAD, dict(line=True, load=True, gen=False, coup=False, op=True, cl=False, topo=False)),
    (ActionType.CLOSE_COUPLING, dict(line=False, load=False, gen=False, coup=True, op=False, cl=True, topo=True)),
    (ActionType.OPEN_COUPLING, dict(line=False, load=False, gen=False, coup=True, op=True, cl=False, topo=True)),
    (ActionType.OPEN_GEN, dict(line=False, load=False, gen=True, coup=False, op=True, cl=False, topo=False)),
    (ActionType.PST_TAP, dict(line=False, load=False, gen=False, coup=False, op=False, cl=False, topo=True)),
])
def test_category_predicates_match_substring_semantics(t, expected):
    assert t.involves_line is expected["line"]
    assert t.involves_load is expected["load"]
    assert t.involves_gen is expected["gen"]
    assert t.involves_coupling is expected["coup"]
    assert t.is_open is expected["op"]
    assert t.is_close is expected["cl"]
    assert t.is_topological is expected["topo"]


def test_coerce_roundtrips_and_defaults_unknown():
    assert coerce("open_line") is ActionType.OPEN_LINE
    assert coerce(ActionType.PST_TAP) is ActionType.PST_TAP
    assert coerce("not_a_type") is ActionType.UNKNOWN


# --- classify_by_description mirrors the old cascade precedence ---

def _no_line_load():
    return (False, False)


def test_coupling_precedence_over_ouverture():
    # "COUPL" wins even though "Ouverture" is present.
    assert classify_by_description("Ouverture COUPL S1", _no_line_load) is ActionType.OPEN_COUPLING
    assert classify_by_description("Fermeture COUPL S1", _no_line_load) is ActionType.CLOSE_COUPLING
    assert classify_by_description("TRO. quelque chose", _no_line_load) is ActionType.CLOSE_COUPLING


def test_pst_tap_keywords():
    assert classify_by_description("Variation de slot de 2", _no_line_load) is ActionType.PST_TAP
    assert classify_by_description("change TAP now", _no_line_load) is ActionType.PST_TAP


def test_opening_branches_use_lazy_has_line_load():
    assert classify_by_description("Ouverture centrale X", _no_line_load) is ActionType.OPEN_GEN
    assert classify_by_description("Ouverture ligne", lambda: (True, False)) is ActionType.OPEN_LINE
    assert classify_by_description("Ouverture charge", lambda: (False, True)) is ActionType.OPEN_LOAD
    assert classify_by_description("Ouverture both", lambda: (True, True)) is ActionType.OPEN_LINE_LOAD
    assert classify_by_description("Ouverture nothing", _no_line_load) is ActionType.UNKNOWN


def test_closing_branches():
    assert classify_by_description("Fermeture ligne", lambda: (True, False)) is ActionType.CLOSE_LINE
    assert classify_by_description("reconnection", lambda: (True, True)) is ActionType.CLOSE_LINE_LOAD


def test_no_keyword_is_unknown():
    assert classify_by_description("", _no_line_load) is ActionType.UNKNOWN
    assert classify_by_description("random text", _no_line_load) is ActionType.UNKNOWN
