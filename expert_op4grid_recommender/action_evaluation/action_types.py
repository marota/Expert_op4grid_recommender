# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Typed action-type vocabulary (revision R5).

The classifier historically returned free-form strings (``"open_line"``,
``"close_coupling"``, …) that downstream code matched by **substring**
(``"line" in action_type``, ``"coupling" in action_type``) — brittle, and the
source of the C7 rules bypass. :class:`ActionType` gives that vocabulary a
single typed home. Every enum *value* is byte-identical to the historical
string, so ``ActionType.OPEN_LINE.value == "open_line"`` and existing string
consumers are unaffected.

The description-keyword classification is expressed as a declarative, ordered
rule cascade in :func:`classify_by_description` — one place, read top to bottom,
matching the precedence the classifier used to inline.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable, Tuple


class ActionType(Enum):
    """Canonical action-type tokens. Values equal the historical strings."""

    OPEN_LINE = "open_line"
    OPEN_LINE_LOAD = "open_line_load"
    OPEN_LOAD = "open_load"
    OPEN_GEN = "open_gen"
    CLOSE_LINE = "close_line"
    CLOSE_LINE_LOAD = "close_line_load"
    CLOSE_LOAD = "close_load"
    OPEN_COUPLING = "open_coupling"
    CLOSE_COUPLING = "close_coupling"
    PST_TAP = "pst_tap"
    LOAD_POWER_REDUCTION = "load_power_reduction"
    GEN_POWER_REDUCTION = "gen_power_reduction"
    GEN_REDISPATCH = "gen_redispatch"
    UNKNOWN = "unknown"

    # -- Category predicates: the single home for what the scattered
    #    ``"<kw>" in action_type`` substring checks used to test. Defined on
    #    ``.value`` so they are provably identical to the historical semantics.
    @property
    def involves_line(self) -> bool:
        return "line" in self.value

    @property
    def involves_load(self) -> bool:
        return "load" in self.value

    @property
    def involves_gen(self) -> bool:
        return "gen" in self.value

    @property
    def involves_coupling(self) -> bool:
        return "coupling" in self.value

    @property
    def is_open(self) -> bool:
        return "open" in self.value

    @property
    def is_close(self) -> bool:
        return "close" in self.value

    @property
    def is_topological(self) -> bool:
        """A topological (non-injection) action — the gate for the expert rule
        block (historically ``"load" not in t and "gen" not in t``)."""
        return not self.involves_load and not self.involves_gen


def coerce(action_type: "str | ActionType") -> ActionType:
    """Return an :class:`ActionType` for a raw string or enum (``UNKNOWN`` on
    an unrecognised string)."""
    if isinstance(action_type, ActionType):
        return action_type
    try:
        return ActionType(action_type)
    except ValueError:
        return ActionType.UNKNOWN


def classify_by_description(
    description: str,
    has_line_load: Callable[[], Tuple[bool, bool]],
) -> ActionType:
    """Classify an action from its human-readable description.

    Ordered, first-match-wins cascade — byte-identical to the historical inline
    ``if/elif`` chain in ``ActionClassifier.identify_action_type``. ``has_line_load``
    is a lazy callback returning ``(has_line, has_load)``; it is only invoked for
    the open/close branches that need it (matching the original).
    """
    desc = description or ""
    if "COUPL" in desc or "TRO." in desc:
        return ActionType.OPEN_COUPLING if "Ouverture" in desc else ActionType.CLOSE_COUPLING
    if "Variation de slot" in desc or "tap" in desc.lower():
        return ActionType.PST_TAP
    if "Ouverture" in desc or "deconnection" in desc:
        low = desc.lower()
        if "generator" in low or "production" in low or "centrale" in low:
            return ActionType.OPEN_GEN
        has_line, has_load = has_line_load()
        if has_load and has_line:
            return ActionType.OPEN_LINE_LOAD
        if has_line:
            return ActionType.OPEN_LINE
        if has_load:
            return ActionType.OPEN_LOAD
        return ActionType.UNKNOWN
    if "Fermeture" in desc or "reconnection" in desc:
        has_line, has_load = has_line_load()
        if has_load and has_line:
            return ActionType.CLOSE_LINE_LOAD
        if has_line:
            return ActionType.CLOSE_LINE
        if has_load:
            return ActionType.CLOSE_LOAD
        return ActionType.UNKNOWN
    return ActionType.UNKNOWN
