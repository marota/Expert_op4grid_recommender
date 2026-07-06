# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Data model for action discovery — the ``FamilyResult`` store (revision R5).

Before this module, ``ActionDiscoverer`` carried the per-family discovery
outcome as ~40 hand-repeated instance attributes (``identified_reconnections``,
``scores_reconnections``, ``params_reconnections``, … one 5-attribute
"quintuplet" per family, plus a PST family that was *never initialised* and so
had to be read through ``getattr`` in the orchestrator). The ``action_scores``
assembly and the two prioritization phases then re-listed all eight families by
hand — 100+ lines of copy-paste in which a duplicated call had already slipped
in twice.

This module collapses that to **one typed record per family** kept in
``self.results: Dict[str, FamilyResult]``, described by a declarative
:data:`FAMILY_SPECS` registry. The registry drives:

* :func:`install_family_result_properties` — installs back-compat ``@property``
  bridges so every existing ``self.identified_reconnections`` /
  ``self.scores_splits_dict`` / ``self.scores_pst_actions`` read *and* write
  (in the family mixins and in the tests) transparently targets
  ``self.results[key].<field>``. No mixin or test changes; the store is unified.
* the data-driven ``action_scores`` build and prioritization loops in the
  orchestrator.

The legacy attribute stems are irregular by history (``splits`` exposes its
scores as ``scores_splits_dict``, PST as ``scores_pst_actions``); the registry
records that mapping explicitly so behaviour is byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FamilyResult:
    """Discovery outcome for a single action family.

    ``identified`` maps ``action_id -> action`` (the candidates kept). ``scores``
    and ``params`` are the per-action score / hypothesis maps surfaced in
    ``action_scores``. ``effective`` / ``ineffective`` are the informational
    per-candidate simulation verdicts (consumed only for logging).
    ``non_convergence`` is filled downstream (reassessment) and is always an
    empty dict at discovery time.
    """

    identified: Dict[str, Any] = field(default_factory=dict)
    effective: List[Any] = field(default_factory=list)
    ineffective: List[Any] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    non_convergence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FamilySpec:
    """Declarative description of one action family.

    ``*_attr`` are the historical (irregular) instance-attribute names the
    property bridge must reproduce; ``scores_key`` is the family's key in the
    returned ``action_scores`` dict.
    """

    key: str
    identified_attr: str
    effective_attr: str
    ineffective_attr: str
    scores_attr: str
    params_attr: str
    scores_key: str


# Canonical family registry. The ``key`` is the stable internal token used for
# ``self.results[key]``; the ``*_attr`` strings are the legacy names bridged for
# backward compatibility.
FAMILY_SPECS: Dict[str, FamilySpec] = {
    spec.key: spec
    for spec in (
        FamilySpec("reconnections", "identified_reconnections",
                   "effective_reconnections", "ineffective_reconnections",
                   "scores_reconnections", "params_reconnections",
                   "line_reconnection"),
        FamilySpec("merges", "identified_merges", "effective_merges",
                   "ineffective_merges", "scores_merges", "params_merges",
                   "close_coupling"),
        FamilySpec("splits", "identified_splits", "effective_splits",
                   "ineffective_splits", "scores_splits_dict",
                   "params_splits_dict", "open_coupling"),
        FamilySpec("disconnections", "identified_disconnections",
                   "effective_disconnections", "ineffective_disconnections",
                   "scores_disconnections", "params_disconnections",
                   "line_disconnection"),
        FamilySpec("pst", "identified_pst_actions", "effective_pst_actions",
                   "ineffective_pst_actions", "scores_pst_actions",
                   "params_pst_actions", "pst_tap"),
        FamilySpec("load_shedding", "identified_load_shedding",
                   "effective_load_shedding", "ineffective_load_shedding",
                   "scores_load_shedding", "params_load_shedding",
                   "load_shedding"),
        FamilySpec("renewable_curtailment", "identified_renewable_curtailment",
                   "effective_renewable_curtailment",
                   "ineffective_renewable_curtailment",
                   "scores_renewable_curtailment",
                   "params_renewable_curtailment", "renewable_curtailment"),
        FamilySpec("redispatch", "identified_redispatch", "effective_redispatch",
                   "ineffective_redispatch", "scores_redispatch",
                   "params_redispatch", "redispatch"),
    )
}

#: Order in which families are emitted into the ``action_scores`` dict. Kept
#: byte-identical to the historical hand-written literal (reconnection,
#: disconnection, open, close, pst, load_shedding, renewable_curtailment,
#: redispatch) so any consumer comparing serialised output is unaffected.
ACTION_SCORES_ORDER: List[str] = [
    "reconnections",
    "disconnections",
    "splits",
    "merges",
    "pst",
    "load_shedding",
    "renewable_curtailment",
    "redispatch",
]

#: Prioritization runs in two ordered passes. The **min phase** enforces the
#: per-type MIN_* floors under a cap that admits every floor; the **fill phase**
#: tops up to ``n_action_max`` with per-family fill caps. Both are one entry per
#: family — which is the R5 fix: the historical hand-written calls had
#: ``renewable_curtailment`` slipped in *twice* per phase (a real latent
#: double-add), impossible to express here. The two orders differ on purpose
#: (PST is 3rd in the floor pass, 5th in the fill pass); both are load-bearing.
MIN_PHASE_ORDER: List[str] = [
    "reconnections", "merges", "pst", "splits", "disconnections",
    "renewable_curtailment", "load_shedding", "redispatch",
]
FILL_PHASE_ORDER: List[str] = [
    "reconnections", "merges", "splits", "disconnections", "pst",
    "renewable_curtailment", "load_shedding", "redispatch",
]

#: Per-family ``config`` attribute holding its guaranteed-minimum count, read in
#: the min phase (defensively via ``getattr(config, attr, 0)``).
FAMILY_MIN_CONFIG_ATTR: Dict[str, str] = {
    "reconnections": "MIN_LINE_RECONNECTIONS",
    "merges": "MIN_CLOSE_COUPLING",
    "pst": "MIN_PST",
    "splits": "MIN_OPEN_COUPLING",
    "disconnections": "MIN_LINE_DISCONNECTIONS",
    "renewable_curtailment": "MIN_RENEWABLE_CURTAILMENT",
    "load_shedding": "MIN_LOAD_SHEDDING",
    "redispatch": "MIN_REDISPATCH",
}

#: Field name on :class:`FamilyResult` for each of the five bridged legacy
#: attribute roles.
_BRIDGED_FIELDS = (
    ("identified_attr", "identified"),
    ("effective_attr", "effective"),
    ("ineffective_attr", "ineffective"),
    ("scores_attr", "scores"),
    ("params_attr", "params"),
)


def new_results() -> Dict[str, FamilyResult]:
    """Return a fresh ``{family_key: FamilyResult()}`` store for all families."""
    return {key: FamilyResult() for key in FAMILY_SPECS}


def _make_bridge_property(family_key: str, result_field: str) -> property:
    """A ``property`` proxying a legacy attribute to ``results[key].<field>``."""

    def getter(self):
        return getattr(self.results[family_key], result_field)

    def setter(self, value):
        setattr(self.results[family_key], result_field, value)

    return property(getter, setter)


def install_family_result_properties(cls: type) -> None:
    """Install the legacy-attribute ``@property`` bridges onto ``cls``.

    Every historical per-family attribute (e.g. ``identified_reconnections``,
    ``scores_splits_dict``, ``params_pst_actions``) becomes a read/write proxy
    onto ``self.results[key].<field>``, so existing family-mixin writes and test
    reads keep working unchanged while the canonical store is the typed
    ``FamilyResult``. Idempotent.
    """
    for spec in FAMILY_SPECS.values():
        for spec_attr, result_field in _BRIDGED_FIELDS:
            legacy_name = getattr(spec, spec_attr)
            setattr(cls, legacy_name, _make_bridge_property(spec.key, result_field))


@dataclass(frozen=True)
class DisconnectionBounds:
    """Memoised flow bounds shared by disconnection and PST scoring (A5 fix).

    Computed once per discovery run from immutable instance state (the
    discoverer is created fresh per Step-2 run), replacing the previous
    ``self._disco_bounds`` / ``self._disco_capacity_map`` pair that PST and line
    disconnection lazily created and disconnection *deleted* at entry/exit — a
    cross-file, order-sensitive ``del`` / ``not hasattr`` protocol that made PST
    scoring silently dependent on disconnections running immediately before it.

    * ``max_overload_flow`` — capacity (pre-cut flow) of the most-loaded
      overloaded line; the reference flow that scores 1.0.
    * ``min_redispatch`` — flow needed to bring the worst overload below 100 %.
    * ``max_redispatch`` — flow the system absorbs before a new line hits 100 %
      (``inf`` when unconstrained).
    * ``capacity_map`` — ``{line_name: max_abs_capacity_MW}`` on the overflow
      graph.
    """

    max_overload_flow: float
    min_redispatch: float
    max_redispatch: float
    capacity_map: Dict[str, float]
