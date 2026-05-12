# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Pluggable recommendation model interface.

Defines the contract every recommendation model must honour so it can be
plugged into the analysis pipeline interchangeably with the built-in
expert system.

Design principles
-----------------
1. **Strategy pattern** — same input / output contract for every model.
2. **Capability flags** — ``requires_overflow_graph`` lets the caller
   skip the expensive step-1 graph build when the model does not need it.
3. **Param introspection** — :meth:`params_spec` enumerates parameters
   the model actually consumes; clients (UI, REST API) hide everything
   else, so the operator never sets a knob the model will ignore.
4. **Homogeneous output** — ``{action_id -> action_object}``; reassessment
   into action cards lives downstream in
   :mod:`expert_op4grid_recommender.utils.reassessment` and works for any
   model that emits raw actions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Literal, Optional


@dataclass
class RecommenderInputs:
    """Every input a recommendation model may consume.

    Always populated by the pipeline:

    - ``obs``                       initial network observation (N state)
    - ``obs_defaut``                post-fault observation (N-K state)
    - ``lines_defaut``              names of the lines forming the contingency
    - ``lines_overloaded_names``    names of lines under constraint
    - ``lines_overloaded_ids``      indices into ``obs_defaut.name_line``
    - ``dict_action``               action dictionary
    - ``env``                       simulation environment
    - ``classifier``                :class:`ActionClassifier` instance
    - ``timestep``                  current timestep

    Network handles (each paired with an observation):

    - ``network``         pypowsybl :class:`pypowsybl.network.Network`
                          paired with ``obs`` (N state). Exposed alongside
                          the observation so models can read topology /
                          device properties (lines, generators, voltage
                          levels, transformers, switches, ...) without
                          digging through the ``env`` backend internals.
                          May be ``None`` on Grid2Op-only paths that do
                          not expose a pypowsybl network.
    - ``network_defaut``  pypowsybl :class:`Network` paired with
                          ``obs_defaut`` (post-fault N-K state). On the
                          pypowsybl backend this is the same underlying
                          :class:`Network` as ``network`` but with the
                          contingency variant active; sourced from
                          ``obs_defaut._network_manager.network`` when
                          present, otherwise from ``env``. May be
                          ``None`` when no pypowsybl network is exposed.

    Optional — set only when the caller asked for the overflow graph
    (``compute_overflow_graph=True``) AND the chosen model declared
    ``requires_overflow_graph=True``:

    - ``overflow_graph``                  alphaDeesp overflow graph
    - ``distribution_graph``              structured overload distribution graph
    - ``overflow_sim``                    associated alphaDeesp simulator
    - ``hubs``                            hub substation names
    - ``node_name_mapping``               internal index → name mapping
    - ``non_connected_reconnectable_lines``
    - ``lines_non_reconnectable``
    - ``lines_we_care_about``
    - ``maintenance_to_reco_at_t``
    - ``act_reco_maintenance``
    - ``use_dc``
    - ``filtered_candidate_actions``      action IDs retained by the expert
                                          rule filter — exposed so downstream
                                          models can sample among them
    """

    obs: Any
    obs_defaut: Any
    lines_defaut: List[str]
    lines_overloaded_names: List[str]
    lines_overloaded_ids: List[int]
    dict_action: dict
    env: Any
    classifier: Any

    # --- Optional from here on -----------------------------------------
    # Pypowsybl Network handles, paired with the corresponding
    # observations. Both default to None so existing call sites that
    # did not pass them remain valid.
    network: Any = None          # paired with ``obs``         (N state)
    network_defaut: Any = None   # paired with ``obs_defaut``  (N-K state)
    timestep: int = 0

    overflow_graph: Any = None
    distribution_graph: Any = None
    overflow_sim: Any = None
    hubs: Optional[List[str]] = None
    node_name_mapping: Any = None
    non_connected_reconnectable_lines: Optional[List[str]] = None
    lines_non_reconnectable: Optional[List[str]] = None
    lines_we_care_about: Any = None
    maintenance_to_reco_at_t: Any = None
    act_reco_maintenance: Any = None
    use_dc: bool = False
    filtered_candidate_actions: Optional[List[str]] = None

    is_pypowsybl: bool = True
    fast_mode: bool = False

    # Private escape hatch for the expert model — it needs many internal
    # helpers (rho-reduction check, baseline simulation, ...) that would
    # otherwise pollute this DTO. External models must not rely on this
    # field.
    _context: Optional[dict] = None


@dataclass
class SimulatedAction:
    """A single action enriched with its simulated state.

    Schema preserved verbatim from the historical reassessment output so
    every existing action-card consumer keeps working unchanged.
    """

    action: Any
    description_unitaire: Optional[str]
    rho_before: Any
    rho_after: Any
    max_rho: float
    max_rho_line: str
    is_rho_reduction: bool
    observation: Any
    non_convergence: Optional[str] = None


@dataclass
class RecommenderOutput:
    """What every model returns.

    ``prioritized_actions`` is ``{action_id: action_object}`` — the raw
    actions selected by the model, NOT enriched. The reassessment step
    in :mod:`expert_op4grid_recommender.utils.reassessment` wraps each
    one with simulated rho / observation / etc.

    ``action_scores`` is free-form and may be empty. When present, it
    follows the legacy expert schema:
    ``{category: {scores: {...}, params: {...}, non_convergence: {...}}}``.
    """

    prioritized_actions: dict
    action_scores: dict = field(default_factory=dict)


@dataclass
class ParamSpec:
    """Declarative spec of a single config parameter consumed by a model.

    Clients (REST API, web UI) read this to decide which controls to
    show and which ones to hide for a given model.
    """

    name: str
    label: str
    kind: Literal["int", "float", "bool"]
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    description: Optional[str] = None
    group: Optional[str] = None


class RecommenderModel(ABC):
    """Plug-in interface for recommendation models.

    Concrete subclasses MUST set ``name`` and ``label``, and implement
    :meth:`params_spec` and :meth:`recommend`.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    requires_overflow_graph: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def params_spec(cls) -> List[ParamSpec]:
        """Parameters this model consumes."""

    @abstractmethod
    def recommend(self, inputs: RecommenderInputs, params: dict) -> RecommenderOutput:
        """Produce a set of candidate actions.

        Args:
            inputs: read-only DTO with everything the pipeline gathered.
                Two paired (observation, network) handles are available:
                ``(inputs.obs, inputs.network)`` for the N state and
                ``(inputs.obs_defaut, inputs.network_defaut)`` for the
                post-fault N-K state. Use the network handle for
                topology / device-level queries and the observation for
                state-dependent values (flows, voltages, ...).
            params: ``{ParamSpec.name -> value}`` from the operator.
                Models should look only at keys they declared.

        Returns:
            A :class:`RecommenderOutput` with at least
            ``prioritized_actions``.
        """
