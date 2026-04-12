# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""ActionDiscoverer package — split by action family.

Re-exports :class:`ActionDiscoverer`, which composes per-family mixins so behavior matches the previous single-file implementation.
"""
from expert_op4grid_recommender.action_evaluation.discovery._base import DiscovererBase
from expert_op4grid_recommender.action_evaluation.discovery._line_reconnection import LineReconnectionMixin
from expert_op4grid_recommender.action_evaluation.discovery._line_disconnection import LineDisconnectionMixin
from expert_op4grid_recommender.action_evaluation.discovery._node_splitting import NodeSplittingMixin
from expert_op4grid_recommender.action_evaluation.discovery._node_merging import NodeMergingMixin
from expert_op4grid_recommender.action_evaluation.discovery._pst import PSTMixin
from expert_op4grid_recommender.action_evaluation.discovery._load_shedding import LoadSheddingMixin
from expert_op4grid_recommender.action_evaluation.discovery._renewable_curtailment import RenewableCurtailmentMixin
from expert_op4grid_recommender.action_evaluation.discovery._orchestrator import OrchestratorMixin


class ActionDiscoverer(
    OrchestratorMixin,
    LineReconnectionMixin,
    LineDisconnectionMixin,
    NodeSplittingMixin,
    NodeMergingMixin,
    PSTMixin,
    LoadSheddingMixin,
    RenewableCurtailmentMixin,
    DiscovererBase,
):
    """Discovers, evaluates, and prioritizes corrective actions for grid overloads.

    This class composes one mixin per action family (line reconnection, line
    disconnection, node splitting, node merging, PST, load shedding, renewable
    curtailment) on top of :class:`DiscovererBase`, which holds the shared
    state (observations, caches, graph references) populated by ``__init__``.
    Splitting the methods across mixins keeps each module focused on a single
    action family while preserving a single public class.
    """
    pass

__all__ = ["ActionDiscoverer"]
