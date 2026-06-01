# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Pluggable recommendation models.

The analysis pipeline does not hardcode the expert system anymore: it
consumes any class implementing :class:`RecommenderModel`. The expert
implementation lives here next to the contract for convenience; canonical
random examples live in the Co-Study4Grid backend; external (user)
models can be shipped from any third-party package.
"""
from expert_op4grid_recommender.models.base import (
    ParamSpec,
    RecommenderInputs,
    RecommenderModel,
    RecommenderOutput,
    SimulatedAction,
)
from expert_op4grid_recommender.models.expert import ExpertRecommender

__all__ = [
    "ExpertRecommender",
    "ParamSpec",
    "RecommenderInputs",
    "RecommenderModel",
    "RecommenderOutput",
    "SimulatedAction",
]
