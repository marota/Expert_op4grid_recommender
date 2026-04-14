#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on
# ExpertOp4Grid principles.
"""Alternative *basic* configuration snapshot.

This module mirrors :mod:`expert_op4grid_recommender.config` but provides a
reference snapshot pointing at the full-France ``env_dijon_v2_assistant``
environment with a smaller, five-action prioritized output. It is consumed by
ad-hoc scripts and notebooks; nothing in the main entry points imports it.

Like the main config, the values below are validated by the pydantic
:class:`Settings` class and re-exported at module level, and each knob can be
overridden via an environment variable with the ``EXPERT_OP4GRID_`` prefix.
"""

from __future__ import annotations

from pathlib import Path

from expert_op4grid_recommender.config import (
    Settings,
    apply_settings_to_namespace,
)

# --- Get Project Root Directory ---
CONFIG_DIR: Path = Path(__file__).parent.resolve()
PROJECT_ROOT: Path = CONFIG_DIR.parent.resolve()

# --- Instantiate a Settings tailored to the *basic* scenario ---
# Explicit overrides here win over env vars, so this module always produces
# the documented snapshot regardless of the calling shell environment.
settings: Settings = Settings(
    ENV_NAME="env_dijon_v2_assistant",
    FILE_ACTION_SPACE_DESC="reduced_model_actions.json",
    CHECK_ACTION_SIMULATION=True,
    N_PRIORITIZED_ACTIONS=5,
    IGNORE_RECONNECTIONS=False,
    IGNORE_LINES_MONITORING=False,
    DO_VISUALIZATION=True,
    MAX_RHO_BOTH_EXTREMITIES=False,
    MIN_LINE_RECONNECTIONS=0,
    MIN_CLOSE_COUPLING=0,
    MIN_OPEN_COUPLING=0,
    MIN_LINE_DISCONNECTIONS=0,
    MIN_PST=0,
    MIN_LOAD_SHEDDING=0,
    MIN_RENEWABLE_CURTAILMENT=0,
)

apply_settings_to_namespace(settings, globals())

# --- Derived values (kept as plain module attributes for backwards compat) ---
CASE_NAME: str = "defaut_" + "_".join(map(str, settings.LINES_DEFAUT)) + "_t" + str(
    settings.TIMESTEP
)

ENV_FOLDER: Path = PROJECT_ROOT / "data"
ENV_PATH: Path = ENV_FOLDER / settings.ENV_NAME
ACTION_SPACE_FOLDER: Path = ENV_FOLDER / "action_space"
ACTION_FILE_PATH: Path = ACTION_SPACE_FOLDER / settings.FILE_ACTION_SPACE_DESC
SAVE_FOLDER_VISUALIZATION: Path = PROJECT_ROOT / "Overflow_Graph"
