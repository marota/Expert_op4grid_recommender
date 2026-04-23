#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on
# ExpertOp4Grid principles.
"""Pydantic-backed configuration module.

All runtime knobs used to live as bare module-level constants. They are now
validated by a :class:`Settings` (``pydantic_settings.BaseSettings``) class
and re-exported at module level for backwards compatibility, so existing
``from expert_op4grid_recommender.config import DATE`` and
``config.ENV_NAME = ...`` call sites keep working unchanged.

Every field on :class:`Settings` can be overridden at process startup via an
environment variable with the prefix ``EXPERT_OP4GRID_``, for example::

    EXPERT_OP4GRID_TIMESTEP=12 python -m expert_op4grid_recommender.main
    EXPERT_OP4GRID_LINES_DEFAUT='["BEON L31CPVAN"]' python ...

Lists and dictionaries accept JSON payloads from env vars;
``LINES_DEFAUT`` additionally accepts a bare line name.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, MutableMapping, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# --- Get Project Root Directory ---
CONFIG_DIR: Path = Path(__file__).parent.resolve()
PROJECT_ROOT: Path = CONFIG_DIR.parent.resolve()

_DEFAULT_PARAM_OPTIONS_EXPERT_OP: Dict[str, Any] = {
    # 0.05 is 5 percent of the max overload flow
    "ThresholdReportOfLine": 0.05,
    # 10 percent de la surcharge max
    "ThersholdMinPowerOfLoop": 0.1,
    # If at least a loop is detected, only keep the ones with a flow of at
    # least 25 percent of the biggest one.
    "ratioToKeepLoop": 0.25,
    # Ratio percentage for reconsidering the flow direction.
    "ratioToReconsiderFlowDirection": 0.75,
    # Max unused lines.
    "maxUnusedLines": 3,
    # Total number of simulated topologies at the final simulation step.
    "totalnumberofsimulatedtopos": 30,
    # Number of simulated topologies per node at the final simulation step.
    "numberofsimulatedtopospernode": 10,
}


class Settings(BaseSettings):
    """Validated configuration for :mod:`expert_op4grid_recommender`.

    Instances pick up values from (in order of precedence): explicit
    constructor arguments, environment variables prefixed with
    ``EXPERT_OP4GRID_``, values loaded from the file pointed at by
    ``EXPERT_OP4GRID_ENV_FILE`` (if set), and finally the defaults below.
    """

    model_config = SettingsConfigDict(
        env_prefix="EXPERT_OP4GRID_",
        env_file=None,
        case_sensitive=True,
        extra="ignore",
        validate_assignment=True,
    )

    # -------------------
    #  Case configuration
    # -------------------
    DATE: datetime = datetime(2024, 12, 7)
    TIMESTEP: int = Field(default=9, ge=0)
    LINES_DEFAUT: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["CHALOL61CPVAN"]
    )

    # -------------------
    #  Environment & paths
    # -------------------
    ENV_NAME: str = "bare_env_small_grid_test"
    FILE_ACTION_SPACE_DESC: str = "reduced_model_actions_test.json"
    LINES_MONITORING_FILE: Optional[str] = None

    # -------------------
    #  User parameters
    # -------------------
    USE_EVALUATION_CONFIG: bool = True
    USE_DC_LOAD_FLOW: bool = False
    DO_CONSOLIDATE_GRAPH: bool = False
    DO_RECO_MAINTENANCE: bool = False
    CHECK_WITH_ACTION_DESCRIPTION: bool = True
    DRAW_ONLY_SIGNIFICANT_EDGES: bool = True
    USE_GRID_LAYOUT: bool = False
    DO_FORCE_OVERLOAD_GRAPH_EVEN_IF_GRAPH_BROKEN_APART: bool = False
    DO_SAVE_DATA_FOR_TEST: bool = False
    CHECK_ACTION_SIMULATION: bool = False
    N_PRIORITIZED_ACTIONS: int = Field(default=20, ge=0)
    IGNORE_RECONNECTIONS: bool = False
    IGNORE_LINES_MONITORING: bool = True
    DO_VISUALIZATION: bool = True
    # Output format for the overflow graph visualization. "pdf" saves the
    # static PDF (default); "html" saves the interactive HTML viewer
    # introduced by ExpertOp4Grid PR #74.
    VISUALIZATION_FORMAT: Literal["pdf", "html"] = "pdf"
    # Only possible for now with the pypowsybl backend.
    MAX_RHO_BOTH_EXTREMITIES: bool = True
    # Factor applied to permanent thermal limits when loading from
    # operational limits.
    MONITORING_FACTOR_THERMAL_LIMITS: float = Field(default=0.95, gt=0.0)
    # 2% — pre-existing overloads are excluded from the analysis unless the
    # current increased by more than this fraction.
    PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD: float = Field(default=0.02, ge=0.0)
    PYPOWSYBL_FAST_MODE: bool = False
    # When True, compute_all_pairs_superposition also runs an actual
    # simulation of each combined action pair and prints the gap between
    # the estimated max rho / max-rho line and the simulated ones. Useful
    # for validating the superposition approximation; adds one extra
    # simulation per pair so leave off for production runs.
    VERIFY_SUPERPOSITION_MAX_RHO: bool = False

    # -------------------
    #  Minimum prioritized actions per type
    # -------------------
    MIN_LINE_RECONNECTIONS: int = Field(default=2, ge=0)
    MIN_CLOSE_COUPLING: int = Field(default=3, ge=0)
    MIN_OPEN_COUPLING: int = Field(default=2, ge=0)
    MIN_LINE_DISCONNECTIONS: int = Field(default=3, ge=0)
    MIN_PST: int = Field(default=2, ge=0)
    MIN_LOAD_SHEDDING: int = Field(default=2, ge=0)
    MIN_RENEWABLE_CURTAILMENT: int = Field(default=2, ge=0)

    # -------------------
    #  Load shedding parameters
    # -------------------
    # 5% safety margin on top of the minimum shedding volume.
    LOAD_SHEDDING_MARGIN: float = Field(default=0.05, ge=0.0)
    # Ignore trivial shedding below this volume (MW).
    LOAD_SHEDDING_MIN_MW: float = Field(default=1.0, ge=0.0)

    # -------------------
    #  Renewable curtailment parameters
    # -------------------
    RENEWABLE_CURTAILMENT_MARGIN: float = Field(default=0.05, ge=0.0)
    RENEWABLE_CURTAILMENT_MIN_MW: float = Field(default=1.0, ge=0.0)
    RENEWABLE_ENERGY_SOURCES: List[str] = Field(
        default_factory=lambda: ["WIND", "SOLAR"]
    )

    # -------------------
    #  Expert system parameters
    # -------------------
    PARAM_OPTIONS_EXPERT_OP: Dict[str, Any] = Field(
        default_factory=lambda: dict(_DEFAULT_PARAM_OPTIONS_EXPERT_OP)
    )

    @field_validator("LINES_DEFAUT", mode="before")
    @classmethod
    def _normalize_lines_defaut(cls, value: Any) -> Any:
        """Accept both a bare line name and a JSON list from env vars.

        ``LINES_DEFAUT`` is annotated with :class:`NoDecode` so
        ``pydantic-settings`` hands the raw env-var string to this validator
        unchanged. We fall back to JSON parsing when the value looks like a
        JSON list, otherwise we treat it as a single line name. Python values
        (lists already, ``None``) pass through untouched.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                import json

                return json.loads(stripped)
            return [stripped]
        return value


def apply_settings_to_namespace(
    settings: Settings, namespace: MutableMapping[str, Any]
) -> None:
    """Promote each Settings field to a module-level attribute.

    This is the compatibility shim that lets pre-pydantic call sites such as
    ``from expert_op4grid_recommender.config import DATE`` or
    ``config.ENV_NAME = env_name`` continue to work unchanged.
    """
    for name, value in settings.model_dump().items():
        namespace[name] = value


# --- Instantiate the default Settings object and populate module globals ---
settings: Settings = Settings()
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
