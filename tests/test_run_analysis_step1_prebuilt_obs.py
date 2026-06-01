# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Signature-level guard for the ``prebuilt_obs_simu_defaut`` kwarg added
to :func:`run_analysis_step1`.

The kwarg lets a host application (Co-Study4Grid) skip the redundant
contingency load-flow when it has already produced the post-contingency
observation while building the N-1 diagram. Co-Study4Grid introspects
the signature with :func:`inspect.signature` and only forwards the
kwarg when present, so the test below is a static contract guard:
"the parameter exists, with the right default, on the public function".

We do not exercise the body — the function pulls in heavy pypowsybl
machinery and is covered end-to-end by the host integration tests.
"""
from __future__ import annotations

import inspect

import pytest

# pypowsybl is a hard import on ``expert_op4grid_recommender.main`` —
# skip the whole module when it isn't available (e.g. tooling CI
# without the JVM stack). The host integration tests cover the
# behaviour end-to-end.
pypowsybl = pytest.importorskip("pypowsybl")  # noqa: F841

from expert_op4grid_recommender.main import run_analysis_step1  # noqa: E402


def test_run_analysis_step1_accepts_prebuilt_obs_simu_defaut():
    """Co-Study4Grid introspects ``run_analysis_step1`` to decide
    whether to forward its cached post-contingency observation. The
    kwarg must stay reachable by ``inspect.signature`` lookups."""
    sig = inspect.signature(run_analysis_step1)
    params = sig.parameters
    assert "prebuilt_obs_simu_defaut" in params, (
        "run_analysis_step1 must accept ``prebuilt_obs_simu_defaut`` so "
        "Co-Study4Grid can reuse the obs it already built for the N-1 "
        "diagram and skip the redundant contingency load-flow."
    )
    # Default must be None so a caller that forwards
    # ``prebuilt_obs_simu_defaut=None`` keeps the legacy behaviour
    # (run ``simulate_contingency_pypowsybl`` normally).
    assert params["prebuilt_obs_simu_defaut"].default is None
