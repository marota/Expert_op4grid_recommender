# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for the ``extra_lines_to_cut_ids`` plumbing.

These tests verify that operator-supplied extra line indices are:

1. Appended to ``ltc`` when ``Grid2opSimulation`` (or its pypowsybl
   counterpart) is constructed, so the cut still happens.
2. Forwarded as ``extra_lines_to_cut`` to the ``OverFlowGraph`` so it
   can stamp ``is_extra_cut`` and skip the overload classification.
3. De-duplicated against ``overloaded_line_ids`` (an extra that is
   already an overload should not be doubled).
4. Defaulted to an empty list when ``None`` is passed.
5. Read from ``context["extra_lines_to_cut_ids"]`` by
   ``run_analysis_step2_graph`` and propagated through to
   ``build_overflow_graph``.

The actual flow simulation is mocked out — these are pure plumbing
tests that don't require a real Grid2Op or pypowsybl environment.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_flow_dataframe():
    """A minimal flow dataframe matching what ``Grid2opSimulation.get_dataframe``
    returns — empty rows but with the expected columns so
    ``inhibit_swapped_flows`` can run without errors.
    """
    return pd.DataFrame({
        "new_flows_swapped": pd.Series([], dtype=bool),
        "delta_flows": pd.Series([], dtype=float),
        "idx_or": pd.Series([], dtype=int),
        "idx_ex": pd.Series([], dtype=int),
    })


def _make_sim_mock():
    sim = MagicMock()
    sim.get_dataframe.return_value = _empty_flow_dataframe()
    sim.topo = MagicMock()
    return sim


def _make_overflow_graph_mock():
    g_overflow = MagicMock()
    g_overflow.g = nx.DiGraph()
    return g_overflow


def _make_distribution_mock():
    dist = MagicMock()
    # Empty red components so consolidate_graph branch is skipped.
    dist.g_only_red_components.nodes = []
    dist.constrained_path.full_n_constrained_path.return_value = []
    dist.get_hubs.return_value = []
    return dist


# ---------------------------------------------------------------------------
# graph_analysis.builder.build_overflow_graph
# ---------------------------------------------------------------------------


@patch("expert_op4grid_recommender.graph_analysis.builder.Structured_Overload_Distribution_Graph")
@patch("expert_op4grid_recommender.graph_analysis.builder.OverFlowGraph")
@patch("expert_op4grid_recommender.graph_analysis.builder.Grid2opSimulation")
def test_builder_appends_extras_to_ltc_and_forwards_to_overflow_graph(
    MockSim, MockOFG, MockSODG
):
    """Extras must be appended to ``ltc`` and forwarded as
    ``extra_lines_to_cut`` to ``OverFlowGraph``."""
    from expert_op4grid_recommender.graph_analysis.builder import (
        build_overflow_graph,
    )

    MockSim.return_value = _make_sim_mock()
    MockOFG.return_value = _make_overflow_graph_mock()
    MockSODG.return_value = _make_distribution_mock()

    env = MagicMock()
    obs = MagicMock()
    obs.name_line = np.array([])
    obs.name_sub = np.array(["sub0", "sub1"])

    build_overflow_graph(
        env,
        obs,
        overloaded_line_ids=[0, 1],
        non_connected_reconnectable_lines=[],
        lines_non_reconnectable=[],
        timestep=0,
        do_consolidate_graph=False,
        node_renaming=False,
        extra_lines_to_cut_ids=[2, 3],
    )

    # Grid2opSimulation must receive the union as ``ltc``.
    sim_kwargs = MockSim.call_args.kwargs
    assert sim_kwargs["ltc"] == [0, 1, 2, 3]

    # OverFlowGraph must receive only the extras as ``extra_lines_to_cut``,
    # and the full union as its second positional ``ltc`` argument.
    ofg_args, ofg_kwargs = MockOFG.call_args
    # ltc positional argument (index 1) is the full list.
    assert list(ofg_args[1]) == [0, 1, 2, 3]
    assert ofg_kwargs["extra_lines_to_cut"] == [2, 3]


@patch("expert_op4grid_recommender.graph_analysis.builder.Structured_Overload_Distribution_Graph")
@patch("expert_op4grid_recommender.graph_analysis.builder.OverFlowGraph")
@patch("expert_op4grid_recommender.graph_analysis.builder.Grid2opSimulation")
def test_builder_dedupes_extras_already_in_overloads(MockSim, MockOFG, MockSODG):
    """If an extra is already in ``overloaded_line_ids`` it must not be
    duplicated in either ``ltc`` or ``extra_lines_to_cut``."""
    from expert_op4grid_recommender.graph_analysis.builder import (
        build_overflow_graph,
    )

    MockSim.return_value = _make_sim_mock()
    MockOFG.return_value = _make_overflow_graph_mock()
    MockSODG.return_value = _make_distribution_mock()

    env = MagicMock()
    obs = MagicMock()
    obs.name_line = np.array([])
    obs.name_sub = np.array(["sub0", "sub1"])

    build_overflow_graph(
        env,
        obs,
        overloaded_line_ids=[1, 5],
        non_connected_reconnectable_lines=[],
        lines_non_reconnectable=[],
        timestep=0,
        do_consolidate_graph=False,
        node_renaming=False,
        # 1 is already an overload → must be filtered out
        extra_lines_to_cut_ids=[1, 7],
    )

    sim_kwargs = MockSim.call_args.kwargs
    assert sim_kwargs["ltc"] == [1, 5, 7]

    ofg_args, ofg_kwargs = MockOFG.call_args
    assert list(ofg_args[1]) == [1, 5, 7]
    assert ofg_kwargs["extra_lines_to_cut"] == [7]


@patch("expert_op4grid_recommender.graph_analysis.builder.Structured_Overload_Distribution_Graph")
@patch("expert_op4grid_recommender.graph_analysis.builder.OverFlowGraph")
@patch("expert_op4grid_recommender.graph_analysis.builder.Grid2opSimulation")
def test_builder_extras_none_defaults_to_empty(MockSim, MockOFG, MockSODG):
    """``extra_lines_to_cut_ids=None`` must behave like ``[]`` — no
    extras forwarded, ``ltc`` unchanged."""
    from expert_op4grid_recommender.graph_analysis.builder import (
        build_overflow_graph,
    )

    MockSim.return_value = _make_sim_mock()
    MockOFG.return_value = _make_overflow_graph_mock()
    MockSODG.return_value = _make_distribution_mock()

    env = MagicMock()
    obs = MagicMock()
    obs.name_line = np.array([])
    obs.name_sub = np.array(["sub0", "sub1"])

    build_overflow_graph(
        env,
        obs,
        overloaded_line_ids=[0, 1],
        non_connected_reconnectable_lines=[],
        lines_non_reconnectable=[],
        timestep=0,
        do_consolidate_graph=False,
        node_renaming=False,
        extra_lines_to_cut_ids=None,
    )

    sim_kwargs = MockSim.call_args.kwargs
    assert sim_kwargs["ltc"] == [0, 1]

    ofg_args, ofg_kwargs = MockOFG.call_args
    assert list(ofg_args[1]) == [0, 1]
    assert ofg_kwargs["extra_lines_to_cut"] == []


@patch("expert_op4grid_recommender.graph_analysis.builder.Structured_Overload_Distribution_Graph")
@patch("expert_op4grid_recommender.graph_analysis.builder.OverFlowGraph")
@patch("expert_op4grid_recommender.graph_analysis.builder.Grid2opSimulation")
def test_builder_default_kwarg_is_empty(MockSim, MockOFG, MockSODG):
    """Calling without ``extra_lines_to_cut_ids`` at all keeps legacy
    behavior (no extras forwarded)."""
    from expert_op4grid_recommender.graph_analysis.builder import (
        build_overflow_graph,
    )

    MockSim.return_value = _make_sim_mock()
    MockOFG.return_value = _make_overflow_graph_mock()
    MockSODG.return_value = _make_distribution_mock()

    env = MagicMock()
    obs = MagicMock()
    obs.name_line = np.array([])
    obs.name_sub = np.array(["sub0", "sub1"])

    build_overflow_graph(
        env,
        obs,
        overloaded_line_ids=[42],
        non_connected_reconnectable_lines=[],
        lines_non_reconnectable=[],
        timestep=0,
        do_consolidate_graph=False,
        node_renaming=False,
    )

    sim_kwargs = MockSim.call_args.kwargs
    assert sim_kwargs["ltc"] == [42]
    ofg_kwargs = MockOFG.call_args.kwargs
    assert ofg_kwargs["extra_lines_to_cut"] == []


# ---------------------------------------------------------------------------
# pypowsybl_backend.overflow_analysis.build_overflow_graph_pypowsybl
# ---------------------------------------------------------------------------


@patch("expert_op4grid_recommender.pypowsybl_backend.overflow_analysis.AlphaDeespAdapter")
def test_pypowsybl_builder_appends_extras_and_forwards_to_overflow_graph(
    MockAdapter,
):
    """Same plumbing assertions as the grid2op variant, but for the
    pypowsybl ``build_overflow_graph_pypowsybl`` entrypoint."""
    # The pypowsybl variant imports ``OverFlowGraph`` and
    # ``Structured_Overload_Distribution_Graph`` lazily inside the
    # function body, so we patch the alphaDeesp module directly.
    with patch(
        "alphaDeesp.core.graphsAndPaths.OverFlowGraph"
    ) as MockOFG, patch(
        "alphaDeesp.core.graphsAndPaths.Structured_Overload_Distribution_Graph"
    ) as MockSODG:
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import (
            build_overflow_graph_pypowsybl,
        )

        adapter = MagicMock()
        adapter.get_dataframe.return_value = _empty_flow_dataframe()
        adapter.topo = MagicMock()
        MockAdapter.return_value = adapter

        MockOFG.return_value = _make_overflow_graph_mock()
        MockSODG.return_value = _make_distribution_mock()

        env = MagicMock()
        obs = MagicMock()
        obs.name_line = np.array([])
        obs.name_sub = np.array(["sub0", "sub1"])

        build_overflow_graph_pypowsybl(
            env,
            obs,
            overloaded_line_ids=[0, 1],
            non_connected_reconnectable_lines=[],
            lines_non_reconnectable=[],
            timestep=0,
            do_consolidate_graph=False,
            extra_lines_to_cut_ids=[1, 4],  # 1 is duplicate, 4 is new
        )

        # AlphaDeespAdapter (drop-in replacement for Grid2opSimulation)
        # must receive deduplicated full ltc.
        adapter_kwargs = MockAdapter.call_args.kwargs
        assert adapter_kwargs["ltc"] == [0, 1, 4]

        # OverFlowGraph must receive only the *new* extras and the full
        # union as its ltc positional argument.
        ofg_args, ofg_kwargs = MockOFG.call_args
        assert list(ofg_args[1]) == [0, 1, 4]
        assert ofg_kwargs["extra_lines_to_cut"] == [4]


@patch("expert_op4grid_recommender.pypowsybl_backend.overflow_analysis.AlphaDeespAdapter")
def test_pypowsybl_builder_extras_none_defaults_to_empty(MockAdapter):
    """``extra_lines_to_cut_ids=None`` is a no-op for the pypowsybl
    variant too."""
    with patch(
        "alphaDeesp.core.graphsAndPaths.OverFlowGraph"
    ) as MockOFG, patch(
        "alphaDeesp.core.graphsAndPaths.Structured_Overload_Distribution_Graph"
    ) as MockSODG:
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import (
            build_overflow_graph_pypowsybl,
        )

        adapter = MagicMock()
        adapter.get_dataframe.return_value = _empty_flow_dataframe()
        adapter.topo = MagicMock()
        MockAdapter.return_value = adapter

        MockOFG.return_value = _make_overflow_graph_mock()
        MockSODG.return_value = _make_distribution_mock()

        env = MagicMock()
        obs = MagicMock()
        obs.name_line = np.array([])
        obs.name_sub = np.array(["sub0", "sub1"])

        build_overflow_graph_pypowsybl(
            env,
            obs,
            overloaded_line_ids=[7],
            non_connected_reconnectable_lines=[],
            lines_non_reconnectable=[],
            timestep=0,
            do_consolidate_graph=False,
            extra_lines_to_cut_ids=None,
        )

        adapter_kwargs = MockAdapter.call_args.kwargs
        assert adapter_kwargs["ltc"] == [7]
        ofg_kwargs = MockOFG.call_args.kwargs
        assert ofg_kwargs["extra_lines_to_cut"] == []


# ---------------------------------------------------------------------------
# main.run_analysis_step2_graph context plumbing
# ---------------------------------------------------------------------------


def _make_step2_context(extra_lines_to_cut_ids, is_pypowsybl=False):
    """Builds the minimal context dict that ``run_analysis_step2_graph``
    expects, with all heavyweight bits replaced by mocks."""
    captured = {}

    def fake_build_overflow_graph(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Mimic the 6-tuple return signature of the real function.
        return (
            pd.DataFrame({
                "new_flows_swapped": pd.Series([], dtype=bool),
                "line_name": pd.Series([], dtype=str),
            }),
            MagicMock(),  # overflow_sim
            MagicMock(g=nx.DiGraph()),  # g_overflow
            [],  # hubs
            MagicMock(),  # g_distribution_graph
            {},  # node_name_mapping
        )

    env = MagicMock()
    env.action_space = MagicMock()
    obs = MagicMock()
    obs_simu_defaut = MagicMock()
    obs_simu_defaut.name_line = np.array(["L0"])

    context = {
        "backend": "grid2op" if not is_pypowsybl else "pypowsybl",
        "env": env,
        "obs": obs,
        "obs_simu_defaut": obs_simu_defaut,
        "analysis_date": None,
        "current_timestep": 0,
        "current_lines_defaut": ["LX"],
        "lines_overloaded_ids": [0],
        "lines_overloaded_ids_kept": [0],
        "maintenance_to_reco_at_t": [],
        "act_reco_maintenance": MagicMock(),
        "lines_non_reconnectable": [],
        "lines_we_care_about": [],
        "classifier": MagicMock(),
        "custom_layout": None,
        "chronic_name": "test",
        "pre_existing_rho": 0.0,
        "lines_overloaded_names": ["L0"],
        "non_connected_reconnectable_lines": [],
        "dict_action": {},
        "is_bare_env": True,
        "is_pypowsybl": is_pypowsybl,
        "actual_fast_mode": True,
        # backend-specific function mocks
        "check_simu_overloads": MagicMock(return_value=(True, False)),
        "switch_to_dc": MagicMock(return_value=(env, obs, obs_simu_defaut)),
        "build_overflow_graph": fake_build_overflow_graph,
        "get_env_first_obs": MagicMock(),
        "simulate_contingency": MagicMock(),
        "create_default_action": MagicMock(),
        "check_rho_reduction": MagicMock(),
        "compute_baseline": MagicMock(),
    }
    if extra_lines_to_cut_ids is not None:
        context["extra_lines_to_cut_ids"] = extra_lines_to_cut_ids
    return context, captured


def test_run_analysis_step2_graph_forwards_extras_from_context():
    """The step2 graph builder must read ``context["extra_lines_to_cut_ids"]``
    and pass it through to the (injected) ``build_overflow_graph``."""
    from expert_op4grid_recommender import config
    from expert_op4grid_recommender.main import run_analysis_step2_graph

    context, captured = _make_step2_context(
        extra_lines_to_cut_ids=[10, 20]
    )

    # Disable visualization & test data saving so the function returns
    # immediately after the build_overflow_graph call.
    with patch.object(config, "DO_VISUALIZATION", False), \
         patch.object(config, "DO_SAVE_DATA_FOR_TEST", False):
        run_analysis_step2_graph(context)

    assert captured["kwargs"].get("extra_lines_to_cut_ids") == [10, 20]


def test_run_analysis_step2_graph_missing_key_defaults_to_empty():
    """When the legacy single-step path doesn't populate the key,
    ``run_analysis_step2_graph`` must fall back to ``[]`` — no extras
    forwarded."""
    from expert_op4grid_recommender import config
    from expert_op4grid_recommender.main import run_analysis_step2_graph

    context, captured = _make_step2_context(extra_lines_to_cut_ids=None)
    # Ensure key is genuinely absent (not just None).
    context.pop("extra_lines_to_cut_ids", None)

    with patch.object(config, "DO_VISUALIZATION", False), \
         patch.object(config, "DO_SAVE_DATA_FOR_TEST", False):
        run_analysis_step2_graph(context)

    assert captured["kwargs"].get("extra_lines_to_cut_ids") == []


def test_run_analysis_step2_graph_pypowsybl_path_forwards_extras():
    """The pypowsybl branch (``is_pypowsybl=True``) of step2 graph also
    passes the extras through (as a kwarg, alongside ``fast_mode``)."""
    from expert_op4grid_recommender import config
    from expert_op4grid_recommender.main import run_analysis_step2_graph

    context, captured = _make_step2_context(
        extra_lines_to_cut_ids=[5], is_pypowsybl=True
    )

    with patch.object(config, "DO_VISUALIZATION", False), \
         patch.object(config, "DO_SAVE_DATA_FOR_TEST", False):
        run_analysis_step2_graph(context)

    assert captured["kwargs"].get("extra_lines_to_cut_ids") == [5]
