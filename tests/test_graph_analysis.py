# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for :mod:`expert_op4grid_recommender.graph_analysis`.

These tests focus on the pure-Python helpers that can be exercised without
standing up a full Grid2Op environment or alphaDeesp simulation:

- :func:`graph_analysis.builder.inhibit_swapped_flows` — operates on a pandas
  DataFrame and has no external dependencies.
- :func:`graph_analysis.processor.get_n_connected_components_graph_with_overloads`
  and :func:`graph_analysis.processor.identify_overload_lines_to_keep_overflow_graph_connected`
  — operate on a Grid2Op-compatible observation, which we mock.
- :func:`graph_analysis.processor.get_subs_islanded_by_overload_disconnections`
  — operates on plain sets plus an observation mock.
- :func:`graph_analysis.visualization.get_graph_file_name` — pure string
  assembly function driven by config flags.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from expert_op4grid_recommender.graph_analysis.builder import inhibit_swapped_flows
from expert_op4grid_recommender.graph_analysis.processor import (
    get_n_connected_components_graph_with_overloads,
    get_subs_islanded_by_overload_disconnections,
    identify_overload_lines_to_keep_overflow_graph_connected,
)
from expert_op4grid_recommender.graph_analysis.visualization import get_graph_file_name


# ---------------------------------------------------------------------------
# builder.inhibit_swapped_flows
# ---------------------------------------------------------------------------

def test_inhibit_swapped_flows_swaps_endpoints_and_negates_delta():
    """Rows flagged as ``new_flows_swapped`` should have their endpoints
    exchanged and ``delta_flows`` negated, while untouched rows are left alone.
    """
    df = pd.DataFrame({
        "new_flows_swapped": [True, False, True],
        "delta_flows": [5.0, 3.0, -2.0],
        "idx_or": [1, 2, 3],
        "idx_ex": [10, 20, 30],
    })
    out = inhibit_swapped_flows(df.copy())

    # Row 0: swapped → delta flipped, endpoints exchanged
    assert out.loc[0, "delta_flows"] == -5.0
    assert out.loc[0, "idx_or"] == 10
    assert out.loc[0, "idx_ex"] == 1

    # Row 1: untouched
    assert out.loc[1, "delta_flows"] == 3.0
    assert out.loc[1, "idx_or"] == 2
    assert out.loc[1, "idx_ex"] == 20

    # Row 2: swapped → delta flipped, endpoints exchanged
    assert out.loc[2, "delta_flows"] == 2.0
    assert out.loc[2, "idx_or"] == 30
    assert out.loc[2, "idx_ex"] == 3


def test_inhibit_swapped_flows_is_noop_when_no_rows_flagged():
    df = pd.DataFrame({
        "new_flows_swapped": [False, False],
        "delta_flows": [1.5, -2.5],
        "idx_or": [1, 2],
        "idx_ex": [10, 20],
    })
    out = inhibit_swapped_flows(df.copy())
    pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# processor.get_n_connected_components_graph_with_overloads
# ---------------------------------------------------------------------------

def _make_obs(
    name_line,
    line_or_to_subid,
    line_or_bus,
    line_ex_to_subid,
    line_ex_bus,
    line_status,
    a_or,
    rho,
    name_sub=None,
):
    """Builds a minimal observation-like object for processor helpers."""
    obs = SimpleNamespace(
        name_line=np.array(name_line),
        line_or_to_subid=np.array(line_or_to_subid),
        line_or_bus=np.array(line_or_bus),
        line_ex_to_subid=np.array(line_ex_to_subid),
        line_ex_bus=np.array(line_ex_bus),
        line_status=np.array(line_status),
        a_or=np.array(a_or, dtype=float),
        rho=np.array(rho, dtype=float),
    )
    if name_sub is None:
        n_subs = max(
            list(line_or_to_subid) + list(line_ex_to_subid)
        ) + 1
        obs.name_sub = np.array([f"sub{i}" for i in range(n_subs)])
    else:
        obs.name_sub = np.array(name_sub)
    return obs


def test_get_n_connected_components_single_chain_splits_on_disconnect():
    """Chain sub0-sub1-sub2: removing the middle line should split the graph."""
    obs = _make_obs(
        name_line=["L01", "L12"],
        line_or_to_subid=[0, 1],
        line_or_bus=[1, 1],
        line_ex_to_subid=[1, 2],
        line_ex_bus=[1, 1],
        line_status=[True, True],
        a_or=[100.0, 50.0],
        rho=[0.8, 1.2],  # L12 is overloaded and worst
    )

    comps_init, comps_wo_max, comps_wo_all = (
        get_n_connected_components_graph_with_overloads(obs, [1])
    )

    assert len(comps_init) == 1            # single connected component initially
    assert len(comps_wo_max) == 2          # removing L12 splits the graph
    assert len(comps_wo_all) == 2          # only one overload → same as max


def test_get_n_connected_components_triangle_stays_connected():
    """Triangle sub0-sub1-sub2: removing one edge keeps the graph connected."""
    obs = _make_obs(
        name_line=["L01", "L12", "L02"],
        line_or_to_subid=[0, 1, 0],
        line_or_bus=[1, 1, 1],
        line_ex_to_subid=[1, 2, 2],
        line_ex_bus=[1, 1, 1],
        line_status=[True, True, True],
        a_or=[100.0, 100.0, 100.0],
        rho=[0.5, 1.2, 0.8],  # L12 is the only overload
    )

    comps_init, comps_wo_max, comps_wo_all = (
        get_n_connected_components_graph_with_overloads(obs, [1])
    )

    assert len(comps_init) == 1
    # Removing L12 still leaves a path sub1-sub0-sub2 → 1 component.
    assert len(comps_wo_max) == 1
    assert len(comps_wo_all) == 1


# ---------------------------------------------------------------------------
# processor.identify_overload_lines_to_keep_overflow_graph_connected
# ---------------------------------------------------------------------------

def test_identify_overload_lines_no_islanding_keeps_all():
    """Triangle topology: disconnecting the single overload stays connected,
    so the function returns the full overload list unchanged."""
    obs = _make_obs(
        name_line=["L01", "L12", "L02"],
        line_or_to_subid=[0, 1, 0],
        line_or_bus=[1, 1, 1],
        line_ex_to_subid=[1, 2, 2],
        line_ex_bus=[1, 1, 1],
        line_status=[True, True, True],
        a_or=[100.0, 100.0, 100.0],
        rho=[0.5, 1.2, 0.8],
    )

    kept, islanded = identify_overload_lines_to_keep_overflow_graph_connected(
        obs, [1]
    )
    assert kept == [1]
    assert islanded == []


def test_identify_overload_lines_islanding_returns_none():
    """Single line whose removal creates an island yields ``(None, [...])``."""
    obs = _make_obs(
        name_line=["L01"],
        line_or_to_subid=[0],
        line_or_bus=[1],
        line_ex_to_subid=[1],
        line_ex_bus=[1],
        line_status=[True],
        a_or=[100.0],
        rho=[1.5],
    )
    kept, islanded = identify_overload_lines_to_keep_overflow_graph_connected(
        obs, [0]
    )
    assert kept is None
    # Removing the only line islands sub1 from sub0 (or vice versa).
    assert len(islanded) >= 1


def test_identify_overload_lines_empty_overload_returns_empty():
    obs = _make_obs(
        name_line=["L01"],
        line_or_to_subid=[0],
        line_or_bus=[1],
        line_ex_to_subid=[1],
        line_ex_bus=[1],
        line_status=[True],
        a_or=[100.0],
        rho=[0.3],
    )
    kept, islanded = identify_overload_lines_to_keep_overflow_graph_connected(
        obs, []
    )
    assert kept == []
    assert islanded == []


# ---------------------------------------------------------------------------
# processor.get_subs_islanded_by_overload_disconnections
# ---------------------------------------------------------------------------

def test_get_subs_islanded_returns_substation_names_for_diff():
    """Given two lists of int-based component sets, identifies missing subs."""
    obs = MagicMock()
    obs.name_sub = np.array(["sub0", "sub1", "sub2", "sub3"])
    comps_init = [{0, 1, 2, 3}]
    comps_after = [{0, 1}]
    result = get_subs_islanded_by_overload_disconnections(
        obs, comps_init, comps_after, max_overload_name="Lfoo"
    )
    assert set(result) == {"sub2", "sub3"}


def test_get_subs_islanded_handles_string_node_ids():
    """When the graph components use ``subid_X_bus_Y`` strings, the function
    should parse the substation index out of them."""
    obs = MagicMock()
    obs.name_sub = np.array(["sub0", "sub1", "sub2"])
    comps_init = [{"subid_0_bus_1", "subid_1_bus_1", "subid_2_bus_1"}]
    comps_after = [{"subid_0_bus_1", "subid_1_bus_1"}]
    result = get_subs_islanded_by_overload_disconnections(
        obs, comps_init, comps_after, max_overload_name="Lbar"
    )
    assert result == ["sub2"]


# ---------------------------------------------------------------------------
# visualization.get_graph_file_name
# ---------------------------------------------------------------------------

def test_get_graph_file_name_contains_inputs_and_config_suffixes():
    """The generated file name encodes all inputs plus config flags."""
    name = get_graph_file_name(
        lines_defaut=["L1", "L2"],
        chronic_name="2024-12-07",
        timestep=9,
        use_dc_load_flow=True,
    )
    assert "L1_L2" in name
    assert "chronic_2024-12-07" in name
    assert "timestep_9" in name
    # DC suffix must be present when use_dc_load_flow=True
    assert name.endswith("_in_DC")


def test_get_graph_file_name_ac_has_no_dc_suffix():
    name = get_graph_file_name(
        lines_defaut=["LX"],
        chronic_name="c",
        timestep=0,
        use_dc_load_flow=False,
    )
    assert not name.endswith("_in_DC")
    assert "_in_DC" not in name
