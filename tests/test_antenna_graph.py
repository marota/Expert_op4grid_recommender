# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""Unit tests for the synthetic antenna (islanded-pocket) overflow graph.

These tests are self-contained: they build a tiny grid2op-compatible
observation by hand, so they need neither pypowsybl nor a real environment.
"""

import types

import networkx as nx
import numpy as np
import pytest
from alphaDeesp.core.graphsAndPaths import Structured_Overload_Distribution_Graph

from expert_op4grid_recommender.graph_analysis.antenna_graph import (
    SOURCE_NODE_NAME,
    build_antenna_overflow_graph,
)
from expert_op4grid_recommender.graph_analysis.processor import extract_antenna_context


def _make_obs(antenna_prod=None):
    """Build a fake observation with a radial pocket fed by one overloaded line.

    Topology::

        main grid:  0 - 5 - 6 - 7 - 8          (sub 0 is the antenna root)
        constraint:    0 == 1                  (overloaded, rho = 1.5)
        antenna:       1 - 2,  1 - 3,  3 - 4

    The pocket {1, 2, 3, 4} is islanded as soon as the 0==1 line is cut.
    Loads sit on subs 2/3/4 → net consumer, unless ``antenna_prod`` injects
    generation to flip the pocket into a net producer.
    """
    name_sub = np.array([f"S{i}" for i in range(9)])
    # lines: (or_sub, ex_sub, name)
    lines = [
        (0, 5, "L0_5"), (5, 6, "L5_6"), (6, 7, "L6_7"), (7, 8, "L7_8"),  # main grid
        (0, 1, "CONSTRAINT"),                                            # feed
        (1, 2, "L1_2"), (1, 3, "L1_3"), (3, 4, "L3_4"),                  # antenna
    ]
    n_lines = len(lines)
    name_line = np.array([l[2] for l in lines])
    line_or_to_subid = np.array([l[0] for l in lines])
    line_ex_to_subid = np.array([l[1] for l in lines])
    line_or_bus = np.ones(n_lines, dtype=int)
    line_ex_bus = np.ones(n_lines, dtype=int)
    line_status = np.ones(n_lines, dtype=bool)
    a_or = np.full(n_lines, 100.0)
    rho = np.full(n_lines, 0.5)
    constraint_id = list(name_line).index("CONSTRAINT")
    rho[constraint_id] = 1.5  # the overload
    # active-power flows (MW), magnitudes only matter for the graph
    p_or = np.full(n_lines, 10.0)
    p_or[constraint_id] = 60.0
    p_or[list(name_line).index("L1_2")] = 30.0
    p_or[list(name_line).index("L1_3")] = 30.0
    p_or[list(name_line).index("L3_4")] = 10.0

    # loads on antenna subs 2/3/4
    load_to_subid = np.array([2, 3, 4])
    load_p = np.array([30.0, 20.0, 10.0])

    if antenna_prod is None:
        gen_to_subid = np.array([0])
        gen_p = np.array([0.0])
    else:
        gen_to_subid = np.array([antenna_prod[0]])
        gen_p = np.array([antenna_prod[1]])

    return types.SimpleNamespace(
        name_sub=name_sub, name_line=name_line,
        line_or_to_subid=line_or_to_subid, line_ex_to_subid=line_ex_to_subid,
        line_or_bus=line_or_bus, line_ex_bus=line_ex_bus,
        line_status=line_status, a_or=a_or, rho=rho, p_or=p_or,
        load_to_subid=load_to_subid, load_p=load_p,
        gen_to_subid=gen_to_subid, gen_p=gen_p,
    )


def _edge_colors_by_name(g):
    return {data["name"]: data["color"]
            for _, _, data in g.edges(data=True)}


def test_extract_antenna_context_identifies_pocket():
    obs = _make_obs()
    overloaded = [list(obs.name_line).index("CONSTRAINT")]
    ctx = extract_antenna_context(obs, overloaded)

    assert ctx is not None
    assert ctx["constraint_line_name"] == "CONSTRAINT"
    assert ctx["root_sub_id"] == 0
    assert ctx["antenna_sub_ids"] == [1, 2, 3, 4]
    assert set(ctx["antenna_sub_names"]) == {"S1", "S2", "S3", "S4"}


def test_extract_antenna_context_none_when_no_overloads():
    obs = _make_obs()
    assert extract_antenna_context(obs, []) is None


def test_antenna_graph_classifies_whole_pocket_as_aval():
    obs = _make_obs()
    ctx = extract_antenna_context(obs, [list(obs.name_line).index("CONSTRAINT")])

    df, sim, g_overflow, hubs, sdg, mapping, meta = build_antenna_overflow_graph(
        obs, ctx["constraint_line_id"], ctx["antenna_sub_ids"], ctx["root_sub_id"])

    assert sim is None  # no Grid2opSimulation involved
    cp = sdg.get_constrained_path()
    # The whole pocket is downstream (aval); the root is upstream (amont).
    assert set(cp.n_aval()) == {"S1", "S2", "S3", "S4"}
    assert "S0" in cp.n_amont()

    colors = _edge_colors_by_name(g_overflow.g)
    assert colors["CONSTRAINT"] == "black"
    for branch in ("L1_2", "L1_3", "L3_4"):
        assert colors[branch] == "blue", f"{branch} should be a downstream blue edge"


def test_antenna_meta_consumer_direction_and_net_mw():
    obs = _make_obs()
    ctx = extract_antenna_context(obs, [list(obs.name_line).index("CONSTRAINT")])
    *_, meta = build_antenna_overflow_graph(
        obs, ctx["constraint_line_id"], ctx["antenna_sub_ids"], ctx["root_sub_id"])

    assert meta["direction"] == "consumer"
    assert meta["n_subs"] == 4
    assert meta["total_load_mw"] == pytest.approx(60.0)
    assert meta["total_prod_mw"] == pytest.approx(0.0)
    assert meta["net_mw"] == pytest.approx(-60.0)
    assert set(meta["antenna_sub_names"]) == {"S1", "S2", "S3", "S4"}


def test_antenna_meta_producer_direction():
    # 200 MW of generation on antenna sub 2 outweighs the 60 MW of load.
    obs = _make_obs(antenna_prod=(2, 200.0))
    ctx = extract_antenna_context(obs, [list(obs.name_line).index("CONSTRAINT")])
    *_, meta = build_antenna_overflow_graph(
        obs, ctx["constraint_line_id"], ctx["antenna_sub_ids"], ctx["root_sub_id"])

    assert meta["direction"] == "producer"
    assert meta["total_prod_mw"] == pytest.approx(200.0)
    assert meta["net_mw"] == pytest.approx(140.0)


def test_node_name_mapping_reverts_to_real_substation_indices():
    """The discovery pipeline reverts node names to *real* grid2op sub indices
    (via pre_process_graph_alphadeesp) and uses them to index obs arrays, so the
    mapping must be keyed by real substation ids — not compact 0..k-1 ids."""
    obs = _make_obs()
    ctx = extract_antenna_context(obs, [list(obs.name_line).index("CONSTRAINT")])
    _, _, g_overflow, _, _, mapping, _ = build_antenna_overflow_graph(
        obs, ctx["constraint_line_id"], ctx["antenna_sub_ids"], ctx["root_sub_id"])

    # mapping is {real_sub_id: name} for the 5 real subs + the sentinel source
    # (id == n_subs == 9), which the discovery's ``idx < n_subs`` guards drop.
    assert mapping[0] == "S0" and mapping[4] == "S4"
    assert mapping[9] == SOURCE_NODE_NAME

    # Reverting names -> real indices (what pre_process_graph_alphadeesp does)
    # must NOT crash in integer space and keeps the pocket downstream.
    reverse = {name: sid for sid, name in mapping.items()}
    g_idx = nx.relabel_nodes(g_overflow.g, reverse, copy=True)
    sdg_idx = Structured_Overload_Distribution_Graph(g_idx)
    # downstream nodes are the real substation indices of the pocket
    assert set(sdg_idx.get_constrained_path().n_aval()) == {1, 2, 3, 4}


def test_antenna_graph_is_a_connected_dag_rooted_at_root():
    obs = _make_obs()
    ctx = extract_antenna_context(obs, [list(obs.name_line).index("CONSTRAINT")])
    _, _, g_overflow, _, _, _, _ = build_antenna_overflow_graph(
        obs, ctx["constraint_line_id"], ctx["antenna_sub_ids"], ctx["root_sub_id"])

    g = g_overflow.g
    assert set(g.nodes()) == {SOURCE_NODE_NAME, "S0", "S1", "S2", "S3", "S4"}
    # grid-source anchor + constraint edge + 3 antenna branches
    assert g.number_of_edges() == 5
    assert nx.is_weakly_connected(g)
    # acyclic when collapsed to a simple digraph (radial, oriented away from root)
    assert nx.is_directed_acyclic_graph(nx.DiGraph(g))
