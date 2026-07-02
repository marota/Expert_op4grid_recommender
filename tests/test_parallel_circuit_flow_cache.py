# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Regression test: parallel circuits must not collapse in the discovery edge cache.

The overflow graph is a ``MultiDiGraph``. Twin circuits between the same two
substations (e.g. the RTE L61/L62 pairs) share a ``(u, v)`` node pair, so the
name/label lookup caches must be keyed by the full edge id ``(u, v, k)``.
Previously they were keyed by ``(u, v)`` alone, silently dropping one of every
parallel pair from the flow-influence scoring used by load shedding /
curtailment / redispatch.
"""

import networkx as nx

from expert_op4grid_recommender.action_evaluation.discovery import ActionDiscoverer


class _GraphWrapper:
    """Minimal stand-in for the alphaDeesp overflow-graph object (``.g``)."""

    def __init__(self, g):
        self.g = g


def _make_discoverer(g):
    d = ActionDiscoverer.__new__(ActionDiscoverer)
    d.g_overflow = _GraphWrapper(g)
    return d


def test_edge_cache_keeps_parallel_circuits_distinct():
    g = nx.MultiDiGraph()
    g.add_edge("SubA", "SubB", key=0, name="L61", label=-50.0, capacity=50.0)
    g.add_edge("SubA", "SubB", key=1, name="L62", label=-50.0, capacity=50.0)

    d = _make_discoverer(g)
    d._get_edge_data_cache()

    # Both parallel edges survive, keyed by the full (u, v, k) id.
    assert len(d._cached_edge_names) == 2
    assert set(d._cached_edge_names.values()) == {"L61", "L62"}
    assert all(len(k) == 3 for k in d._cached_edge_names)


def test_node_flow_cache_sums_parallel_circuit_flows():
    g = nx.MultiDiGraph()
    g.add_edge("SubA", "SubB", key=0, name="L61", label=-50.0, capacity=50.0)
    g.add_edge("SubA", "SubB", key=1, name="L62", label=-50.0, capacity=50.0)

    d = _make_discoverer(g)
    d._get_edge_data_cache()
    node_flow = d._build_node_flow_cache(blue_edge_names_set={"L61", "L62"})

    # Both circuits (-50 MW each, blue) contribute; the cache must not drop one.
    assert node_flow["SubA"]["neg_out"] == 100.0
    assert node_flow["SubB"]["neg_in"] == 100.0


def test_non_multigraph_still_supported():
    """A plain DiGraph (no parallel edges) keeps 2-tuple keys and works."""
    g = nx.DiGraph()
    g.add_edge("SubA", "SubB", name="L1", label=-30.0, capacity=30.0)

    d = _make_discoverer(g)
    d._get_edge_data_cache()
    assert set(d._cached_edge_names) == {("SubA", "SubB")}
    node_flow = d._build_node_flow_cache(blue_edge_names_set={"L1"})
    assert node_flow["SubA"]["neg_out"] == 30.0
