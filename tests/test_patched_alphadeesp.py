# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Tests for the alphaDeesp Dijkstra performance patch (patched_alphadeesp).

Covers: patch applied at import, path equivalence original-vs-optimized on a
synthetic graph with unique weights (no tie-break ambiguity), the reverse
(target-side) branch, and the kill-switch / version-guard plumbing.
"""
import networkx as nx
import pytest

# Importing the package applies the patch (mirrors production usage).
import expert_op4grid_recommender  # noqa: F401
from expert_op4grid_recommender import patched_alphadeesp as PA

NFM = pytest.importorskip(
    "alphaDeesp.core.graphs.null_flow_graph"
).NullFlowGraphMixin


def _make_graph():
    """Small MultiDiGraph with unique capacities => unique shortest paths."""
    g = nx.MultiDiGraph()
    edges = [
        ("A", "B", 1.0), ("B", "C", 2.0), ("A", "C", 10.0),
        ("C", "D", 1.5), ("B", "D", 7.0), ("D", "E", 0.5),
        ("C", "E", 9.0), ("E", "F", 1.1), ("A", "F", 30.0),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, capacity=w, name=f"{u}{v}")
    return g


def _prepared(sources, targets, g):
    """Minimal `prepared` dict exercising the pass-through source filter."""
    nodes = set(sources) | set(targets)
    return {
        "source_nodes_in_gc": list(sources),
        "target_nodes_in_gc": list(targets),
        "node_has_incident_interest": {n: True for n in nodes},
        "bfs_cache": {n: ["stub"] for n in nodes},
        "targets_with_bfs": frozenset(targets),
        "any_target_has_interest": True,
    }


def test_patch_is_applied_at_import():
    assert getattr(NFM, PA._PATCH_FLAG, False)
    assert NFM._compute_sssp_paths is PA._optimized_compute_sssp_paths
    assert hasattr(NFM, "_eo4g_orig_compute_sssp_paths")


def test_apply_is_idempotent():
    assert PA.apply_alphadeesp_dijkstra_patch() is True
    # Second call must not re-wrap (original stays the true upstream).
    orig = NFM._eo4g_orig_compute_sssp_paths
    assert PA.apply_alphadeesp_dijkstra_patch() is True
    assert NFM._eo4g_orig_compute_sssp_paths is orig


@pytest.mark.parametrize(
    "sources,targets",
    [
        (["A", "B", "C"], ["E"]),          # |targets| < |sources| -> reverse branch
        (["A"], ["D", "E", "F"]),          # forward branch
        (["A", "B"], ["E", "F"]),          # equal cardinality -> forward branch
    ],
)
def test_paths_equivalent_to_upstream(sources, targets):
    g = _make_graph()
    prepared = _prepared(sources, targets, g)
    # Effective multigraph weights are hop-cost-only (33/100): a single
    # promoted edge keeps every tested shortest path strictly unique, so
    # tie-break order (which may differ between the forward and reverse
    # implementations) cannot mask a real regression.
    edges_of_interest = {("B", "C")}

    self_stub = object.__new__(NFM)
    orig = NFM._eo4g_orig_compute_sssp_paths(
        self_stub, g.copy(), dict(prepared), set(edges_of_interest))
    opt = PA._optimized_compute_sssp_paths(
        self_stub, g.copy(), dict(prepared), set(edges_of_interest))

    # The consumer only reads paths[source].get(target): compare exactly that
    # surface. Unique weights => unique shortest paths => strict equality.
    for s in sources:
        for t in targets:
            if s == t:
                continue
            assert orig.get(s, {}).get(t) == opt.get(s, {}).get(t), (s, t)


def test_negative_capacity_raises_on_plain_graph():
    # Multigraphs never fire the guard (upstream effective behaviour: the
    # capacity is never read there) — the fail-fast check applies to plain
    # graphs, where upstream's callable did read the attribute dict.
    g = nx.DiGraph()
    g.add_edge("A", "B", capacity=1.0)
    g.add_edge("B", "C", capacity=-1.0)
    self_stub = object.__new__(NFM)
    with pytest.raises(ValueError):
        PA._optimized_compute_sssp_paths(
            self_stub, g, _prepared(["A"], ["C"], g), set())


def test_negative_capacity_ignored_on_multigraph():
    g = _make_graph()
    g.add_edge("F", "A", capacity=-1.0, name="bad")
    self_stub = object.__new__(NFM)
    # No raise: on multigraphs the capacity term is not part of the
    # effective upstream weight (hop-cost only).
    PA._optimized_compute_sssp_paths(
        self_stub, g, _prepared(["A"], ["E"], g), set())


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(PA._DISABLE_ENV, "1")
    assert PA.apply_alphadeesp_dijkstra_patch() is False


def test_version_guard_rejects_unknown_body():
    class FakeMixin:
        def _compute_sssp_paths(self):  # body lacks the assumed markers
            return {}
    assert PA._upstream_body_matches_assumption(FakeMixin) is False
