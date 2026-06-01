"""Tests for the global fast path in `detect_non_reconnectable_lines`.

Regression guards for the performance fix that replaced the per-VL
`get_node_breaker_topology()` loop (~2.8 s on PyPSA-EUR France 400 kV)
with a single batch of `get_lines/trafos/switches(all_attributes=True)`
queries (~0.7 s, 4.7× speedup). The integration test
`test_environment_detection.py::test_non_reconnectable_detection_with_date`
already validates numerical correctness on the small real grid; this
module asserts the fast-path code path is the one being exercised
and that the fallback kicks in for bus-breaker-only grids.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from expert_op4grid_recommender.utils import helpers_pypowsybl as hp


def _mk_network_mock(
    *,
    lines_rows,
    trafos_rows,
    switches_rows,
    node_cols_on_lines: bool = True,
    node_cols_on_switches: bool = True,
):
    """Build a MagicMock pypowsybl Network whose `get_lines(all_attributes=True)`,
    `get_2_windings_transformers(all_attributes=True)` and
    `get_switches(all_attributes=True)` return predictable DataFrames."""
    lines_cols = ["voltage_level1_id", "voltage_level2_id", "connected1", "connected2"]
    if node_cols_on_lines:
        lines_cols += ["node1", "node2"]
    lines_df = pd.DataFrame(lines_rows, columns=["id"] + lines_cols).set_index("id")

    trafos_df = pd.DataFrame(trafos_rows, columns=["id"] + lines_cols).set_index("id") if trafos_rows else pd.DataFrame(columns=lines_cols).astype({"connected1": bool, "connected2": bool})

    switch_cols = ["voltage_level_id", "kind", "open"]
    if node_cols_on_switches:
        switch_cols += ["node1", "node2"]
    switches_df = pd.DataFrame(switches_rows, columns=["id"] + switch_cols).set_index("id") if switches_rows else pd.DataFrame(columns=switch_cols)

    network = MagicMock()
    network.get_lines.return_value = lines_df
    network.get_2_windings_transformers.return_value = trafos_df
    network.get_switches.return_value = switches_df
    return network, lines_df, trafos_df, switches_df


def test_fast_path_detects_line_with_both_sides_isolated():
    """Node-breaker grid: one line is disconnected on both sides, its only
    breaker is open, and all disconnectors on the intermediate node are open
    → line must be flagged non-reconnectable."""
    #   Line LINE_A lives at node 0 (vl=V1) and node 10 (vl=V2).
    #   On V1 side: breaker (node 0 ↔ 1, open=True), disconnector (node 1 ↔ 2, open=True)
    #   On V2 side: breaker (node 10 ↔ 11, open=False) — not isolated on this side
    # Expected: non-reconnectable (isolated on side V1 is enough).
    network, *_ = _mk_network_mock(
        lines_rows=[
            # id, vl1, vl2, c1, c2, n1, n2
            ("LINE_A", "V1", "V2", False, False, 0, 10),
        ],
        trafos_rows=[],
        switches_rows=[
            # id, vl, kind, open, n1, n2
            ("SW1", "V1", "BREAKER", True, 0, 1),
            ("SW2", "V1", "DISCONNECTOR", True, 1, 2),
            ("SW3", "V2", "BREAKER", False, 10, 11),
        ],
    )
    result = hp.detect_non_reconnectable_lines(network)

    assert "LINE_A" in result
    # The fallback path would call get_node_breaker_topology — verify we did NOT.
    network.get_node_breaker_topology.assert_not_called()


def test_fast_path_keeps_reconnectable_when_side_has_closed_disconnector():
    """A line whose breaker is open but whose intermediate node still has a
    CLOSED disconnector is still reconnectable (the DA can re-close the
    breaker). Must NOT be flagged."""
    network, *_ = _mk_network_mock(
        lines_rows=[
            ("LINE_B", "V1", "V2", False, False, 0, 10),
        ],
        trafos_rows=[],
        switches_rows=[
            ("SW1", "V1", "BREAKER", True, 0, 1),
            # Intermediate node has a closed disconnector → reconnectable.
            ("SW2", "V1", "DISCONNECTOR", False, 1, 2),
            ("SW3", "V2", "BREAKER", True, 10, 11),
            ("SW4", "V2", "DISCONNECTOR", False, 11, 12),
        ],
    )
    result = hp.detect_non_reconnectable_lines(network)
    assert result == []
    network.get_node_breaker_topology.assert_not_called()


def test_fast_path_ignores_connected_lines():
    """Lines with connected1=True AND connected2=True are not considered
    — they aren't candidates for reconnection detection."""
    network, *_ = _mk_network_mock(
        lines_rows=[
            ("LINE_OK", "V1", "V2", True, True, 0, 10),
        ],
        trafos_rows=[],
        switches_rows=[
            ("SW1", "V1", "BREAKER", True, 0, 1),
        ],
    )
    assert hp.detect_non_reconnectable_lines(network) == []


def test_fallback_triggered_when_node_columns_missing_on_switches():
    """Bus-breaker-only grid (no node1/node2 on switches) → must drop to
    the per-VL `get_node_breaker_topology()` fallback."""
    network, *_ = _mk_network_mock(
        lines_rows=[
            ("LINE_A", "V1", "V2", False, True, 0, 10),
        ],
        trafos_rows=[],
        switches_rows=[],
        node_cols_on_switches=False,  # bus-breaker switches only
    )
    # Fallback will call get_node_breaker_topology per VL.
    topo_mock = MagicMock()
    topo_mock.nodes = pd.DataFrame(columns=["connectable_id"])
    topo_mock.switches = pd.DataFrame(columns=["node1", "node2", "kind", "open"])
    network.get_node_breaker_topology.return_value = topo_mock

    hp.detect_non_reconnectable_lines(network)

    # The fallback MUST have been exercised.
    assert network.get_node_breaker_topology.call_count >= 1


def test_empty_when_no_disconnected_elements():
    """Early-exit short-circuit: nothing to detect when all lines &
    transformers are fully connected."""
    network, *_ = _mk_network_mock(
        lines_rows=[
            ("LINE_A", "V1", "V2", True, True, 0, 10),
            ("LINE_B", "V3", "V4", True, True, 5, 15),
        ],
        trafos_rows=[],
        switches_rows=[],
    )
    assert hp.detect_non_reconnectable_lines(network) == []
    # Neither global nor per-VL work is needed when nothing is disconnected.
    network.get_switches.assert_not_called()
    network.get_node_breaker_topology.assert_not_called()


def test_fallback_triggered_when_all_nodes_are_nan():
    """A pathological case where lines expose `node1`/`node2` columns BUT
    every row is NaN (bus-breaker-only grid with the newer schema) must
    also trigger the fallback — the fast path would produce an empty
    connectable_map otherwise."""
    import numpy as np
    network, *_ = _mk_network_mock(
        lines_rows=[
            ("LINE_A", "V1", "V2", False, False, np.nan, np.nan),
        ],
        trafos_rows=[],
        switches_rows=[
            ("SW1", "V1", "BREAKER", True, np.nan, np.nan),
        ],
    )
    topo_mock = MagicMock()
    topo_mock.nodes = pd.DataFrame(columns=["connectable_id"])
    topo_mock.switches = pd.DataFrame(columns=["node1", "node2", "kind", "open"])
    network.get_node_breaker_topology.return_value = topo_mock

    hp.detect_non_reconnectable_lines(network)

    # Fast path produced an empty connectable_map → fallback kicks in.
    assert network.get_node_breaker_topology.call_count >= 1
