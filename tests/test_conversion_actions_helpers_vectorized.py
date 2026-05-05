"""Tests for the vectorized node-breaker column builders in
`utils.conversion_actions_repas`.

Regression guards for the performance fix that replaced three
`df.iterrows()` loops (~8 400 ms combined on a 85 000-switch grid)
with vectorized pandas string ops (~150 ms, 55× speedup). The fix
preserves strict input/output equivalence; these tests assert it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from expert_op4grid_recommender.utils.conversion_actions_repas import (
    _get_switches_with_topology,
    _get_injection_with_bus_breaker_info,
    _get_branch_with_bus_breaker_info,
    _node_breaker_node_id,
)


def _ref_switches_iterrows(df: pd.DataFrame) -> pd.DataFrame:
    """Reference implementation (the old iterrows code) — used to validate
    the vectorized version produces the SAME output."""
    bus1_ids = [
        _node_breaker_node_id(row['voltage_level_id'], int(row['node1']))
        if pd.notna(row['node1']) else None
        for _, row in df.iterrows()
    ]
    bus2_ids = [
        _node_breaker_node_id(row['voltage_level_id'], int(row['node2']))
        if pd.notna(row['node2']) else None
        for _, row in df.iterrows()
    ]
    return pd.DataFrame({
        'voltage_level_id': df['voltage_level_id'],
        'bus_breaker_bus1_id': bus1_ids,
        'bus_breaker_bus2_id': bus2_ids,
        'open': df['open'],
    }, index=df.index)


class _NetStub:
    """Minimal network stub whose `get_<kind>(all_attributes=True)` returns
    a fixed DataFrame. Used to exercise the three helpers in isolation
    without touching real pypowsybl."""

    def __init__(self, switches=None, loads=None, generators=None,
                 shunt_compensators=None, branches=None, lines=None):
        self._dfs = {
            'get_switches': switches if switches is not None else pd.DataFrame(),
            'get_loads': loads if loads is not None else pd.DataFrame(),
            'get_generators': generators if generators is not None else pd.DataFrame(),
            'get_shunt_compensators': shunt_compensators if shunt_compensators is not None else pd.DataFrame(),
            'get_branches': branches if branches is not None else pd.DataFrame(),
            'get_lines': lines if lines is not None else pd.DataFrame(),
        }

    def __getattr__(self, name):
        if name in self._dfs:
            return lambda *a, **kw: self._dfs[name]
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# _get_switches_with_topology (node-breaker vectorized path)
# ---------------------------------------------------------------------------

def test_switches_vectorized_matches_iterrows_reference():
    """The vectorized node-breaker branch MUST produce a DataFrame
    strictly equivalent to the old iterrows implementation — same rows,
    same values, same NaN handling."""
    df = pd.DataFrame({
        'voltage_level_id': ['VL_A', 'VL_A', 'VL_B', 'VL_C'],
        'node1': [1, 3, 0, 7],
        'node2': [2, 4, 5, 8],
        'kind': ['BREAKER', 'DISCONNECTOR', 'BREAKER', 'BREAKER'],
        'open': [False, True, False, True],
    }, index=['SW1', 'SW2', 'SW3', 'SW4'])
    net = _NetStub(switches=df)

    new = _get_switches_with_topology(net, node_breaker=True)
    ref = _ref_switches_iterrows(df)

    pd.testing.assert_frame_equal(new.reset_index(drop=True),
                                   ref.reset_index(drop=True),
                                   check_dtype=False)


def test_switches_vectorized_handles_nan_nodes():
    """Rows with NaN on node1 or node2 must map to None in the output,
    matching the iterrows behaviour."""
    df = pd.DataFrame({
        'voltage_level_id': ['VL_A', 'VL_B', 'VL_C'],
        'node1': [1, float('nan'), 5],
        'node2': [float('nan'), 3, 6],
        'kind': ['BREAKER', 'BREAKER', 'BREAKER'],
        'open': [False, True, False],
    }, index=['SW1', 'SW2', 'SW3'])
    net = _NetStub(switches=df)

    out = _get_switches_with_topology(net, node_breaker=True)
    assert out.loc['SW1', 'bus_breaker_bus1_id'] == 'VL_A#1'
    assert out.loc['SW1', 'bus_breaker_bus2_id'] is None
    assert out.loc['SW2', 'bus_breaker_bus1_id'] is None
    assert out.loc['SW2', 'bus_breaker_bus2_id'] == 'VL_B#3'
    assert out.loc['SW3', 'bus_breaker_bus1_id'] == 'VL_C#5'
    assert out.loc['SW3', 'bus_breaker_bus2_id'] == 'VL_C#6'


def test_switches_vectorized_empty_dataframe():
    """Empty switches DataFrame passes through without errors."""
    net = _NetStub(switches=pd.DataFrame(
        columns=['voltage_level_id', 'node1', 'node2', 'kind', 'open']
    ))
    out = _get_switches_with_topology(net, node_breaker=True)
    assert list(out.columns) == ['voltage_level_id', 'bus_breaker_bus1_id',
                                  'bus_breaker_bus2_id', 'open']
    assert len(out) == 0


# ---------------------------------------------------------------------------
# _get_injection_with_bus_breaker_info (node-breaker vectorized path)
# ---------------------------------------------------------------------------

def test_injection_vectorized_loads():
    df = pd.DataFrame({
        'voltage_level_id': ['VL_A', 'VL_B', 'VL_C'],
        'node': [10, float('nan'), 20],
    }, index=['LOAD_1', 'LOAD_2', 'LOAD_3'])
    net = _NetStub(loads=df)

    out = _get_injection_with_bus_breaker_info(net, 'get_loads', node_breaker=True)
    assert out.loc['LOAD_1', 'bus_breaker_bus_id'] == 'VL_A#10'
    assert out.loc['LOAD_2', 'bus_breaker_bus_id'] is None
    assert out.loc['LOAD_3', 'bus_breaker_bus_id'] == 'VL_C#20'
    assert list(out['voltage_level_id']) == ['VL_A', 'VL_B', 'VL_C']


def test_injection_vectorized_handles_int_and_float_node_columns():
    """pypowsybl can return `node` as float64 (because of NaN) or int64
    depending on the call — both must produce the same int-suffixed ID."""
    df_float = pd.DataFrame({
        'voltage_level_id': ['VL_A'],
        'node': [5.0],
    }, index=['X'])
    df_int = pd.DataFrame({
        'voltage_level_id': ['VL_A'],
        'node': [5],
    }, index=['X'])

    for df in (df_float, df_int):
        net = _NetStub(generators=df)
        out = _get_injection_with_bus_breaker_info(net, 'get_generators', node_breaker=True)
        assert out.loc['X', 'bus_breaker_bus_id'] == 'VL_A#5'


# ---------------------------------------------------------------------------
# _get_branch_with_bus_breaker_info (node-breaker vectorized path)
# ---------------------------------------------------------------------------

def test_branch_vectorized_lines():
    df = pd.DataFrame({
        'voltage_level1_id': ['VL_A', 'VL_B'],
        'voltage_level2_id': ['VL_B', 'VL_C'],
        'node1': [1, 3],
        'node2': [2, 4],
    }, index=['LINE_1', 'LINE_2'])
    net = _NetStub(lines=df)

    out = _get_branch_with_bus_breaker_info(net, 'get_lines', node_breaker=True)
    assert out.loc['LINE_1', 'bus_breaker_bus1_id'] == 'VL_A#1'
    assert out.loc['LINE_1', 'bus_breaker_bus2_id'] == 'VL_B#2'
    assert out.loc['LINE_2', 'bus_breaker_bus1_id'] == 'VL_B#3'
    assert out.loc['LINE_2', 'bus_breaker_bus2_id'] == 'VL_C#4'


def test_branch_vectorized_handles_nan_on_either_side():
    df = pd.DataFrame({
        'voltage_level1_id': ['VL_A', 'VL_B'],
        'voltage_level2_id': ['VL_B', 'VL_C'],
        'node1': [1, float('nan')],
        'node2': [float('nan'), 4],
    }, index=['BR_1', 'BR_2'])
    net = _NetStub(branches=df)

    out = _get_branch_with_bus_breaker_info(net, 'get_branches', node_breaker=True)
    assert out.loc['BR_1', 'bus_breaker_bus1_id'] == 'VL_A#1'
    assert out.loc['BR_1', 'bus_breaker_bus2_id'] is None
    assert out.loc['BR_2', 'bus_breaker_bus1_id'] is None
    assert out.loc['BR_2', 'bus_breaker_bus2_id'] == 'VL_C#4'
