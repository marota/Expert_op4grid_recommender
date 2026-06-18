# expert_op4grid_recommender/graph_analysis/antenna_graph.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles.
"""Synthetic overflow graph for an islanded radial ("antenne") pocket.

When a contingency islands a radial pocket of substations — disconnecting even
the single most-loaded overloaded line breaks the grid apart — the regular
``build_overflow_graph`` pipeline cannot run: ``alphaDeesp`` computes a flow
*redistribution* that has no meaning on a pocket that simply blacks out.

Instead we build a synthetic downstream ("aval") overflow graph by hand:

* the constraint (overloaded) line becomes the **black** constrained edge,
  oriented from its root substation (on the main grid) into the pocket;
* every branch inside the pocket becomes a **blue** edge, oriented away from the
  root (root → leaves) and carrying ``-|p_or|`` as its reported flow — the
  *negative delta* of the flow it would lose if the constraint were cut.

Because ``OverFlowGraph`` colours an edge blue when its reported flow is
negative, every pocket branch lands on the constrained / downstream path, and
``Structured_Overload_Distribution_Graph`` classifies the whole pocket as
"aval". The recommender's injection-action discovery (load shedding, renewable
curtailment, redispatch) then targets those downstream nodes directly.

The function returns the same 6-tuple shape as
:func:`expert_op4grid_recommender.graph_analysis.builder.build_overflow_graph`
(with ``overflow_sim = None``, since no ``Grid2opSimulation`` is involved) plus
an ``antenna_meta`` dict describing the pocket.
"""

from collections import deque
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from alphaDeesp.core.graphsAndPaths import OverFlowGraph, Structured_Overload_Distribution_Graph


def _sub_injections(obs: Any, sub_ids: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Aggregate production and load (MW) per substation id."""
    prod_by_sub: Dict[int, float] = {sid: 0.0 for sid in sub_ids}
    load_by_sub: Dict[int, float] = {sid: 0.0 for sid in sub_ids}
    sub_set = set(sub_ids)

    gen_p = getattr(obs, "gen_p", None)
    gen_to_subid = getattr(obs, "gen_to_subid", None)
    if gen_p is not None and gen_to_subid is not None:
        for p, sid in zip(gen_p, gen_to_subid):
            sid = int(sid)
            if sid in sub_set:
                prod_by_sub[sid] += float(p)

    load_p = getattr(obs, "load_p", None)
    load_to_subid = getattr(obs, "load_to_subid", None)
    if load_p is not None and load_to_subid is not None:
        for p, sid in zip(load_p, load_to_subid):
            sid = int(sid)
            if sid in sub_set:
                load_by_sub[sid] += float(p)

    return prod_by_sub, load_by_sub


def _orient_pocket(obs: Any, node_sub_ids: List[int], root_sub_id: int,
                   constraint_line_id: int) -> List[Tuple[int, int, int]]:
    """Return pocket branches as (from_sub, to_sub, line_id), oriented root→leaf.

    Orientation is purely *topological* (breadth-first distance from the root),
    so the pocket forms a clean DAG pointing away from the constraint regardless
    of whether the pocket is a net consumer or producer. The constraint line
    itself is excluded — it is added separately as the black edge.
    """
    sub_set = set(node_sub_ids)
    # Build undirected adjacency over connected lines internal to the pocket.
    adj: Dict[int, List[Tuple[int, int]]] = {sid: [] for sid in node_sub_ids}
    line_status = obs.line_status
    for lid in range(len(obs.name_line)):
        if lid == constraint_line_id or not line_status[lid]:
            continue
        a = int(obs.line_or_to_subid[lid])
        b = int(obs.line_ex_to_subid[lid])
        if a in sub_set and b in sub_set and a != b:
            adj[a].append((b, lid))
            adj[b].append((a, lid))

    # BFS depth from the root substation.
    depth: Dict[int, int] = {root_sub_id: 0}
    queue = deque([root_sub_id])
    while queue:
        cur = queue.popleft()
        for nxt, _lid in adj.get(cur, []):
            if nxt not in depth:
                depth[nxt] = depth[cur] + 1
                queue.append(nxt)

    branches: List[Tuple[int, int, int]] = []
    seen_lines = set()
    for sid in node_sub_ids:
        for nxt, lid in adj[sid]:
            if lid in seen_lines:
                continue
            seen_lines.add(lid)
            d_a = depth.get(sid, np.inf)
            d_b = depth.get(nxt, np.inf)
            # Orient from the endpoint closer to the root toward the leaf.
            if d_a <= d_b:
                branches.append((sid, nxt, lid))
            else:
                branches.append((nxt, sid, lid))
    return branches


def build_antenna_overflow_graph(obs: Any, constraint_line_id: int,
                                 antenna_sub_ids: List[int], root_sub_id: int,
                                 float_precision: str = "%.0f"):
    """Build a synthetic downstream overflow graph for an islanded pocket.

    Args:
        obs: Grid2Op-compatible observation (post-contingency state, where the
            constraint line is still connected and overloaded).
        constraint_line_id (int): index of the overloaded line feeding the pocket.
        antenna_sub_ids (list[int]): substation ids inside the pocket.
        root_sub_id (int): the constraint endpoint on the main grid.
        float_precision (str): edge-label precision passed to ``OverFlowGraph``.

    Returns:
        tuple: ``(df, overflow_sim, g_overflow, real_hubs, g_distribution_graph,
        node_name_mapping, antenna_meta)`` — mirrors ``build_overflow_graph``'s
        return (``overflow_sim is None``) with ``antenna_meta`` appended.
    """
    # Compact, contiguous node ids: root first, then the pocket substations.
    node_sub_ids: List[int] = [root_sub_id] + [s for s in antenna_sub_ids if s != root_sub_id]
    sub_to_idx: Dict[int, int] = {sid: i for i, sid in enumerate(node_sub_ids)}
    node_name_mapping: Dict[int, str] = {i: obs.name_sub[sid] for i, sid in enumerate(node_sub_ids)}

    prod_by_sub, load_by_sub = _sub_injections(obs, node_sub_ids)

    # topo["nodes"]: arrays indexed by compact node id (0..k-1). build_nodes
    # pulls from prods_values / loads_values only where are_prods / are_loads.
    are_prods, are_loads, prods_values, loads_values = [], [], [], []
    for sid in node_sub_ids:
        prod = prod_by_sub.get(sid, 0.0)
        load = load_by_sub.get(sid, 0.0)
        are_prods.append(prod > 0)
        are_loads.append(load > 0)
        if prod > 0:
            prods_values.append(prod)
        if load > 0:
            loads_values.append(load)

    topo = {
        "nodes": {
            "are_prods": are_prods,
            "are_loads": are_loads,
            "prods_values": prods_values,
            "loads_values": loads_values,
        },
        "edges": {"idx_or": [], "idx_ex": [], "init_flows": []},
    }

    p_or = obs.p_or
    # Row 0 is the constraint (black) edge, root → pocket entry.
    or_sub = int(obs.line_or_to_subid[constraint_line_id])
    ex_sub = int(obs.line_ex_to_subid[constraint_line_id])
    entry_sub = ex_sub if or_sub == root_sub_id else or_sub
    rows = [{
        "idx_or": sub_to_idx[root_sub_id],
        "idx_ex": sub_to_idx[entry_sub],
        "delta_flows": -abs(float(p_or[constraint_line_id])),
        "gray_edges": False,
        "line_name": obs.name_line[constraint_line_id],
    }]

    # Pocket branches: blue (negative delta = lost report), oriented root→leaf.
    for from_sub, to_sub, lid in _orient_pocket(obs, node_sub_ids, root_sub_id, constraint_line_id):
        rows.append({
            "idx_or": sub_to_idx[from_sub],
            "idx_ex": sub_to_idx[to_sub],
            "delta_flows": -abs(float(p_or[lid])),
            "gray_edges": False,
            "line_name": obs.name_line[lid],
        })

    df = pd.DataFrame(rows, columns=["idx_or", "idx_ex", "delta_flows", "gray_edges", "line_name"])

    # The constraint line (row 0) is the only "cut" line → coloured black.
    g_overflow = OverFlowGraph(topo, [0], df, float_precision=float_precision)
    g_overflow.g = nx.relabel_nodes(g_overflow.g, node_name_mapping, copy=True)

    g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)
    real_hubs = g_distribution_graph.get_hubs()

    total_prod = sum(prod_by_sub.get(s, 0.0) for s in antenna_sub_ids)
    total_load = sum(load_by_sub.get(s, 0.0) for s in antenna_sub_ids)
    net_mw = total_prod - total_load
    antenna_meta = {
        "constraint_line_id": int(constraint_line_id),
        "constraint_line_name": obs.name_line[constraint_line_id],
        "root_sub_id": int(root_sub_id),
        "root_sub_name": obs.name_sub[root_sub_id],
        "antenna_sub_ids": list(antenna_sub_ids),
        "antenna_sub_names": [obs.name_sub[s] for s in antenna_sub_ids],
        "n_subs": len(antenna_sub_ids),
        "total_prod_mw": round(float(total_prod), 2),
        "total_load_mw": round(float(total_load), 2),
        "net_mw": round(float(net_mw), 2),
        # Net exporter (prod > load) → curtailment / redispatch-down help;
        # net importer (load > prod) → load shedding / redispatch-up help.
        "direction": "producer" if net_mw > 0 else "consumer",
    }

    return df, None, g_overflow, real_hubs, g_distribution_graph, node_name_mapping, antenna_meta
