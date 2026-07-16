# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Import-time, version-guarded performance patch for alphaDeesp's
``NullFlowGraphMixin._compute_sssp_paths`` (overflow-graph null-flow-line
search).

Background
----------
Profiling the overflow-graph build on a national-scale network (RTE7000,
~6 400 nodes / ~9 500 branches) shows ~90 % of the non-load-flow time inside
``add_relevant_null_flow_lines_all_paths``, and within it the
``_compute_sssp_paths`` step: one full-graph ``nx.single_source_dijkstra_path``
per *source* node (hundreds of runs), each using a Python weight **callable**
(``incentivized_weight``) invoked on every edge relaxation — the slowest
weight mode networkx supports.

This module applies the two optimizations validated on the RTE7000 benchmark
(x1.6 on ``add_relevant_null_flow_lines`` with a strictly identical output
graph — same edges, same hubs):

1. **Precomputed edge-attribute weight** — one O(E) pass writes
   ``capacity * 1e9 + hop_cost`` into a private edge attribute, then Dijkstra
   runs with ``weight="<attr>"`` (a string), removing the per-relaxation
   Python call. For parallel (multi-)edges networkx takes the minimum over the
   attribute values, exactly as it does over callable results.
2. **Target-side reverse Dijkstra** — the downstream consumer
   (``_collect_paths_of_interest``) only reads ``paths[source][target]`` for
   the prepared *target* nodes. When ``|targets| < |sources|`` the same paths
   are obtained by running Dijkstra from each **target** on the reversed graph
   view (``|targets|`` runs instead of ``|sources|``) and reversing the node
   paths.

Upstream quirk faithfully reproduced (found while testing this patch)
----------------------------------------------------------------------
On a **multigraph** (the overflow graph is a ``MultiDiGraph``), networkx
passes a weight *callable* the ``{key: attrdict}`` mapping of the parallel
edges — not a single edge's attribute dict. Upstream's
``incentivized_weight`` therefore reads ``attr.get("capacity", 0)`` on that
keyed mapping and **always gets 0**: the effective edge weight is the hop
cost alone (33 promoted / 100 normal); the capacity term is silently
ignored, and the negative-capacity guard can never fire. This patch
reproduces that *effective* behaviour exactly on multigraphs (hop-cost-only
precomputed weight) so the output is byte-identical, and keeps the intended
``capacity·1e9 + hop`` formula on plain graphs where the callable did
receive the attribute dict. The quirk itself should be fixed (or blessed)
upstream in alphaDeesp.

Behaviour notes (deliberate, minor deviations):

- On plain (non-multi) graphs the negative-weight guard runs during the
  precompute pass over **all** edges, whereas upstream raised only if the
  offending edge was actually relaxed. Fail-fast is at least as safe.
- A private ``_eo4g_iw`` attribute is written on the working graph's edges
  (overwritten on every call; never read by other code).
- Tie-breaking between equal-weight shortest paths may differ from upstream;
  the retained null-flow line set was verified identical on the benchmark
  (9 615 edges, 12 hubs).

Following the package precedent (``patched_backend.py``, review M5): applied
at import time by ``__init__.py``, idempotent, no-op when alphaDeesp is
absent, self-disabling if the upstream implementation drifts, and removable
via the ``EXPERT_OP4GRID_DISABLE_ALPHADEESP_DIJKSTRA_PATCH`` environment
variable. The proper home for these gains is upstream alphaDeesp
(``expertop4grid``); this patch bridges until then.
"""
from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, Set

_logger = logging.getLogger(__name__)

#: Sentinel attribute stamped on the patched class so we never double-wrap.
_PATCH_FLAG = "_eo4g_dijkstra_patched"

#: Private edge attribute carrying the precomputed incentivised weight.
_WEIGHT_ATTR = "_eo4g_iw"

#: Kill switch (set to any non-empty value to keep the upstream implementation).
_DISABLE_ENV = "EXPERT_OP4GRID_DISABLE_ALPHADEESP_DIJKSTRA_PATCH"

#: Markers we expect in the upstream body — the version guard. If upstream
#: rewrites the method (e.g. ships its own optimization), we stand down.
_ASSUMED_MARKERS = ("single_source_dijkstra_path", "incentivized_weight")

# Constants mirrored from upstream _compute_sssp_paths.
_HUGE_MULTIPLIER = 1_000_000_000
_NORMAL_HOP_COST = 100
_PROMOTED_HOP_COST = 33


def _resolve_null_flow_mixin():
    """Return alphaDeesp's ``NullFlowGraphMixin`` class, or ``None`` if absent."""
    try:
        from alphaDeesp.core.graphs.null_flow_graph import NullFlowGraphMixin
        return NullFlowGraphMixin
    except Exception as exc:  # ImportError or a partial install
        _logger.debug("alphaDeesp null_flow_graph unavailable, skip patch: %s", exc)
        return None


def _upstream_body_matches_assumption(cls) -> bool:
    """Best-effort version guard: the upstream method must still be the
    per-source Dijkstra loop with the Python weight callable."""
    try:
        src = inspect.getsource(cls._compute_sssp_paths)
    except (OSError, TypeError, AttributeError) as exc:
        _logger.debug("Cannot read upstream _compute_sssp_paths source: %s", exc)
        return False
    return all(marker in src for marker in _ASSUMED_MARKERS)


def _optimized_compute_sssp_paths(
    self,
    g_c,
    prepared: Dict[str, Any],
    edges_of_interest: Set[Any],
) -> Dict[Any, Any]:
    """Drop-in replacement for ``NullFlowGraphMixin._compute_sssp_paths``.

    Same contract: a ``{source: {target_or_node: node_path}}`` mapping whose
    entries the caller reads via ``paths[source].get(target)``.
    """
    import networkx as nx

    promoted = set(edges_of_interest)
    is_multi = g_c.is_multigraph()
    # G1 — one O(E) pass: precompute the incentivised weight as an attribute.
    if is_multi:
        # Reproduce the upstream *effective* multigraph behaviour: the weight
        # callable received the {key: attrdict} mapping, so its capacity read
        # always yielded 0 — the effective weight is the hop cost alone (see
        # module docstring). Byte-identical routing, no capacity term.
        for u, v, _k, data in g_c.edges(keys=True, data=True):
            data[_WEIGHT_ATTR] = (
                _PROMOTED_HOP_COST if (u, v) in promoted else _NORMAL_HOP_COST)
    else:
        for u, v, data in g_c.edges(data=True):
            real_weight = data.get("capacity", 0)
            if real_weight < 0:
                raise ValueError("Negative weights not allowed.")
            hop_cost = _PROMOTED_HOP_COST if (u, v) in promoted else _NORMAL_HOP_COST
            data[_WEIGHT_ATTR] = (real_weight * _HUGE_MULTIPLIER) + hop_cost

    node_has_incident_interest = prepared["node_has_incident_interest"]
    bfs_cache = prepared["bfs_cache"]
    targets_with_bfs = prepared["targets_with_bfs"]
    any_target_has_interest = prepared["any_target_has_interest"]

    # Same source filter as upstream.
    sources = [
        s for s in set(prepared["source_nodes_in_gc"])
        if (node_has_incident_interest[s] or any_target_has_interest)
        and (bfs_cache[s] or targets_with_bfs)
    ]
    targets = list(set(prepared["target_nodes_in_gc"]))

    sssp_paths_cache: Dict[Any, Any] = {}
    if 0 < len(targets) < len(sources):
        # G2 — fewer runs from the target side on the reversed view; a
        # target->source path there is the source->target path reversed.
        g_rev = g_c.reverse(copy=False)
        per_target: Dict[Any, Dict[Any, Any]] = {}
        for t in targets:
            try:
                per_target[t] = nx.single_source_dijkstra_path(
                    g_rev, t, weight=_WEIGHT_ATTR)
            except Exception:
                per_target[t] = {}
        for s in sources:
            paths_for_s = {}
            for t in targets:
                rev_path = per_target[t].get(s)
                if rev_path:
                    paths_for_s[t] = list(reversed(rev_path))
            sssp_paths_cache[s] = paths_for_s
    else:
        for s in sources:
            try:
                sssp_paths_cache[s] = nx.single_source_dijkstra_path(
                    g_c, s, weight=_WEIGHT_ATTR)
            except Exception:
                sssp_paths_cache[s] = {}
    return sssp_paths_cache


def apply_alphadeesp_dijkstra_patch() -> bool:
    """Install the optimized ``_compute_sssp_paths`` on alphaDeesp's
    ``NullFlowGraphMixin``. Returns ``True`` iff the patch is in place
    (idempotent). No-op when alphaDeesp is absent, when the kill-switch
    environment variable is set, or when upstream no longer matches the
    assumed implementation."""
    if os.environ.get(_DISABLE_ENV):
        _logger.info("alphaDeesp Dijkstra patch disabled via %s", _DISABLE_ENV)
        return False
    cls = _resolve_null_flow_mixin()
    if cls is None:
        return False
    if getattr(cls, _PATCH_FLAG, False):
        return True
    if not hasattr(cls, "_compute_sssp_paths"):
        _logger.debug("Upstream _compute_sssp_paths missing — skip patch.")
        return False
    if not _upstream_body_matches_assumption(cls):
        _logger.info(
            "alphaDeesp _compute_sssp_paths no longer matches the assumed "
            "implementation (upstream change?) — Dijkstra patch NOT applied."
        )
        return False
    cls._eo4g_orig_compute_sssp_paths = cls._compute_sssp_paths
    cls._compute_sssp_paths = _optimized_compute_sssp_paths
    setattr(cls, _PATCH_FLAG, True)
    _logger.debug("alphaDeesp NullFlowGraphMixin._compute_sssp_paths patched")
    return True
