#!/usr/bin/env python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
"""Try measurable improvement axes of the recommender on the benchmark's
non-solved cases (:mod:`benchmark_recommender_cases` journal).

Axes:

- **A. loop-guard** (``--axis loop-guard``): the alphaDeesp overflow graph
  crashes on contingencies whose graph carries NO red loop
  (``red_loops.Path.sum()`` returns a float on an empty frame). A defensive
  monkeypatch returns the constrained-path-only dispatch context instead,
  letting discovery continue. Reruns every ``error_step2`` case of the
  journal with the guard and reports how many now produce actions.
- **B. pair-validation** (``--axis pairs``): for every ``partial`` /
  ``none`` case, take the best GST-predicted pair of prioritized actions
  and VALIDATE it by true simulation (grid2op action composition), i.e.
  measure the real post-pair max rho — does combining rescue the case?
- **C. budget** (``--axis budget``): rerun the failing cases with
  ``N_PRIORITIZED_ACTIONS`` doubled (20 -> 40) and every per-family
  minimum raised, to test whether failures are a ranking/budget artefact
  or a real absence of solution in the action space.

Usage::

    /workspace/venv_reco/bin/python scripts/test_improvement_axes.py \
        --journal /workspace/out_benchmark/cases.jsonl \
        --axis loop-guard pairs budget --out /workspace/out_benchmark/axes.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

BASE_DELTAS = dict(
    ENV_NAME="env_dijon_v2_assistant",
    FILE_ACTION_SPACE_DESC="reduced_model_actions.json",
    CHECK_ACTION_SIMULATION=True,
    N_PRIORITIZED_ACTIONS=20,
    IGNORE_LINES_MONITORING=False,
    DO_VISUALIZATION=False,
    MAX_RHO_BOTH_EXTREMITIES=False,
)


def install_loop_guard() -> None:
    """Defensive patch of alphaDeesp's ``get_dispatch_edges_nodes``: an
    overflow graph without red loops must yield an empty dispatch set, not
    a crash (``DataFrame.Path.sum()`` on an empty frame returns ``0.0``)."""
    from alphaDeesp.core.graphs import structured_overload_graph as sog

    orig = sog.Structured_Overload_Distribution_Graph.get_dispatch_edges_nodes

    def guarded(self, only_loop_paths: bool = False):
        try:
            return orig(self, only_loop_paths=only_loop_paths)
        except TypeError:
            # no red loop in the graph: empty (lines, nodes) dispatch path
            return [], []
    sog.Structured_Overload_Distribution_Graph.get_dispatch_edges_nodes = guarded


def _run(case_key: str, extra_deltas: dict | None = None):
    """One pipeline run (step1+step2) for a journal case key."""
    from expert_op4grid_recommender import config
    from expert_op4grid_recommender.backends import Backend
    from expert_op4grid_recommender.pipeline import (
        AnalysisResult,
        run_analysis_step1,
        run_analysis_step2_discovery,
        run_analysis_step2_graph,
    )
    config.override_settings(**{**BASE_DELTAS, **(extra_deltas or {})})
    date, ts, defaut = case_key.split("_", 2)
    ts = int(ts[2:])
    outcome = run_analysis_step1(
        analysis_date=datetime.fromisoformat(date), current_timestep=ts,
        current_lines_defaut=[defaut], backend=Backend.GRID2OP)
    if isinstance(outcome, AnalysisResult):
        return None, outcome
    ctx = run_analysis_step2_graph(outcome)
    return ctx, run_analysis_step2_discovery(ctx)


def axis_loop_guard(rows: list[dict]) -> dict:
    errs = [r for r in rows if r.get("outcome") == "error_step2"]
    install_loop_guard()
    out = {"axis": "loop-guard", "cases": len(errs), "recovered": 0,
           "solved": 0, "detail": []}
    for r in errs:
        try:
            ctx, res = _run(r["key"])
        except Exception as exc:  # noqa: BLE001
            out["detail"].append({"key": r["key"],
                                  "error": str(exc)[:120]})
            continue
        rhos = [float(v.max_rho) for v in (res.prioritized_actions or {}).values()
                if v.max_rho == v.max_rho and not v.non_convergence]
        best = min(rhos) if rhos else None
        out["recovered"] += bool(rhos)
        out["solved"] += bool(best is not None and best < 1.0)
        out["detail"].append({"key": r["key"], "n_actions": len(rhos),
                              "best_max_rho": round(best, 4) if best else None})
    return out


def axis_pairs(rows: list[dict]) -> dict:
    """Validate the best GST pair by TRUE simulation on failing cases."""
    fails = [r for r in rows if r.get("resolution") in ("partial", "none")
             and r.get("best_pair")]
    out = {"axis": "pairs", "cases": len(fails), "validated_rescues": 0,
           "prediction_gap": [], "detail": []}
    for r in fails:
        try:
            ctx, res = _run(r["key"])
        except Exception as exc:  # noqa: BLE001
            out["detail"].append({"key": r["key"], "error": str(exc)[:120]})
            continue
        # best predicted pair among the rerun's combined_actions
        best_pk, best_mr = None, None
        for pk, pv in (res.combined_actions or {}).items():
            mr = pv.get("max_rho")
            if mr is not None and (best_mr is None or mr < best_mr):
                best_pk, best_mr = pk, mr
        if best_pk is None:
            out["detail"].append({"key": r["key"], "note": "no pair"})
            continue
        a_name, b_name = best_pk.split("+", 1)
        pa = res.prioritized_actions
        if a_name not in pa or b_name not in pa:
            out["detail"].append({"key": r["key"], "note": "pair member missing"})
            continue
        combined = pa[a_name].action + pa[b_name].action
        backend = ctx.backend
        act_defaut = backend.create_default_action(
            ctx.env.action_space, ctx.current_lines_defaut)
        ok, obs2 = backend.check_rho_reduction(
            ctx.obs, ctx.current_timestep, act_defaut, combined,
            ctx.lines_overloaded_ids, ctx.act_reco_maintenance,
            ctx.lines_we_care_about)
        real = float(obs2.rho.max()) if obs2 is not None else None
        rescued = bool(real is not None and real < 1.0)
        out["validated_rescues"] += rescued
        if real is not None and best_mr is not None:
            out["prediction_gap"].append(round(real - best_mr, 4))
        out["detail"].append({
            "key": r["key"], "pair": best_pk,
            "pair_pred_max_rho": round(float(best_mr), 4),
            "pair_simulated_max_rho": round(real, 4) if real else None,
            "single_best_max_rho": r.get("best_action_max_rho"),
            "rescued": rescued})
    return out


def axis_budget(rows: list[dict]) -> dict:
    fails = [r for r in rows if r.get("resolution") in ("partial", "none")]
    deltas = dict(N_PRIORITIZED_ACTIONS=40, MIN_LINE_RECONNECTIONS=3,
                  MIN_CLOSE_COUPLING=5, MIN_OPEN_COUPLING=5,
                  MIN_LINE_DISCONNECTIONS=5, MIN_LOAD_SHEDDING=5,
                  MIN_RENEWABLE_CURTAILMENT=3, MIN_REDISPATCH=3)
    out = {"axis": "budget", "cases": len(fails), "solved_now": 0,
           "improved": 0, "detail": []}
    for r in fails:
        try:
            ctx, res = _run(r["key"], deltas)
        except Exception as exc:  # noqa: BLE001
            out["detail"].append({"key": r["key"], "error": str(exc)[:120]})
            continue
        rhos = [float(v.max_rho) for v in (res.prioritized_actions or {}).values()
                if v.max_rho == v.max_rho and not v.non_convergence]
        best = min(rhos) if rhos else None
        prev = r.get("best_action_max_rho")
        out["solved_now"] += bool(best is not None and best < 1.0)
        out["improved"] += bool(best is not None and prev is not None
                                and best < prev - 0.005)
        out["detail"].append({"key": r["key"], "n_actions": len(rhos),
                              "best_before": prev,
                              "best_now": round(best, 4) if best else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--axis", nargs="+", default=["loop-guard", "pairs", "budget"],
                    choices=["loop-guard", "pairs", "budget"])
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.journal).read_text().splitlines()
            if x.strip()]
    results = []
    for axis in args.axis:
        fn = {"loop-guard": axis_loop_guard, "pairs": axis_pairs,
              "budget": axis_budget}[axis]
        res = fn(rows)
        results.append(res)
        print(json.dumps({k: v for k, v in res.items() if k != "detail"},
                         ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1,
                                             ensure_ascii=False))


if __name__ == "__main__":
    main()
