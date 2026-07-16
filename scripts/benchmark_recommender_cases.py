#!/usr/bin/env python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
"""Benchmark the RECOMMENDER over a (date x timestep x contingency) case grid.

For every case on the ``env_dijon_v2_assistant`` environment (the two shipped
chronic days x sampled timesteps x the ``lignes_a_deconnecter.csv``
contingencies), runs the analysis pipeline in two explicit steps so the bench
can record what ``run_analysis`` alone does not expose:

- **constraint depth** — the N-1 rho of each retained overload (from
  ``context.obs_simu_defaut``), plus the count of overloads, the dropped
  pre-existing ones and the island-preventing exclusions;
- **per-action efficacy** — every prioritized action's reassessed
  ``max_rho`` / ``is_rho_reduction`` / ``non_convergence`` (the
  ``SimulatedAction`` payload), classified into its family from the
  ``action_scores`` categories (reco/disco/open_coupling/close_coupling/
  load shedding/curtailment/redispatch);
- **combination headroom** — the best GST pair of ``combined_actions``
  (betas + predicted max_rho), to measure what pairing adds on cases a
  single action cannot solve;
- **outcome class** — no_overload / pre_existing_only / dead_end (grid
  breaks apart, no antenna) / antenna / analyzed, and for analyzed cases
  solved (best max_rho < 1) / partial (improved but >= 1) / none.

Journal: one JSON line per case (resumable — journaled case keys are
skipped). Usage::

    /workspace/venv_reco/bin/python scripts/benchmark_recommender_cases.py \
        --out out_benchmark/cases.jsonl --ts-step 24 [--jobs 3]

Summary only::

    ... benchmark_recommender_cases.py --out out_benchmark/cases.jsonl --summary
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CONFIG_DELTAS = dict(
    ENV_NAME="env_dijon_v2_assistant",
    FILE_ACTION_SPACE_DESC="reduced_model_actions.json",
    CHECK_ACTION_SIMULATION=True,     # reassess every prioritized action
    N_PRIORITIZED_ACTIONS=20,
    IGNORE_LINES_MONITORING=False,    # the operational monitoring perimeter
    DO_VISUALIZATION=False,
    MAX_RHO_BOTH_EXTREMITIES=False,   # grid2op backend
)


#: The shipped chronics carry 48 half-hour steps (24 h) with perfect
#: forecasts — ``current_timestep`` is a forecast horizon from the first
#: observation, so the valid grid is 0..47.
N_TIMESTEPS = 48


def case_grid(ts_step: int) -> list[dict]:
    dates = ["2024-08-28", "2024-11-27"]
    lines = [ln.strip() for ln in
             (REPO / "data" / "lignes_a_deconnecter.csv").read_text().splitlines()[1:]
             if ln.strip()]
    cases = []
    for d in dates:
        for ts in range(0, N_TIMESTEPS, ts_step):
            for line in lines:
                cases.append({"date": d, "timestep": ts, "defaut": line,
                              "key": f"{d}_ts{ts}_{line}"})
    return cases


def _family_of(key: str, cat_of_key: dict[str, str]) -> str:
    """Action family from the scored-category map, falling back to the
    naming conventions used across the pipeline."""
    if key in cat_of_key:
        return cat_of_key[key]
    if key.startswith("reco_"):
        return "line_reconnection"
    if key.startswith("disco_"):
        return "line_disconnection"
    if key.startswith("node_merging_"):
        return "close_coupling"
    if key.startswith("load_shedding_"):
        return "load_shedding"
    if key.startswith("curtailment_") or key.startswith("renewable_"):
        return "renewable_curtailment"
    if key.startswith("redispatch_"):
        return "redispatch"
    if "_variant_" in key or key.count("-") >= 4:   # uuid-named node actions
        return "open_coupling"
    return "other"


def run_case(case: dict) -> dict:
    from expert_op4grid_recommender.backends import Backend
    from expert_op4grid_recommender.pipeline import (
        AnalysisResult,
        run_analysis_step1,
        run_analysis_step2_discovery,
        run_analysis_step2_graph,
    )

    out = {"key": case["key"], "date": case["date"], "timestep": case["timestep"],
           "defaut": case["defaut"]}
    t0 = time.time()
    try:
        outcome = run_analysis_step1(
            analysis_date=datetime.fromisoformat(case["date"]),
            current_timestep=case["timestep"],
            current_lines_defaut=[case["defaut"]],
            backend=Backend.GRID2OP)
    except Exception as exc:  # noqa: BLE001 — contingency may diverge etc.
        out["outcome"] = "error_step1"
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    if isinstance(outcome, AnalysisResult):
        # short-circuit: no actionable overload, or dead-end (grid apart)
        out["overloads"] = outcome.lines_overloaded_names
        out["n_overloads"] = len(outcome.lines_overloaded_names)
        out["pre_existing"] = outcome.pre_existing_overloads
        out["outcome"] = ("dead_end" if outcome.lines_overloaded_names
                          else ("pre_existing_only" if outcome.pre_existing_overloads
                                else "no_overload"))
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    ctx = outcome
    rho_n1 = ctx.obs_simu_defaut.rho
    names = list(ctx.obs_simu_defaut.name_line)
    depths = {names[i]: round(float(rho_n1[i]), 4)
              for i in (ctx.lines_overloaded_ids or [])}
    # antenna mode leaves lines_overloaded_ids_kept as None/empty
    kept = [names[i] for i in (ctx.lines_overloaded_ids_kept or [])]
    out.update({
        "outcome": "antenna" if ctx.antenna_mode else "analyzed",
        "overloads": ctx.lines_overloaded_names,
        "n_overloads": len(ctx.lines_overloaded_ids or []),
        "n_overloads_kept": len(ctx.lines_overloaded_ids_kept or []),
        "overloads_kept": kept,
        "depths": depths,
        "max_depth": max(depths.values()) if depths else None,
        "sum_excess": round(sum(v - 1.0 for v in depths.values()), 4) if depths else None,
        "pre_existing": ctx.get("pre_existing_rho") and [
            names[i] for i in ctx.get("pre_existing_rho", {})] or [],
    })
    try:
        ctx = run_analysis_step2_graph(ctx)
        res = run_analysis_step2_discovery(ctx)
    except Exception as exc:  # noqa: BLE001
        out["outcome"] = "error_step2"
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    cat_of_key = {}
    for cat, block in (res.action_scores or {}).items():
        for k in (block.get("scores") or {}):
            cat_of_key[k] = cat
    actions = []
    for k, v in (res.prioritized_actions or {}).items():
        try:
            actions.append({
                "name": k,
                "family": _family_of(k, cat_of_key),
                "max_rho": round(float(v.max_rho), 4) if v.max_rho == v.max_rho else None,
                "max_rho_line": v.max_rho_line,
                "is_rho_reduction": bool(v.is_rho_reduction),
                "non_convergence": v.non_convergence,
            })
        except Exception:  # noqa: BLE001 — keep the row minimal on oddities
            actions.append({"name": k, "family": _family_of(k, cat_of_key)})
    out["actions"] = actions
    out["n_actions"] = len(actions)

    # combination headroom (GST pairs): best predicted pair
    best_pair = None
    for pk, pv in (res.combined_actions or {}).items():
        mr = pv.get("max_rho_prediction", pv.get("max_rho"))
        if mr is None:
            continue
        if best_pair is None or mr < best_pair["max_rho"]:
            best_pair = {"pair": pk, "max_rho": round(float(mr), 4),
                         "rho_reduction": bool(pv.get("rho_reduction", False))}
    out["best_pair"] = best_pair
    out["n_pairs"] = len(res.combined_actions or {})

    rhos = [a["max_rho"] for a in actions
            if a.get("max_rho") is not None and not a.get("non_convergence")]
    out["best_action_max_rho"] = min(rhos) if rhos else None
    if rhos and out["max_depth"] is not None:
        best = min(rhos)
        out["resolution"] = ("solved" if best < 1.0 else
                            "partial" if best < out["max_depth"] - 1e-3 else "none")
    else:
        out["resolution"] = "no_action"
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def _worker(case):
    try:
        return run_case(case)
    except Exception as exc:  # noqa: BLE001 — a case must never kill the pool
        return {"key": case["key"], "date": case["date"],
                "timestep": case["timestep"], "defaut": case["defaut"],
                "outcome": "error_driver",
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def summarize(rows: list[dict]) -> dict:
    import collections
    c = collections.Counter(r.get("outcome") for r in rows)
    res = collections.Counter(r.get("resolution") for r in rows
                              if r.get("outcome") in ("analyzed", "antenna"))
    return {"cases": len(rows), "outcomes": dict(c), "resolutions": dict(res)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out_benchmark/cases.jsonl")
    ap.add_argument("--ts-step", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    if args.summary:
        print(json.dumps(summarize(list(done.values())), indent=1,
                         ensure_ascii=False))
        return

    from expert_op4grid_recommender import config
    config.override_settings(**CONFIG_DELTAS)

    cases = [c for c in case_grid(args.ts_step) if c["key"] not in done]
    print(f"{len(done)} journaled, {len(cases)} to run")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.jobs <= 1:
        results = map(run_case, cases)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.jobs, initializer=_init_worker,
                        maxtasksperchild=40)
        results = pool.imap_unordered(_worker, cases)
    n = len(done)
    fresh: list[dict] = []
    with out.open("a") as f:
        for r in results:
            n += 1
            fresh.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f'[{n}] {r["key"]:44s} {r.get("outcome", "?"):16s} '
                  f'depth={r.get("max_depth")} res={r.get("resolution", "-")} '
                  f'{r.get("elapsed_s")}s')
    print(json.dumps(summarize(list(done.values()) + fresh), indent=1))


def _init_worker():
    from expert_op4grid_recommender import config
    config.override_settings(**CONFIG_DELTAS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        traceback.print_exc()
