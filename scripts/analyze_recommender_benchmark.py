#!/usr/bin/env python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
"""Analyse the recommender case benchmark journal
(:mod:`benchmark_recommender_cases`): constraint depth/count profile,
per-action-family efficacy, failure typology, combination headroom.

Usage::

    python scripts/analyze_recommender_benchmark.py \
        --journal /workspace/out_benchmark/cases.jsonl \
        [--json out.json] [--md out.md]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

DEPTH_BINS = [(1.0, 1.05, "1.00-1.05"), (1.05, 1.10, "1.05-1.10"),
              (1.10, 1.20, "1.10-1.20"), (1.20, 99.0, ">1.20")]

FAMILIES = ["open_coupling", "close_coupling", "line_disconnection",
            "line_reconnection", "load_shedding", "renewable_curtailment",
            "redispatch", "pst_tap", "other"]


def depth_bin(d: float) -> str:
    for lo, hi, name in DEPTH_BINS:
        if lo <= d < hi:
            return name
    return "<1.00"


def load(journal: Path) -> list[dict]:
    return [json.loads(x) for x in journal.read_text().splitlines() if x.strip()]


def analyze(rows: list[dict]) -> dict:
    out: dict = {"cases": len(rows)}
    out["outcomes"] = dict(collections.Counter(r.get("outcome") for r in rows))

    ana = [r for r in rows if r.get("outcome") in ("analyzed", "antenna")]
    out["analyzed_cases"] = len(ana)
    out["unique_constrained_situations"] = len(
        {(r["date"], r["timestep"], r["defaut"]) for r in ana})

    # ---- constraint profile -------------------------------------------------
    depths = [r["max_depth"] for r in ana if r.get("max_depth")]
    counts = [r["n_overloads"] for r in ana if r.get("n_overloads") is not None]
    out["constraints"] = {
        "max_depth": {"med": round(float(np.median(depths)), 3),
                      "p90": round(float(np.percentile(depths, 90)), 3),
                      "max": round(max(depths), 3)} if depths else None,
        "depth_bins": dict(collections.Counter(depth_bin(d) for d in depths)),
        "n_overloads": dict(collections.Counter(counts)),
        "multi_overload_cases": sum(1 for c in counts if c > 1),
    }

    # ---- resolution x depth / count ----------------------------------------
    res_by_bin: dict = collections.defaultdict(collections.Counter)
    res_by_count: dict = collections.defaultdict(collections.Counter)
    for r in ana:
        if r.get("max_depth"):
            res_by_bin[depth_bin(r["max_depth"])][r.get("resolution")] += 1
        if r.get("n_overloads"):
            key = str(r["n_overloads"]) if r["n_overloads"] <= 2 else ">=3"
            res_by_count[key][r.get("resolution")] += 1
    out["resolution"] = dict(collections.Counter(
        r.get("resolution") for r in ana))
    out["resolution_by_depth"] = {k: dict(v) for k, v in sorted(res_by_bin.items())}
    out["resolution_by_count"] = {k: dict(v) for k, v in sorted(res_by_count.items())}

    # ---- per-family efficacy ------------------------------------------------
    fam: dict = {f: {"cases_present": 0, "cases_best": 0, "solves": 0,
                     "reduces": 0, "non_convergence": 0, "n_actions": 0,
                     "solo_solves": 0}
                 for f in FAMILIES}
    for r in ana:
        actions = r.get("actions") or []
        if not actions:
            continue
        by_f: dict = collections.defaultdict(list)
        for a in actions:
            f = a.get("family", "other")
            fam.setdefault(f, {"cases_present": 0, "cases_best": 0, "solves": 0,
                               "reduces": 0, "non_convergence": 0,
                               "n_actions": 0, "solo_solves": 0})
            fam[f]["n_actions"] += 1
            if a.get("non_convergence"):
                fam[f]["non_convergence"] += 1
            if a.get("max_rho") is not None and not a.get("non_convergence"):
                by_f[f].append(a["max_rho"])
        valid = {f: min(v) for f, v in by_f.items() if v}
        if not valid:
            continue
        best_f = min(valid, key=valid.get)
        fam[best_f]["cases_best"] += 1
        solvers = [f for f, v in valid.items() if v < 1.0]
        for f, v in valid.items():
            fam[f]["cases_present"] += 1
            if v < 1.0:
                fam[f]["solves"] += 1
            if v < (r.get("max_depth") or 9) - 1e-3:
                fam[f]["reduces"] += 1
        if len(solvers) == 1:
            fam[solvers[0]]["solo_solves"] += 1
    out["families"] = {f: v for f, v in fam.items() if v["n_actions"]}

    # ---- failure typology ---------------------------------------------------
    fails = [r for r in ana if r.get("resolution") in ("partial", "none",
                                                       "no_action")]
    out["failures"] = {
        "n": len(fails),
        "partial": sum(1 for r in fails if r["resolution"] == "partial"),
        "none": sum(1 for r in fails if r["resolution"] == "none"),
        "no_action": sum(1 for r in fails if r["resolution"] == "no_action"),
        "antenna": sum(1 for r in fails if r["outcome"] == "antenna"),
        "multi_overload": sum(1 for r in fails if (r.get("n_overloads") or 0) > 1),
        "depth_med": round(float(np.median([r["max_depth"] for r in fails
                                            if r.get("max_depth")])), 3)
        if any(r.get("max_depth") for r in fails) else None,
        "best_gap_med": round(float(np.median(
            [r["best_action_max_rho"] - 1.0 for r in fails
             if r.get("best_action_max_rho")])), 4)
        if any(r.get("best_action_max_rho") for r in fails) else None,
        "situations": sorted({f'{r["date"]} ts{r["timestep"]} {r["defaut"]}'
                              for r in fails}),
    }

    # ---- combination headroom (GST pairs) -----------------------------------
    pair_rescues = pair_cases = 0
    for r in fails:
        bp = r.get("best_pair")
        if bp and bp.get("max_rho") is not None:
            pair_cases += 1
            if bp["max_rho"] < 1.0:
                pair_rescues += 1
    solved_pairs_better = 0
    for r in ana:
        bp, ba = r.get("best_pair"), r.get("best_action_max_rho")
        if bp and ba is not None and bp.get("max_rho") is not None \
                and bp["max_rho"] < ba - 0.01:
            solved_pairs_better += 1
    out["combination_headroom"] = {
        "failing_cases_with_pairs": pair_cases,
        "pair_rescues_predicted": pair_rescues,
        "cases_where_best_pair_beats_best_action": solved_pairs_better,
    }

    # ---- errors / timing ----------------------------------------------------
    errs = [r for r in rows if str(r.get("outcome", "")).startswith("error")]
    out["errors"] = {
        "n": len(errs),
        "by_message": dict(collections.Counter(
            (r.get("error") or "?")[:80] for r in errs)),
    }
    ts = [r["elapsed_s"] for r in rows if r.get("elapsed_s")]
    tsc = [r["elapsed_s"] for r in ana if r.get("elapsed_s")]
    out["timing_s"] = {
        "all_med": round(float(np.median(ts)), 2) if ts else None,
        "constrained_med": round(float(np.median(tsc)), 2) if tsc else None,
        "constrained_p90": round(float(np.percentile(tsc, 90)), 2) if tsc else None,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = load(Path(args.journal))
    rep = analyze(rows)
    text = json.dumps(rep, indent=1, ensure_ascii=False)
    print(text)
    if args.json:
        Path(args.json).write_text(text)


if __name__ == "__main__":
    main()
