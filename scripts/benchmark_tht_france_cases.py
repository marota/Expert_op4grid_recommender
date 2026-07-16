#!/usr/bin/env python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
"""Benchmark the recommender (pypowsybl backend) on the reconstructed
France-THT cases (Grid_snapshot_reconstruct, THT-only 225/400 kV mode).

Per solved THT snapshot (one per canonical 2021-2023 instant):

1. **N-1 screening** (DC security analysis over every THT line): keep the
   contingencies that create at least one post-contingency overload
   (>= ``RHO_MIN`` of the file's own permanent limits x the monitoring
   factor) on another line — depth and count recorded for every one;
2. **recommender run** on the ``--per-case`` deepest screened
   contingencies through the pypowsybl backend (static network, real
   seasonal limits). The RTE7000 node-action space (REPAS) is not
   available here, so the exercised families are line disconnection /
   reconnection, load shedding (on the THT->HT equivalent loads!) and
   renewable curtailment — which is itself a finding: what the
   recommender can carry on a THT-only national grid without a nodal
   action space.

Journal: JSONL, same schema as :mod:`benchmark_recommender_cases` (with
``source="france_tht"``), resumable.

Usage::

    /workspace/venv_reco/bin/python scripts/benchmark_tht_france_cases.py \
        --cases-dir /workspace/out_benchmark/tht_cases \
        --out /workspace/out_benchmark/cases_tht_france.jsonl --per-case 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RHO_MIN = 1.0


def screen_case(xiidm: Path, per_case: int) -> list[dict]:
    """DC N-1 screening: contingencies creating overloads, deepest first."""
    import numpy as np
    import pypowsybl as pp
    import pypowsybl.loadflow as lf

    net = pp.network.load(str(xiidm))
    lf.run_dc(net, parameters=lf.Parameters(distributed_slack=True))
    lines = net.get_lines()
    lines = lines[lines["connected1"] & lines["connected2"]]
    # permanent limits (A) of every line, most restrictive side
    lim = net.get_operational_limits()
    perm = lim[lim.index.get_level_values("acceptable_duration") == -1]
    amp = perm.groupby(level=0)["value"].min()
    monitored = [lid for lid in lines.index if amp.get(lid, np.nan) > 0]
    sa = pp.security.create_analysis()
    for lid in monitored:
        sa.add_single_element_contingency(lid, lid)
    sa.add_monitored_elements(branch_ids=monitored)
    res = sa.run_dc(net)
    br = res.branch_results.reset_index()
    br = br[br["contingency_id"] != ""]      # post-contingency rows only
    br["rho"] = br.apply(
        lambda r: abs(r["i1"]) / amp[r["branch_id"]]
        if r["branch_id"] in amp.index and amp[r["branch_id"]] > 0 else 0.0,
        axis=1)
    out = []
    for cid, grp in br.groupby("contingency_id"):
        over = grp[(grp["rho"] >= RHO_MIN) & (grp["branch_id"] != cid)]
        if not len(over):
            continue
        out.append({
            "defaut": cid,
            "n_overloads": int(len(over)),
            "max_depth_dc": round(float(over["rho"].max()), 4),
            "overloads": {r["branch_id"]: round(float(r["rho"]), 4)
                          for _, r in over.iterrows()},
        })
    out.sort(key=lambda x: -x["max_depth_dc"])
    return out[:per_case], len(out)


def run_case(xiidm: Path, label: str, defaut: str) -> dict:
    from expert_op4grid_recommender import config
    from expert_op4grid_recommender.backends import Backend
    from expert_op4grid_recommender.pipeline import (
        AnalysisResult,
        run_analysis_step1,
        run_analysis_step2_discovery,
        run_analysis_step2_graph,
    )
    config.override_settings(
        ENV_NAME=xiidm.name,
        CHECK_ACTION_SIMULATION=True,
        N_PRIORITIZED_ACTIONS=20,
        IGNORE_LINES_MONITORING=True,       # no monitoring file: watch all
        DO_VISUALIZATION=False,
        MAX_RHO_BOTH_EXTREMITIES=True,
        USE_VOLTAGE_LEVEL_NAMES_IN_GRAPH=False,
        PYPOWSYBL_FAST_MODE=True,
    )
    # derived paths are plain module attributes (see benchmark_pipeline.py)
    config.ENV_FOLDER = xiidm.parent
    config.ENV_PATH = xiidm
    # no nodal action space for RTE7000 snapshots (REPAS absent): empty dict
    config.ACTION_FILE_PATH = xiidm.parent / "empty_actions.json"
    out = {"key": f"{label}_{defaut}", "source": "france_tht",
           "case": label, "defaut": defaut}
    t0 = time.time()
    try:
        outcome = run_analysis_step1(
            analysis_date=None, current_timestep=0,
            current_lines_defaut=[defaut], backend=Backend.PYPOWSYBL)
    except Exception as exc:  # noqa: BLE001
        out.update(outcome="error_step1", error=f"{type(exc).__name__}: {exc}"[:200],
                   elapsed_s=round(time.time() - t0, 2))
        return out
    if isinstance(outcome, AnalysisResult):
        out["overloads"] = outcome.lines_overloaded_names
        out["n_overloads"] = len(outcome.lines_overloaded_names)
        out["outcome"] = ("dead_end" if outcome.lines_overloaded_names
                          else ("pre_existing_only" if outcome.pre_existing_overloads
                                else "no_overload"))
        out["pre_existing"] = outcome.pre_existing_overloads
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out
    ctx = outcome
    rho_n1 = ctx.obs_simu_defaut.rho
    names = list(ctx.obs_simu_defaut.name_line)
    depths = {names[i]: round(float(rho_n1[i]), 4)
              for i in (ctx.lines_overloaded_ids or [])}
    out.update({
        "outcome": "antenna" if ctx.antenna_mode else "analyzed",
        "overloads": ctx.lines_overloaded_names,
        "n_overloads": len(ctx.lines_overloaded_ids or []),
        "n_overloads_kept": len(ctx.lines_overloaded_ids_kept or []),
        "depths": depths,
        "max_depth": max(depths.values()) if depths else None,
        "sum_excess": round(sum(v - 1.0 for v in depths.values()), 4)
        if depths else None,
    })
    try:
        ctx = run_analysis_step2_graph(ctx)
        res = run_analysis_step2_discovery(ctx)
    except Exception as exc:  # noqa: BLE001
        out.update(outcome="error_step2",
                   error=f"{type(exc).__name__}: {exc}"[:200],
                   elapsed_s=round(time.time() - t0, 2))
        return out
    cat_of_key = {}
    for cat, block in (res.action_scores or {}).items():
        for k in (block.get("scores") or {}):
            cat_of_key[k] = cat
    from benchmark_recommender_cases import _family_of
    actions = []
    for k, v in (res.prioritized_actions or {}).items():
        try:
            actions.append({
                "name": k, "family": _family_of(k, cat_of_key),
                "max_rho": round(float(v.max_rho), 4)
                if v.max_rho == v.max_rho else None,
                "max_rho_line": v.max_rho_line,
                "is_rho_reduction": bool(v.is_rho_reduction),
                "non_convergence": v.non_convergence,
            })
        except Exception:  # noqa: BLE001
            actions.append({"name": k, "family": _family_of(k, cat_of_key)})
    out["actions"] = actions
    out["n_actions"] = len(actions)
    best_pair = None
    for pk, pv in (res.combined_actions or {}).items():
        mr = pv.get("max_rho")
        if mr is not None and (best_pair is None or mr < best_pair["max_rho"]):
            best_pair = {"pair": pk, "max_rho": round(float(mr), 4)}
    out["best_pair"] = best_pair
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-dir", default="/workspace/out_benchmark/tht_cases")
    ap.add_argument("--out", default="/workspace/out_benchmark/cases_tht_france.jsonl")
    ap.add_argument("--per-case", type=int, default=12)
    ap.add_argument("--screen-out",
                    default="/workspace/out_benchmark/tht_screening.json")
    args = ap.parse_args()
    cases = sorted(Path(args.cases_dir).glob("*_v11.xiidm"))
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(x)["key"] for x in out.read_text().splitlines()
                if x.strip()}
    screening: dict = {}
    sp = Path(args.screen_out)
    if sp.exists():
        screening = json.loads(sp.read_text())
    with out.open("a") as f:
        for xiidm in cases:
            label = xiidm.stem.replace("_v11", "")
            if label not in screening:
                picked, n_total = screen_case(xiidm, args.per_case)
                screening[label] = {"n_constraining_contingencies": n_total,
                                    "picked": picked}
                sp.write_text(json.dumps(screening, indent=1,
                                         ensure_ascii=False))
                print(f"{label}: {n_total} contingences contraignantes, "
                      f"{len(picked)} retenues")
            for c in screening[label]["picked"]:
                key = f"{label}_{c['defaut']}"
                if key in done:
                    continue
                r = run_case(xiidm, label, c["defaut"])
                r["dc_screen"] = {"n_overloads": c["n_overloads"],
                                  "max_depth_dc": c["max_depth_dc"]}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f'  {key:60s} {r.get("outcome", "?"):12s} '
                      f'depth={r.get("max_depth")} res={r.get("resolution", "-")} '
                      f'{r.get("elapsed_s")}s')


if __name__ == "__main__":
    main()
