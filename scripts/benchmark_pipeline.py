#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Benchmark the pypowsybl analysis pipeline, phase by phase.

Instruments ``PypowsyblObservation.simulate`` (the pypowsybl load-flow call —
the dominant cost) and attributes every call to the pipeline phase that made
it: **step1** (contingency + overload detection), **graph** (overflow graph),
**discovery** (candidate identification + per-candidate effectiveness check),
and **reassessment** (the final per-action re-simulation). Prints, per phase,
the number of load flows and the wall time — so you can see where the time
goes, in particular how much the end-of-run action reassessment costs.

Usage (defaults mirror Co-Study4Grid's ``config.default.json``)::

    python scripts/benchmark_pipeline.py \
        --network  data/pypsa_eur_eur220_225_380_400/network.xiidm \
        --actions  data/pypsa_eur_eur220_225_380_400/actions.json \
        --defaut   LANNEL61PRAGN \
        --n-prioritized 15

Run it inside the environment where the analysis already completes (e.g. the
Co-Study4Grid checkout with the pypsa-eur data), so the pipeline reaches the
reassessment phase.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", required=True, help="Path to the .xiidm network file")
    ap.add_argument("--actions", required=True, help="Path to the actions JSON file")
    ap.add_argument("--defaut", nargs="+", required=True, help="Contingency line name(s)")
    ap.add_argument("--n-prioritized", type=int, default=15)
    ap.add_argument("--min-load-shedding", type=int, default=2)
    ap.add_argument("--min-redispatch", type=int, default=2)
    ap.add_argument("--min-line-disconnections", type=int, default=3)
    ap.add_argument("--ignore-lines-monitoring", action="store_true", default=True)
    ap.add_argument("--no-fast-mode", action="store_true")
    args = ap.parse_args()

    net = Path(args.network).resolve()
    from expert_op4grid_recommender import config
    config.ENV_NAME = net.name
    config.ENV_FOLDER = net.parent
    config.ENV_PATH = net
    config.ACTION_FILE_PATH = Path(args.actions).resolve()
    config.DO_VISUALIZATION = False
    config.CHECK_ACTION_SIMULATION = True
    config.N_PRIORITIZED_ACTIONS = args.n_prioritized
    config.MIN_LOAD_SHEDDING = args.min_load_shedding
    config.MIN_REDISPATCH = args.min_redispatch
    config.MIN_LINE_DISCONNECTIONS = args.min_line_disconnections
    config.IGNORE_LINES_MONITORING = args.ignore_lines_monitoring
    config.LINES_MONITORING_FILE = None
    config.PYPOWSYBL_FAST_MODE = not args.no_fast_mode

    # --- instrument simulate(): count + time per phase -----------------------
    import expert_op4grid_recommender.pypowsybl_backend.observation as obsmod
    phase = {"name": "setup"}
    stats: dict[str, list] = {}
    orig_sim = obsmod.PypowsyblObservation.simulate

    def timed_sim(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        t = time.perf_counter()
        try:
            return orig_sim(self, *a, **k)
        finally:
            s = stats.setdefault(phase["name"], [0, 0.0])
            s[0] += 1
            s[1] += time.perf_counter() - t

    obsmod.PypowsyblObservation.simulate = timed_sim

    import expert_op4grid_recommender.utils.reassessment as reass
    orig_reass = reass.reassess_prioritized_actions

    def timed_reass(*a, **k):  # noqa: ANN002, ANN003
        phase["name"] = "reassessment"
        t = time.perf_counter()
        try:
            return orig_reass(*a, **k)
        finally:
            print(f"[bench] reassessment wall: {time.perf_counter() - t:.2f}s")

    reass.reassess_prioritized_actions = timed_reass

    from expert_op4grid_recommender.main import (
        AnalysisResult,
        Backend,
        run_analysis_step1,
        run_analysis_step2_graph,
        run_analysis_step2_discovery,
    )

    phase_times: dict[str, float] = {}
    phase["name"] = "step1"
    t = time.perf_counter()
    # run_analysis_step1 returns an AnalysisContext (proceed) or an
    # AnalysisResult (no-overload short-circuit) — no longer a 2-tuple.
    outcome = run_analysis_step1(None, 0, list(args.defaut), backend=Backend.PYPOWSYBL)
    phase_times["step1"] = time.perf_counter() - t
    if isinstance(outcome, AnalysisResult):
        print("Early exit at step1 (no actionable overflow graph). "
              "Overloaded:", outcome.get("lines_overloaded_names"))
        return
    ctx = outcome

    phase["name"] = "graph"
    t = time.perf_counter()
    ctx = run_analysis_step2_graph(ctx)
    phase_times["graph"] = time.perf_counter() - t

    phase["name"] = "discovery"
    t = time.perf_counter()
    res = run_analysis_step2_discovery(ctx)
    phase_times["discovery+reassessment"] = time.perf_counter() - t

    print("\n==================== BENCHMARK ====================")
    print("Contingency:", args.defaut)
    print("Overloaded lines:", res.get("lines_overloaded_names"))
    print("Prioritized actions:", len(res.get("prioritized_actions", {})))
    print("\n-- phase wall time --")
    for k, v in phase_times.items():
        print(f"  {k:28s} {v:8.2f}s")
    print("\n-- load flows (simulate calls) & time by phase --")
    tot_c = 0
    tot_t = 0.0
    for k, (c, dt) in sorted(stats.items(), key=lambda x: -x[1][1]):
        print(f"  {k:16s} calls={c:5d}  time={dt:8.2f}s")
        tot_c += c
        tot_t += dt
    print(f"  {'TOTAL':16s} calls={tot_c:5d}  time={tot_t:8.2f}s")


if __name__ == "__main__":
    main()
