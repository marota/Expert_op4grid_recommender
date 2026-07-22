#!/usr/bin/env python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
"""Minimal reproduction for the env-corruption bug documented in
docs/reviews/2026-07_env_corruption_overflow_graph.md.

Building the overflow graph (run_analysis_step2_graph) for a stressed contingency
runs a non-converging flow-change load flow that destroys the *base* variant's AC
voltage state (buses left NaN) below the pypowsybl variant layer. A subsequent
run_analysis_step1 on the SAME reused env then warm-starts from those NaN voltages,
diverges, and silently reports no overload — even though the contingency clearly
overloads a line on a fresh env.

Fixture: docs/reviews/hiver_pic_2021.xiidm (a reconstructed RTE7000 France THT
operating point). Run from the repo root, in the package's environment:

    python docs/reviews/repro_env_corruption_overflow_graph.py

Expected output (bug present):

    fresh probe(AVALLL61VNOL): CONTEXT (overload detected)   base NaN v_mag: 35
    ... grade AVALLL61J.VIL (step1 + step2_graph) ...
    post-graph probe(AVALLL61VNOL): no-context (BUG)         base NaN v_mag: 1701
    -> variant purge / re-clone base from InitialState / cold base LF do NOT recover;
       only a full env reload does.
"""
import os
import numpy as np

from expert_op4grid_recommender import config
# Default config already targets 95% (IGNORE_LINES_MONITORING=True,
# MONITORING_FACTOR_THERMAL_LIMITS=0.95); just silence visualization.
config.override_settings(DO_VISUALIZATION=False, IGNORE_LINES_MONITORING=True,
                         MONITORING_FACTOR_THERMAL_LIMITS=0.95,
                         ENABLE_ANTENNA_RECOMMENDATIONS=True)
from expert_op4grid_recommender.environment_pypowsybl import setup_environment_configs_pypowsybl
from expert_op4grid_recommender.pipeline import (run_analysis_step1, run_analysis_step2_graph,
    AnalysisContext, set_thermal_limits)
from expert_op4grid_recommender.backends import Backend

XIIDM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hiver_pic_2021.xiidm")

# A contingency that clearly overloads a monitored line (the "probe").
PROBE = "AVALLL61VNOL"
# A stressed contingency whose overflow-graph flow-change LF does not converge
# (disconnecting its overloads islands the grid) and corrupts the base variant.
GRADE = "AVALLL61J.VIL"


def main():
    config.override_settings(ENV_NAME=XIIDM)
    env, obs0, pc, cn, cl, da, lnr, lwa = setup_environment_configs_pypowsybl(
        analysis_date=None, env_folder=XIIDM, env_name="")
    # Match a normal CLI run: reload limits from operational_limits when huge.
    if float(np.mean(env.get_thermal_limit())) >= 1e4:
        set_thermal_limits(env.network_manager.network, env, thresold_thermal_limit=0.95)
    ctx0 = {'env': env, 'path_chronic': pc, 'chronic_name': cn, 'custom_layout': cl,
            'lines_non_reconnectable': lnr, 'lines_we_care_about': lwa}
    nm = env.network_manager
    net = nm.network

    def probe(cid):
        out = run_analysis_step1(analysis_date=None, current_timestep=0,
                                 current_lines_defaut=[cid], backend=Backend.PYPOWSYBL,
                                 dict_action={}, prebuilt_env_context=ctx0)
        return isinstance(out, AnalysisContext)

    def base_nan():
        nm.set_working_variant(nm.base_variant_id)
        return int(np.isnan(net.get_buses()["v_mag"].values).sum())

    ok_before = probe(PROBE)
    print(f"fresh probe({PROBE}): {'CONTEXT (overload detected)' if ok_before else 'no-context'}"
          f"   base NaN v_mag: {base_nan()}")

    # Build the overflow graph for the stressed contingency (this is what corrupts).
    out = run_analysis_step1(analysis_date=None, current_timestep=0,
                             current_lines_defaut=[GRADE], backend=Backend.PYPOWSYBL,
                             dict_action={}, prebuilt_env_context=ctx0)
    assert isinstance(out, AnalysisContext), f"{GRADE} should be actionable on a fresh env"
    run_analysis_step2_graph(out)

    ok_after = probe(PROBE)
    print(f"post-graph probe({PROBE}): {'CONTEXT' if ok_after else 'no-context (BUG)'}"
          f"        base NaN v_mag: {base_nan()}")

    if ok_before and not ok_after:
        print("\nBUG REPRODUCED: step2_graph corrupted the reused env; step1 now misses the "
              "overload it detected on a fresh env. See docs/reviews/"
              "2026-07_env_corruption_overflow_graph.md for the root cause and fix directions.")
        raise SystemExit(1)
    print("\nBug NOT reproduced on this build (already fixed?).")


if __name__ == "__main__":
    main()
