#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Command-line entry point for the expert-system analysis.

Split out of the historical ``main.py`` so the pipeline no longer doubles as
the CLI: the dependency direction is ``cli → pipeline → models →
action_evaluation/graph_analysis → utils``. Because this is the top of the
stack (nothing imports it), the ``print`` statements here are appropriate —
this is the user-facing command, not library code.
"""
import os
import sys
import argparse

import pypowsybl as pp

from expert_op4grid_recommender import config
from expert_op4grid_recommender.config import DATE as DEFAULT_DATE, TIMESTEP as DEFAULT_TIMESTEP, \
    LINES_DEFAUT as DEFAULT_LINES_DEFAUT
from expert_op4grid_recommender.backends import Backend
from expert_op4grid_recommender.pipeline import run_analysis
from expert_op4grid_recommender.utils.helpers import Timer
from expert_op4grid_recommender.utils.action_rebuilder import run_rebuild_actions
from expert_op4grid_recommender.data_loader import load_actions


def main():
    """Main function to run the expert system analysis from the command line."""
    default_date_str = DEFAULT_DATE.strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="Run ExpertOp4Grid analysis for a specific contingency.")
    parser.add_argument("--date", default=default_date_str,
        help=f"Date for the chronic in YYYY-MM-DD format (default: {default_date_str}). Pass 'None' to use the bare environment without a specific date.")
    parser.add_argument("--timestep", type=int, default=DEFAULT_TIMESTEP,
        help=f"Timestep index within the chronic (default: {DEFAULT_TIMESTEP})")
    parser.add_argument("--lines-defaut", nargs='+', default=DEFAULT_LINES_DEFAUT,
        help=f"One or more line names for the N-1 contingency (default: {' '.join(DEFAULT_LINES_DEFAUT)})")
    parser.add_argument("--backend", choices=["grid2op", "pypowsybl"], default="grid2op",
        help="Simulation backend to use (default: grid2op)")
    parser.add_argument("--rebuild-actions", action='store_true',
        help="If set, rebuilds the action dictionary from REPAS files based on the current grid snapshot before analysis. Stops analysis after rebuilding.")
    parser.add_argument("--repas-file",
        default=os.path.join("data", "action_space", "allLogics.2024.12.10.json"),
        help="Path to the REPAS actions file (default: data/action_space/allLogics.2024.12.10.json)")
    parser.add_argument("--grid-snapshot-file",
        default=os.path.join("data", "snapshot", "pf_20240828T0100Z_20240828T0100Z.xiidm"),
        help="Path to the snapshot grid file in detailed topology format with switches, to rebuild action dictionary on")
    parser.add_argument("--voltage-threshold", type=float, default=300.0,
        help="Voltage filter threshold for REPAS actions (default: 300)")
    parser.add_argument("--pypowsybl-format", action='store_true',
        help="When used with --rebuild-actions and an empty action file (from scratch), outputs the action dictionary in pypowsybl format.")
    parser.add_argument("--ignore-lines-monitoring", action='store_true',
        help="If set, ignores the lignes_a_monitorer.csv file and monitors all lines.")
    parser.add_argument("--fast-mode", action='store_true',
        help="If set, uses pypowsybl fast mode (no voltage control) for grid simulations.")
    args = parser.parse_args()

    if args.ignore_lines_monitoring:
        config.IGNORE_LINES_MONITORING = True

    sum_min_actions = (config.MIN_LINE_RECONNECTIONS +
                       config.MIN_CLOSE_COUPLING +
                       config.MIN_OPEN_COUPLING +
                       config.MIN_LINE_DISCONNECTIONS)

    if sum_min_actions > config.N_PRIORITIZED_ACTIONS:
        print(f"Warning: The sum of minimum actions per type ({sum_min_actions}) exceeds the "
              f"maximum number of prioritized actions overall ({config.N_PRIORITIZED_ACTIONS}). "
              f"Some minimums will not be respected.", file=sys.stderr)

    date_arg = args.date
    if date_arg == "None":
        date_arg = None

    backend = Backend.GRID2OP if args.backend == "grid2op" else Backend.PYPOWSYBL

    try:
        with Timer("Total Execution"):
            if args.rebuild_actions:
                grid_snapshot_file_path = args.grid_snapshot_file
                n_grid = pp.network.load(grid_snapshot_file_path)

                dict_action = {}
                do_from_scratch = False
                if config.ACTION_FILE_PATH is not None and os.path.exists(config.ACTION_FILE_PATH):
                    dict_action = load_actions(config.ACTION_FILE_PATH)
                else:
                    do_from_scratch = True

                try:
                    dict_action = run_rebuild_actions(n_grid, do_from_scratch, args.repas_file,
                                                       dict_action_to_filter_on=dict_action,
                                                       voltage_filter_threshold=args.voltage_threshold,
                                                       output_file_base_name="reduced_model_actions",
                                                       pypowsybl_format=args.pypowsybl_format)
                except Exception as exc:  # print acceptable in the CLI entry point
                    # run_rebuild_actions now RE-RAISES on failure (M6); surface
                    # it and stop instead of printing "complete" over a failure.
                    print(f"Action rebuilding failed: {exc}", file=sys.stderr)
                    sys.exit(1)

                print("Action rebuilding process complete. Stopping analysis as requested.")
                return
            else:
                run_analysis(
                    analysis_date=date_arg,
                    current_timestep=args.timestep,
                    current_lines_defaut=args.lines_defaut,
                    backend=backend,
                    fast_mode=args.fast_mode
                )
    except (ValueError, RuntimeError, TypeError) as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
