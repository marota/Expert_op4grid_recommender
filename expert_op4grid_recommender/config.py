# expert_op4grid_recommender/config.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

from datetime import datetime
import os

# -------------------
#  Case Configuration
# -------------------
# Define the specific scenario to analyze
DATE = datetime(2024, 9, 19)
TIMESTEP = 1
LINES_DEFAUT = ["CPVANL61ZMAGN"]
CASE_NAME = "defaut_" + "_".join(map(str, LINES_DEFAUT)) + "_t" + str(TIMESTEP)

# -------------------
#  Environment & Paths
# -------------------
ENV_FOLDER = "../data"
ENV_NAME = "env_dijon_v2_assistant"
ENV_PATH = os.path.join(ENV_FOLDER, ENV_NAME)
ACTION_SPACE_FOLDER = os.path.join(ENV_FOLDER,"action_space")
FILE_ACTION_SPACE_DESC = "reduced_model_actions.json"
ACTION_FILE_PATH = os.path.join(ACTION_SPACE_FOLDER, FILE_ACTION_SPACE_DESC)
SAVE_FOLDER_VISUALIZATION = "../Overflow_Graph" # Directory to save graph visualizations

# -------------------
#  User Parameters
# -------------------
USE_EVALUATION_CONFIG = True
USE_DC_LOAD_FLOW = False
DO_CONSOLIDATE_GRAPH = False
DO_RECO_MAINTENANCE = False
CHECK_WITH_ACTION_DESCRIPTION = True
DRAW_ONLY_SIGNIFICANT_EDGES = True
USE_GRID_LAYOUT = False
DO_FORCE_OVERLOAD_GRAPH_EVEN_IF_GRAPH_BROKEN_APART = False
DO_SAVE_DATA_FOR_TEST = False
CHECK_ACTION_SIMULATION = True
N_PRIORITIZED_ACTIONS = 5

# -------------------
#  Expert System Parameters
# -------------------
PARAM_OPTIONS_EXPERT_OP = {
    # 0.05 is 5 percent of the max overload flow
    "ThresholdReportOfLine": 0.05,
    # 10 percent de la surcharge max
    "ThersholdMinPowerOfLoop": 0.1,
    # If at least a loop is detected, only keep the ones with a flow of at least 25 percent the biggest one
    "ratioToKeepLoop": 0.25,
    # Ratio percentage for reconsidering the flow direction
    "ratioToReconsiderFlowDirection": 0.75,
    # max unused lines
    "maxUnusedLines": 3,
    # number of simulated topologies node at the final simulation step
    "totalnumberofsimulatedtopos": 30,
    # number of simulated topologies per node at the final simulation step
    "numberofsimulatedtopospernode": 10
}