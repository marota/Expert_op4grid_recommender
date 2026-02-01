# expert_op4grid_recommender/data_loader.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

import os
import json
from expert_op4grid_recommender.utils.load_training_data import load_interesting_lines as exop_load_interesting_lines, DELETED_LINE_NAME as EXOP_DELETED_LINE_NAME

# To avoid circular dependency issues and keep data loading self-contained
DELETED_LINE_NAME = EXOP_DELETED_LINE_NAME

def load_actions(file_path):
    """
    Loads action definitions from a specified JSON file.

    This function reads a JSON file expected to contain a dictionary
    representing the action space or a set of predefined actions.

    Args:
        file_path (str): The full path to the JSON file containing the actions.

    Returns:
        dict: The dictionary loaded from the JSON file.

    Raises:
        FileNotFoundError: If the file specified by `file_path` does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The action file {file_path} does not exist.")
    with open(file_path, 'r') as file:
        dict_actions = json.load(file)
        for action_id,action in dict_actions.items():
            if "switches" in action and "content" in action:
                action["content"]["switches"]=action["switches"]#make switches directly available in action content as well
        return dict_actions

def load_interesting_lines(path=None, file_name=None):
    """
   Loads a list of line names from a file, typically used for monitoring or exclusion lists.

   This function acts as a wrapper around the `load_interesting_lines` function
   provided by the `expertop4grid` library (or its local copy within the utils).
   It reads a file (usually CSV) specified by either `path` and `file_name`
   or just `file_name` (if it's a full path or in the current directory)
   and returns a list of line names contained within.

   Args:
       path (str, optional): The directory path containing the file. Defaults to None.
       file_name (str, optional): The name of the file (e.g., "non_reconnectable_lines.csv").
                                   Can also be a full path if `path` is None. Defaults to None.

   Returns:
       list[str]: A list of line names read from the specified file.
    """
    return exop_load_interesting_lines(path=path, file_name=file_name)