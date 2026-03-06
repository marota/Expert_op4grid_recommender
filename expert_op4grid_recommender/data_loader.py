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
import re
import logging
from expert_op4grid_recommender.utils.load_training_data import load_interesting_lines as exop_load_interesting_lines, DELETED_LINE_NAME as EXOP_DELETED_LINE_NAME

# To avoid circular dependency issues and keep data loading self-contained
DELETED_LINE_NAME = EXOP_DELETED_LINE_NAME

logger = logging.getLogger(__name__)

# Regex to extract line name from disco action descriptions
_DISCO_LINE_RE = re.compile(r"ligne\s+'([^']+)'")


def _build_disco_content(action_id: str, action_data: dict) -> dict:
    """Build content for a ``disco_*`` line disconnection action.

    Extracts the line name from either the action ID (``disco_<LINE>``) or
    from the ``description_unitaire`` field, and returns the trivial
    disconnection content.
    """
    line_name = None

    # Try extracting from action ID first (most reliable)
    if action_id.startswith("disco_"):
        line_name = action_id[len("disco_"):]

    # Fallback: extract from description
    if not line_name:
        desc = action_data.get("description_unitaire", action_data.get("description", ""))
        m = _DISCO_LINE_RE.search(desc)
        if m:
            line_name = m.group(1)

    if not line_name:
        return {"set_bus": {}}

    return {
        "set_bus": {
            "lines_or_id": {line_name: -1},
            "lines_ex_id": {line_name: -1},
            "loads_id": {},
            "generators_id": {},
        }
    }


class LazyActionDict(dict):
    """A dict subclass that computes 'content' lazily from 'switches' on first access.

    When action JSON files omit the pre-computed ``content`` field, this wrapper
    intercepts accesses to ``"content"`` and derives the ``set_bus`` payload on
    the fly using a shared :class:`NetworkTopologyCache`.

    For ``disco_*`` line disconnection actions (no switches), content is built
    from the action ID / description instead.
    """

    def __init__(self, data: dict, topology_cache=None, action_id: str = ""):
        super().__init__(data)
        self._topology_cache = topology_cache
        self._action_id = action_id
        self._content_computed = "content" in data

    def _compute_content(self):
        """Compute content.set_bus from switches using NetworkTopologyCache."""
        if self._content_computed:
            return
        self._content_computed = True

        switches = super().get("switches")

        # disco_* actions: no switches, derive content from action ID / description
        if not switches:
            content = _build_disco_content(self._action_id, dict(self))
            super().__setitem__("content", content)
            return

        if not self._topology_cache:
            super().__setitem__("content", {"set_bus": {}})
            return

        # Derive impacted voltage levels from the switch IDs
        impacted_vl_ids = set()
        for switch_id in switches:
            vl = self._topology_cache._switch_to_vl.get(switch_id)
            if vl:
                impacted_vl_ids.add(vl)

        if not impacted_vl_ids:
            logger.warning("No voltage levels found for switches %s", list(switches.keys()))
            super().__setitem__("content", {"set_bus": {}})
            return

        try:
            node_to_bus = self._topology_cache.compute_bus_assignments(switches, impacted_vl_ids)
            set_bus = self._topology_cache.get_element_bus_assignments(node_to_bus, impacted_vl_ids)
        except Exception:
            logger.exception("Failed to compute bus assignments for switches %s", list(switches.keys()))
            set_bus = {}

        content = {"set_bus": set_bus, "switches": switches}
        super().__setitem__("content", content)

    def __getitem__(self, key):
        if key == "content" and not self._content_computed:
            self._compute_content()
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "content" and not self._content_computed:
            self._compute_content()
        return super().get(key, default)

    def __contains__(self, key):
        if key == "content":
            return True  # content is always logically available
        return super().__contains__(key)


def enrich_actions_lazy(dict_actions: dict, n_grid) -> dict:
    """Wrap action dicts with :class:`LazyActionDict` so ``content`` is computed on demand.

    A single :class:`NetworkTopologyCache` is built from *n_grid* and shared
    across all action wrappers.

    Args:
        dict_actions: Action dictionary as returned by :func:`load_actions`.
        n_grid: A ``pypowsybl.network.Network`` object.

    Returns:
        dict: Same structure with :class:`LazyActionDict` values.
    """
    from expert_op4grid_recommender.utils.conversion_actions_repas import NetworkTopologyCache

    cache = NetworkTopologyCache(n_grid)

    result = {}
    for action_id, action_data in dict_actions.items():
        result[action_id] = LazyActionDict(action_data, topology_cache=cache, action_id=action_id)
    return result


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