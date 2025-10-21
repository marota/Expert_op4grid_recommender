#!/usr/bin/python3
# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

# --- CRUCIAL FIX: Add Project Root to Python Path ---
# This code block makes the test file self-sufficient by telling Python where to find your package.
# It solves the import errors that cause the "assert False is False" issue.
import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#from make_assistant_env import make_grid2op_assistant_env
#from make_training_env import make_grid2op_training_env
#from load_evaluation_data import list_all_chronics, get_first_obs_on_chronic
from datetime import datetime

import numpy as np
from pathlib import Path
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph,Structured_Overload_Distribution_Graph

import json
# --- Imports from the refactored project ---
from expert_op4grid_recommender.action_evaluation.classifier import *
from expert_op4grid_recommender.action_evaluation.discovery import *
from expert_op4grid_recommender.action_evaluation.rules import *
from expert_op4grid_recommender.graph_analysis.processor import *
from expert_op4grid_recommender.utils.simulation import *
from expert_op4grid_recommender.utils.helpers import *
from expert_op4grid_recommender.environment import make_grid2op_training_env, make_grid2op_assistant_env, \
    get_first_obs_on_chronic
from expert_op4grid_recommender.utils.load_evaluation_data import list_all_chronics,load_interesting_lines
from expert_op4grid_recommender.graph_analysis.builder import build_overflow_graph
from expert_op4grid_recommender.action_evaluation.discovery import (_get_line_substations, _find_paths_for_line,
                                                                    _get_active_edges_between, _has_blocking_disconnected_line,
                                                                    _is_sublist)

#from Expert_rule_action_verification import is_line_reconnection,is_load_disconnection,is_line_disconnection,is_nodale_grid2op_action,find_relevant_disconnections, verify_relevant_reconnections, get_maintenance_timestep
#from Expert_rule_action_verification import find_relevant_node_merging,find_relevant_node_splitting, compute_node_splitting_action_score, identify_and_score_node_splitting_actions,check_other_reconnectable_line_on_path
#from Expert_rule_action_verification import build_overflow_graph, categorize_action_space, identify_action_type, check_rules, identify_grid2op_action_type,check_simu_overloads, load_interesting_lines,get_n_connected_components_graph_with_overloads, identify_overload_lines_to_keep_overflow_graph_connected, get_subs_islanded_by_overload_disconnections, get_maintenance_timestep
#from Expert_rule_action_verification import is_sublist,_get_line_substations,_find_paths_for_line,_get_active_edges_between,_has_blocking_disconnected_line,check_other_reconnectable_line_on_path
from packaging.version import Version as version_packaging
from importlib.metadata import version

#: name of the powerline that are now removed (non existant)
#: but are in the environment because we need them when we will use
#: historical dataset
#Note: reimporté directement ici pour éviter les problèmes de dépendances, en effet DELETED_LINE_NAME changeait de valeur après l'éxécution de categorize_action_space dans test_overflow_graph_actions_filtered
DELETED_LINE_NAME = ['BXNE L32BXNE5', 'BXNE5L32MTAGN', 'BXNE5L32CORGO', 'BXNE5L31MTAGN']

EXOP_MIN_VERSION = version_packaging("0.2.6")
if version_packaging(version("expertop4grid")) < EXOP_MIN_VERSION:
    raise RuntimeError(f"Incompatible version found for expertOp4Grid, make sureit is >= {EXOP_MIN_VERSION}")

##########################
## UNITARY TESTS
class MockAction:
    """
    Simple mock of a Grid2Op action object.

    Supports:
    - Line attributes (used in is_line_reconnection / is_line_disconnection)
    - Load attributes (used in is_load_disconnection)
    - Topology vectors (used in is_nodale_grid2op_action)

    Any attribute not relevant for a test defaults to "no action".
    """

    def __init__(self,
                 # Line attributes
                 line_or_change=None,
                 line_ex_change=None,
                 line_change_status=None,
                 line_or_set=None,
                 line_ex_set=None,
                 line_set_status=None,
                 # Load attributes
                 load_change_bus=None,
                 load_set_bus=None,
                 # Nodal attributes
                 set_topo_vect=None,
                 topo_vect_to_sub=None):
        # Line attributes
        self.line_or_change_bus = np.array(line_or_change) if line_or_change is not None else np.array([0])
        self.line_ex_change_bus = np.array(line_ex_change) if line_ex_change is not None else np.array([0])
        self.line_change_status = np.array(line_change_status) if line_change_status is not None else np.array([0])
        self.line_or_set_bus = np.array(line_or_set) if line_or_set is not None else np.array([0])
        self.line_ex_set_bus = np.array(line_ex_set) if line_ex_set is not None else np.array([0])
        self.line_set_status = np.array(line_set_status) if line_set_status is not None else np.array([0])

        # Load attributes
        self.load_change_bus = np.array(load_change_bus) if load_change_bus is not None else np.array([0])
        self.load_set_bus = np.array(load_set_bus) if load_set_bus is not None else np.array([0])

        # Nodal attributes
        self._set_topo_vect = np.array(set_topo_vect) if set_topo_vect is not None else np.array([0])
        self._topo_vect_to_sub = np.array(topo_vect_to_sub) if topo_vect_to_sub is not None else np.array([0])

        def __add__(self, other):
            """
            Allow action concatenation (e.g., act_deco_overloads + act_reco_maintenance).
            This just bundles them in a list for mocks.
            """
            if isinstance(other, MockAction):
                return [self, other]
            elif isinstance(other, list):
                return [self] + other
            return NotImplemented


def test_no_reconnection():
    """Should return False when no reconnection happens."""
    action = MockAction(
        line_or_change=[0], line_ex_change=[0], line_change_status=[0],
        line_or_set=[0], line_ex_set=[0], line_set_status=[0]
    )
    assert not is_line_reconnection(action) #is False


def test_reconnection_with_status():
    """Should return True when line_set_status explicitly reconnects a line."""
    action = MockAction(
        line_or_change=[0], line_ex_change=[0], line_change_status=[0],
        line_or_set=[0], line_ex_set=[0], line_set_status=[1]
    )
    assert is_line_reconnection(action) #is True


def test_reconnection_with_both_buses():
    """Should return True when both buses are set (origin and extremity)."""
    action = MockAction(
        line_or_change=[0], line_ex_change=[0], line_change_status=[0],
        line_or_set=[1], line_ex_set=[1], line_set_status=[0]
    )
    assert is_line_reconnection(action) #is True


def test_warning_on_unsupported_actions(capfd):
    """Should print a warning if unsupported action types are used."""
    action = MockAction(
        line_or_change=[1], line_ex_change=[0], line_change_status=[0],
        line_or_set=[0], line_ex_set=[0], line_set_status=[0]
    )
    result = is_line_reconnection(action)
    out, _ = capfd.readouterr()
    assert "WARNING" in out
    assert not result #is False


def test_no_load_disconnection():
    """Should return False when no load is disconnected."""
    action = MockAction(load_change_bus=[0], load_set_bus=[0])
    assert not is_load_disconnection(action) #is False


def test_load_disconnection_with_set_bus():
    """Should return True when a load is explicitly disconnected (set to -1)."""
    action = MockAction(load_change_bus=[0], load_set_bus=[-1])
    assert is_load_disconnection(action) #is True


def test_warning_on_unsupported_actions_load_disconnection(capfd):
    """Should print a warning if load_change_bus is used."""
    action = MockAction(load_change_bus=[1], load_set_bus=[0])
    result = is_load_disconnection(action)
    out, _ = capfd.readouterr()
    assert "WARNING" in out
    assert result==False


def test_multiple_loads_mixed():
    """Should return True if at least one load is disconnected among several."""
    action = MockAction(load_change_bus=[0, 0, 0], load_set_bus=[0, -1, 1])
    assert is_load_disconnection(action)==True

def test_no_disconnection():
    """Should return False when no line is disconnected."""
    action = MockAction(
        line_or_change=[0], line_ex_change=[0], line_change_status=[0],
        line_or_set=[0], line_ex_set=[0], line_set_status=[0]
    )
    assert is_line_disconnection(action)==False


def test_line_disconnection_with_status():
    """Should return True when line_set_status is explicitly -1."""
    action = MockAction(line_set_status=[-1])
    assert is_line_disconnection(action)==True


def test_line_disconnection_with_origin_bus():
    """Should return True when origin bus is explicitly set to -1."""
    action = MockAction(line_or_set=[-1], line_ex_set=[1])
    assert is_line_disconnection(action)==True


def test_line_disconnection_with_extremity_bus():
    """Should return True when extremity bus is explicitly set to -1."""
    action = MockAction(line_or_set=[1], line_ex_set=[-1])
    assert is_line_disconnection(action)==True


def test_multiple_lines_disconnections_mixed():
    """Should return True if at least one line among many is disconnected."""
    action = MockAction(
        line_or_set=[1, 0, -1],
        line_ex_set=[1, 0, 1],
        line_set_status=[0, 0, 0]
    )
    assert is_line_disconnection(action)==True


def test_warning_on_unsupported_actions_line_disconnection(capfd):
    """Should print a warning if unsupported action types are used."""
    action = MockAction(line_or_change=[1])
    result = is_line_disconnection(action)
    out, _ = capfd.readouterr()
    assert "WARNING" in out
    assert result==False

def test_no_nodale_action():
    """Should return False if no substation has multiple set elements."""
    act = MockAction(
        set_topo_vect=[0, 1, 0],
        topo_vect_to_sub=[0, 1, 2]  # each element belongs to a different substation
    )
    result, subs, splits = is_nodale_grid2op_action(act)
    assert result==False
    assert subs == []
    assert splits == []


def test_single_substation_with_two_elements():
    """Should return True if two elements in the same substation are set."""
    act = MockAction(
        set_topo_vect=[1, 1, 0],
        topo_vect_to_sub=[0, 0, 1]  # first two elements in same substation
    )
    result, subs, splits = is_nodale_grid2op_action(act)
    assert result==True
    assert subs == [0]  # substation 0 is concerned
    assert splits == [False]  # all set to the same bus → not splitting


def test_substation_with_splitting():
    """Should detect a substation where elements are set to different buses."""
    act = MockAction(
        set_topo_vect=[1, 2, 0],
        topo_vect_to_sub=[0, 0, 1]  # substation 0 has two elements set
    )
    result, subs, splits = is_nodale_grid2op_action(act)
    assert result==True
    assert subs == [0]
    assert splits == [True]  # bus 1 and bus 2 → splitting


def test_multiple_substations_mixed():
    """Should detect multiple substations, with mixed splitting behavior."""
    act = MockAction(
        set_topo_vect=[1, 1, 2, 1],
        topo_vect_to_sub=[0, 0, 1, 1]  # substations 0 and 1 both have multiple sets
    )
    result, subs, splits = is_nodale_grid2op_action(act)
    assert result==True
    assert set(subs) == {0, 1}
    # substation 0: all bus 1 → not splitting
    # substation 1: bus 1 and bus 2 → splitting
    assert splits == [False, True]

class MockLoadP:
    """Fake load power vector with .sum() method."""
    def __init__(self, values):
        self.values = np.array(values)

    def sum(self):
        return self.values.sum()

class MockObservation:

    def __init__(self, name_sub, sub_topologies=None, sub_info=None, topo_vect=None,name_line=None,load_values=None, simulate_return = None,line_or_to_subid=None,line_ex_to_subid=None):
        self.name_sub = name_sub
        self.sub_topologies = sub_topologies or {}
        self.sub_info = sub_info
        if sub_info is None:
            self.sub_info = np.array([1] * len(name_sub))
        self.topo_vect = topo_vect
        if topo_vect is None:
            np.array([])
        self.load_p = MockLoadP(load_values)
        self._simulate_return = simulate_return
        self.name_line = name_line
        if name_line is None:
            self.name_line=np.array(["L0", "L1", "L2"])
        self.line_or_to_subid = line_or_to_subid  # A, B, C
        self.line_ex_to_subid = line_ex_to_subid

    def simulate(self, action, time_step):
        return self._simulate_return

    def sub_topology(self, sub_id):
        return self.sub_topologies[sub_id]

    # Allow adding an action
    def __add__(self, action):
        # return new observation with updated topo_vect
        new_topo = self.topo_vect.copy()
        for sub_id, topo in action.substations_id:
            start = int(np.sum(self.sub_info[:sub_id]))
            length = int(self.sub_info[sub_id])
            new_topo[start:start+length] = topo
        return MockObservation(self.name_sub, self.sub_topologies, self.sub_info, new_topo)

# --- Mock Action returned by action_space ---
class MockActionObject:
    """Represents a Grid2Op action for testing."""
    def __init__(self, substations_id=None, lines_ex_id=None, lines_or_id=None, lines_status=None):
        self.substations_id = substations_id or []
        self.lines_ex_id = lines_ex_id or {}
        self.lines_or_id = lines_or_id or {}
        self.lines_status= lines_status or {}
        self.action_id = str(substations_id)+str(lines_or_id)

    def get_topological_impact(self):
        # Return dummy lists, just to satisfy the interface
        lines_impacted = list(self.lines_ex_id.keys()) + list(self.lines_or_id.keys())
        sub_id=self.substations_id[0][0]
        subs_impacted = np.array([False for i in range(sub_id+1)])  # arbitrary array
        subs_impacted[sub_id]=True
        return lines_impacted, subs_impacted

    def __add__(self, other):
        """
        Allow action concatenation (e.g., act_deco_overloads + act_reco_maintenance).
        This just bundles them in a list for mocks.
        """
        return self
#

class MockActionSpace:
    """Returns a MockActionObject when called."""
    def __call__(self, action_dict):
        if "set_bus" in action_dict:
            set_bus = action_dict["set_bus"]
            if "substations_id" in set_bus:
                return MockActionObject(substations_id=set_bus["substations_id"])
            else:
                # Handles lines_ex_id / lines_or_id for act_defaut
                return MockActionObject(lines_ex_id=set_bus.get("lines_ex_id"),
                                    lines_or_id=set_bus.get("lines_or_id"))
        elif "set_line_status" in action_dict:
            set_line_status=action_dict["set_line_status"]
            return MockActionObject(lines_status=set_line_status)


def mock_action_space(content):
    act = MockAction()
    act.action_id = list(content["set_bus"]["lines_ex_id"].keys())[0]  # use line id as action_id
    return act

# Mock AlphaDeesp_warmStart
class MockAlphaDeesp:
    def rank_current_topo_at_node_x(self, g, sub_id, isSingleNode, topo_vect, is_score_specific_substation):
        # Simple mock: return score 1 for sub_id 0, 0.5 for others
        return 1.0 if sub_id == 0 else 0.5

# Mock identify_action_type
def mock_identify_action_type(action_desc, by_description, grid2op_action_space):
    return ["open_coupling"]  # treat all as open_coupling for testing

def test_no_islanded_subs():
    """If all nodes remain in the main component, no substation is islanded."""
    obs = MockObservation(name_sub=np.array(["Sub0", "Sub1", "Sub2"]))
    comps_init = [set([0, 1, 2])]
    comp_overloads = [set([0, 1, 2])]  # no change
    result = get_subs_islanded_by_overload_disconnections(obs, comps_init, comp_overloads, "LineA")
    assert result == []


def test_single_islanded_sub():
    """A single substation is disconnected from the main component."""
    obs = MockObservation(name_sub=np.array(["Sub0", "Sub1", "Sub2"]))
    comps_init = [set([0, 1, 2])]
    comp_overloads = [set([0, 1])]  # Sub2 disconnected
    result = get_subs_islanded_by_overload_disconnections(obs, comps_init, comp_overloads, "LineB")
    assert result == ["Sub2"]


def test_multiple_islanded_subs():
    """Multiple substations disconnected after overload."""
    obs = MockObservation(name_sub=np.array(["Sub0", "Sub1", "Sub2", "Sub3"]))
    comps_init = [set([0, 1, 2, 3])]
    comp_overloads = [set([0, 1])]  # Sub2 and Sub3 disconnected
    result = get_subs_islanded_by_overload_disconnections(obs, comps_init, comp_overloads, "LineC")
    assert set(result) == {"Sub2", "Sub3"}


def test_non_identified_nodes_printed(capfd):
    """Nodes outside the substation range are printed as non-identified."""
    obs = MockObservation(name_sub=np.array(["Sub0", "Sub1"]))
    comps_init = [set([0, 1, 2])]  # node 2 has no corresponding substation
    comp_overloads = [set([0, 1])]  # node 2 disconnected
    result = get_subs_islanded_by_overload_disconnections(obs, comps_init, comp_overloads, "LineD")
    assert result == []  # no valid substation identified
    out, _ = capfd.readouterr()
    assert "non-identified" in out

def mock_check_rho_reduction(obs, timestep, act_defaut, action, overload_ids, act_reco_maintenance,lines_we_care_about=None, rho_tolerance=0.02):
    """
    Simulate rho reduction:
    - sub_id 0 → effective
    - sub_id 3 → ineffective
    """
    #sub_id = action["set_bus"]["substations_id"][0][0]
    if len(action.lines_ex_id)!=0 and "L1" in action.lines_ex_id:
    #if hasattr(action, "lines_ex_id") and "L1" in action.lines_ex_id:
        return True, obs
    elif len(action.lines_status) != 0 and "L1" in action.lines_status[0][0]:
    #elif hasattr(action, "lines_status") and "L1" in action.lines_status[0][0]:
        return True, obs
    elif len(action.substations_id) != 0 and action.substations_id[0][0] == 0:
    #elif hasattr(action, "sub_id") and action.substations_id[0][0] == 0:
        return True, obs
    else:
        return False, obs

@pytest.fixture
def discovery_mocks(monkeypatch):
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.check_rho_reduction",
                        mock_rho_reduction_discovery)
    monkeypatch.setattr("expert_op4grid_recommender.utils.simulation.create_default_action",
                        lambda *args: MockActionObject())
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.AlphaDeesp_warmStart",
                        lambda *args: MockAlphaDeesp())

    def mock_id(desc, **kwargs): return desc.get("type", "unknown")

    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.identify_action_type", mock_id)

    def mock_sort(map_score):
        items = sorted(map_score.items(), key=lambda item: item[1]['score'], reverse=True)
        return {k: v['action'] for k, v in items}, [v.get('sub_impacted') or v.get('line_impacted') for k, v in
                                                    items], [v['score'] for k, v in items]

    monkeypatch.setattr("expert_op4grid_recommender.utils.helpers.sort_actions_by_score", mock_sort)

def test_find_relevant_node_merging(monkeypatch):
    # Setup observation with four substations
    obs = MockObservation(
        name_sub=np.array(["Sub0", "Sub1", "Sub2", "Sub3"]),
        sub_topologies={
            0: [1, 2, 2],    # Sub0: eligible, effective
            1: [1, 1],       # Sub1: ignored
            2: [0, 2, -1],   # Sub2: ignored (only 1 eligible node)
            3: [2, 1, 2]     # Sub3: eligible, ineffective
        }
    )

    nodes_dispatch_path = ["Sub0", "Sub1", "Sub2", "Sub3"]
    timestep = 0
    defauts = []
    overload_ids = []
    act_reco_maintenance = []

    action_space = MockActionSpace()
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.check_rho_reduction", mock_check_rho_reduction)

    identified, effective, ineffective = find_relevant_node_merging(
        nodes_dispatch_path, obs, timestep, action_space, defauts, overload_ids, act_reco_maintenance
    )

    # Expectations
    # Identified actions → only Sub0 and Sub3 have >=2 eligible nodes
    assert len(identified) == 2
    identified_sub_ids = [act.substations_id[0][0] for act in identified.values()]
    assert set(identified_sub_ids) == {0, 3}

    # Effective actions → only Sub0
    assert len(effective) == 1
    assert effective[0].substations_id[0][0] == 0

    # Ineffective actions → only Sub3
    assert len(ineffective) == 1
    assert ineffective[0].substations_id[0][0] == 3

    # Optional: verify target bus assignments
    for act in identified.values():
        topo_target = act.substations_id[0][1]
        # All nodes on bus >=2 should be merged to bus 1
        assert all(bus == 1 or bus <= 1 for bus in topo_target)

def test_identify_and_score_node_splitting_actions(monkeypatch):
    # Actions to test
    actions_unfiltered = {0: "a0", 1: "a1"}
    dict_action = {
        0: {"content": {"set_bus": {"substations_id": [(0, [1, 2])]}}},
        1: {"content": {"set_bus": {"substations_id": [(1, [1, 2])]}}}
    }

    hubs = ["Sub0"]
    nodes_blue_path = ["Sub1"]

    obs_defaut = MockObservation(
        name_sub=np.array(["Sub0", "Sub1"]),
        sub_topologies={0: [1, 2], 1: [1, 1]},
        sub_info=np.array([2, 2]),
        topo_vect=np.array([1, 2, 1, 1])
    )

    # Patch identify_action_type to always return open_coupling
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.identify_action_type", lambda desc, **kwargs: ["open_coupling"])

    map_score, ignored = identify_and_score_node_splitting_actions(
        dict_action, hubs, nodes_blue_path,
        obs_defaut, MockActionSpace(), MockAlphaDeesp(), g=None
    )

    # Check results
    assert 0 in map_score
    assert 1 in map_score
    assert map_score[0]["score"] == 1.0  # sub_id 0
    assert map_score[1]["score"] == 0.5  # sub_id 1
    assert ignored == []  # both actions are in hubs or blue path

def test_compute_node_splitting_action_score_global():
    # Create a mock observation
    obs = MockObservation(
        name_sub=np.array(["Sub0", "Sub1"]),
        sub_topologies={0: [1, 2], 1: [1, 1]},
        sub_info=np.array([2, 2]),
        topo_vect=np.array([1, 2, 1, 1])
    )

    # Create a mock action targeting Sub0
    action = MockActionObject(substations_id=[(0, [1, 2])])

    # Use global mock AlphaDeesp
    ranker = MockAlphaDeesp()

    score = compute_node_splitting_action_score(action, 0, obs, ranker, g=None)
    assert score == 1.0  # MockAlphaDeesp returns 1.0 for sub_id 0

    score_sub1 = compute_node_splitting_action_score(MockActionObject(substations_id=[(1, [1, 1])]), 1, obs, ranker, g=None)
    assert score_sub1 == 0.5  # MockAlphaDeesp returns 0.5 for sub_id != 0

def test_find_relevant_node_splitting(monkeypatch):
    # Observation setup
    obs_t0 = MockObservation(
        name_sub=np.array(["Sub0", "Sub1"]),
        sub_topologies={0: [1, 2], 1: [1, 1]},
        sub_info=np.array([2, 2]),
        topo_vect=np.array([1, 2, 1, 1])
    )
    obs_defaut = obs_t0

    # Candidate actions
    actions_unfiltered = {0: "action0", 1: "action1"}
    dict_action = {
        0: {"content": {"set_bus": {"substations_id": [(0, [1, 2])]}}},
        1: {"content": {"set_bus": {"substations_id": [(1, [1, 2])]}}}
    }

    hubs = ["Sub0"]
    nodes_blue_path = ["Sub1"]

    g = None
    g_distribution_graph = None
    simulator_data = None
    timestep = 0
    defauts = []
    overload_ids = []
    act_reco_maintenance = []

    action_space = MockActionSpace()
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.AlphaDeesp_warmStart", lambda g, g_dist, sim_data: MockAlphaDeesp())
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.check_rho_reduction", mock_check_rho_reduction)
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.identify_action_type", mock_identify_action_type)

    # Run function
    identified, effective, ineffective, ignored, scores = find_relevant_node_splitting(
        actions_unfiltered, dict_action, hubs, nodes_blue_path,
        g, g_distribution_graph, simulator_data,
        obs_t0, obs_defaut, timestep, action_space,
        defauts, overload_ids, act_reco_maintenance
    )

    # Assertions
    assert len(identified) == 2        # both actions identified
    assert len(effective) == 1         # only Sub0 effective
    assert len(ineffective) == 1       # Sub1 ineffective
    assert len(ignored) == 0           # none ignored
    assert scores[0] >= scores[1]      # sorted by descending score


def test_find_relevant_disconnections(monkeypatch):
    # Candidate actions
    actions_unfiltered = {"action_0": {}, "action_1": {}, "action_2": {}}

    dict_actions = {
        "action_0": {  # Effective: disconnects L1
            "type": "open_line",
            "substations_id": [[0]],
            "content": {"set_bus": {"lines_ex_id": {"L1": -1}, "lines_or_id": {}}},
        },
        "action_1": {  # Ineffective: disconnects L2
            "type": "open_line",
            "substations_id": [[3]],
            "content": {"set_bus": {"lines_ex_id": {"L2": -1}, "lines_or_id": {}}},
        },
        "action_2": {  # Ignored: not an open_line
            "type": "other_action",
            "substations_id": [[1]],
            "content": {"set_bus": {"lines_ex_id": {}, "lines_or_id": {}}},
        },
    }

    lines_constrained_path = ["L1", "L2"]

    obs = MockObservation(
        name_sub=np.array(["Sub0", "Sub1"]),
        sub_topologies={0: [1, 2], 1: [1, 1]},
        sub_info=np.array([2, 2]),
        topo_vect=np.array([1, 2, 1, 1])
    )
    timestep = 0
    defauts = []
    overload_ids = []
    act_reco_maintenance = None

    # Patch dependencies
    def mock_identify_action_type(action_desc, by_description, grid2op_action_space):
        return action_desc.get("type", "")
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.identify_action_type", lambda desc, **_: desc["type"])
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.check_rho_reduction", mock_check_rho_reduction)

    # Run function
    identified, effective, ineffective, ignored = find_relevant_disconnections(
        actions_unfiltered,
        dict_actions,
        lines_constrained_path,
        obs,
        timestep,
        MockActionSpace(),
        defauts,
        overload_ids,
        act_reco_maintenance
    )

    # ---- Assertions ----
    # Both action_0 and action_1 are identified (open_line and in constrained path)
    assert len(identified.keys()) == 2

    # action_0 disconnects L1 → effective
    assert "action_0" in effective
    assert "action_1" not in effective

    # action_1 disconnects L2 → ineffective
    assert "action_1" in ineffective
    assert "action_0" not in ineffective

    # action_2 is ignored (not open_line)
    assert "action_2" in ignored

class MockGraph:
    def __init__(self, edge_data=None):
        self.edge_data = edge_data or {}

    def get_edge_data(self, node1, node2):
        # Simulate graph connections
        return self.edge_data.get((node1, node2), None)


class MockOverflowGraph:
    def __init__(self, edge_data=None):
        self.g = MockGraph(edge_data=edge_data)  # expose .g as expected


# --- Helper function tests ---

def test_is_sublist_true_and_false():
    assert _is_sublist(["B", "C"], ["A", "B", "C", "D"]) is True
    assert _is_sublist(["C", "A"], ["A", "B", "C", "D"]) is False


def test_get_line_substations():
    obs = MockObservation(name_line=np.array(["L1", "L2", "L3"]), name_sub=np.array(["A", "B", "C", "D"]),
                          line_or_to_subid=np.array([0, 1, 2]),  # A, B, C)
                          line_ex_to_subid=np.array([1, 2, 3])  # B, C, D
                          )
    sub_or, sub_ex = _get_line_substations(obs, "L2")
    assert (sub_or, sub_ex) == ("B", "C")


def test_find_paths_for_line():
    line_subs = ("A", "B")
    paths = [["A", "B", "C"], ["B", "C", "D"], ["C", "D", "E"]]
    result = _find_paths_for_line(line_subs, paths)
    assert result == [["A", "B", "C"]]


def test_get_active_edges_between_active_and_inactive():
    g_overflow = MockOverflowGraph(edge_data={
        ("A", "B"): {
            0: {"name": "L1", "style": "solid"},
            1: {"name": "L2", "style": "dashed"}  # inactive
        },
        ("B", "A"): {
            0: {"name": "L3"}  # active (no style)
        }
    })
    active_edges = _get_active_edges_between(g_overflow, "A", "B")
    assert "L1" in active_edges
    assert "L3" in active_edges
    assert "L2" not in active_edges


def test_has_blocking_disconnected_line_blocked():
    """Should detect L2 as a blocking disconnected line with no active parallel edge."""
    obs = MockObservation(name_line=np.array(["L1", "L2", "L3"]), name_sub=np.array(["A", "B", "C", "D"]),
                          line_or_to_subid=np.array([0, 1, 2]),  # A, B, C)
                          line_ex_to_subid=np.array([1, 2, 3])  # B, C, D
                          )
    g_overflow = MockOverflowGraph(edge_data={
        ("A", "B"): {0: {"name": "L1"}},
        ("B", "C"): None,  # No parallel edges for L2
    })
    found_path = ["A", "B", "C"]
    all_disconnected_lines = ["L2"]
    blocked, blocker = _has_blocking_disconnected_line(obs, found_path, "L1", all_disconnected_lines, g_overflow)
    assert blocked is True
    assert blocker == "L2"


def test_has_blocking_disconnected_line_not_blocked():
    """L2 has a parallel active edge, so path should not be blocked."""
    obs = MockObservation(name_line = np.array(["L1", "L2", "L3"]),name_sub = np.array(["A", "B", "C", "D"]),
                          line_or_to_subid=np.array([0, 1, 2]),  # A, B, C)
                          line_ex_to_subid = np.array([1, 2, 3])# B, C, D
                          )
    g_overflow = MockOverflowGraph(edge_data={
        ("B", "C"): {0: {"name": "L2"}, 1: {"name": "L3", "style": "solid"}},  # active parallel
    })
    found_path = ["A", "B", "C"]
    all_disconnected_lines = ["L2"]
    blocked, blocker = _has_blocking_disconnected_line(obs, found_path, "L1", all_disconnected_lines, g_overflow)
    assert blocked is False
    assert blocker is None


# --- Main function tests ---

def test_check_other_reconnectable_line_on_path_no_block():
    """L1 is in path, no other disconnections → should be effective."""
    obs = MockObservation(name_line=np.array(["L1", "L2", "L3"]), name_sub=np.array(["A", "B", "C", "D"]),
                          line_or_to_subid=np.array([0, 1, 2]),  # A, B, C)
                          line_ex_to_subid=np.array([1, 2, 3])  # B, C, D
                          )
    g_overflow = MockOverflowGraph(edge_data={
        ("A", "B"): {0: {"name": "L1"}},  # active
        ("B", "A"): {0: {"name": "L1"}},
    })
    red_loop_paths = [["A", "B", "C"]]
    all_disconnected = []

    has_path, other_line = check_other_reconnectable_line_on_path(
        obs, "L1", all_disconnected, red_loop_paths, g_overflow
    )
    assert has_path is True
    assert other_line is None


def test_check_other_reconnectable_line_on_path_with_blocker():
    """L1 is in path but blocked by L2 (no parallel edge)."""
    obs = MockObservation(name_line=np.array(["L1", "L2", "L3"]), name_sub=np.array(["A", "B", "C", "D"]),
                          line_or_to_subid=np.array([0, 1, 2]),  # A, B, C)
                          line_ex_to_subid=np.array([1, 2, 3])  # B, C, D
                          )
    g_overflow = MockOverflowGraph(edge_data={
        ("A", "B"): {0: {"name": "L1"}},
        ("B", "C"): None  # no active parallel for L2
    })
    red_loop_paths = [["A", "B", "C"]]
    all_disconnected = ["L2"]

    has_path, other_line = check_other_reconnectable_line_on_path(
        obs, "L1", all_disconnected, red_loop_paths, g_overflow
    )
    assert has_path is False
    assert other_line == "L2"


def test_check_other_reconnectable_line_on_path_not_in_path():
    """L3 connects C-D, not in red loop paths."""
    obs = MockObservation(name_line=np.array(["L1", "L2", "L3"]), name_sub=np.array(["A", "B", "C", "D"]),
                          line_or_to_subid=np.array([0, 1, 2]),  # A, B, C)
                          line_ex_to_subid=np.array([1, 2, 3])  # B, C, D
                          )
    g_overflow = MockOverflowGraph()
    red_loop_paths = [["A", "B", "C"]]  # does not include D
    all_disconnected = []

    has_path, other_line = check_other_reconnectable_line_on_path(
        obs, "L3", all_disconnected, red_loop_paths, g_overflow
    )
    assert has_path is False
    assert other_line is None


def test_check_other_reconnectable_line_on_path_empty_paths():
    """No paths provided → should directly return (False, None)."""
    obs = MockObservation(name_line=np.array(["L1", "L2", "L3"]), name_sub=np.array(["A", "B", "C", "D"]),
                          line_or_to_subid=np.array([0, 1, 2]),  # A, B, C)
                          line_ex_to_subid=np.array([1, 2, 3])  # B, C, D
                          )
    g_overflow = MockOverflowGraph()
    has_path, other_line = check_other_reconnectable_line_on_path(
        obs, "L1", [], [], g_overflow
    )
    assert has_path is False
    assert other_line is None

def test_verify_relevant_reconnections(monkeypatch):
    """
    Test verify_relevant_reconnections with updated logic:
    - includes path filtering
    - action scoring
    - sorting
    - rho reduction check
    """

    # --- Mocks for helper functions ---
    def mock_check_other_reconnectable_line_on_path(obs_defaut, line_reco, all_disconected_lines, red_loop_paths, g_overflow):
        # Always mark L1 as relevant, L2 as irrelevant (to test filtering)
        if line_reco == "L1":
            return True, None
        return False, "L2"

    def mock_get_delta_theta_line(obs_defaut, line_id):
        # Return fixed values per line index
        return [0.5, 0.1][line_id]

    def mock_sort_actions_by_score(map_action_score):
        # Just preserve insertion order
        actions = {k: v["action"] for k, v in map_action_score.items()}
        lines = [v["line_impacted"] for v in map_action_score.values()]
        scores = [v["score"] for v in map_action_score.values()]
        return actions, lines, scores

    #def mock_check_rho_reduction(
    #    obs, timestep, act_defaut, action, overload_ids, act_reco_maintenance, lines_we_care_about, rho_tolerance
    #):
    #    # Simulate L1 effective, L2 ineffective
    #    line_name = action.args[0][0] if hasattr(action, "args") else "L1"
    #    return (line_name == "L1"), None

    def mock_create_default_action(action_space, defauts):
        return "mock_default_action"

    # --- Apply patches ---
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.check_other_reconnectable_line_on_path", mock_check_other_reconnectable_line_on_path)
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.get_delta_theta_line", mock_get_delta_theta_line)
    monkeypatch.setattr("expert_op4grid_recommender.utils.helpers.sort_actions_by_score", mock_sort_actions_by_score)
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.check_rho_reduction", mock_check_rho_reduction)
    monkeypatch.setattr("expert_op4grid_recommender.utils.simulation.create_default_action", mock_create_default_action)

    # --- Setup minimal mocks for objects ---
    obs = MockObservation(name_sub=["Sub0", "Sub1"],name_line=np.array(["L1", "L2"]))
    obs_defaut = obs  # can reuse
    action_space = MockActionSpace()  # mimic callable action space
    timestep = 0
    defauts = []
    overload_ids = ["O1"]
    act_reco_maintenance = MockAction()

    # Candidate reconnections
    relevant_line_reconnections = ["L1", "L2"]

    # Extra parameters
    g_overflow = None
    lines_we_care_about = ["L1", "L2"]
    red_loop_paths = []
    all_disconected_lines = []

    # --- Run ---
    effective, ineffective, identified_action = verify_relevant_reconnections(
        obs,
        obs_defaut,
        timestep,
        action_space,
        defauts,
        overload_ids,
        act_reco_maintenance,
        g_overflow,
        relevant_line_reconnections,
        red_loop_paths,
        all_disconected_lines,
        lines_we_care_about=lines_we_care_about,
        check_action_simulation=True
    )

    # --- Assertions ---
    assert "L1" in effective
    assert "L2" not in effective
    assert len(effective) == 1
    assert len(ineffective) == 0  # L2 is skipped due to path filtering
    assert "reco_L1" in identified_action


def test_check_simulation_with_exception():
    obs = MockObservation([100], simulate_return=(None, None, None, {"exception": ["error"]}))
    obs_defaut = MockObservation([100])
    action_space = MockActionSpace()
    timestep=0

    has_converged, has_lost_load = check_simu_overloads(
        obs, obs_defaut, lines_defaut=["L1"], lines_overloaded_ids=[0],
        lines_reco_maintenance=[], action_space=action_space,timestep=timestep
    )

    assert has_converged is False
    assert has_lost_load is False


def test_check_simulation_with_load_shedding():
    obs_simu_overloads = MockObservation(name_sub=[],load_values=[50])  # same load, no loss
    obs = MockObservation(name_sub=[],load_values=[100], simulate_return=(obs_simu_overloads, None, None, {"exception": []}))
    obs_defaut = MockObservation(name_sub=[],load_values=[100])
    action_space = MockActionSpace()
    timestep = 0

    has_converged, has_lost_load = check_simu_overloads(
        obs, obs_defaut, lines_defaut=["L1"], lines_overloaded_ids=[0],
        lines_reco_maintenance=[], action_space=action_space,timestep=timestep
    )

    assert has_converged is True
    assert has_lost_load is True


def test_check_simulation_successful_no_load_loss():
    obs_simu_overloads = MockObservation(name_sub=[],load_values=[100])  # same load, no loss
    obs = MockObservation(name_sub=[],load_values=[100], simulate_return=(obs_simu_overloads, None, None, {"exception": []}))
    obs_defaut = MockObservation(name_sub=[],load_values=[100])
    action_space = MockActionSpace()
    timestep = 0

    has_converged, has_lost_load = check_simu_overloads(
        obs, obs_defaut, lines_defaut=["L1"], lines_overloaded_ids=[0],
        lines_reco_maintenance=[], action_space=action_space,timestep=timestep
    )

    assert has_converged is True
    assert has_lost_load is False

class MockEnv:
    """
    Mock environment containing:
    - name_line: list of line names
    - chronics_handler.real_data.data.maintenance_handler.array: maintenance status array
    - action_space: callable to build actions
    """
    def __init__(self, name_line, maintenance_array):
        self.name_line = name_line
        # simulate chronics handler structure
        self.chronics_handler = type('Chronics', (), {})()
        self.chronics_handler.real_data = type('RealData', (), {})()
        self.chronics_handler.real_data.data = type('Data', (), {})()
        self.chronics_handler.real_data.data.maintenance_handler = type('Maintenance', (), {})()
        self.chronics_handler.real_data.data.maintenance_handler.array = maintenance_array
        self.action_space = MockActionSpace()


def test_get_maintenance_timestep():
    """
    Test get_maintenance_timestep for various reconnection scenarios.
    """

    # Setup mock environment
    name_line = ["L1", "L2", "L3"]
    # 2D array: rows are timesteps, columns are lines; True = in maintenance
    maintenance_array = np.array([
        [True, True, False],   # t=0
        [False, True, False],  # t=1
        [False, False, False], # t=2
    ])
    env = MockEnv(name_line, maintenance_array)

    lines_non_reconnectable = ["L2"]

    # --- Case 1: do_reco_maintenance = True ---
    act, reconnected = get_maintenance_timestep(
        timestep=1,
        lines_non_reconnectable=lines_non_reconnectable,
        env=env,
        do_reco_maintenance=True
    )

    # Only L1 should be reconnected (L2 is non-reconnectable, L3 not in maintenance)
    assert reconnected == ["L1"]
    assert act.content["set_line_status"] == [("L1", 1)]

    # --- Case 2: do_reco_maintenance = False ---
    act, reconnected = get_maintenance_timestep(
        timestep=1,
        lines_non_reconnectable=lines_non_reconnectable,
        env=env,
        do_reco_maintenance=False
    )

    # No lines should be reconnected
    assert reconnected == []
    assert act.content == {}

    # --- Case 3: timestep where all lines are out of maintenance ---
    act, reconnected = get_maintenance_timestep(
        timestep=2,
        lines_non_reconnectable=lines_non_reconnectable,
        env=env,
        do_reco_maintenance=True
    )

    # No lines eligible for reconnection at this timestep
    assert reconnected == []
    assert act.content == {}

#################################
### TEST Section
@pytest.mark.slow
def test_overflow_graph_construction():
    """
    This function tests the construction of the overflow graph and the identification of constrained and dispatch paths.
    """
    date = datetime(2024, 8, 28)  # we choose a date for the chronic
    timestep=1#36
    line_defaut="P.SAOL31RONCI"#"FRON5L31LOUHA"
    env_name = "env_dijon_v2_assistant"
    non_connected_reconnectable_lines = ['BOISSL61GEN.P', 'CHALOL31LOUHA', 'CRENEL71VIELM', 'CURTIL61ZCUR5',
                                         'GEN.PL73VIELM',
                                         'P.SAOL31RONCI',
                                         'PYMONL61VOUGL', 'BUGEYY715', 'CPVANY632', 'GEN.PY762', 'PYMONY632']

    #env = grid2op.make(env_name, backend=backend, n_busbar=6, param=p)

    current_folder = Path(__file__).parent.resolve()
    env_folder=env_folder=os.path.join(os.path.dirname(current_folder),"data")#two level up
    env = make_grid2op_training_env(env_folder, env_name)#make_grid2op_assistant_env(".", env_name)

    # make the environment
    chronics_name = list_all_chronics(env)
    print("chronics names are:")
    print(chronics_name)

    # we get the first observation for the chronic at the desired date
    obs = get_first_obs_on_chronic(date, env)

    act_deco_defaut=env.action_space({"set_line_status": [(line_defaut, -1)]})



    obs_simu, reward, done, info=obs.simulate(act_deco_defaut,time_step=timestep)

    lines_overloaded_ids=[i for i,rho in enumerate(obs_simu.rho) if rho>=1]

    param_options_test={
        # 2 percent of the max overload flow
        "ThresholdReportOfLine": 0.2,  # 0.05,#
        # 10 percent de la surcharge max
        "ThersholdMinPowerOfLoop": 0.1,
        # If at least a loop is detected, only keep the ones with a flow  of at least 25 percent the biggest one
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
    inhibit_swapped_flow_reversion=False
    do_consolidate_graph=True
    lines_non_reconnectable = []  # TO DELETE, just for test
    df_of_g, overflow_sim, g_overflow, hubs, g_distribution_graph, node_name_mapping = build_overflow_graph(env,obs_simu,
                                                            lines_overloaded_ids, non_connected_reconnectable_lines,lines_non_reconnectable,
                                                            timestep,do_consolidate_graph, inhibit_swapped_flow_reversion,param_options=param_options_test)

    ##########
    # get useful paths for action verification

    lines_constrained_path, nodes_constrained_path,other_blue_edges, other_blue_nodes = g_distribution_graph.get_constrained_edges_nodes()

    lines_redispatch,list_nodes_dispatch_path = g_distribution_graph.get_dispatch_edges_nodes()

    ############
    # Pour tests
    list_nodes_constrained_path_test=['NAVILP3','CPVANP6','CPVANP3','CHALOP6','GROSNP6', '1GROSP7',
                                      'GROSNP7', 'VIELMP7', 'H.PAUP7', 'SSV.OP7', 'ZCUR5P6', 'H.PAUP6', '2H.PAP7',
                                      'COUCHP6', 'VIELMP6', '1VIELP7', 'COMMUP6', 'ZMAGNP6', 'C.REGP6', 'BEON P3', 'P.SAOP3']

    list_lines_contrained_path_test=['GROSNY761','COMMUL61VIELM', 'GROSNY771', 'COUCHL61CPVAN', 'VIELMY771', 'VIELMY763', 'GROSNY762',
                                     'H.PAUL61ZCUR5', 'VIELMY762', 'CPVANY632', 'GROSNL61ZCUR5', 'C.REGL61VIELM', 'H.PAUL71VIELM',
                                     'H.PAUY762', 'CPVANY633', 'C.REGL62VIELM', 'CHALOL62GROSN', 'CHALOL61CPVAN', 'C.REGL61ZMAGN',
                                     'COMMUL61H.PAU', 'CHALOL61GROSN', 'GROSNL71SSV.O', 'CPVANL61ZMAGN', 'COUCHL61VIELM',
                                     'VIELMY761', 'BEON L31P.SAO','BEON L31CPVAN', 'H.PAUY772', 'CPVANY631','NAVILL31P.SAO']

    list_nodes_dispatch_path_test=['VOUGLP3', 'MAGNYP3', 'FRON5P3', 'C.FOUP3', 'ZCUR5P6', 'SSV.OP7', 'SSUSUP3', '1GEN.P7', 'PYMONP3', 'AISERP3',
                                   'NAVILP3', 'GROSNP6', 'ZMAGNP6', 'SAISSP3', 'VIELMP7', 'VOUGLP6', 'CREYSP7', 'LOUHAP3', 'RONCIP3', 'GEN.PP7',
                                   'GEN.PP6', 'IZERNP6', 'FLEYRP6', 'H.PAUP7', 'P.SAOP3', 'CHALOP3', 'BOISSP6', 'CURTIP6', 'MERVAP3', 'CHALOP6',
                                   'CIZE P6', 'ZJOUXP6', 'MACONP6', 'CUISEP3', 'G.CHEP3', 'MAGNYP6']



    list_lines_redispatch_path_test=['GEN.PY761', 'AISERL31RONCI', 'BOISSL61GEN.P', 'C.FOUL31NAVIL', 'CHALOL31LOUHA', 'CHALOY631', 'CHALOY632',
                                     'CHALOY633', 'CIZE L61FLEYR', 'CREYSL71GEN.P', 'CREYSL72GEN.P', 'CUISEL31G.CHE', 'CURTIL61GROSN', 'FLEYRL61VOUGL',
                                     'FRON5L31LOUHA', 'FRON5L31G.CHE', 'GEN.PL61IZERN', 'GEN.PL61VOUGL', 'GEN.PY762', 'GEN.PY771', 'GROSNL61MACON',
                                     'H.PAUL71SSV.O', 'CIZE L61IZERN', 'LOUHAL31SSUSU', 'MACONL61ZJOUX', 'AISERL31MAGNY', 'MAGNYY633', 'C.FOUL31MERVA',
                                     'LOUHAL31PYMON', 'P.SAOL31RONCI', 'PYMONL31SAISS', 'MERVAL31SSUSU', 'CREYSL71SSV.O', 'CREYSL72SSV.O', 'GEN.PL71VIELM',
                                     'GEN.PL72VIELM', 'GEN.PL73VIELM', 'CUISEL31VOUGL', 'SAISSL31VOUGL', 'VOUGLY631', 'VOUGLY632', 'CURTIL61ZCUR5', 'BOISSL61ZJOUX', 'MAGNYL61ZMAGN']


    list_hubs_test=[ 'VIELMP7', 'H.PAUP7', 'SSV.OP7','NAVILP3']#[ 'CPVANP6', 'CHALOP6', 'GROSNP6', 'VIELMP7', 'H.PAUP7', 'SSV.OP7','NAVILP3']#'P.SAOP3',

    assert(set(list_nodes_constrained_path_test).intersection(set(nodes_constrained_path))==set(nodes_constrained_path))
    assert (set(list_lines_contrained_path_test).intersection(set(lines_constrained_path))==set(lines_constrained_path))
    assert (set(list_nodes_dispatch_path_test).intersection(set(list_nodes_dispatch_path))==set(list_nodes_dispatch_path_test))
    assert (set(list_lines_redispatch_path_test).intersection(set(lines_redispatch))==set(list_lines_redispatch_path_test))
    assert (set(list_hubs_test).intersection(set(hubs))==set(hubs))

@pytest.mark.slow
def test_overflow_graph_actions_filtered(check_with_action_description=True):
    """
    This function tests the filtering of actions based on the overflow graph and the identification of constrained and dispatch paths.
    It verifies that the actions are correctly categorized into filtered and unfiltered actions based on the expert rules.

    The test involves the following steps:
    1. Setting up the environment and loading the necessary data.
    2. Simulating the environment to get the initial observation.
    3. Identifying overloaded lines and building the overflow graph.
    4. Extracting constrained and dispatch paths from the overflow graph.
    5. Categorizing the actions based on the expert rules.
    6. Asserting that the number of actions and their categorization match the expected values.

    The test ensures that the expert rules are correctly applied to filter out inappropriate actions.
    """
    date = datetime(2024, 8, 28)  # we choose a date for the chronic
    timestep = 1  # 36
    line_defaut = "P.SAOL31RONCI"  # "FRON5L31LOUHA"
    lines_defaut = [line_defaut]
    env_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
    env_name = "env_dijon_v2_assistant"

    action_space_folder = os.path.join(env_folder,"action_space")
    file_action_space_desc = "actions_repas_most_frequent_topologies_revised.json"
    file_path = os.path.join(action_space_folder, file_action_space_desc)

    # Load actions
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        dict_action = json.load(file)

    # Make the environment
    env = make_grid2op_training_env(env_folder, env_name)

    chronics_name = list_all_chronics(env)
    print("chronics names are:")
    print(chronics_name)

    # Get the first observation for the chronic at the desired date
    obs = get_first_obs_on_chronic(date, env)

    # read non reconnectable lines
    path_chronic = [path for path in env.chronics_handler.real_data.subpaths if date.strftime('%Y%m%d') in path][0]
    lines_non_reconnectable = list(load_interesting_lines(path=path_chronic,file_name="non_reconnectable_lines.csv"))
    lines_should_not_reco_2024_and_beyond =DELETED_LINE_NAME


    # simulate contingency tp detect overloads
    act_deco_defaut = env.action_space({"set_line_status": [(line_defaut, -1)]})

    obs_simu, reward, done, info = obs.simulate(act_deco_defaut, time_step=timestep)

    non_connected_reconnectable_lines = [l_name for i,l_name in enumerate(env.name_line)
                                         if l_name not in lines_non_reconnectable+lines_should_not_reco_2024_and_beyond and not obs_simu.line_status[i]]
    param_options_test = {
        # 2 percent of the max overload flow
        "ThresholdReportOfLine": 0.2,  # 0.05,#
        # 10 percent de la surcharge max
        "ThersholdMinPowerOfLoop": 0.1,
        # If at least a loop is detected, only keep the ones with a flow  of at least 25 percent the biggest one
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

    inhibit_swapped_flow_reversion = True  # Cancel the swapped edge direction for swapped flows (possibly not needed anymore given the new consolidate graph functions)
    lines_overloaded_ids = [i for i, rho in enumerate(obs_simu.rho) if rho >= 1]
    do_consolidate_graph = True
    lines_non_reconnectable=[]
    df_of_g, overflow_sim, g_overflow, hubs, g_distribution_graph, node_name_mapping = build_overflow_graph(env,obs_simu,
                                                            lines_overloaded_ids, non_connected_reconnectable_lines,lines_non_reconnectable,
                                                            timestep,do_consolidate_graph, inhibit_swapped_flow_reversion,param_options=param_options_test)

    ##########
    # Get useful paths for action verification
    lines_constrained_path, nodes_constrained_path,other_blue_edges, other_blue_nodes = g_distribution_graph.get_constrained_edges_nodes()

    lines_dispatch, nodes_dispatch_path = g_distribution_graph.get_dispatch_edges_nodes()

    #########
    # Check rules for each action
    lines_reco_maintenance=[]
    paths=((lines_constrained_path, nodes_constrained_path), (lines_dispatch, nodes_dispatch_path))
    actions_to_filter, actions_unfiltered = categorize_action_space(dict_action, hubs,paths, obs, timestep, lines_defaut, env.action_space, lines_overloaded_ids,lines_reco_maintenance,by_description=check_with_action_description)

    n_actions = len(dict_action.keys())
    n_actions_filtered = len(actions_to_filter.keys())
    n_actions_unfiltered = len(actions_unfiltered.keys())
    n_actions_badly_filtered = len([id for id, act_filter_content in actions_to_filter.items() if act_filter_content["is_rho_reduction"]])

    # Could also directly compare to saved dictionaries "actions_to_filter_expert_rules.json" and "actions_unfiltered_expert_rules.json"
    assert(n_actions == 102)
    assert(n_actions_filtered == 65)
    assert(n_actions_unfiltered == n_actions - n_actions_filtered)
    assert(n_actions_badly_filtered == 4)  # Opening OC 'MAGNY3TR633 DJ_OC' in the substation 'MAGNYP3'. This action is filtered because of the significant delta flow threshold.
    # If "ThresholdReportOfLine" is reduced from 0.2 to 0.05, then this action is not filtered anymore, and everything works as expected

def test_action_type_open_coupling():
    actions_desc={
        "description": " Ouverture OC 'VOUGL6COUPL DJ_OC' dans le poste 'VOUGLP6'",
        "description_unitaire": " Ouverture OC 'VOUGL6COUPL DJ_OC' dans le poste 'VOUGLP6'",
        "content": {
            "set_bus": {
                "lines_or_id": {
                    "VOUGLY612": 1,
                    "VOUGLY631": 1,
                    "VOUGLY632": 2
                },
                "lines_ex_id": {
                    "FLEYRL61VOUGL": 2,
                    "GEN.PL61VOUGL": 1,
                    "PYMONL61VOUGL": 2
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
        },
        "VoltageLevelId": "VOUGLP6"
    }

    action_type=identify_action_type(actions_desc, by_description=True)
    assert(action_type=="open_coupling")

def test_action_type_open_line():
    actions_desc={
        "description": "Ouverture OC 'PYMON3TR632 DJ_OC' dans le poste 'PYMONP3'",
        "description_unitaire": "Ouverture OC 'PYMON3TR632 DJ_OC' dans le poste 'PYMONP3'",
        "content": {
            "set_bus": {
                "lines_or_id": {},
                "lines_ex_id": {
                    "PYMONY632": -1
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
        },
        "VoltageLevelId": "PYMONP3"
    }

    action_type=identify_action_type(actions_desc, by_description=True)
    assert(action_type=="open_line")

def test_action_type_open_line_load():
    actions_desc={
        "description": " Ouverture OC 'GEN.P6CHAV6.1 DJ_OC' dans le poste 'GEN.PP6'\n- Ouverture OC 'GEN.P6AT762 DJ_OC' dans le poste 'GEN.PP6'",
        "description_unitaire": " Ouverture OC 'GEN.P6CHAV6.1 DJ_OC' dans le poste 'GEN.PP6'\n- Ouverture OC 'GEN.P6AT762 DJ_OC' dans le poste 'GEN.PP6'",
        "content": {
            "set_bus": {
                "lines_or_id": {},
                "lines_ex_id": {
                    "GEN.PY762": -1
                },
                "loads_id": {
                    "CHAV6L61GEN.P":-1
                },
                "generators_id": {},
                "shunts_id": {}
            }
        },
        "VoltageLevelId": "GEN.PP6"
    }

    action_type=identify_action_type(actions_desc, by_description=True)
    assert(action_type=="open_line_load")

def test_action_type_close_line():
    actions_desc={
        "description": "Fermeture OC 'PYMON6CPVAN.1 DJ_OC' dans le poste 'PYMONP6'(reconnection sur noeuds 1 aux 2 postes extremites)",
        "description_unitaire": "Fermeture OC 'PYMON6CPVAN.1 DJ_OC' dans le poste 'PYMONP6'(reconnection sur noeuds 1 aux 2 postes extremites)",
        "content": {
            "set_bus": {
                "lines_or_id": {
                    "CPVANL61PYMON": 1
                },
                "lines_ex_id": {
                    "CPVANL61PYMON": 1
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
        },
        "VoltageLevelId": "PYMONP6"
    }

    action_type=identify_action_type(actions_desc, by_description=True)
    assert(action_type=="close_line")

def test_action_type_close_coupling():
    actions_desc = {
        "description": "Fermeture OC 'CPVAN3COUPL DJ_OC' dans le poste 'CPVANP3'",
        "description_unitaire": "Fermeture OC 'CPVAN3COUPL DJ_OC' dans le poste 'CPVANP3'",
        "content": {
            "set_bus": {
                "lines_or_id": {
                    "CPVANL31RIBAU": 1
                },
                "lines_ex_id": {
                    "BEON L31CPVAN": 1,
                    "CPVANY631": 1,
                    "CPVANY632": 1,
                    "CPVANY633": 1
                },
                "loads_id": {
                    "ARBOIL31CPVAN": 1,
                    "BREVAL31CPVAN": 1,
                    "CPDIVL32CPVAN": 1,
                    "CPVANL31MESNA": 1,
                    "CPVANL31ZBRE6": 1,
                    "CPVAN3TR312": 1,
                    "CPVAN3TR311": 1
                },
                "shunts_id": {},
                "generators_id": {}
            }
        },
        "VoltageLevelId": "CPVANP3"
    }

    action_type = identify_action_type(actions_desc, by_description=True)
    assert (action_type == "close_coupling")

def test_action_types():
    test_action_type_close_line()
    test_action_type_open_line()
    test_action_type_open_coupling()
    test_action_type_close_coupling()
    test_action_type_open_line_load()

def _test_grid2op_action_type_open_line_load(action_space):
    actions_desc={
            "set_bus": {
                "lines_or_id": {},
                "lines_ex_id": {
                    "GEN.PY762": -1
                },
                "loads_id": {
                    "CHAV6L61GEN.P":-1
                },
                "generators_id": {},
                "shunts_id": {}
            }
    }

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)
    assert(action_type=="open_line_load")

def _test_grid2op_action_type_close_line(action_space):
    actions_desc={
            "set_bus": {
                "lines_or_id": {
                    "CPVANL61PYMON": 1
                },
                "lines_ex_id": {
                    "CPVANL61PYMON": 1
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
    }

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)
    assert(action_type=="close_line")

def _test_grid2op_action_type_open_line(action_space):
    actions_desc={
            "set_bus": {
                "lines_or_id": {},
                "lines_ex_id": {
                    "PYMONY632": -1
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
    }

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)
    assert(action_type=="open_line")


def _test_grid2op_action_type_open_coupling(action_space):
    actions_desc={
            "set_bus": {
                "lines_or_id": {
                    "VOUGLY612": 1,
                    "VOUGLY631": 1,
                    "VOUGLY632": 2
                },
                "lines_ex_id": {
                    "FLEYRL61VOUGL": 2,
                    "GEN.PL61VOUGL": 1,
                    "PYMONL61VOUGL": 2
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
    }

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)
    assert(action_type=="open_coupling")


def _test_grid2op_action_type_close_coupling(action_space):
    actions_desc={'set_bus': {'lines_or_id': {'CPVANL31RIBAU': 1},
          'lines_ex_id': {'BEON L31CPVAN': 1,
           'CPVANY631': 1,
           'CPVANY632': 1,
           'CPVANY633': 1},
          'loads_id': {'ARBOIL31CPVAN': 1,
           'BREVAL31CPVAN': 1,
           'CPDIVL32CPVAN': 1,
           'CPVANL31MESNA': 1,
           'CPVANL31ZBRE6': 1,
           'CPVAN3TR312': 1,
           'CPVAN3TR311': 1},
          'shunts_id': {},
          'generators_id': {}}}

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)

    assert (action_type == "close_coupling")

@pytest.mark.slow
def test_grid2op_action_types():
    #load action space
    env_name = "env_dijon_v2_assistant"
    env_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
    env = make_grid2op_training_env(env_folder, env_name)  # make_grid2op_assistant_env(".", env_name)
    action_space=env.action_space

    # run tests
    _test_grid2op_action_type_open_line_load(action_space)
    _test_grid2op_action_type_close_line(action_space)
    _test_grid2op_action_type_open_line(action_space)
    _test_grid2op_action_type_close_coupling(action_space)
    _test_grid2op_action_type_open_coupling(action_space)

def test_no_broken_rule_multi_node_dispatch_path():
    action_type="open_coupling"
    localization="dispatch_path"
    subs_topology=[[1,1,2,2,2,1]]#topology already in multi nodes
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action==False)
    assert (broken_rule is None)

def test_broken_rule_open_line_dispatch_path():
    action_type="open_line"
    localization="dispatch_path"
    subs_topology=[]#topology already in multi nodes
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action)
    assert (broken_rule=="No line disconnection on dispatch path")

def test_broken_rule_close_line_constrained_path():
    action_type="close_line"
    localization="constrained_path"
    subs_topology=[]#topology already in multi nodes
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action)
    assert (broken_rule=="No line reconnection on constrained path")

def test_broken_rule_close_coupling_constrained_path():
    action_type="close_coupling"
    localization="constrained_path"
    subs_topology=[]#topology already in multi nodes
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action)
    assert (broken_rule=="No node merging on constrained path")

def test_broken_rule_open_coupling_dispatch_path():
    action_type="open_coupling"
    localization="dispatch_path"
    subs_topology=[[1,1,1,1]]#topology in one node
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action)
    assert (broken_rule=="No node splitting on dispatch path")

def test_load_action_no_filter():
    action_type = "open_line_load"
    localization = ""
    subs_topology = []

    do_filter_action, broken_rule = check_rules(action_type, localization, subs_topology)

    assert(do_filter_action==False)
    assert (broken_rule is None)

@pytest.mark.slow
def test_identify_overload_lines_to_keep_overflow_graph_connected():
    """
        Test function to verify that only rho of lines_overloaded_ids will be considered in this function to detect the max rho
    """
    # Setup test environment
    env_name = "env_dijon_v2_assistant"
    env_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
    env = make_grid2op_assistant_env(env_folder, env_name)
    obs=env.get_obs()

    l1_id=0
    l2_id=1

    obs.rho[l1_id]=1.3
    obs.rho[l2_id]=1.5

    lines_overloaded_ids=[l1_id]
    lines_overloaded_ids_to_keep, _ = identify_overload_lines_to_keep_overflow_graph_connected(obs, lines_overloaded_ids)

    assert (l1_id in lines_overloaded_ids_to_keep)
    assert (l2_id not in lines_overloaded_ids_to_keep)

@pytest.mark.slow
def test_get_n_connected_components_graph_with_overloads():
    """
        Logical test to test get_n_connected_components_graph_with_overloads: when you target a line considered overloaded that is actually an antenna on the grid, ...
    """
    env_name = "env_dijon_v2_assistant"
    env_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
    env = make_grid2op_assistant_env(env_folder, env_name)
    obs = env.get_obs()

    # tester la deconnexion d'une antenne
    line_overloaded = "AUXONL31TILLE"  # we make as this this line is overloaded, to test its disconnection, that should lead to the islanding of AUXONP3

    lines_overloaded_ids = [i for i, l in enumerate(obs.name_line) if l == line_overloaded]
    comps_init, comps_wo_max_overload, comp_overloads = get_n_connected_components_graph_with_overloads(
        obs, lines_overloaded_ids)

    assert(comps_wo_max_overload == comp_overloads)#in that case they should be equal as we consider only one "overload"
    assert(len(comps_init)+1 == len(comps_wo_max_overload))

    islanded_sub=obs.name_sub[comps_wo_max_overload[2].pop()]
    assert(islanded_sub=='TILLEP3')

@pytest.mark.slow
def test_get_n_connected_components_graph_with_overloads_2():
    """
        Known situation with BEON L31CPVAN as a contingency and C.FOUL31MERVA as an overload that then island several substations.
        Also adding the fictitious overload of "AUXONL31TILLE" as we would expect one more islanded component when considering all overloads then
    """
    #get_n_connected_components_graph_with_overloads(obs_simu, lines_overloaded_ids)

    env_name = "env_dijon_v2_assistant"
    env_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
    env = make_grid2op_assistant_env(env_folder, env_name)
    date = datetime(2024, 11,
                    27)  # datetime(2024, 9, 19)#datetime(2024, 8, 28)#datetime(2024, 11, 27)#datetime(2024, 12, 9)#datetime(2024, 12, 7)#datetime(2024, 12, 7)#datetime(2024, 11, 27)#datetime(2024, 9, 19)#datetime(2024, 11, 27)#datetime(2024, 9, 19)#datetime(2024, 11, 25)#datetime(2024, 11, 25)#datetime(2024, 11, 25)#datetime(2024, 12, 9)#datetime(2024, 12, 2)#datetime(2024, 8, 28)  # we choose a date for the chronic
    timestep = 36  # 1#47#18#13#22#9#15#1#35#10#14#14#13#22 #1 # 36
    line_defaut = "BEON L31CPVAN"
    path_chronic = [path for path in env.chronics_handler.real_data.subpaths if date.strftime('%Y%m%d') in path][0]

    obs = get_first_obs_on_chronic(date, env, path_thermal_limits=path_chronic)
    act_deco_defaut = env.action_space({"set_line_status": [(line_defaut, -1)]})
    obs_simu, reward, done, info = obs.simulate(act_deco_defaut, time_step=timestep)

    line_overloaded = "C.FOUL31MERVA"
    fictitious_line_overload = "AUXONL31TILLE"
    lines_overloaded_ids = [i for i, l in enumerate(obs.name_line) if l == line_overloaded or l==fictitious_line_overload]
    comps_init, comps_wo_max_overload, comp_overloads = get_n_connected_components_graph_with_overloads(
        obs_simu, lines_overloaded_ids)

    assert (len(comps_init) + 1 == len(comps_wo_max_overload))
    assert (len(comps_wo_max_overload) + 1 == len(comp_overloads))

    islanded_subs_1 = set([obs.name_sub[i] for i in comps_wo_max_overload[1]])

    islanded_sub_2= obs.name_sub[list(comp_overloads[2])[0]]
    islanded_subs_1_prime = set([obs.name_sub[i] for i in comp_overloads[1]])
    assert (islanded_subs_1 == islanded_subs_1_prime == set(['NAVILP3', 'P.SAOP3', 'RONCIP3', 'AISERP3', 'BEON P3', 'C.FOUP3'])) #TODO
    assert (islanded_sub_2 == 'TILLEP3')

@pytest.mark.slow
def test_get_subs_islanded_by_overload_disconnections():
    """
        Logical test to test get_subs_islanded_by_overload_disconnections: when you target a line considered overloaded that is actually an antenna on the grid, you expect one of its neighboring substation to be islanded
    """
    #get_subs_islanded_by_overload_disconnections(comps_init, comp_overloads, max_rho):
    #BEONL31CPVAN, nov 27 2024
    # Setup test environment
    env_name = "env_dijon_v2_assistant"
    env_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    env = make_grid2op_assistant_env(env_folder, env_name)
    obs = env.get_obs()

    # tester la deconnexion d'une antenne
    line_overloaded= "AUXONL31TILLE" #we make as this this line is overloaded, to test its disconnection, that should lead to the islanding of AUXONP3

    lines_overloaded_ids = [i for i, l in enumerate(obs.name_line) if l == line_overloaded]
    comps_init, comps_wo_max_overload, comp_overloads = get_n_connected_components_graph_with_overloads(
        obs, lines_overloaded_ids)

    identified_subs_broken_apart = get_subs_islanded_by_overload_disconnections(obs, comps_init, comp_overloads,
                                                                                line_overloaded)

    identified_subs_broken_apart
    assert(identified_subs_broken_apart==['TILLEP3'])

@pytest.mark.slow
def test_get_subs_islanded_by_overload_disconnections_2():
    """
        Known situation with BEON L31CPVAN as a contingency and C.FOUL31MERVA as an overload that then island several substations
    """
    env_name = "env_dijon_v2_assistant"
    env_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
    env = make_grid2op_assistant_env(env_folder, env_name)
    date = datetime(2024, 11,
                    27)  # datetime(2024, 9, 19)#datetime(2024, 8, 28)#datetime(2024, 11, 27)#datetime(2024, 12, 9)#datetime(2024, 12, 7)#datetime(2024, 12, 7)#datetime(2024, 11, 27)#datetime(2024, 9, 19)#datetime(2024, 11, 27)#datetime(2024, 9, 19)#datetime(2024, 11, 25)#datetime(2024, 11, 25)#datetime(2024, 11, 25)#datetime(2024, 12, 9)#datetime(2024, 12, 2)#datetime(2024, 8, 28)  # we choose a date for the chronic
    timestep = 36  # 1#47#18#13#22#9#15#1#35#10#14#14#13#22 #1 # 36
    line_defaut = "BEON L31CPVAN"
    path_chronic = [path for path in env.chronics_handler.real_data.subpaths if date.strftime('%Y%m%d') in path][0]

    obs = get_first_obs_on_chronic(date, env,path_thermal_limits=path_chronic)
    act_deco_defaut = env.action_space({"set_line_status": [(line_defaut, -1)]})
    obs_simu, reward, done, info = obs.simulate(act_deco_defaut, time_step=timestep)

    line_overloaded = "C.FOUL31MERVA"
    lines_overloaded_ids=[i for i, l in enumerate(obs.name_line) if l==line_overloaded]
    comps_init, comps_wo_max_overload, comp_overloads = get_n_connected_components_graph_with_overloads(
        obs_simu, lines_overloaded_ids)

    identified_subs_broken_apart = get_subs_islanded_by_overload_disconnections(obs, comps_init, comp_overloads,line_overloaded)

    assert(set(identified_subs_broken_apart)==set(['NAVILP3', 'P.SAOP3', 'RONCIP3', 'AISERP3', 'BEON P3', 'C.FOUP3']))

@pytest.mark.slow
def test_get_maintenance_timestep():
    #WARNING env folder expected too be at the root level where the package is
    env_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")#two level up
    env_name = "env_dijon_v2_assistant"
    env = make_grid2op_assistant_env(env_folder, env_name)
    # datetime(2024, 9, 19)#datetime(2024, 8, 28)#datetime(2024, 11, 27)#datetime(2024, 12, 9)#datetime(2024, 12, 7)#datetime(2024, 12, 7)#datetime(2024, 11, 27)#datetime(2024, 9, 19)#datetime(2024, 11, 27)#datetime(2024, 9, 19)#datetime(2024, 11, 25)#datetime(2024, 11, 25)#datetime(2024, 11, 25)#datetime(2024, 12, 9)#datetime(2024, 12, 2)#datetime(2024, 8, 28)  # we choose a date for the chronic
    do_reco_maintenance=True


    #create fictitious maintenance for the first timestep only
    l_id=0
    env.chronics_handler.real_data.data.maintenance_handler.array[0][l_id]=True
    nb_timesteps=len(env.chronics_handler.real_data.data.maintenance_handler.array)
    for t in range(1,nb_timesteps):
        env.chronics_handler.real_data.data.maintenance_handler.array[t][l_id] = False

    lines_non_reconnectable=[]

    #at first timestep, the line is in maintenance and should not be reconnected
    timestep = 0
    act_reco_maintenance, maintenance_to_reco_at_t=get_maintenance_timestep(timestep,lines_non_reconnectable,env,do_reco_maintenance)
    assert(act_reco_maintenance.as_dict()=={})
    assert(maintenance_to_reco_at_t==[])

    # at second timestep, the line is not in maintenance and can be reconnected
    timestep=1
    act_reco_maintenance, maintenance_to_reco_at_t = get_maintenance_timestep(timestep, lines_non_reconnectable, env,
                                                                              do_reco_maintenance)
    assert(act_reco_maintenance.as_dict()['set_line_status']['connected_id'][0]==l_id)
    assert(env.name_line[l_id] in maintenance_to_reco_at_t)

    lines_non_reconnectable = [env.name_line[l_id]]
    # at second timestep, the line is not in maintenance but can not be reconnected because it is not reconnectable
    act_reco_maintenance, maintenance_to_reco_at_t = get_maintenance_timestep(timestep, lines_non_reconnectable, env,
                                                                             do_reco_maintenance)
    assert (act_reco_maintenance.as_dict() == {})
    assert (maintenance_to_reco_at_t == [])


#def test_rules():
#    test_no_broken_rule_multi_node_dispatch_path()
#    test_broken_rule_open_line_dispatch_path()
#    test_broken_rule_close_line_constrained_path()
#    test_broken_rule_close_coupling_constrained_path()
#    test_broken_rule_open_coupling_dispatch_path()
#    test_load_action_no_filter()
#
#if __name__ == "__main__":
#
#    print("STARTING TESTS")
#    test_get_subs_islanded_by_overload_disconnections()
#    test_get_subs_islanded_by_overload_disconnections_2()
#    test_get_n_connected_components_graph_with_overloads()
#    test_get_n_connected_components_graph_with_overloads_2()
#    test_identify_overload_lines_to_keep_overflow_graph_connected()
#    test_overflow_graph_actions_filtered()
#    test_overflow_graph_actions_filtered(check_with_action_description=False)
#    test_grid2op_action_types()
#    test_overflow_graph_construction()
#    test_action_types()
#    test_rules()
#    print("ENDING TESTS")