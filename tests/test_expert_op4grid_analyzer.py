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
from expert_op4grid_recommender.environment import switch_to_dc_load_flow,setup_environment_configs,get_env_first_obs
from tests import config_test
from expert_op4grid_recommender.main import run_analysis,Backend

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
                                   'CIZE P6', 'ZJOUXP6', 'MACONP6', 'CUISEP3', 'G.CHEP3', 'MAGNYP6','PYMONP6']



    list_lines_redispatch_path_test=['GEN.PY761', 'AISERL31RONCI', 'BOISSL61GEN.P', 'C.FOUL31NAVIL', 'CHALOL31LOUHA', 'CHALOY631', 'CHALOY632',
                                     'CHALOY633', 'CIZE L61FLEYR', 'CREYSL71GEN.P', 'CREYSL72GEN.P', 'CUISEL31G.CHE', 'CURTIL61GROSN', 'FLEYRL61VOUGL',
                                     'FRON5L31LOUHA', 'FRON5L31G.CHE', 'GEN.PL61IZERN', 'GEN.PL61VOUGL', 'GEN.PY762', 'GEN.PY771', 'GROSNL61MACON',
                                     'H.PAUL71SSV.O', 'CIZE L61IZERN', 'LOUHAL31SSUSU', 'MACONL61ZJOUX', 'AISERL31MAGNY', 'MAGNYY633', 'C.FOUL31MERVA',
                                     'LOUHAL31PYMON', 'P.SAOL31RONCI', 'PYMONL31SAISS', 'MERVAL31SSUSU', 'CREYSL71SSV.O', 'CREYSL72SSV.O', 'GEN.PL71VIELM',
                                     'GEN.PL72VIELM', 'GEN.PL73VIELM', 'CUISEL31VOUGL', 'SAISSL31VOUGL', 'VOUGLY631', 'VOUGLY632', 'CURTIL61ZCUR5', 'BOISSL61ZJOUX', 'MAGNYL61ZMAGN',
                                     'PYMONL61VOUGL', 'PYMONY631', 'PYMONY632']


    list_hubs_test=[ 'VIELMP7', 'H.PAUP7', 'SSV.OP7','NAVILP3']#[ 'CPVANP6', 'CHALOP6', 'GROSNP6', 'VIELMP7', 'H.PAUP7', 'SSV.OP7','NAVILP3']#'P.SAOP3',

    assert(set(list_nodes_constrained_path_test).intersection(set(nodes_constrained_path))==set(nodes_constrained_path))
    assert (set(list_lines_contrained_path_test).intersection(set(lines_constrained_path))==set(lines_constrained_path))
    assert (set(list_nodes_dispatch_path_test).intersection(set(list_nodes_dispatch_path))==set(list_nodes_dispatch_path))
    assert (set(list_lines_redispatch_path_test).intersection(set(lines_redispatch))==set(lines_redispatch))
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
    # --- Instantiate Classifier ---
    classifier = ActionClassifier(grid2op_action_space=env.action_space)

    # Check rules for each action
    lines_reco_maintenance=[]
    paths=((lines_constrained_path, nodes_constrained_path), (lines_dispatch, nodes_dispatch_path))

    # Instantiate the validator with context
    validator = ActionRuleValidator(
        obs=obs,
        action_space=env.action_space,
        classifier=classifier,
        hubs=hubs,
        paths=paths,
        by_description=check_with_action_description
    )

    # Call the categorization method, passing simulation context
    actions_to_filter, actions_unfiltered = validator.categorize_actions(
        dict_action=dict_action,
        timestep=timestep,
        defauts=lines_defaut,
        overload_ids=lines_overloaded_ids,
        lines_reco_maintenance=lines_reco_maintenance
    )

    n_actions = len(dict_action.keys())
    n_actions_filtered = len(actions_to_filter.keys())
    n_actions_unfiltered = len(actions_unfiltered.keys())
    n_actions_badly_filtered = len([id for id, act_filter_content in actions_to_filter.items() if act_filter_content["is_rho_reduction"]])

    # Could also directly compare to saved dictionaries "actions_to_filter_expert_rules.json" and "actions_unfiltered_expert_rules.json"
    assert(n_actions == 102)
    #assert(n_actions_filtered == 65)
    assert (n_actions_filtered == 71)
    assert(n_actions_unfiltered == n_actions - n_actions_filtered)
    #assert(n_actions_badly_filtered == 4)
    assert(n_actions_badly_filtered == 7)# Opening OC 'MAGNY3TR633 DJ_OC' in the substation 'MAGNYP3'. This action is filtered because of the significant delta flow threshold.
    # If "ThresholdReportOfLine" is reduced from 0.2 to 0.05, then this action is not filtered anymore, and everything works as expected


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

    isolated_subs_init=set([comp.pop() for comp in comps_init if len(comp)==1])
    isolated_subs_wo_overload = set([comp.pop() for comp in comps_wo_max_overload if len(comp) == 1])

    islanded_subs=[obs.name_sub[int(sub_text.split('_')[1])] for sub_text in isolated_subs_wo_overload-isolated_subs_init]
    assert('TILLEP3' in islanded_subs)

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

    islanded_subs_test=set(['NAVILP3', 'P.SAOP3', 'RONCIP3', 'AISERP3', 'BEON P3', 'C.FOUP3'])
    n_subs_test=len(islanded_subs_test)
    islanded_subs_1_comp=[comp for comp in comps_wo_max_overload if len(comp)==n_subs_test][0]
    islanded_subs_1 = set([obs.name_sub[int(sub_text.split('_')[1])] for sub_text in islanded_subs_1_comp])

    islanded_sub_2= obs.name_sub[int(list(comp_overloads[2])[0].split('_')[1])]

    islanded_subs_1_prime_comp=[comp for comp in comp_overloads if len(comp)==n_subs_test][0]
    islanded_subs_1_prime = set([obs.name_sub[int(sub_text.split('_')[1])] for sub_text in islanded_subs_1_prime_comp])
    assert (islanded_subs_1 == islanded_subs_1_prime == islanded_subs_test) #TODO

    isolated_subs_init = set([comp.pop() for comp in comps_init if len(comp) == 1])
    isolated_subs_wo_overload = set([comp.pop() for comp in comp_overloads if len(comp) == 1])
    islanded_subs = [obs.name_sub[int(sub_text.split('_')[1])] for sub_text in
                     isolated_subs_wo_overload - isolated_subs_init]
    assert ('TILLEP3' in islanded_subs)

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

    assert('TILLEP3' in identified_subs_broken_apart)

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

    islanded_subs_test=set(['NAVILP3', 'P.SAOP3', 'RONCIP3', 'AISERP3', 'BEON P3', 'C.FOUP3'])
    n_subs_test=len(islanded_subs_test)
    identified_subs_broken_apart = get_subs_islanded_by_overload_disconnections(obs, comps_init, comp_overloads,line_overloaded)
    #islanded_subs_1_comp=[comp for comp in comps_wo_max_overload if len(comp)==n_subs_test][0]


    assert(set(identified_subs_broken_apart)==islanded_subs_test)

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


###########
# TEST reproductibilité prioritized actions
#1)
#date = datetime(2024, 12, 7)
#timestep = 22
#lines_defaut = ["CPVANY633"]
#['180c19aa-762d-4d6f-a74c-4fd5432aa5d1', '3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_18', '3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_75', '623e78ac-ae42-48d7-a5a3-7ec3fca7bb9b', 'disco_BEON L31CPVAN']

# --- Test Case Definitions ---
# Each tuple represents: (test_id, date_str, timestep, lines_defaut_list, expected_keys_set)
# Add your known scenarios and their expected prioritized action keys here.
REPRODUCIBILITY_CASES = [
    (
        "Case_CPVANY633_T22",
        "2024-12-7",
        22,
        ["CPVANY633"],
        # Replace with expected keys for this second case
        {'180c19aa-762d-4d6f-a74c-4fd5432aa5d1', '3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_18', '3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_75', '623e78ac-ae42-48d7-a5a3-7ec3fca7bb9b', 'disco_BEON L31CPVAN'}
        #TODO inspect this
        #warning not always reproductible...
       #'623e78ac-ae42-48d7-a5a3-7ec3fca7bb9b' vs 'disco_CPVANL61ZMAGN',
    ),
    (
        "Case_C.REGL61ZMAGN_T1",
        "2024-9-19",
        1,
        ["C.REGL61ZMAGN"],
        # Replace with expected keys for this second case
        {'reco_CHALOL31LOUHA','reco_BOISSL61GEN.P','466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_8',
         '466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_23','5e41ee90-d8cc-4900-800a-ebb8fe30bd20_variant_52'}
         #'3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40','01495164-2bf3-48d4-8b51-9276ca50386e_VIELMP6'}

    ),
        #  Simulating effectiveness...
        #✅ Rho reduction from [1.02 1.03] to [0.98 0.99]. New max rho is 0.99 on line C.SAUL31ZCRIM.
        #    Effective: CHALOL31LOUHA (Score: 0.13)
        #✅ Rho reduction from [1.02 1.03] to [1.01 1.01]. New max rho is 1.01 on line C.SAUL31ZCRIM.
        #    Effective: BOISSL61GEN.P (Score: 0.08)
        #✅ Rho reduction from [1.02 1.03] to [0.62 0.63]. New max rho is 0.69 on line P.SAOL31RONCI.
        #    Effective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_23 at C.REGP6 (hub: False)
        #✅ Rho reduction from [1.02 1.03] to [0.59 0.59]. New max rho is 0.68 on line P.SAOL31RONCI.
        #    Effective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_8 at C.REGP6 (hub: False)
        #✅ Rho reduction from [1.02 1.03] to [0.95 0.96]. New max rho is 0.96 on line C.SAUL31ZCRIM.
        #    Effective node split: 5e41ee90-d8cc-4900-800a-ebb8fe30bd20_variant_52 at PYMONP6 (hub: False)
        #✅ Rho reduction from [1.02 1.03] to [0.95 0.96]. New max rho is 0.96 on line C.SAUL31ZCRIM.
        #    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40 at VIELMP6 (hub: True)
        #✅ Rho reduction from [1.02 1.03] to [0.99 1.  ]. New max rho is 1.00 on line C.SAUL31ZCRIM.
        #    Effective node split: 256668ce-2a62-46c0-ba88-8001837b497f_VOUGLP6_variant_6 at VOUGLP6 (hub: False)
        #✅ Rho reduction from [1.02 1.03] to [1.01 1.02]. New max rho is 1.02 on line C.SAUL31ZCRIM.
        #    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_42 at VIELMP6 (hub: True)
        #    Ineffective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_75 at CPVANP6 (hub: False)
        #✅ Rho reduction from [1.02 1.03] to [0.95 0.95]. New max rho is 0.95 on line C.SAUL31ZCRIM.
        #    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_18 at CPVANP6 (hub: False)
    (
        "Case_CPVANL61ZMAGN_T1",
        "2024-9-19",
        1,
        ["CPVANL61ZMAGN"],
    {'reco_CHALOL31LOUHA','reco_BOISSL61GEN.P', '466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_8',
     '3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40','466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_23'}
    ),
#    Simulating effectiveness...
#✅ Rho reduction from [1.03] to [0.92]. New max rho is 0.92 on line P.SAOL31RONCI.
#    Effective: CHALOL31LOUHA (Score: 0.12)
#✅ Rho reduction from [1.03] to [1.01]. New max rho is 1.01 on line P.SAOL31RONCI.
#    Effective: BOISSL61GEN.P (Score: 0.08)
#✅ Rho reduction from [1.03] to [0.5]. New max rho is 0.72 on line C.SAUL31ZCRIM.
#    Effective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_8 at C.REGP6 (hub: False)
#✅ Rho reduction from [1.03] to [0.98]. New max rho is 0.98 on line P.SAOL31RONCI.
#    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40 at VIELMP6 (hub: True)
#✅ Rho reduction from [1.03] to [0.96]. New max rho is 0.96 on line P.SAOL31RONCI.
#    Effective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_23 at C.REGP6 (hub: False)
#✅ Rho reduction from [1.03] to [0.99]. New max rho is 0.99 on line P.SAOL31RONCI.
#    Effective node split: 5e41ee90-d8cc-4900-800a-ebb8fe30bd20_variant_52 at PYMONP6 (hub: False)
#✅ Rho reduction from [1.03] to [1.01]. New max rho is 1.01 on line P.SAOL31RONCI.
#    Effective node split: 256668ce-2a62-46c0-ba88-8001837b497f_VOUGLP6_variant_6 at VOUGLP6 (hub: False)
#✅ Rho reduction from [1.03] to [1.01]. New max rho is 1.01 on line P.SAOL31RONCI.
#    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_42 at VIELMP6 (hub: True)
#    Ineffective node split: 463179d8-be4e-4293-b42c-b53db763272b_variant_203 at GEN.PP6 (hub: False)
#✅ Rho reduction from [1.03] to [0.91]. New max rho is 0.91 on line P.SAOL31RONCI.
#    Effective node split: 180c19aa-762d-4d6f-a74c-4fd5432aa5d1 at CPVANP3 (hub: False)

    (
        "Case_MAGNYY633_T14",
        "2024-12-9",
        14,
        ["MAGNYY633"],
        {'reco_CHALOL31LOUHA','reco_BOISSL61GEN.P', '466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_23', '3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40', '5e41ee90-d8cc-4900-800a-ebb8fe30bd20_variant_52'}#,'01495164-2bf3-48d4-8b51-9276ca50386e_VIELMP6'}
    ),

#Evaluating 2 potential reconnections...
#  Simulating effectiveness...
#✅ Rho reduction from [1.03 1.03] to [0.98 0.98]. New max rho is 0.98 on line C.SAUL31ZCRIM.
#    Effective: CHALOL31LOUHA (Score: 0.18)
#✅ Rho reduction from [1.03 1.03] to [1.02 1.02]. New max rho is 1.02 on line C.SAUL31ZCRIM.
#    Effective: BOISSL61GEN.P (Score: 0.17)
#✅ Rho reduction from [1.03 1.03] to [0.85 0.85]. New max rho is 0.85 on line C.SAUL31ZCRIM.
#    Effective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_23 at C.REGP6 (hub: True)
#✅ Rho reduction from [1.03 1.03] to [1. 1.]. New max rho is 1.00 on line C.SAUL31ZCRIM.
#    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40 at VIELMP6 (hub: True)
#    Ineffective node split: 5e41ee90-d8cc-4900-800a-ebb8fe30bd20_variant_52 at PYMONP6 (hub: False)
#    Ineffective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_42 at VIELMP6 (hub: True)
#    Ineffective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_8 at C.REGP6 (hub: True)
#✅ Rho reduction from [1.03 1.03] to [1.02 1.02]. New max rho is 1.02 on line C.SAUL31ZCRIM.
#    Effective node split: 256668ce-2a62-46c0-ba88-8001837b497f_VOUGLP6_variant_6 at VOUGLP6 (hub: False)

    (
        "Case_CHALOY631_T32",
        "2024-8-29",
        32,
        ["CHALOY631"],
        {'reco_CHALOY632', "reco_CURTIL61ZCUR5",#'reco_BOISSL61GEN.P' in the case max_null_flow_path_length=10 for add_relevant_null_flow_lines - red_only
         '3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_42', "3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_18",#'f344b395-9908-43c2-bca0-75c5f298465e_variant_190',
         'f344b395-9908-43c2-bca0-75c5f298465e_variant_18'}
    ),
#  Simulating effectiveness...
#✅ Rho reduction from [1.01 1.01] to [0.77 0.77]. New max rho is 0.77 on line CORGOL32MTAGN.
#    Effective: CHALOY632 (Score: 0.15)
#    Ineffective: CURTIL61ZCUR5 (Score: 0.01)
#✅ Rho reduction from [1.01 1.01] to [0.88 0.88]. New max rho is 0.88 on line CORGOL32MTAGN.
#    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_42 at VIELMP6 (hub: True)
#✅ Rho reduction from [1.01 1.01] to [0.88 0.88]. New max rho is 0.88 on line CORGOL32N.GEO.
#    Effective node split: f344b395-9908-43c2-bca0-75c5f298465e_variant_18 at COUCHP6 (hub: True)
#    Ineffective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_18 at CPVANP6 (hub: False)
#✅ Rho reduction from [1.01 1.01] to [0.94 0.94]. New max rho is 0.94 on line CORGOL32MTAGN.
#    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40 at VIELMP6 (hub: True)
#✅ Rho reduction from [1.01 1.01] to [0.82 0.82]. New max rho is 0.99 on line C.REGL61ZMAGN.
#    Effective node split: f344b395-9908-43c2-bca0-75c5f298465e_variant_190 at COUCHP6 (hub: True)
#✅ Rho reduction from [1.01 1.01] to [0.95 0.95]. New max rho is 0.95 on line CORGOL32N.GEO.
#    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_75 at CPVANP6 (hub: False)


    (
        "Case_P.SAOL31RONCI_T47",
        "2024-8-28",
        47,
        ["P.SAOL31RONCI"],
        {'reco_CHALOL31LOUHA', 'reco_GEN.PY762', 'node_merging_PYMONP3', '3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_42','466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_8'}
    ),
#Evaluating 4 potential reconnections...
#  Skipping CHALOY632: Path potentially blocked by CHALOL31LOUHA.
#  Simulating effectiveness...
#✅ Rho reduction from [1.15] to [0.91]. New max rho is 0.91 on line BEON L31CPVAN.
#    Effective: CHALOL31LOUHA (Score: 0.14)
#✅ Rho reduction from [1.15] to [1.11]. New max rho is 1.11 on line BEON L31CPVAN.
#    Effective: GEN.PY762 (Score: 0.03)
#    Ineffective: BOISSL61GEN.P (Score: 0.03)
#-- Verifying relevant node merging ---
#Evaluating node merging for 18 substations...
#✅ Rho reduction from [1.15] to [0.87]. New max rho is 0.87 on line BEON L31CPVAN.
#  Effective node merge: node_merging_PYMONP3
#
#✅ Rho reduction from [1.15] to [1.08]. New max rho is 1.08 on line BEON L31CPVAN.
#    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_42 at VIELMP6 (hub: True)
#✅ Rho reduction from [1.15] to [1.04]. New max rho is 1.04 on line BEON L31CPVAN.
#    Effective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_8 at C.REGP6 (hub: False)
#✅ Rho reduction from [1.15] to [1.09]. New max rho is 1.09 on line BEON L31CPVAN.
#    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40 at VIELMP6 (hub: True)
#✅ Rho reduction from [1.15] to [1.08]. New max rho is 1.08 on line BEON L31CPVAN.
#    Effective node split: f344b395-9908-43c2-bca0-75c5f298465e_variant_190 at COUCHP6 (hub: False)
#    Ineffective node split: 6ba7e0a3-2c1a-4dd5-b85c-5cf83d8c0358_variant_53 at GROSNP6 (hub: False)
#    Ineffective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_75 at CPVANP6 (hub: False)
#✅ Rho reduction from [1.15] to [1.13]. New max rho is 1.13 on line BEON L31CPVAN.
#    Effective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_18 at CPVANP6 (hub: False)
#    Ineffective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_23 at C.REGP6 (hub: False)
#    Ineffective node split: 180c19aa-762d-4d6f-a74c-4fd5432aa5d1 at CPVANP3 (hub: True)
#    Ineffective node split: f344b395-9908-43c2-bca0-75c5f298465e_variant_18 at COUCHP6 (hub: False)
#✅ Rho reduction from [1.15] to [1.08]. New max rho is 1.08 on line BEON L31CPVAN.
#    Effective node split: 256668ce-2a62-46c0-ba88-8001837b497f_VOUGLP6_variant_6 at VOUGLP6 (hub: False) => WARNING = illegal PYMONL61VOUGL reconnection



    (
        "Case_COUCHL31VOSNE_T18",
        "2024-11-27",
        18,
        ["COUCHL31VOSNE"],
        {'reco_CHALOY632', 'reco_VIELMY762', '3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_42', 'f344b395-9908-43c2-bca0-75c5f298465e_variant_18', '3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40'}
    ),#divergence now with new library versions in load flow after contingency

        (
        "Case_CHALOL61CPVAN_T9",
        "2024-12-7",
        9,
        ["CHALOL61CPVAN"],
        {'reco_BOISSL61GEN.P', '466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_8', '3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_75',#'256668ce-2a62-46c0-ba88-8001837b497f_VOUGLP6_variant_6',
         '5e41ee90-d8cc-4900-800a-ebb8fe30bd20_variant_52', '01495164-2bf3-48d4-8b51-9276ca50386e_VIELMP6'}
    )
    #sometimes in this case, non reproductibility
    #"01495164-2bf3-48d4-8b51-9276ca50386e_VIELMP6" not considered and "disco_CPVANL61ZMAGN" added
    #Evaluating 1 potential reconnections...
    #  Simulating effectiveness...
    #✅ Rho reduction from [1.04] to [0.99]. New max rho is 0.99 on line C.REGL61ZMAGN.
    #    Effective: BOISSL61GEN.P (Score: 0.20)
    #Simulating effectiveness...
    #-- Verifying relevant node splitting ---
    #  Simulating effectiveness...
    #ERROR: Candidate action simulation failed in check_rho_reduction: [Grid2OpException BackendError DivergingPowerflow DivergingPowerflow()]
    #    Ineffective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_8 at C.REGP6 (hub: True)
    #    Ineffective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_75 at CPVANP6 (hub: True)
    #✅ Rho reduction from [1.04] to [0.71]. New max rho is 0.71 on line C.REGL61ZMAGN.
    #    Effective node split: 5e41ee90-d8cc-4900-800a-ebb8fe30bd20_variant_52 at PYMONP6 (hub: False)
    #✅ Rho reduction from [1.04] to [0.82]. New max rho is 0.82 on line C.REGL61ZMAGN.
    #    Effective node split: 256668ce-2a62-46c0-ba88-8001837b497f_VOUGLP6_variant_6 at VOUGLP6 (hub: False)
    #    Ineffective node split: 466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6_variant_23 at C.REGP6 (hub: True)
    #ERROR: Candidate action simulation failed in check_rho_reduction: [Grid2OpException BackendError DivergingPowerflow DivergingPowerflow()]
    #    Ineffective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_40 at VIELMP6 (hub: True)
    #ERROR: Candidate action simulation failed in check_rho_reduction: [Grid2OpException BackendError DivergingPowerflow DivergingPowerflow()]
    #    Ineffective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_CPVANP6_variant_18 at CPVANP6 (hub: True)
    #    Ineffective node split: 3617076a-a7f5-4f8a-9009-127ac9b85cff_VIELMP6_variant_42 at VIELMP6 (hub: True)
    #✅ Rho reduction from [1.04] to [1.01]. New max rho is 1.01 on line C.REGL61ZMAGN.
    #    Effective node split: 463179d8-be4e-4293-b42c-b53db763272b_variant_203 at GEN.PP6 (hub: True)

    # Add more test cases here following the same tuple format
    # (test_id, date_str, timestep, lines_defaut_list, expected_keys_set),

]


# --- Pytest Parametrization ---
@pytest.mark.parametrize(
    "test_id, date_str, timestep, lines_defaut, expected_keys_set",
    REPRODUCIBILITY_CASES
)
@pytest.mark.slow # Mark as slow because it loads the environment and runs analysis
def test_reproducibility(test_id, date_str, timestep, lines_defaut, expected_keys_set):

    prioritized_actions=run_analysis(
        analysis_date=date_str,
        current_timestep=timestep,
        current_lines_defaut=lines_defaut,
        backend=Backend.GRID2OP
    )

    # --- Assertion ---
    actual_keys_set = set(prioritized_actions.keys())

    assert actual_keys_set == expected_keys_set, \
        f"[{test_id}] Prioritized action keys mismatch.\nExpected: {sorted(list(expected_keys_set))}\nActual:   {sorted(list(actual_keys_set))}"

    print(f"--- Test Passed: {test_id} ---")


@pytest.mark.slow
def test_reproducibility_bare_env_small_grid_test():
    """
    Reproducibility test using bare_env_small_grid_test environment.
    
    This test overrides config_test.py settings to use:
    - Environment: bare_env_small_grid_test
    - Line defaut: P.SAOL31RONCI
    - Timestep: 0
    - Action space file: reduced_model_actions_test.json
    - Expected prioritized actions: 5 specific actions (see expected_keys_set)
    
    NOTE: conftest.py replaces expert_op4grid_recommender.config with tests.config_test
    via sys.modules, so we are actually modifying tests/config_test.py values here.
    Config overrides are restored after test execution via try/finally.
    """
    # NOTE: Due to conftest.py replacing the config module via sys.modules,
    # 'expert_op4grid_recommender.config' actually points to 'tests.config_test'
    import expert_op4grid_recommender.config as config
    
    # Test parameters
    test_id = "Case_BareEnvSmallGrid_T0"
    timestep = 0
    lines_defaut = ["P.SAOL31RONCI"]
    
    # OVERRIDE config settings for this test
    # Save original values to restore after test
    original_env_name = config.ENV_NAME
    original_file_action_space = config.FILE_ACTION_SPACE_DESC
    original_action_file_path = config.ACTION_FILE_PATH
    original_timestep = config.TIMESTEP
    original_lines_defaut = config.LINES_DEFAUT
    
    try:
        # Override config for this test
        # These overrides modify the config that run_analysis will use
        config.ENV_NAME = "bare_env_small_grid_test"  # Override environment
        config.FILE_ACTION_SPACE_DESC = "reduced_model_actions_test.json"  # Override action file
        # CRITICAL: Must recompute ACTION_FILE_PATH since it depends on FILE_ACTION_SPACE_DESC
        config.ACTION_FILE_PATH = config.ACTION_SPACE_FOLDER / config.FILE_ACTION_SPACE_DESC
        config.TIMESTEP = timestep
        config.LINES_DEFAUT = lines_defaut
        
        # Expected prioritized actions for this scenario
        expected_keys_set = {
            '466f2c03-90ce-401e-a458-fa177ad45abc_C.REGP6', 'f344b395-9908-43c2-bca0-75c5f298465e_COUCHP6',
             'node_merging_PYMONP3', 'reco_CHALOL31LOUHA', 'reco_GEN.PL73VIELM'
        }
        
        print(f"\n--- Running: {test_id} ---")
        print(f"Config file being used: {config.__file__}")
        print(f"Environment: {config.ENV_NAME}")
        print(f"Action file path: {config.ACTION_FILE_PATH}")
        print(f"Timestep: {timestep}")
        print(f"Line defaut: {lines_defaut}")
        
        # Verify the config values are what we expect BEFORE calling run_analysis
        assert config.ENV_NAME == "bare_env_small_grid_test", \
            f"Config override failed! ENV_NAME is '{config.ENV_NAME}' instead of 'bare_env_small_grid_test'"
        assert "reduced_model_actions_test.json" in str(config.ACTION_FILE_PATH), \
            f"Config override failed! ACTION_FILE_PATH is '{config.ACTION_FILE_PATH}'"
        
        # Run analysis - will now use overridden config values
        prioritized_actions = run_analysis(
            analysis_date=None,  # None means bare environment
            current_timestep=timestep,
            current_lines_defaut=lines_defaut,
            backend=Backend.GRID2OP
        )
        
        # Get actual keys from result
        actual_keys_set = set(prioritized_actions.keys())
        
        # Print results for debugging
        print(f"\nExpected actions: {sorted(list(expected_keys_set))}")
        print(f"Actual actions:   {sorted(list(actual_keys_set))}")
        
        # Assertion with detailed error message
        assert actual_keys_set == expected_keys_set, \
            f"[{test_id}] Prioritized action keys mismatch.\n" \
            f"Expected ({len(expected_keys_set)} actions): {sorted(list(expected_keys_set))}\n" \
            f"Actual ({len(actual_keys_set)} actions):   {sorted(list(actual_keys_set))}\n" \
            f"Missing: {sorted(list(expected_keys_set - actual_keys_set))}\n" \
            f"Extra: {sorted(list(actual_keys_set - expected_keys_set))}"
        
        print(f"\n✅ Test Passed: {test_id}")
        print(f"All {len(actual_keys_set)} prioritized actions match expected output.")
        
    finally:
        # IMPORTANT: Restore original config values after test
        # This ensures other tests are not affected by our overrides
        config.ENV_NAME = original_env_name
        config.FILE_ACTION_SPACE_DESC = original_file_action_space
        config.ACTION_FILE_PATH = original_action_file_path
        config.TIMESTEP = original_timestep
        config.LINES_DEFAUT = original_lines_defaut