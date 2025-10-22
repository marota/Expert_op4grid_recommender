# tests/test_expert_op4grid_analyzer.py

import pytest
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, List, Tuple, Optional, Callable, Set

# --- Test Setup: Add Project Root to Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mock Objects ---
# (Keep all Mock classes: MockAction, MockActionObject, MockActionSpace, MockObservation, MockEnv, MockAlphaDeesp, MockGraph, MockOverflowGraph, MockDistributionGraph)
# ... (Previous Mock class definitions are assumed here) ...
class MockAction:
    def __init__(self, **kwargs):
        defaults = {'line_or_set_bus': [0], 'line_ex_set_bus': [0], 'line_set_status': [0], 'load_set_bus': [0], 'line_or_change_bus': [0], 'line_ex_change_bus': [0], 'line_change_status': [0], 'load_change_bus': [0], '_set_topo_vect': [0], '_topo_vect_to_sub': [0]}
        for key, value in defaults.items(): setattr(self, key, np.array(kwargs.get(key, value)))
class MockActionObject:
    def __init__(self, substations_id=None, lines_ex_id=None, lines_or_id=None, set_line_status=None):
        self.substations_id = substations_id or []; self.lines_ex_id = lines_ex_id or {}; self.lines_or_id = lines_or_id or {}; self.set_line_status = set_line_status or []; self.content = {'set_line_status': self.set_line_status, 'set_bus': {'substations_id': self.substations_id}}
    def __add__(self, other): return self
    def get_topological_impact(self):
        subs_impacted = np.zeros(10, dtype=bool)
        if self.substations_id:
            sub_id = self.substations_id[0][0];
            if sub_id < len(subs_impacted): subs_impacted[sub_id] = True
        return [], subs_impacted
    def impact_on_objects(self): # Added method
        assigned_bus_list = []
        if self.substations_id:
            sub_id = self.substations_id[0][0]
            assigned_bus_list.append({'substation': sub_id})
        return {"topology": {"assigned_bus": assigned_bus_list}}
    def as_dict(self): return self.content
class MockActionSpace:
    def __call__(self, action_dict):
        set_bus = action_dict.get("set_bus", {}); # Return MockActionObject for discoverer tests
        return MockActionObject(substations_id=set_bus.get("substations_id"), lines_ex_id=set_bus.get("lines_ex_id"), lines_or_id=set_bus.get("lines_or_id"), set_line_status=action_dict.get("set_line_status"))
class MockObservation:
    def __init__(self, **kwargs):
        self.name_sub = np.array(kwargs.get('name_sub', [])); self.sub_topologies = kwargs.get('sub_topologies', {}); self.topo_vect = np.array(kwargs.get('topo_vect', [])); sub_info_arg = kwargs.get('sub_info'); self.sub_info = np.array(sub_info_arg) if sub_info_arg is not None else np.ones(len(self.name_sub), dtype=int); self.name_line = np.array(kwargs.get('name_line', [])); num_lines = len(self.name_line); self.rho = kwargs.get('rho', np.zeros(num_lines)); self.line_or_to_subid = np.array(kwargs.get('line_or_to_subid', np.zeros(num_lines, dtype=int))); self.line_ex_to_subid = np.array(kwargs.get('line_ex_to_subid', np.zeros(num_lines, dtype=int))); self.line_or_bus = np.array(kwargs.get('line_or_bus', np.ones(num_lines, dtype=int))); self.line_ex_bus = np.array(kwargs.get('line_ex_bus', np.ones(num_lines, dtype=int))); self.theta_or = np.array(kwargs.get('theta_or', np.zeros(num_lines))); self.theta_ex = np.array(kwargs.get('theta_ex', np.zeros(num_lines))); self.line_status = np.ones(num_lines, dtype=bool)
        class MockLoadP:
            def __init__(self, values): self._values = np.array(values if values is not None else [100.0])
            def sum(self): return self._values.sum()
        self.load_p = MockLoadP(kwargs.get('load_values')); self._simulate_return = kwargs.get('simulate_return')
    def sub_topology(self, sub_id): return self.sub_topologies.get(sub_id, [])
    def simulate(self, action, time_step): return self._simulate_return or (self, 0.0, False, {"exception": []})
    def __add__(self, action):
        new_attrs = self.__dict__.copy(); new_topo = self.topo_vect.copy()
        if hasattr(action, 'substations_id') and action.substations_id:
            for sub_id, topo in action.substations_id:
                if sub_id < len(self.sub_info):
                    start = int(np.sum(self.sub_info[:sub_id])); length = int(self.sub_info[sub_id])
                    if start + length <= len(new_topo): new_topo[start:start+length] = topo
        new_attrs['topo_vect'] = new_topo;
        if '_simulate_return' in new_attrs: del new_attrs['_simulate_return']
        if 'load_p' in new_attrs: del new_attrs['load_p']
        return MockObservation(**new_attrs, load_values=self.load_p._values)
    def get_energy_graph(self): return None
    def get_obj_connect_to(self, substation_id):
         if substation_id < 2: return {'lines_or_id': [substation_id], 'lines_ex_id': [max(0, substation_id-1)]}
         else: return {'lines_or_id': [], 'lines_ex_id': []}
class MockEnv:
    def __init__(self, name_line=None, maintenance_array=None, name_sub=None):
        self.name_line = name_line if name_line is not None else ["L1", "L2", "L3"]; self.name_sub = name_sub if name_sub is not None else ["S1", "S2"]; maint_arr = maintenance_array if maintenance_array is not None else np.zeros((2, len(self.name_line)), dtype=bool)
        self.chronics_handler = type('Chronics', (), {'real_data': type('RealData', (), {'data': type('Data', (), {'maintenance_handler': type('Maintenance', (), {'array': maint_arr})()})()})()})(); self.action_space = MockActionSpace()
    def get_obs(self): return MockObservation(name_line=self.name_line, name_sub=self.name_sub)
class MockAlphaDeesp:
    def rank_current_topo_at_node_x(self, g, sub_id, **kwargs): return 1.0 if sub_id == 0 else 0.5
class MockGraph:
    def __init__(self, edge_data=None): self.edge_data = edge_data or {}; self._edges = list(self.edge_data.keys())
    def get_edge_data(self, u, v): return self.edge_data.get((u, v))
    def has_edge(self, u, v): return (u,v) in self._edges or (v,u) in self._edges
class MockOverflowGraph:
    def __init__(self, edge_data=None): self.g = MockGraph(edge_data=edge_data)
class MockDistributionGraph:
    def __init__(self): self.red_loops = pd.DataFrame(columns=["Path"])
    def get_dispatch_edges_nodes(self, only_loop_paths=False): return [], []
    def get_constrained_edges_nodes(self): return [], [], [], []


# Import the new validator class
from expert_op4grid_recommender.action_evaluation.rules import ActionRuleValidator
from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

## Section 3: ActionRuleValidator Method Unit Tests ##
# These tests now instantiate ActionRuleValidator

# Fixture to create a validator instance with basic mocks
@pytest.fixture
def basic_validator():
    mock_obs = MockObservation(name_sub=["S0", "S1", "S2"], name_line=["L1", "L2", "L3"])
    mock_paths = ( (["L1"], ["S0"]), (["L2"], ["S1"]) ) # (constrained), (dispatch)
    return ActionRuleValidator(
        obs=mock_obs,
        action_space=MockActionSpace(),
        classifier=ActionClassifier(MockActionSpace()),
        hubs=["S0"],
        paths=mock_paths,
        by_description=True
    )

def test_validator_localize_line_action(basic_validator):
    assert basic_validator.localize_line_action(["L1"]) == "constrained_path"
    assert basic_validator.localize_line_action(["L2"]) == "dispatch_path"
    assert basic_validator.localize_line_action(["L3"]) == "out_of_graph"

def test_validator_localize_coupling_action(basic_validator):
    assert basic_validator.localize_coupling_action(["S0"]) == "hubs"
    # Need to adjust paths fixture if we want to test constrained path node localization
    # basic_validator.nodes_constrained_path = ["S1"] # Example adjustment
    # assert basic_validator.localize_coupling_action(["S1"]) == "constrained_path"
    assert basic_validator.localize_coupling_action(["S1"]) == "dispatch_path" # Based on fixture paths
    assert basic_validator.localize_coupling_action(["S2"]) == "out_of_graph"

def test_validator_check_rules(basic_validator):
    assert basic_validator.check_rules("open_line", "dispatch_path", [])[1] == "No line disconnection on dispatch path"
    assert basic_validator.check_rules("close_line", "constrained_path", [])[1] == "No line reconnection on constrained path"
    assert basic_validator.check_rules("open_coupling", "dispatch_path", [[1, 1]])[1] == "No node splitting on dispatch path" # Check single node case
    assert basic_validator.check_rules("open_coupling", "dispatch_path", [[1, 2]])[0] is False # Multi-node case is allowed

def test_validator_verify_action_basic_checks(basic_validator):
    # Test disconnect already disconnected
    basic_validator.obs.line_status = np.array([False, True, True]) # L1 is disconnected
    # FIX: Use French keyword "Ouverture"
    action_desc_open_l1 = {"description_unitaire": "Ouverture L1",
                           "content": {"set_bus": {"lines_ex_id": {"L1": -1}}}}
    filter_open, rule_open = basic_validator.verify_action(action_desc_open_l1, [])
    assert filter_open is True and rule_open == "No disconnection of a line already disconnected"

    # Test reconnect already connected
    basic_validator.obs.line_status = np.array([True, True, True]) # All connected
    # FIX: Use French keyword "Fermeture"
    action_desc_close_l1 = {"description_unitaire": "Fermeture L1",
                            "content": {"set_bus": {"lines_or_id": {"L1": 1}, "lines_ex_id": {"L1": 1}}}}
    filter_close, rule_close = basic_validator.verify_action(action_desc_close_l1, [])
    assert filter_close is True and rule_close == "No reconnection of a line already connected"

def test_validator_verify_action_rules_check(): # Removed fixture injection for isolation
    """Tests that the rule 'No line disconnection on dispatch path' is correctly applied."""
    # Setup mocks specifically for this test
    mock_obs = MockObservation(
        name_sub=["S0", "S1", "S2"],
        name_line=["L1", "L2", "L3"],
        line_status=np.array([True, True, True]) # All lines connected
    )
    mock_paths = (
        (["L1"], ["S0"]),  # Constrained: Line L1, Node S0
        (["L2"], ["S1"])   # Dispatch: Line L2, Node S1
    )
    validator = ActionRuleValidator(
        obs=mock_obs,
        action_space=MockActionSpace(),
        classifier=ActionClassifier(MockActionSpace()),
        hubs=["S0"],
        paths=mock_paths,
        by_description=True
    )

    # Action description using the correct keyword
    action_desc_open_l2 = {
        "description_unitaire": "Ouverture L2", # Correct keyword
        "content": {"set_bus": {"lines_ex_id": {"L2": -1}}}
    }

    # Call the method on the locally created instance
    filter_rule, rule_name = validator.verify_action(action_desc_open_l2, [])

    # Assert the expected outcome
    assert filter_rule is True, f"Expected filter_rule to be True, but got {filter_rule}"
    assert rule_name == "No line disconnection on dispatch path", f"Expected rule 'No line disconnection on dispatch path', but got {rule_name}"

