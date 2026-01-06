# tests/test_expert_op4grid_analyzer.py

import pytest
import os
import sys
import numpy as np
import pandas as pd # Import pandas for MockDistributionGraph
import networkx as nx

# --- Test Setup: Add Project Root to Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mock Objects ---
# (Mock classes: MockAction, MockActionObject, MockActionSpace, MockObservation, MockEnv, MockAlphaDeesp, MockGraph, MockOverflowGraph remain the same as the previous version)
class MockAction:
    def __init__(self, **kwargs):
        defaults = {'line_or_set_bus': [0], 'line_ex_set_bus': [0], 'line_set_status': [0], 'load_set_bus': [0], 'line_or_change_bus': [0], 'line_ex_change_bus': [0], 'line_change_status': [0], 'load_change_bus': [0], '_set_topo_vect': [0], '_topo_vect_to_sub': [0]}
        for key, value in defaults.items(): setattr(self, key, np.array(kwargs.get(key, value)))

class MockActionObject:
    """Mocks the higher-level action object returned by an action_space."""
    def __init__(self, substations_id=None, lines_ex_id=None, lines_or_id=None, set_line_status=None):
        self.substations_id = substations_id or []
        self.lines_ex_id = lines_ex_id or {}
        self.lines_or_id = lines_or_id or {}
        self.set_line_status = set_line_status or []
        self.content = {'set_line_status': self.set_line_status, 'set_bus': {'substations_id': self.substations_id}}

    def __add__(self, other):
        return self # For tests, a simple return is sufficient.

    # Modify this method
    def impact_on_objects(self):
        """Mocks the impact_on_objects method, focusing on topology impact."""
        assigned_bus_list = []
        if self.substations_id:
            # Assumes the action affects only one substation for simplicity in mock
            sub_id = self.substations_id[0][0]
            # Create a dummy assignment structure expected by the code
            # We just need one entry with the correct substation ID
            assigned_bus_list.append({'substation': sub_id})
            # Add more dummy assignments if the code iterates over elements within the sub

        # Return a dictionary with the expected "assigned_bus" key
        return {
            "topology": {
                "assigned_bus": assigned_bus_list
                # 'substation_ids' key might no longer be needed if the code only uses 'assigned_bus'
            }
            # Add other keys like 'powerlines', 'loads' if needed
        }

    def as_dict(self):
        return self.content

class MockActionSpace:
    """Mocks the Grid2Op action_space callable."""
    def __call__(self, action_dict):
        set_bus = action_dict.get("set_bus", {})
        # Return MockActionObject containing necessary attributes
        return MockActionObject(
            substations_id=set_bus.get("substations_id"),
            lines_ex_id=set_bus.get("lines_ex_id"),
            lines_or_id=set_bus.get("lines_or_id"),
            set_line_status=action_dict.get("set_line_status") # Pass this through
        )# --- Updated MockObservation ---
class MockObservation:
    """A comprehensive mock for a Grid2Op observation object."""
    def __init__(self, **kwargs):
        self.name_sub = np.array(kwargs.get('name_sub', []))
        self.sub_topologies = kwargs.get('sub_topologies', {})
        self.topo_vect = np.array(kwargs.get('topo_vect', []))
        self.sub_info = np.array(kwargs.get('sub_info', np.ones(len(self.name_sub))))#[k for k, v in sorted(data.items()) for _ in v]
        self.name_line = np.array(kwargs.get('name_line', []))
        num_lines = len(self.name_line)
        self.rho = kwargs.get('rho', np.zeros(num_lines))
        # Add attributes needed for get_delta_theta_line
        self.line_or_to_subid = np.array(kwargs.get('line_or_to_subid', np.zeros(num_lines, dtype=int)))
        self.line_ex_to_subid = np.array(kwargs.get('line_ex_to_subid', np.zeros(num_lines, dtype=int)))
        self.line_or_bus = np.array(kwargs.get('line_or_bus', np.ones(num_lines, dtype=int)))
        self.line_ex_bus = np.array(kwargs.get('line_ex_bus', np.ones(num_lines, dtype=int)))
        self.theta_or = np.array(kwargs.get('theta_or', np.zeros(num_lines)))
        self.theta_ex = np.array(kwargs.get('theta_ex', np.zeros(num_lines)))
        # --- End added attributes ---
        self.line_status = np.ones(num_lines, dtype=bool)

        class MockLoadP:
            def __init__(self, values): self._values = np.array(values if values is not None else [100.0])
            def sum(self): return self._values.sum()
        self.load_p = MockLoadP(kwargs.get('load_values'))

        self._simulate_return = kwargs.get('simulate_return')

    def sub_topology(self, sub_id): return np.array(self.sub_topologies.get(sub_id, []))
    def simulate(self, action, time_step): return self._simulate_return or (self, 0.0, False, {"exception": []})
    def __add__(self, action):
        if hasattr(action, 'substations_id') and action.substations_id:
             new_topo = self.topo_vect.copy(); return MockObservation(**{**self.__dict__, 'topo_vect': new_topo}) # Pass existing attrs
        return self
    def get_energy_graph(self): return None
    # Add mock for get_obj_connect_to needed by get_theta_node
    def get_obj_connect_to(self, substation_id):
         # Return dummy indices, assuming lines 0, 1 connect if sub_id is 0 or 1
         if substation_id < 2:
             return {'lines_or_id': [substation_id], 'lines_ex_id': [substation_id-1 if substation_id > 0 else 1]}
         else:
             return {'lines_or_id': [], 'lines_ex_id': []}

class MockEnv:
    def __init__(self, name_line=None, maintenance_array=None, name_sub=None):
        self.name_line = name_line if name_line is not None else ["L1", "L2", "L3"]; self.name_sub = name_sub if name_sub is not None else ["S1", "S2"]; maint_arr = maintenance_array if maintenance_array is not None else np.zeros((2, len(self.name_line)), dtype=bool)
        self.chronics_handler = type('Chronics', (), {'real_data': type('RealData', (), {'data': type('Data', (), {'maintenance_handler': type('Maintenance', (), {'array': maint_arr})()})()})()})(); self.action_space = MockActionSpace()
    def get_obs(self): return MockObservation(name_line=self.name_line, name_sub=self.name_sub)

class MockAlphaDeesp:
    def rank_current_topo_at_node_x(self, g, sub_id, **kwargs): return 1.0 if sub_id == 0 else 0.5

class MockGraph(nx.MultiDiGraph):
    def __init__(self, edge_data=None, **attr):
        super().__init__(**attr)
        if edge_data:
            for (u, v), data in edge_data.items():
                # Check if data is a dictionary of attributes or a dictionary of edge_ids
                # If 'data' has integer keys (like 0, 1), it's likely {edge_key: {attributes}}

                if isinstance(data, dict):
                    for key, attributes in data.items():
                        if isinstance(key, int) and isinstance(attributes, dict):
                            # Case: {0: {'name': 'L1'}} -> key=0 is the edge_key (multigraph index)
                            self.add_edge(u, v, key=key, **attributes)
                        else:
                            # Case: {'weight': 10} -> standard attributes
                            # We can't unpack 'data' directly if it has int keys, so we check carefully
                            self.add_edge(u, v, **data)
                            break  # Stop loop if we treated 'data' as flat attributes
                else:
                    self.add_edge(u, v)

class MockOverflowGraph:
    def __init__(self, name_line=None,edge_data=None):
        self.g = MockGraph(edge_data=edge_data)

class MockDistributionGraph:
    def __init__(self): self.red_loops = pd.DataFrame(columns=["Path"])
    def get_dispatch_edges_nodes(self, only_loop_paths=False): return [], []
    def get_constrained_edges_nodes(self): return [], [], [], []

from expert_op4grid_recommender.action_evaluation.discovery import ActionDiscoverer
from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier
## Section 3: ActionDiscoverer Internal Method Tests ##
# These tests now use an instance of ActionDiscoverer

# Fixture to create a discoverer instance with basic mocks
@pytest.fixture
def basic_discoverer():
    mock_obs = MockObservation(name_sub=["A", "B", "C", "D"], name_line=["L1", "L2", "L3"], line_or_to_subid=[0, 1, 2], line_ex_to_subid=[1, 2, 3])
    mock_env = MockEnv(name_line=mock_obs.name_line, name_sub=mock_obs.name_sub)
    mock_g_overflow = MockOverflowGraph(edge_data={ (0, 1): {0:{"name":"L1"}}, (1, 2): {0:{"name":"L2"}} }) # Edges A-B, B-C
    mock_g_dist = MockDistributionGraph()
    return ActionDiscoverer( env=mock_env, obs=mock_obs,classifier=ActionClassifier(MockActionSpace()), obs_defaut=mock_obs, timestep=0, lines_defaut=[], lines_overloaded_ids=[], act_reco_maintenance=MockActionObject(), non_connected_reconnectable_lines=[], all_disconnected_lines=[], dict_action={}, actions_unfiltered=set(), hubs=[], g_overflow=mock_g_overflow, g_distribution_graph=mock_g_dist, simulator_data={})

def test_internal_is_sublist(basic_discoverer):
    assert basic_discoverer._is_sublist(["B", "C"], ["A", "B", "C", "D"]) is True
    assert basic_discoverer._is_sublist(["C", "A"], ["A", "B", "C", "D"]) is False

def test_internal_get_line_substations(basic_discoverer):
    assert basic_discoverer._get_line_substations("L2") == ("B", "C")

def test_internal_find_paths_for_line(basic_discoverer):
    line_subs = ("A", "B")
    paths = [["A", "B", "C"], ["B", "C", "D"], ["C", "D", "E"]]
    assert basic_discoverer._find_paths_for_line(line_subs, paths) == [["A", "B", "C"]]

def test_internal_get_active_edges_between(basic_discoverer):
    # Update discoverer's graph mock for this test
    basic_discoverer.g_overflow = MockOverflowGraph(edge_data={ (0, 1): {0: {"name": "L1"}, 1: {"name": "L1_dashed", "style": "dashed"}}, (1, 0): {0: {"name": "L1_rev"}} }) # A-B edges
    active = basic_discoverer._get_active_edges_between("A", "B")
    assert "L1" in active
    assert "L1_rev" in active
    assert "L1_dashed" not in active

def test_internal_has_blocking_disconnected_line(basic_discoverer):
    basic_discoverer.all_disconnected_lines = ["L2"]
    # Graph has A-B edge, but no B-C edge implicitly
    basic_discoverer.g_overflow = MockOverflowGraph(edge_data={ (0, 1): {0:{"name":"L1"}} })
    blocked, blocker = basic_discoverer._has_blocking_disconnected_line(["A", "B", "C"], "L1")
    assert blocked is True and blocker == "L2"
    # Now add a parallel active edge for L2 (B-C)
    basic_discoverer.g_overflow = MockOverflowGraph(edge_data={ (0, 1): {0:{"name":"L1"}}, (1,2): {0:{"name":"L3"}} })
    blocked, blocker = basic_discoverer._has_blocking_disconnected_line(["A", "B", "C"], "L1")
    assert blocked is False and blocker is None

def test_internal_check_other_reconnectable_line(basic_discoverer):
    basic_discoverer.all_disconnected_lines = ["L2"]
    # Path A-B-C exists, L1 connects A-B, L2 connects B-C
    # Case 1: No parallel edge for L2 -> blocked
    basic_discoverer.g_overflow = MockOverflowGraph(edge_data={ (0, 1): {0:{"name":"L1"}}})
    has_path, blocker = basic_discoverer._check_other_reconnectable_line_on_path("L1", [["A", "B", "C"]])
    assert has_path is False and blocker == "L2"
    # Case 2: Add parallel edge L3 for L2 -> not blocked
    basic_discoverer.g_overflow = MockOverflowGraph(edge_data={ (0, 1): {0:{"name":"L1"}}, (1,2): {0:{"name":"L3"}} })
    has_path, blocker = basic_discoverer._check_other_reconnectable_line_on_path("L1", [["A", "B", "C"]])
    assert has_path is True and blocker is None

## Section 4: Discovery Class Method Unit Tests (using fixture) ##

# Mock for rho reduction checks specific to discovery tests
def mock_rho_reduction_discovery(obs, timestep, act_defaut, action, *args, **kwargs):
    """Predictable mock: effective if action involves Sub0, disconnects L1, or reconnects L1."""
    is_effective = False
    # Check for node merge/split of Sub0
    if hasattr(action, 'substations_id') and action.substations_id and action.substations_id[0][0] == 0:
        is_effective = True
    # Check for disconnection of L1
    elif hasattr(action, 'lines_ex_id') and action.lines_ex_id.get("L1") == -1:
         is_effective = True
    elif hasattr(action, 'lines_or_id') and action.lines_or_id.get("L1") == -1:
         is_effective = True
    # Check for reconnection of L1
    #elif hasattr(action, 'set_line_status') and action.set_line_status == [("L1", 1)]:
    elif hasattr(action, 'lines_ex_id') and action.lines_ex_id.get("L1") == 1:
         if hasattr(action, 'lines_or_id') and action.lines_or_id.get("L1") == 1:
            is_effective = True

    return is_effective, obs

def mock_compute_node_splitting_action_score_value(overflow_graph, g_distribution_graph, node: int,
                                                  dict_edge_names_buses=None): return 1.0 if node == 0 else 0.5

def mock_edge_names_buses_dict(obs, action_topo_vect, sub_impacted_id): return {line_name:1 for line_name in obs.name_line}

@pytest.fixture
def discoverer_instance(monkeypatch):
    """Provides a configured ActionDiscoverer instance with mocks for testing discovery methods."""
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.check_rho_reduction", mock_rho_reduction_discovery)
    monkeypatch.setattr("expert_op4grid_recommender.utils.simulation.create_default_action", lambda *args: MockActionObject())
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery.AlphaDeesp_warmStart", lambda *args: MockAlphaDeesp())
    #def mock_id(desc, **kwargs): return desc.get("type", "unknown")
    #monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.classifier.identify_action_type", mock_id)
    def mock_sort(map_score):
        items = sorted(map_score.items(), key=lambda item: item[1]['score'], reverse=True)
        return {k: v['action'] for k,v in items}, [v.get('sub_impacted') or v.get('line_impacted') for k,v in items], [v['score'] for k,v in items]
    monkeypatch.setattr("expert_op4grid_recommender.utils.helpers.sort_actions_by_score", mock_sort)

    name_line=["L1", "L2"]
    mock_obs = MockObservation(name_sub=["Sub0", "Sub1", "Sub3"], name_line=name_line, sub_topologies={0: [1, 2, 2], 1: [1, 1], 2: [2, 1, 2]}, sub_info=[3,2,3], topo_vect=np.array([1,2,1, 1,1, 2,1,1]))
    mock_env = MockEnv(name_line=mock_obs.name_line, name_sub=mock_obs.name_sub)
    mock_g_overflow = MockOverflowGraph(edge_data={ (0, 1): {0:{"name":"L1","capacity":1}}, (1, 2): {0:{"name":"L2","capacity":2}} })
    mock_g_dist = MockDistributionGraph()

    discoverer = ActionDiscoverer( env=mock_env, obs=mock_obs, obs_defaut=mock_obs,classifier=ActionClassifier(MockActionSpace()),
                                   timestep=0, lines_defaut=[], lines_overloaded_ids=[0], act_reco_maintenance=MockActionObject(), non_connected_reconnectable_lines=["L1"], all_disconnected_lines=[],
                                   dict_action={
                                       "reco_L1": {
                                           "type": "close_line",
                                           "description_unitaire": "Fermeture L1",  # Added
                                           "content": {"set_bus": {"lines_ex_id": {"L1": 1}},
                                                       "lines_or_id": {"L1": 1}}},#{"set_line_status": [("L1", 1)], "set_bus": {}}},
                                       "disco_L1": {
                                           "type": "open_line",
                                           "description_unitaire": "Ouverture L1",  # Added
                                           "content": {"set_bus": {"lines_ex_id": {"L1": -1}}}},
                                       "merge_S0": {
                                           "type": "close_coupling",
                                           "description_unitaire": "Fermeture COUPL S0",  # Added
                                           "content": {"set_bus": {"substations_id": [(0, [1, 1, 1])]}}},
                                       "split_S0": {
                                           "type": "open_coupling",
                                           "description_unitaire": "Ouverture COUPL S0",  # Added
                                           "content": {"set_bus": {"substations_id": [(0, [1, 2, 2])]}}},
                                       "split_S1": {
                                           "type": "open_coupling",
                                           "description_unitaire": "Ouverture COUPL S1",  # Added
                                           "content": {"set_bus": {"substations_id": [(1, [1, 2])]}}},
                                       "disco_L2": {
                                           "type": "open_line",
                                           "description_unitaire": "Ouverture L2",  # Added
                                           "content": {"set_bus": {"lines_ex_id": {"L2": -1}}}},
                                   },
                      actions_unfiltered={"reco_L1", "disco_L1", "merge_S0", "split_S0", "split_S1", "disco_L2"}, hubs=["Sub0"], g_overflow=mock_g_overflow, g_distribution_graph=mock_g_dist, simulator_data={}, check_action_simulation=True )

    monkeypatch.setattr(discoverer, '_edge_names_buses_dict', mock_edge_names_buses_dict)
    monkeypatch.setattr(discoverer, 'compute_node_splitting_action_score_value', mock_compute_node_splitting_action_score_value)
    return discoverer

def test_discoverer_find_relevant_node_merging(discoverer_instance):
    discoverer_instance.find_relevant_node_merging(["Sub0", "Sub1", "Sub3"])
    assert len(discoverer_instance.identified_merges) == 2
    assert len(discoverer_instance.effective_merges) == 1
    assert discoverer_instance.effective_merges[0].substations_id[0][0] == 0

def test_discoverer_verify_relevant_reconnections(discoverer_instance, monkeypatch):
    def mock_path_check(*args): return True, None # Assume path is always clear
    monkeypatch.setattr(discoverer_instance, "_check_other_reconnectable_line_on_path", mock_path_check)
    discoverer_instance.verify_relevant_reconnections(lines_to_reconnect={"L1"}, red_loop_paths=[])
    assert "reco_L1" in discoverer_instance.identified_reconnections
    assert "L1" in discoverer_instance.effective_reconnections

def test_discoverer_find_relevant_disconnections(discoverer_instance):
    discoverer_instance.find_relevant_disconnections(lines_constrained_path_names=["L1"])
    assert "disco_L1" in discoverer_instance.identified_disconnections
    assert "disco_L1" in discoverer_instance.effective_disconnections
    assert "disco_L2" not in discoverer_instance.identified_disconnections # Not in constrained path

def test_discoverer_find_relevant_node_splitting(discoverer_instance):
    discoverer_instance.find_relevant_node_splitting(hubs_names=["Sub0"], nodes_blue_path_names=["Sub1"])
    assert "split_S0" in discoverer_instance.identified_splits
    assert "split_S1" in discoverer_instance.identified_splits
    assert len(discoverer_instance.effective_splits) == 1
    assert discoverer_instance.effective_splits[0].substations_id[0][0] == 0
    assert discoverer_instance.scores_splits[0] >= discoverer_instance.scores_splits[1] # Check sort order


# --- Add these new tests to your test_expert_op4grid_analyzer.py file ---

def test_internal_compute_node_splitting_action_score(discoverer_instance):
    """
    Dedicated test for the internal _compute_node_splitting_action_score method.
    Verifies that the score calculation logic works correctly based on mocks.
    """
    # Use the discoverer_instance fixture which already has mocks set up
    discoverer = discoverer_instance
    alpha_ranker = MockAlphaDeesp() # We know its behavior

    # Mock an action targeting Sub0 (index 0)
    action_sub0 = MockActionObject(substations_id=[(0, [1, 2])]) # Splitting action
    # Mock an action targeting Sub1 (index 1)
    action_sub1 = MockActionObject(substations_id=[(1, [1, 1])]) # Non-splitting/irrelevant action content

    # --- Test Case 1: Substation 0 (should get score 1.0 from MockAlphaDeesp) ---
    # Manually set up obs_defaut to ensure sub_info and topo_vect are suitable
    discoverer.obs_defaut = MockObservation(name_sub=["Sub0", "Sub1"], sub_info=[2, 2], topo_vect=[1, 1, 1, 1])
    score0 = discoverer.compute_node_splitting_action_score(action_sub0, 0, alpha_ranker)
    assert score0 == 1.0, "Score for Sub0 should be 1.0 based on mock ranker"

    # --- Test Case 2: Substation 1 (should get score 0.5 from MockAlphaDeesp) ---
    discoverer.obs_defaut = MockObservation(name_sub=["Sub0", "Sub1"], sub_info=[2, 2], topo_vect=[1, 1, 1, 1])
    score1 = discoverer.compute_node_splitting_action_score(action_sub1, 1, alpha_ranker)
    assert score1 == 0.5, "Score for Sub1 should be 0.5 based on mock ranker"

def test_internal_identify_and_score_node_splitting_actions(discoverer_instance, monkeypatch):
    """
    Dedicated test for the internal _identify_and_score_node_splitting_actions method.
    Verifies action identification, relevance check (hubs/blue path), and scoring.
    """
    discoverer = discoverer_instance
    alpha_ranker = MockAlphaDeesp()

    # Override dict_action and unfiltered actions for this specific test
    discoverer.dict_action["split_S2"]={"type": "open_coupling",
                                           "description_unitaire": "Ouverture COUPL S2",  # Added
                                           "content": {"set_bus": {"substations_id": [(2, [1, 2])]}}}
    #discoverer.dict_action = {
    #    "split_S0": {"type": "open_coupling", "content": {"set_bus": {"substations_id": [(0, [1, 2])]}}}, # Relevant (Hub)
    #    "split_S1": {"type": "open_coupling", "content": {"set_bus": {"substations_id": [(1, [1, 2])]}}}, # Relevant (Blue Path)
    #    "split_S2": {"type": "open_coupling", "content": {"set_bus": {"substations_id": [(2, [1, 2])]}}}, # Irrelevant
    #    "merge_S0": {"type": "close_coupling", "content": {"set_bus": {"substations_id": [(0, [1, 1])]}}}, # Wrong type
    #}
    discoverer.actions_unfiltered = set(discoverer.dict_action.keys()) # Assume all passed initial filter

    # Define hubs and blue path for relevance check
    hubs_names = ["Sub0"]
    nodes_blue_path_names = ["Sub1"]

    # Mock obs_defaut needed for name lookup
    discoverer.obs_defaut = MockObservation(name_sub=["Sub0", "Sub1", "Sub2"])

    # --- Execute the method under test ---
    map_score, ignored = discoverer.identify_and_score_node_splitting_actions(
        hubs_names, nodes_blue_path_names, alpha_ranker
    )

    # --- Assertions ---
    # Check identified and scored actions
    assert len(map_score) == 2, "Should identify 2 relevant splitting actions"
    assert "split_S0" in map_score
    assert "split_S1" in map_score
    assert map_score["split_S0"]["score"] == 1.0 # From mock ranker for sub_id 0
    assert map_score["split_S0"]["sub_impacted"] == "Sub0"
    assert map_score["split_S1"]["score"] == 0.5 # From mock ranker for sub_id 1
    assert map_score["split_S1"]["sub_impacted"] == "Sub1"

    # Check ignored actions
    assert len(ignored) == 5, "Should ignore 5 actions"
    ignored_ids = {k for k, v in discoverer.dict_action.items() if v in ignored}
    assert "reco_L1" in ignored_ids
    assert "disco_L1" in ignored_ids
    assert "disco_L2" in ignored_ids
    assert "merge_S0" in ignored_ids # Ignored because it's not "open_coupling" type
    assert "split_S2" in ignored_ids #Ignored because not on constrained path