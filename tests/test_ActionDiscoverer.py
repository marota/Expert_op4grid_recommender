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

        # Ensure content reflects all input arguments so .get() returns expected structure
        self.content = {
            'set_line_status': self.set_line_status,
            'set_bus': {
                'substations_id': self.substations_id,
                'lines_ex_id': self.lines_ex_id,
                'lines_or_id': self.lines_or_id
            }
        }

    # Add this method to satisfy the .get() call in the code under test
    def get(self, key, default=None):
        return self.content.get(key, default)

    def __add__(self, other):
        return self  # For tests, a simple return is sufficient.

    def impact_on_objects(self):
        """Mocks the impact_on_objects method, focusing on topology impact."""
        assigned_bus_list = []
        if self.substations_id:
            # Assumes the action affects only one substation for simplicity in mock
            sub_id = self.substations_id[0][0]
            # Create a dummy assignment structure expected by the code
            assigned_bus_list.append({'substation': sub_id})

        return {
            "topology": {
                "assigned_bus": assigned_bus_list
            }
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
        # Load and generator attributes
        self.name_load = np.array(kwargs.get('name_load', []))
        self.name_gen = np.array(kwargs.get('name_gen', []))
        self.load_to_subid = np.array(kwargs.get('load_to_subid', []), dtype=int) if kwargs.get('load_to_subid') is not None else np.array([], dtype=int)
        self.gen_to_subid = np.array(kwargs.get('gen_to_subid', []), dtype=int) if kwargs.get('gen_to_subid') is not None else np.array([], dtype=int)
        # Optional: pre-computed per-substation element lists for get_obj_connect_to
        self._obj_connect_to = kwargs.get('obj_connect_to', None)

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
    def get_obj_connect_to(self, substation_id):
         """Return element indices per substation. Uses explicit mapping if provided."""
         if self._obj_connect_to and substation_id in self._obj_connect_to:
             entry = self._obj_connect_to[substation_id]
             return {
                 'loads_id': entry.get('loads_id', []),
                 'generators_id': entry.get('generators_id', []),
                 'lines_or_id': entry.get('lines_or_id', []),
                 'lines_ex_id': entry.get('lines_ex_id', []),
             }
         # Fallback: auto-derive from line_or/ex_to_subid
         lines_or = [i for i, s in enumerate(self.line_or_to_subid) if s == substation_id]
         lines_ex = [i for i, s in enumerate(self.line_ex_to_subid) if s == substation_id]
         loads = [i for i, s in enumerate(self.load_to_subid) if s == substation_id]
         gens = [i for i, s in enumerate(self.gen_to_subid) if s == substation_id]
         return {'loads_id': loads, 'generators_id': gens,
                 'lines_or_id': lines_or, 'lines_ex_id': lines_ex}

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
    has_path,path, blocker = basic_discoverer._check_other_reconnectable_line_on_path("L1", [["A", "B", "C"]])
    assert has_path is False and blocker == "L2"
    # Case 2: Add parallel edge L3 for L2 -> not blocked
    basic_discoverer.g_overflow = MockOverflowGraph(edge_data={ (0, 1): {0:{"name":"L1"}}, (1,2): {0:{"name":"L3"}} })
    has_path,path, blocker = basic_discoverer._check_other_reconnectable_line_on_path("L1", [["A", "B", "C"]])
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
    monkeypatch.setattr("expert_op4grid_recommender.action_evaluation.discovery._default_check_rho_reduction", mock_rho_reduction_discovery)
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
    # Verify scores dict is populated for identified merges
    assert len(discoverer_instance.scores_merges) == 2
    for action_id, score in discoverer_instance.scores_merges.items():
        assert isinstance(score, float)

def test_discoverer_verify_relevant_reconnections(discoverer_instance, monkeypatch):
    def mock_path_check(*args): return True, ["Sub0", "Sub1", "Sub3"], None # Assume path is always clear
    monkeypatch.setattr(discoverer_instance, "_check_other_reconnectable_line_on_path", mock_path_check)
    discoverer_instance.verify_relevant_reconnections(lines_to_reconnect={"L1"}, red_loop_paths=[])
    assert "reco_L1" in discoverer_instance.identified_reconnections
    assert "L1" in discoverer_instance.effective_reconnections
    # Verify scores dict is populated
    assert "reco_L1" in discoverer_instance.scores_reconnections
    assert isinstance(discoverer_instance.scores_reconnections["reco_L1"], float)

def test_discoverer_find_relevant_disconnections(discoverer_instance):
    discoverer_instance.find_relevant_disconnections(lines_constrained_path_names=["L1"])
    assert "disco_L1" in discoverer_instance.identified_disconnections
    assert "disco_L1" in discoverer_instance.effective_disconnections
    assert "disco_L2" not in discoverer_instance.identified_disconnections # Not in constrained path
    # Verify scores dict is populated for identified disconnections
    assert "disco_L1" in discoverer_instance.scores_disconnections
    assert isinstance(discoverer_instance.scores_disconnections["disco_L1"], float)
    # disco_L2 should not be scored (not on constrained path)
    assert "disco_L2" not in discoverer_instance.scores_disconnections

def test_discoverer_find_relevant_node_splitting(discoverer_instance):
    discoverer_instance.find_relevant_node_splitting(hubs_names=["Sub0"], nodes_blue_path_names=["Sub1"])
    assert "split_S0" in discoverer_instance.identified_splits
    assert "split_S1" in discoverer_instance.identified_splits
    assert len(discoverer_instance.effective_splits) == 1
    assert discoverer_instance.effective_splits[0].substations_id[0][0] == 0
    assert discoverer_instance.scores_splits[0] >= discoverer_instance.scores_splits[1] # Check sort order
    # Verify scores dict is populated
    assert "split_S0" in discoverer_instance.scores_splits_dict
    assert "split_S1" in discoverer_instance.scores_splits_dict
    assert discoverer_instance.scores_splits_dict["split_S0"] == 1.0
    assert discoverer_instance.scores_splits_dict["split_S1"] == 0.5


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
    score0, details0 = discoverer.compute_node_splitting_action_score(action_sub0, 0, alpha_ranker)
    assert score0 == 1.0, "Score for Sub0 should be 1.0 based on mock ranker"
    assert isinstance(details0, dict)

    # --- Test Case 2: Substation 1 (should get score 0.5 from MockAlphaDeesp) ---
    discoverer.obs_defaut = MockObservation(name_sub=["Sub0", "Sub1"], sub_info=[2, 2], topo_vect=[1, 1, 1, 1])
    score1, details1 = discoverer.compute_node_splitting_action_score(action_sub1, 1, alpha_ranker)
    assert score1 == 0.5, "Score for Sub1 should be 0.5 based on mock ranker"
    assert isinstance(details1, dict)

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


# =============================================================================
# Tests for _get_subs_impacted_from_action_desc (new backend-agnostic method)
# =============================================================================

class TestGetSubsImpactedFromActionDesc:
    """Tests for the _get_subs_impacted_from_action_desc method in ActionDiscoverer."""
    
    @pytest.fixture
    def discoverer_for_subs_impacted(self):
        """Create a discoverer with proper observation setup for testing."""
        mock_obs = MockObservation(
            name_sub=["Sub0", "Sub1", "Sub2", "Sub3"],
            name_line=["L1", "L2", "L3"],
            line_or_to_subid=[0, 1, 2],  # L1->Sub0, L2->Sub1, L3->Sub2
            line_ex_to_subid=[1, 2, 3]   # L1->Sub1, L2->Sub2, L3->Sub3
        )
        mock_env = MockEnv(name_line=mock_obs.name_line, name_sub=mock_obs.name_sub)
        mock_g_overflow = MockOverflowGraph()
        mock_g_dist = MockDistributionGraph()
        
        discoverer = ActionDiscoverer(
            env=mock_env,
            obs=mock_obs,
            obs_defaut=mock_obs,
            classifier=ActionClassifier(MockActionSpace()),
            timestep=0,
            lines_defaut=[],
            lines_overloaded_ids=[],
            act_reco_maintenance=MockActionObject(),
            non_connected_reconnectable_lines=[],
            all_disconnected_lines=[],
            dict_action={},
            actions_unfiltered=set(),
            hubs=[],
            g_overflow=mock_g_overflow,
            g_distribution_graph=mock_g_dist,
            simulator_data={}
        )
        return discoverer
    
    def test_substations_id_direct(self, discoverer_for_subs_impacted):
        """Test extraction from substations_id (direct substation topology changes)."""
        action_desc = {
            "content": {
                "set_bus": {
                    "substations_id": [(0, [1, 2, 1]), (2, [1, 1])]
                }
            }
        }
        
        subs = discoverer_for_subs_impacted._get_subs_impacted_from_action_desc(action_desc)
        
        assert set(subs) == {0, 2}
    
    def test_lines_or_id(self, discoverer_for_subs_impacted):
        """Test extraction from lines_or_id (line origin bus changes)."""
        action_desc = {
            "content": {
                "set_bus": {
                    "lines_or_id": {"L1": 1, "L2": 2}
                }
            }
        }
        
        subs = discoverer_for_subs_impacted._get_subs_impacted_from_action_desc(action_desc)
        
        # L1 origin is Sub0, L2 origin is Sub1
        assert set(subs) == {0, 1}
    
    def test_lines_ex_id(self, discoverer_for_subs_impacted):
        """Test extraction from lines_ex_id (line extremity bus changes)."""
        action_desc = {
            "content": {
                "set_bus": {
                    "lines_ex_id": {"L1": -1, "L3": 1}
                }
            }
        }
        
        subs = discoverer_for_subs_impacted._get_subs_impacted_from_action_desc(action_desc)
        
        # L1 extremity is Sub1, L3 extremity is Sub3
        assert set(subs) == {1, 3}
    
    def test_combined_lines_or_and_ex(self, discoverer_for_subs_impacted):
        """Test extraction from both lines_or_id and lines_ex_id."""
        action_desc = {
            "content": {
                "set_bus": {
                    "lines_or_id": {"L1": 1},
                    "lines_ex_id": {"L2": 2}
                }
            }
        }
        
        subs = discoverer_for_subs_impacted._get_subs_impacted_from_action_desc(action_desc)
        
        # L1 origin is Sub0, L2 extremity is Sub2
        assert set(subs) == {0, 2}
    
    def test_empty_action_desc(self, discoverer_for_subs_impacted):
        """Test with empty action description."""
        action_desc = {"content": {}}
        
        subs = discoverer_for_subs_impacted._get_subs_impacted_from_action_desc(action_desc)
        
        assert subs == []
    
    def test_empty_set_bus(self, discoverer_for_subs_impacted):
        """Test with empty set_bus dictionary."""
        action_desc = {"content": {"set_bus": {}}}
        
        subs = discoverer_for_subs_impacted._get_subs_impacted_from_action_desc(action_desc)
        
        assert subs == []
    
    def test_unknown_line_name(self, discoverer_for_subs_impacted):
        """Test with unknown line name (should be skipped)."""
        action_desc = {
            "content": {
                "set_bus": {
                    "lines_or_id": {"UNKNOWN_LINE": 1, "L1": 1}
                }
            }
        }
        
        subs = discoverer_for_subs_impacted._get_subs_impacted_from_action_desc(action_desc)
        
        # Should only include Sub0 from L1, ignore unknown line
        assert set(subs) == {0}
    
    def test_all_fields_combined(self, discoverer_for_subs_impacted):
        """Test with all possible fields combined."""
        action_desc = {
            "content": {
                "set_bus": {
                    "substations_id": [(3, [1, 1])],
                    "lines_or_id": {"L1": 1},
                    "lines_ex_id": {"L2": 2}
                }
            }
        }
        
        subs = discoverer_for_subs_impacted._get_subs_impacted_from_action_desc(action_desc)
        
        # Sub3 from substations_id, Sub0 from L1 origin, Sub2 from L2 extremity
        assert set(subs) == {0, 2, 3}
    
    def test_duplicate_substations_deduplicated(self, discoverer_for_subs_impacted):
        """Test that duplicate substations are deduplicated."""
        action_desc = {
            "content": {
                "set_bus": {
                    "lines_or_id": {"L1": 1},  # Sub0
                    "lines_ex_id": {"L1": 1}   # Sub1
                }
            }
        }

        subs = discoverer_for_subs_impacted._get_subs_impacted_from_action_desc(action_desc)

        # Should be deduplicated (using set internally)
        assert len(subs) == len(set(subs))


# =============================================================================
# Tests for disconnection scoring (_asymmetric_bell_score, compute_disconnection_score)
# =============================================================================

class TestAsymmetricBellScore:
    """Tests for the static _asymmetric_bell_score method."""

    def test_zero_at_min_boundary(self):
        """Score should be zero at the min_flow boundary."""
        score = ActionDiscoverer._asymmetric_bell_score(10.0, 10.0, 50.0)
        assert score == 0.0

    def test_zero_at_max_boundary(self):
        """Score should be zero at the max_flow boundary."""
        score = ActionDiscoverer._asymmetric_bell_score(50.0, 10.0, 50.0)
        assert score == 0.0

    def test_positive_inside_range(self):
        """Score should be positive between min and max."""
        score = ActionDiscoverer._asymmetric_bell_score(30.0, 10.0, 50.0)
        assert score > 0.0

    def test_peak_closer_to_max(self):
        """Score should be higher closer to max_flow than to min_flow."""
        # With default alpha=3.0, beta=1.5, peak is at x=0.8
        score_low = ActionDiscoverer._asymmetric_bell_score(20.0, 10.0, 50.0)   # x=0.25
        score_high = ActionDiscoverer._asymmetric_bell_score(40.0, 10.0, 50.0)  # x=0.75
        assert score_high > score_low

    def test_negative_below_min(self):
        """Score should be negative below min_flow."""
        score = ActionDiscoverer._asymmetric_bell_score(5.0, 10.0, 50.0)
        assert score < 0.0

    def test_negative_above_max(self):
        """Score should be negative above max_flow."""
        score = ActionDiscoverer._asymmetric_bell_score(60.0, 10.0, 50.0)
        assert score < 0.0

    def test_increasingly_negative_further_from_range(self):
        """Score should become more negative the further from the range."""
        score_close = ActionDiscoverer._asymmetric_bell_score(8.0, 10.0, 50.0)
        score_far = ActionDiscoverer._asymmetric_bell_score(0.0, 10.0, 50.0)
        assert score_far < score_close < 0.0

    def test_degenerate_range_returns_zero(self):
        """Score should be zero when max <= min (no valid range)."""
        assert ActionDiscoverer._asymmetric_bell_score(5.0, 10.0, 10.0) == 0.0
        assert ActionDiscoverer._asymmetric_bell_score(5.0, 10.0, 5.0) == 0.0

    def test_peak_value_normalized_to_one(self):
        """Peak score should be approximately 1.0."""
        # Peak is at x=0.8 for alpha=3.0, beta=1.5 → observed_flow = 10 + 0.8*40 = 42
        score_peak = ActionDiscoverer._asymmetric_bell_score(42.0, 10.0, 50.0)
        assert abs(score_peak - 1.0) < 0.01


class TestUnconstrainedLinearScore:
    """Tests for _unconstrained_linear_score (disconnection scoring when no new overloads arise)."""

    def test_one_at_max_flow(self):
        """Score should be exactly 1.0 at max_flow."""
        assert ActionDiscoverer._unconstrained_linear_score(100.0, 20.0, 100.0) == 1.0

    def test_zero_at_min_flow(self):
        """Score should be exactly 0.0 at min_flow."""
        assert ActionDiscoverer._unconstrained_linear_score(20.0, 20.0, 100.0) == 0.0

    def test_linear_in_between(self):
        """Score should be linear between min and max (midpoint = 0.5)."""
        score = ActionDiscoverer._unconstrained_linear_score(60.0, 20.0, 100.0)
        assert abs(score - 0.5) < 1e-9

    def test_negative_below_min(self):
        """Score should be negative below min_flow."""
        score = ActionDiscoverer._unconstrained_linear_score(10.0, 20.0, 100.0)
        assert score < 0.0

    def test_capped_at_one_above_max(self):
        """Score should be capped at 1.0 even beyond max_flow."""
        score = ActionDiscoverer._unconstrained_linear_score(120.0, 20.0, 100.0)
        assert score == 1.0

    def test_degenerate_range_returns_zero(self):
        """Should return 0 when max_flow <= min_flow."""
        assert ActionDiscoverer._unconstrained_linear_score(50.0, 50.0, 50.0) == 0.0
        assert ActionDiscoverer._unconstrained_linear_score(50.0, 60.0, 50.0) == 0.0

    def test_increasingly_negative_further_below_min(self):
        """Score should become more negative the further below min_flow."""
        s1 = ActionDiscoverer._unconstrained_linear_score(15.0, 20.0, 100.0)
        s2 = ActionDiscoverer._unconstrained_linear_score(0.0, 20.0, 100.0)
        assert s2 < s1 < 0.0


class TestComputeDisconnectionFlowBounds:
    """Tests for _compute_disconnection_flow_bounds with obs_linecut fix (issue #30)."""

    def _make_discoverer(self, rho_defaut, rho_linecut=None, lines_overloaded_ids=None):
        """Helper to build a discoverer with controlled rho values."""
        n_lines = len(rho_defaut)
        name_line = [f"L{i}" for i in range(n_lines)]
        name_sub = [f"S{i}" for i in range(n_lines + 1)]

        mock_obs_defaut = MockObservation(
            name_sub=name_sub,
            name_line=name_line,
            rho=np.array(rho_defaut, dtype=float),
            line_or_to_subid=list(range(n_lines)),
            line_ex_to_subid=list(range(1, n_lines + 1)),
        )
        mock_obs_linecut = None
        if rho_linecut is not None:
            mock_obs_linecut = MockObservation(
                name_sub=name_sub,
                name_line=name_line,
                rho=np.array(rho_linecut, dtype=float),
                line_or_to_subid=list(range(n_lines)),
                line_ex_to_subid=list(range(1, n_lines + 1)),
            )

        mock_env = MockEnv(name_line=name_line, name_sub=name_sub)
        # Overflow graph: edge 0->1 named L0 with capacity 100, edge 1->2 named L1 with capacity 50
        edge_data = {}
        for i in range(min(n_lines, 2)):
            edge_data[(i, i + 1)] = {0: {"name": f"L{i}", "capacity": 100.0 / (i + 1)}}
        mock_g_overflow = MockOverflowGraph(edge_data=edge_data)
        mock_g_dist = MockDistributionGraph()

        return ActionDiscoverer(
            env=mock_env,
            obs=mock_obs_defaut,
            obs_defaut=mock_obs_defaut,
            obs_linecut=mock_obs_linecut,
            classifier=ActionClassifier(MockActionSpace()),
            timestep=0,
            lines_defaut=[],
            lines_overloaded_ids=lines_overloaded_ids or [0],
            act_reco_maintenance=MockActionObject(),
            non_connected_reconnectable_lines=[],
            all_disconnected_lines=[],
            dict_action={},
            actions_unfiltered=set(),
            hubs=[],
            g_overflow=mock_g_overflow,
            g_distribution_graph=mock_g_dist,
            simulator_data={},
            check_action_simulation=False,
        )

    def test_unconstrained_when_obs_linecut_is_none(self):
        """When obs_linecut is None, max_redispatch must be inf (unconstrained regime)."""
        d = self._make_discoverer(rho_defaut=[1.2, 0.8], rho_linecut=None)
        _, _, max_redispatch = d._compute_disconnection_flow_bounds()
        assert max_redispatch == float('inf')

    def test_unconstrained_when_obs_linecut_equals_obs_defaut(self):
        """When obs_linecut == obs_defaut and no lines are overloaded, max_redispatch is inf."""
        # In practice, overloaded lines are always cut in obs_linecut (rho → 0).
        # This test covers the case where no lines cross 100% in either observation.
        rho = [0.8, 0.7]
        d = self._make_discoverer(rho_defaut=rho, rho_linecut=rho)
        _, _, max_redispatch = d._compute_disconnection_flow_bounds()
        assert max_redispatch == float('inf')

    def test_unconstrained_when_obs_linecut_has_higher_but_not_overloaded(self):
        """Loading increase that stays below 1.0 must NOT constrain max_redispatch."""
        # L0 goes 0.7 → 0.9 (increased but < 1.0) → should NOT trigger constraint
        # L1 goes 0.5 → 0.6 (increased but < 1.0) → should NOT trigger constraint
        d = self._make_discoverer(rho_defaut=[0.7, 0.5], rho_linecut=[0.9, 0.6])
        _, _, max_redispatch = d._compute_disconnection_flow_bounds()
        assert max_redispatch == float('inf')

    def test_constrained_when_obs_linecut_causes_new_overload(self):
        """When obs_linecut shows a line is NEWLY overloaded, max_redispatch must be finite."""
        # L0: capacity=100, rho_defaut=0.7, rho_linecut=1.2  (newly overloaded)
        #   ratio = 100 * (1 - 0.7) / (1.2 - 0.7) = 100 * 0.3 / 0.5 = 60
        # L1: capacity=50, rho_defaut=0.5, rho_linecut=0.8   (NOT overloaded)
        #   does NOT constrain
        # binding = 60
        d = self._make_discoverer(rho_defaut=[0.7, 0.5], rho_linecut=[1.2, 0.8])
        _, _, max_redispatch = d._compute_disconnection_flow_bounds()
        assert max_redispatch == pytest.approx(60.0, rel=1e-6)

    def test_constrained_when_existing_overload_not_relieved(self):
        """An existing overload that is NOT relieved in obs_linecut (rho_after > 1) fully constrains."""
        # L0: rho_defaut=1.1 (overloaded), rho_linecut=1.3 (still overloaded) → max_redispatch = 0
        d = self._make_discoverer(rho_defaut=[1.1, 0.5], rho_linecut=[1.3, 0.6])
        _, _, max_redispatch = d._compute_disconnection_flow_bounds()
        assert max_redispatch == 0.0

    def test_unconstrained_when_existing_overload_is_relieved(self):
        """An existing overload that IS relieved in obs_linecut (rho_after < 1) must NOT constrain."""
        # L0: rho_defaut=1.2 (overloaded), rho_linecut=0.5 (relieved) → rho_after < 1.0 → no constraint
        d = self._make_discoverer(rho_defaut=[1.2, 0.5], rho_linecut=[0.5, 0.6])
        _, _, max_redispatch = d._compute_disconnection_flow_bounds()
        assert max_redispatch == float('inf')

    def test_max_redispatch_uses_obs_defaut_as_rho_before(self):
        """rho_before in the ratio formula must come from obs_defaut, not obs."""
        # Deliberately set obs (self.obs) differently from obs_defaut to confirm the
        # method reads obs_defaut for rho_before, not obs.
        # L0: rho_defaut=0.6, rho_linecut=1.2 → newly overloaded
        rho_defaut = [0.6, 0.5]
        rho_linecut = [1.2, 0.5]  # L0 becomes newly overloaded
        d = self._make_discoverer(rho_defaut=rho_defaut, rho_linecut=rho_linecut)
        # Patch self.obs to have a very different rho to ensure it is NOT used
        d.obs = MockObservation(
            name_sub=d.obs.name_sub,
            name_line=d.obs.name_line,
            rho=np.array([0.1, 0.1]),  # very different from obs_defaut
            line_or_to_subid=list(range(2)),
            line_ex_to_subid=list(range(1, 3)),
        )
        _, _, max_redispatch = d._compute_disconnection_flow_bounds()
        # Expected: capacity_L0=100, rho_before=0.6 (obs_defaut), rho_after=1.2 (obs_linecut)
        # ratio = 100 * (1 - 0.6) / (1.2 - 0.6) = 100 * 0.4 / 0.6 = 66.67
        assert max_redispatch == pytest.approx(66.67, rel=1e-3)

    def test_min_redispatch_uses_obs_defaut(self):
        """min_redispatch must reflect the worst overload in obs_defaut."""
        # L0 rho=1.3 → min_redispatch = (1.3-1)*100 = 30
        d = self._make_discoverer(rho_defaut=[1.3, 0.5], rho_linecut=None, lines_overloaded_ids=[0])
        _, min_redispatch, _ = d._compute_disconnection_flow_bounds()
        assert min_redispatch == pytest.approx(30.0, rel=1e-6)

    def test_max_overload_flow_is_overloaded_line_capacity(self):
        """max_overload_flow must equal the overloaded line's capacity, not the global max."""
        # L0 is overloaded (rho=1.2, lines_overloaded_ids=[0], capacity=100).
        # L1 has a smaller capacity (50). max_overload_flow = capacity of L0 = 100.
        d = self._make_discoverer(rho_defaut=[1.2, 0.5], rho_linecut=None)
        max_overload_flow, _, _ = d._compute_disconnection_flow_bounds()
        assert max_overload_flow == pytest.approx(100.0, rel=1e-6)

    def test_max_overload_flow_uses_overloaded_line_not_global_max(self):
        """When the overloaded line is NOT the highest-capacity edge, use its capacity."""
        # L0 has capacity 100 (higher), L1 (overloaded, lines_overloaded_ids=[1]) has capacity 50.
        # max_overload_flow must be 50 (L1's capacity), not 100 (L0's capacity).
        d = self._make_discoverer(rho_defaut=[0.5, 1.2], rho_linecut=None, lines_overloaded_ids=[1])
        max_overload_flow, _, _ = d._compute_disconnection_flow_bounds()
        assert max_overload_flow == pytest.approx(50.0, rel=1e-6)


class TestNodeMergingScore:
    """Tests for the compute_node_merging_score method."""

    @pytest.fixture
    def merging_discoverer(self):
        """Create a discoverer with meaningful theta values for node merging tests."""
        # Sub0 has 2 buses. L1 originates at Sub0 (bus 1), L2 also at Sub0 (bus 2).
        # Load0 on bus 1, Gen0 on bus 2 at Sub0.
        # L3 connects Sub1->Sub2.
        # Overflow graph: edge (0,1) with L1 has POSITIVE capacity (red loop flow on bus 1),
        # edge (0,1) with L2 has NEGATIVE capacity (on bus 2).
        #
        # sub_topology order for Sub0: [Load0_bus, Gen0_bus, L1_or_bus, L2_or_bus]
        #                              = [1, 2, 1, 2]
        mock_obs = MockObservation(
            name_sub=["Sub0", "Sub1", "Sub2"],
            name_line=["L1", "L2", "L3"],
            name_load=["Load0"],
            name_gen=["Gen0"],
            load_to_subid=[0],
            gen_to_subid=[0],
            sub_topologies={0: [1, 2, 1, 2], 1: [1, 1], 2: [1, 1]},
            sub_info=[4, 3, 3],
            topo_vect=np.array([1, 2, 1, 2, 1, 1, 1, 1, 1, 1]),
            line_or_to_subid=[0, 0, 1],
            line_ex_to_subid=[1, 1, 2],
            line_or_bus=[1, 2, 1],
            line_ex_bus=[1, 1, 1],
            # L1: theta_or=-5.0 (at Sub0 bus 1), theta_ex=-3.0 (at Sub1 bus 1)
            # L2: theta_or=-1.0 (at Sub0 bus 2), theta_ex=-3.0 (at Sub1 bus 1)
            # L3: theta_or=-3.0 (at Sub1 bus 1), theta_ex=-1.0 (at Sub2 bus 1)
            theta_or=np.array([-5.0, -1.0, -3.0]),
            theta_ex=np.array([-3.0, -3.0, -1.0]),
        )
        mock_env = MockEnv(name_line=list(mock_obs.name_line), name_sub=list(mock_obs.name_sub))
        # Edge (0,1) with L1 has POSITIVE capacity (red loop on bus 1),
        # Edge (0,1) with L2 has NEGATIVE capacity (on bus 2)
        mock_g_overflow = MockOverflowGraph(
            edge_data={(0, 1): {0: {"name": "L1", "capacity": 10},
                                1: {"name": "L2", "capacity": -5}},
                       (1, 2): {0: {"name": "L3", "capacity": 3}}}
        )
        mock_g_dist = MockDistributionGraph()

        return ActionDiscoverer(
            env=mock_env, obs=mock_obs, obs_defaut=mock_obs,
            classifier=ActionClassifier(MockActionSpace()),
            timestep=0, lines_defaut=[], lines_overloaded_ids=[0],
            act_reco_maintenance=MockActionObject(),
            non_connected_reconnectable_lines=[], all_disconnected_lines=[],
            dict_action={}, actions_unfiltered=set(), hubs=[],
            g_overflow=mock_g_overflow, g_distribution_graph=mock_g_dist,
            simulator_data={}, check_action_simulation=False
        )

    def test_returns_tuple(self, merging_discoverer):
        """compute_node_merging_score must return a (float, dict) tuple."""
        result = merging_discoverer.compute_node_merging_score(0, [1, 2])
        assert isinstance(result, tuple) and len(result) == 2
        score, details = result
        assert isinstance(score, float)
        assert isinstance(details, dict)

    def test_red_loop_bus_identified_by_positive_capacity(self, merging_discoverer):
        """Bus carrying positive capacity edges should be identified as red loop bus."""
        # Sub0 bus 1 has L1 (line_or_bus[0]=1), and L1 has positive capacity=10
        # Sub0 bus 2 has L2 (line_or_bus[1]=2), and L2 has negative capacity=-5
        # So bus 1 is the red loop bus (theta1), bus 2 is the other (theta2)
        # theta1 = get_theta_node(obs, 0, 1) = median of theta_or[0] = -5.0
        # theta2 = get_theta_node(obs, 0, 2) = median of theta_or[1] = -1.0
        # score = theta2 - theta1 = -1.0 - (-5.0) = 4.0
        score, details = merging_discoverer.compute_node_merging_score(0, [1, 2])
        assert score > 0.0  # theta2 > theta1 means flow towards red loop
        # Red loop bus assets should contain L1 (on bus 1)
        assert "L1" in details["targeted_node_assets"]["lines"]

    def test_single_bus_returns_zero(self, merging_discoverer):
        """Should return (0.0, {}) when fewer than 2 buses."""
        score, details = merging_discoverer.compute_node_merging_score(0, [1])
        assert score == 0.0
        assert details == {}

    def test_details_contains_targeted_node_assets(self, merging_discoverer):
        """Details dict must contain targeted_node_assets with lines on the red loop bus."""
        _, details = merging_discoverer.compute_node_merging_score(0, [1, 2])
        assert "targeted_node_assets" in details
        assets = details["targeted_node_assets"]
        assert "lines" in assets
        assert "loads" in assets
        assert "generators" in assets
        # L1 is on bus 1 (red loop bus) at Sub0 origin
        assert "L1" in assets["lines"]
        # L2 is on bus 2 (not red loop bus), should not appear
        assert "L2" not in assets["lines"]


# =============================================================================
# Tests for params storage alongside scores in each discovery method
# =============================================================================

class TestDiscoveryParamsStorage:
    """Tests that each discovery method stores params alongside scores."""

    def test_reconnection_params_populated(self, discoverer_instance, monkeypatch):
        """verify_relevant_reconnections must populate params_reconnections with threshold and max flow."""
        def mock_path_check(*args): return True, ["Sub0", "Sub1", "Sub3"], None
        monkeypatch.setattr(discoverer_instance, "_check_other_reconnectable_line_on_path", mock_path_check)
        discoverer_instance.verify_relevant_reconnections(lines_to_reconnect={"L1"}, red_loop_paths=[])
        params = discoverer_instance.params_reconnections
        assert "percentage_threshold_min_dispatch_flow" in params
        assert "max_dispatch_flow" in params
        assert isinstance(params["percentage_threshold_min_dispatch_flow"], (int, float))
        assert isinstance(params["max_dispatch_flow"], (int, float))
        assert params["max_dispatch_flow"] > 0

    def test_disconnection_params_populated(self, discoverer_instance):
        """find_relevant_disconnections must populate params_disconnections with redispatch bounds."""
        discoverer_instance.find_relevant_disconnections(lines_constrained_path_names=["L1"])
        params = discoverer_instance.params_disconnections
        assert isinstance(params, dict)
        if params:
            assert "regime" in params
            assert "min_redispatch" in params
            if params["regime"] == "constrained":
                assert "max_redispatch" in params
                assert "peak_redispatch" in params
                expected_peak = params["min_redispatch"] + 0.8 * (params["max_redispatch"] - params["min_redispatch"])
                assert abs(params["peak_redispatch"] - expected_peak) < 1e-9
            else:
                assert params["regime"] == "unconstrained"
                assert "max_overload_flow" in params

    def test_splitting_params_populated_per_action(self, discoverer_instance):
        """find_relevant_node_splitting must store per-action details in params_splits_dict."""
        discoverer_instance.find_relevant_node_splitting(hubs_names=["Sub0"], nodes_blue_path_names=["Sub1"])
        params = discoverer_instance.params_splits_dict
        assert isinstance(params, dict)
        # Each scored action should have a details entry
        for action_id in discoverer_instance.scores_splits_dict:
            assert action_id in params
            assert isinstance(params[action_id], dict)

    def test_merging_params_populated_per_action(self, discoverer_instance):
        """find_relevant_node_merging must store per-action params with assets."""
        discoverer_instance.find_relevant_node_merging(["Sub0", "Sub1", "Sub3"])
        params = discoverer_instance.params_merges
        assert isinstance(params, dict)
        # Each scored action should have a per-action details entry
        for action_id in discoverer_instance.scores_merges:
            assert action_id in params
            assert isinstance(params[action_id], dict)
            assert "targeted_node_assets" in params[action_id]
            assets = params[action_id]["targeted_node_assets"]
            assert "lines" in assets
            assert "loads" in assets
            assert "generators" in assets


# =============================================================================
# Tests for action_scores dictionary structure and rounding
# =============================================================================

class TestActionScoresStructureAndRounding:
    """Tests for the nested action_scores dict assembled by discover_and_prioritize."""

    @pytest.fixture
    def scores_discoverer(self):
        """Create a discoverer with pre-populated scores/params to test assembly and rounding."""
        mock_obs = MockObservation(
            name_sub=["Sub0", "Sub1"],
            name_line=["L1", "L2"],
            sub_topologies={0: [1, 2], 1: [1, 1]},
            sub_info=[2, 2],
            topo_vect=np.array([1, 2, 1, 1]),
            line_or_to_subid=[0, 1],
            line_ex_to_subid=[1, 0],
            line_or_bus=[1, 1],
            line_ex_bus=[1, 1],
            theta_or=np.array([0.0, 0.0]),
            theta_ex=np.array([0.0, 0.0]),
        )
        mock_env = MockEnv(name_line=list(mock_obs.name_line), name_sub=list(mock_obs.name_sub))
        mock_g_overflow = MockOverflowGraph(
            edge_data={(0, 1): {0: {"name": "L1", "capacity": 10}}}
        )
        mock_g_dist = MockDistributionGraph()

        discoverer = ActionDiscoverer(
            env=mock_env, obs=mock_obs, obs_defaut=mock_obs,
            classifier=ActionClassifier(MockActionSpace()),
            timestep=0, lines_defaut=[], lines_overloaded_ids=[0],
            act_reco_maintenance=MockActionObject(),
            non_connected_reconnectable_lines=[], all_disconnected_lines=[],
            dict_action={}, actions_unfiltered=set(), hubs=[],
            g_overflow=mock_g_overflow, g_distribution_graph=mock_g_dist,
            simulator_data={}, check_action_simulation=False
        )

        # Pre-populate with values that have many decimals to test rounding
        discoverer.scores_reconnections = {"reco_1": 3.14159, "reco_2": -1.23456}
        discoverer.params_reconnections = {
            "percentage_threshold_min_dispatch_flow": 0.10000001,
            "max_dispatch_flow": 123.456789,
        }
        discoverer.scores_disconnections = {"disco_1": 0.87654321}
        discoverer.params_disconnections = {
            "regime": "constrained",
            "min_redispatch": 10.111, "max_redispatch": 50.999, "peak_redispatch": 42.8888,
        }
        discoverer.scores_splits_dict = {"split_1": 0.99999, "split_2": -0.12345}
        discoverer.params_splits_dict = {
            "split_1": {"node_type": "amont",
                        "targeted_node_assets": {"lines": ["L1"], "loads": [], "generators": []},
                        "in_negative_flows": 12.3456, "out_negative_flows": 78.9012,
                        "in_positive_flows": 0.0, "out_positive_flows": 5.55555},
            "split_2": {"node_type": "aval",
                        "targeted_node_assets": {"lines": ["L2"], "loads": ["Load_X"], "generators": ["Gen_Y"]},
                        "in_negative_flows": 99.9999, "out_negative_flows": 1.11111,
                        "in_positive_flows": 3.33333, "out_positive_flows": 0.0},
        }
        discoverer.scores_merges = {"merge_1": 2.71828}
        discoverer.params_merges = {
            "merge_1": {
                "targeted_node_assets": {"lines": ["L1"], "loads": ["Load_A"], "generators": []},
            },
        }
        return discoverer

    def test_action_scores_has_all_four_types(self, scores_discoverer):
        """action_scores must have exactly the four expected action type keys."""
        # Manually call the assembly logic (extract from discover_and_prioritize)
        scores_discoverer.prioritized_actions = {}
        action_scores = self._build_action_scores(scores_discoverer)
        expected_types = {"line_reconnection", "line_disconnection", "open_coupling", "close_coupling"}
        assert set(action_scores.keys()) == expected_types

    def test_each_type_has_scores_and_params(self, scores_discoverer):
        """Each action type entry must have 'scores' and 'params' keys."""
        action_scores = self._build_action_scores(scores_discoverer)
        for action_type, entry in action_scores.items():
            assert "scores" in entry, f"Missing 'scores' in {action_type}"
            assert "params" in entry, f"Missing 'params' in {action_type}"

    def test_scores_sorted_descending(self, scores_discoverer):
        """Scores within each type must be sorted in descending order."""
        action_scores = self._build_action_scores(scores_discoverer)
        for action_type, entry in action_scores.items():
            scores = list(entry["scores"].values())
            assert scores == sorted(scores, reverse=True), \
                f"Scores not sorted descending in {action_type}: {scores}"

    def test_scores_rounded_to_two_decimals(self, scores_discoverer):
        """All score values must be rounded to 2 decimal places."""
        action_scores = self._build_action_scores(scores_discoverer)
        assert action_scores["line_reconnection"]["scores"]["reco_1"] == 3.14
        assert action_scores["line_reconnection"]["scores"]["reco_2"] == -1.23
        assert action_scores["line_disconnection"]["scores"]["disco_1"] == 0.88
        assert action_scores["open_coupling"]["scores"]["split_1"] == 1.0
        assert action_scores["open_coupling"]["scores"]["split_2"] == -0.12
        assert action_scores["close_coupling"]["scores"]["merge_1"] == 2.72

    def test_flat_params_rounded_to_two_decimals(self, scores_discoverer):
        """Flat params (reconnections, disconnections) must have floats rounded."""
        action_scores = self._build_action_scores(scores_discoverer)
        reco_params = action_scores["line_reconnection"]["params"]
        assert reco_params["percentage_threshold_min_dispatch_flow"] == 0.1
        assert reco_params["max_dispatch_flow"] == 123.46

        disco_params = action_scores["line_disconnection"]["params"]
        assert disco_params["regime"] == "constrained"  # String preserved
        assert disco_params["min_redispatch"] == 10.11
        assert disco_params["max_redispatch"] == 51.0
        assert disco_params["peak_redispatch"] == 42.89

    def test_nested_params_rounded_to_two_decimals(self, scores_discoverer):
        """Per-action nested params (open/close_coupling) must have floats rounded, non-floats preserved."""
        action_scores = self._build_action_scores(scores_discoverer)
        split_params = action_scores["open_coupling"]["params"]

        s1 = split_params["split_1"]
        assert s1["node_type"] == "amont"  # String preserved
        assert isinstance(s1["targeted_node_assets"], dict)  # Assets dict preserved
        assert s1["in_negative_flows"] == 12.35
        assert s1["out_negative_flows"] == 78.9
        assert s1["out_positive_flows"] == 5.56

        s2 = split_params["split_2"]
        assert s2["in_negative_flows"] == 100.0
        assert s2["out_negative_flows"] == 1.11

        # close_coupling (merges) is now per-action too
        merge_params = action_scores["close_coupling"]["params"]
        m1 = merge_params["merge_1"]
        assert isinstance(m1["targeted_node_assets"], dict)  # Assets dict preserved

    def test_empty_scores_produce_empty_entries(self, scores_discoverer):
        """When a category has no scored actions, its scores and params should be empty."""
        scores_discoverer.scores_reconnections = {}
        scores_discoverer.params_reconnections = {}
        action_scores = self._build_action_scores(scores_discoverer)
        assert action_scores["line_reconnection"]["scores"] == {}
        assert action_scores["line_reconnection"]["params"] == {}

    @staticmethod
    def _build_action_scores(discoverer):
        """Replicate the action_scores assembly logic from discover_and_prioritize."""
        def _round_scores(d):
            return {k: round(v, 2) for k, v in d.items()}

        def _round_params(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
                elif isinstance(v, float):
                    out[k] = round(v, 2)
                else:
                    out[k] = v
            return out

        return {
            "line_reconnection": {
                "scores": _round_scores(dict(sorted(discoverer.scores_reconnections.items(), key=lambda x: x[1], reverse=True))),
                "params": _round_params(discoverer.params_reconnections),
            },
            "line_disconnection": {
                "scores": _round_scores(dict(sorted(discoverer.scores_disconnections.items(), key=lambda x: x[1], reverse=True))),
                "params": _round_params(discoverer.params_disconnections),
            },
            "open_coupling": {
                "scores": _round_scores(dict(sorted(discoverer.scores_splits_dict.items(), key=lambda x: x[1], reverse=True))),
                "params": _round_params(discoverer.params_splits_dict),
            },
            "close_coupling": {
                "scores": _round_scores(dict(sorted(discoverer.scores_merges.items(), key=lambda x: x[1], reverse=True))),
                "params": _round_params(discoverer.params_merges),
            },
        }