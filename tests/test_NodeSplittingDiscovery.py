import pytest
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, patch

# --- Import the actual class ---
# Ensure this path matches your project structure
from expert_op4grid_recommender.action_evaluation.discovery import ActionDiscoverer


# --- Fixtures ---

@pytest.fixture
def action_discoverer():
    """
    Fixture that returns an ActionDiscoverer instance with a patched __init__.
    This avoids loading the real Grid2Op environment while allowing access to methods.
    """
    with patch.object(ActionDiscoverer, '__init__', return_value=None):
        discoverer = ActionDiscoverer()
        # Mock attributes usually set in __init__ that are accessed by methods
        discoverer.obs_defaut = MagicMock()
        yield discoverer


# --- 1. Bus Identification Logic ---

@pytest.mark.parametrize("node_type, neg_in, neg_out, expected_bus", [
    # Amont: Expects max (NegOut - NegIn). Bus 1: (100-10)=90, Bus 2: (20-10)=10 -> Pick 1
    ("amont", [10, 10], [100, 20], 1),
    # Aval: Expects max (NegIn - NegOut). Bus 1: (100-10)=90, Bus 2: (20-10)=10 -> Pick 1
    ("aval", [100, 20], [10, 10], 1),
    # Fallback/Loop: Expects max abs(Diff). Bus 2 has larger diff -> Pick 2
    ("loop", [10, 100], [10, 10], 2),
])
def test_identify_bus_of_interest(action_discoverer, node_type, neg_in, neg_out, expected_bus):
    """Test bus identification logic for Amont, Aval, and fallback types."""
    buses = [1, 2]
    result = action_discoverer.identify_bus_of_interest_in_node_splitting_(
        node_type, buses, neg_in, neg_out
    )
    assert result == expected_bus


# --- 2. Graph Flow Parsing ---

def test_computing_buses_values_basic(action_discoverer):
    """Test basic aggregation of flow attributes from graph to bus vectors."""
    G = nx.MultiDiGraph()
    # Node 0 is target.
    # Line_A: -100 (Inflow) -> Bus 1
    # Line_B: +50  (Inflow) -> Bus 1
    # Line_C: -20  (Outflow)-> Bus 2
    G.add_edge(10, 0, key=0, label='-100', name='Line_A')
    G.add_edge(11, 0, key=0, label='50', name='Line_B')
    G.add_edge(0, 12, key=0, label='-20', name='Line_C')

    dict_edge_names_buses = {'Line_A': 1, 'Line_B': 1, 'Line_C': 2}

    buses, neg_in, neg_out, pos_in, pos_out = action_discoverer.computing_buses_values_of_interest(
        G, 0, dict_edge_names_buses
    )

    idx_1 = buses.index(1)
    idx_2 = buses.index(2)

    assert neg_in[idx_1] == 100.0
    assert pos_in[idx_1] == 50.0
    assert neg_out[idx_2] == 20.0


def test_computing_buses_parallel_and_disconnected(action_discoverer):
    """
    Test robust aggregation:
    1. Parallel lines (MultiGraph keys 0 and 1) should be summed.
    2. Disconnected buses (0 or -1) should be ignored.
    """
    G = nx.MultiDiGraph()
    # Parallel lines: Line A and Line A_prime both go into Bus 1
    G.add_edge(10, 0, key=0, label='-50', name='Line_A')
    G.add_edge(10, 0, key=1, label='-50', name='Line_A_prime')

    # Disconnected line: Line B on bus 0
    G.add_edge(11, 0, key=0, label='-100', name='Line_B')

    dict_edge_names_buses = {
        'Line_A': 1,
        'Line_A_prime': 1,
        'Line_B': 0  # Should be ignored
    }

    buses, neg_in, _, _, _ = action_discoverer.computing_buses_values_of_interest(
        G, 0, dict_edge_names_buses
    )

    # Check Bus 1: Should be 50 + 50 = 100
    assert 1 in buses
    assert neg_in[buses.index(1)] == 100.0

    # Check Bus 0: Should NOT be in the results
    assert 0 not in buses


# --- 3. Node Type Classification ---

def test_identify_node_splitting_type(action_discoverer):
    """Test classification of nodes (Amont/Aval/Loop) using mocked graph."""
    mock_graph = MagicMock()

    # Setup Paths
    mock_path = MagicMock()
    mock_path.n_amont.return_value = [10]
    mock_path.n_aval.return_value = [20]
    mock_graph.get_constrained_path.return_value = mock_path

    # Setup Loops
    mock_loops = MagicMock()
    mock_loops.Path = [[30, 31]]
    mock_graph.get_loops.return_value = mock_loops

    assert action_discoverer.identify_node_splitting_type(10, mock_graph) == "amont"
    assert action_discoverer.identify_node_splitting_type(20, mock_graph) == "aval"
    assert action_discoverer.identify_node_splitting_type(30, mock_graph) == "loop"
    assert action_discoverer.identify_node_splitting_type(99, mock_graph) is None


# --- 4. Scoring Logic ---

@pytest.mark.parametrize("scenario_name, node_type, flows, should_be_positive", [
    (
            "Amont Good Split",
            "amont",
            {"neg_in": 0, "neg_out": 100, "pos_in": 0, "pos_out": 10},
            True
    ),
    (
            "Amont Bad Split (Sign Flip)",
            "amont",
            # Repulsion: 10 - 100 = -90 (Negative)
            # Weight: (10 - 200) = -190 (Negative)
            # Result should be flipped to negative
            {"neg_in": 200, "neg_out": 10, "pos_in": 0, "pos_out": 100},
            False
    ),
])
def test_compute_score_standard(action_discoverer, scenario_name, node_type, flows, should_be_positive):
    """Parametrized test for standard scoring scenarios."""
    score = action_discoverer.compute_node_splitting_action_bus_score(
        node_type, 1, [1],
        [flows["neg_in"]], [flows["neg_out"]], [flows["pos_in"]], [flows["pos_out"]]
    )

    if should_be_positive:
        assert score > 0, f"Scenario '{scenario_name}' should have positive score"
    else:
        assert score < 0, f"Scenario '{scenario_name}' should have negative score"


def test_compute_score_division_by_zero(action_discoverer):
    """Ensure code handles cases where total flow is zero (e.g., disconnected bus)."""
    # Note: If your code doesn't handle this, this test expects a ZeroDivisionError.
    # If you fix your code to return 0, change this to: assert score == 0
    with pytest.raises(ZeroDivisionError):
        action_discoverer.compute_node_splitting_action_bus_score(
            node_type="amont", bus_of_interest=1, buses=[1],
            buses_negative_inflow=[0], buses_negative_out_flow=[0],
            buses_positive_inflow=[0], buses_positive_out_flow=[0]
        )


@pytest.mark.parametrize("neg_in, neg_out, expected_behavior", [
    (10, 100, "amont"),  # Outflow dominates -> Treat as Amont
    (100, 10, "aval"),  # Inflow dominates  -> Treat as Aval
])
def test_score_harmonization_logic(action_discoverer, neg_in, neg_out, expected_behavior):
    """
    Test that 'loop' or 'unknown' nodes are correctly harmonized
    based on their dominant flow direction.
    """
    # We pass "loop" to trigger the auto-detection block
    score = action_discoverer.compute_node_splitting_action_bus_score(
        node_type="loop", bus_of_interest=1, buses=[1],
        buses_negative_inflow=[neg_in], buses_negative_out_flow=[neg_out],
        buses_positive_inflow=[0], buses_positive_out_flow=[0]
    )

    # Amont logic produces positive score for (10, 100)
    # Aval logic produces positive score for (100, 10)
    assert score > 0


# --- 5. Topology Helper Tests ---

def test_get_action_topo_vect_masking(action_discoverer):
    """
    Test that the topology vector from the action is correctly merged
    with the 'disconnected' state of the initial topology.
    """
    # Mock Observation behavior
    # Sub 0 has 3 buses, Sub 1 has 2 buses
    action_discoverer.obs_defaut.sub_info = np.array([3, 2])

    # Initial state of Sub 1: [1, -1] (Bus 1 connected, Bus 2 disconnected)
    action_discoverer.obs_defaut.sub_topology.return_value = np.array([1, -1])

    # Simulated Action Result (impact_obs)
    # Global vector: [1,1,1,  2, 2] -> Sub 1 is [2, 2]
    mock_impact_obs = MagicMock()
    mock_impact_obs.topo_vect = np.array([1, 1, 1, 2, 2])

    # Mock addition (obs + action)
    action_discoverer.obs_defaut.__add__.return_value = mock_impact_obs

    # Run logic on Sub 1
    res_topo, is_single = action_discoverer._get_action_topo_vect(sub_impacted_id=1, action=MagicMock())

    # Logic: Should take [2, 2] from action, but mask the 2nd element
    # because initial state was -1. Result: [2, -1]
    np.testing.assert_array_equal(res_topo, np.array([2, -1]))

    # is_single check: set({1, -1}) - {0, -1} is {1}, len=1 -> True
    assert is_single is True


# --- 6. Full Integration Test ---

def test_integration_full_scoring_pipeline(action_discoverer):
    """
    Verifies that the main orchestrator calls helper functions in the correct order
    and passes data correctly.
    """
    mock_graph = MagicMock()
    mock_dist_graph = MagicMock()

    # Patch internal methods to isolate the orchestration logic
    with patch.object(action_discoverer, 'identify_node_splitting_type', return_value="amont") as mock_id_type:
        # Mock return: buses=[1], neg_in=[10], neg_out=[100], pos_in=[0], pos_out=[0]
        fake_flow_data = ([1], [10], [100], [0], [0])
        with patch.object(action_discoverer, 'computing_buses_values_of_interest', return_value=fake_flow_data):
            final_score = action_discoverer.compute_node_splitting_action_score_value(
                overflow_graph=mock_graph,
                g_distribution_graph=mock_dist_graph,
                node=10,
                dict_edge_names_buses={}
            )

            mock_id_type.assert_called_once()
            # With Amont logic and favorable flows, score should be positive
            assert final_score > 0


# --- 7. Topology & Dictionary Helpers ---

def test_get_action_topo_vect_slicing_and_masking(action_discoverer):
    """
    Test _get_action_topo_vect for:
    1. Correct slicing of the global topology vector based on sub_id.
    2. Preservation of disconnected states (masking) from the initial observation.
    3. Correct calculation of 'is_single_node'.
    """
    # SETUP: Mock the default observation
    # Scenario: 2 Substations. Sub 0 has 2 elements, Sub 1 has 3 elements.
    action_discoverer.obs_defaut.sub_info = np.array([2, 3])

    # Initial state of Sub 1 (Indices 2, 3, 4 in global vector)
    # Element 0: Bus 1
    # Element 1: Disconnected (-1) -> This MUST persist
    # Element 2: Bus 1
    action_discoverer.obs_defaut.sub_topology.return_value = np.array([1, -1, 1])

    # SETUP: Mock the Action result (impact_obs)
    mock_impact_obs = MagicMock()
    # Global vector has length 5 (2 + 3).
    # The action attempts to set Sub 1 to [2, 2, 2] (Split to bus 2)
    mock_impact_obs.topo_vect = np.array([1, 1, 2, 2, 2])

    # Mock the addition operator (obs + action)
    action_discoverer.obs_defaut.__add__.return_value = mock_impact_obs

    # EXECUTE
    # We target Substation 1
    res_topo, is_single = action_discoverer._get_action_topo_vect(sub_impacted_id=1, action=MagicMock())

    # ASSERT
    # 1. Slicing: Should grab the last 3 elements from the action ([2, 2, 2])
    # 2. Masking: The middle element was -1 initially, so it should stay -1.
    # Expected Result: [2, -1, 2]
    expected_topo = np.array([2, -1, 2])
    np.testing.assert_array_equal(res_topo, expected_topo)

    # 3. Single Node Check
    # Active buses in result are {2}. Since len({2}) == 1, is_single should be True.
    assert is_single is True


def test_edge_names_buses_dict_parsing(action_discoverer):
    """
    Test _edge_names_buses_dict for:
    1. Mapping line names to bus numbers.
    2. Handling both line origins ('line_or_id') and extremities ('line_ex_id').
    3. Ignoring non-line elements (like Loads or Generators).
    """
    # SETUP: Mock Observation
    mock_obs = MagicMock()
    # Scenario: Substation 0 has 3 elements.
    mock_obs.sub_info = np.array([3])

    # Mock Line Names: ID 10 -> "Line_A", ID 11 -> "Line_B"
    mock_obs.name_line = {10: "Line_A", 11: "Line_B"}

    # Mock Topology Element Metadata
    # This function tells us what object exists at a specific vector index
    def side_effect_topo_element(idx):
        if idx == 0:
            # Index 0: Origin of Line 10 (Line_A)
            return {'line_id': 10, 'line_or_id': 10}
        elif idx == 1:
            # Index 1: A Load (Should be ignored by the dict)
            return {'load_id': 0}
        elif idx == 2:
            # Index 2: Extremity of Line 11 (Line_B)
            return {'line_id': 11, 'line_ex_id': 11}
        return {}

    mock_obs.topo_vect_element.side_effect = side_effect_topo_element

    # SETUP: Action Topology Vector (The inputs we want to map)
    # Bus assignments: Index 0 -> Bus 1, Index 1 -> Bus 1, Index 2 -> Bus 2
    action_topo_vect = np.array([1, 1, 2])

    # EXECUTE
    result_dict = action_discoverer._edge_names_buses_dict(
        obs=mock_obs,
        action_topo_vect=action_topo_vect,
        sub_impacted_id=0
    )

    # ASSERT
    # Line A (Index 0) is on Bus 1
    assert result_dict["Line_A"] == 1
    # Line B (Index 2) is on Bus 2
    assert result_dict["Line_B"] == 2
    # The Load (Index 1) should not appear in the dictionary
    assert len(result_dict) == 2