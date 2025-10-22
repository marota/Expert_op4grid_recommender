import pytest
import os
import sys
import numpy as np
from pathlib import Path
from expert_op4grid_recommender.environment import make_grid2op_training_env
from typing import Dict, Any, Optional, Callable, Tuple, List

# --- Test Setup: Add Project Root to Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mock Objects ---

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

class MockActionSpace:
    """Mocks the Grid2Op action_space callable for object-based tests."""
    def __call__(self, action_dict):
        # Creates a MockAction suitable for internal checks
        set_bus = action_dict.get("set_bus", {})
        return MockAction(
            line_or_set_bus=list(set_bus.get("lines_or_id", {}).values()),
            line_ex_set_bus=list(set_bus.get("lines_ex_id", {}).values()),
            load_set_bus=list(set_bus.get("loads_id", {}).values()),
            line_set_status=action_dict.get("set_line_status", []) # Needs adaptation if status is used
        )

# --- Imports from the refactored project ---
from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

# --- Test Functions for ActionClassifier ---

@pytest.fixture
def classifier_instance():
    """Provides a basic ActionClassifier instance."""
    # Pass a mock action space in case object-based tests need it
    return ActionClassifier(grid2op_action_space=MockActionSpace())

## Tests for internal helper methods (called via instance) ##

def test_internal_no_reconnection(classifier_instance):
    action = MockAction()
    assert not classifier_instance._is_line_reconnection(action) #is False

def test_internal_reconnection_with_status(classifier_instance):
    action = MockAction(line_set_status=[1])
    assert classifier_instance._is_line_reconnection(action) #is True

def test_internal_reconnection_with_both_buses(classifier_instance):
    action = MockAction(line_or_set=[1], line_ex_set=[1])
    assert classifier_instance._is_line_reconnection(action) #is True

def test_internal_warning_on_unsupported_actions_reco(classifier_instance, capfd):
    classifier_instance._is_line_reconnection(MockAction(line_or_change=[1]))
    assert "WARNING" in capfd.readouterr()[0]

def test_internal_no_load_disconnection(classifier_instance):
    action = MockAction()
    assert not classifier_instance._is_load_disconnection(action) #is False

def test_internal_load_disconnection_with_set_bus(classifier_instance):
    action = MockAction(load_set_bus=[-1])
    assert classifier_instance._is_load_disconnection(action) #is True

def test_internal_warning_on_unsupported_actions_load_disco(classifier_instance, capfd):
    classifier_instance._is_load_disconnection(MockAction(load_change_bus=[1]))
    assert "WARNING" in capfd.readouterr()[0]

def test_internal_multiple_loads_mixed(classifier_instance):
    assert classifier_instance._is_load_disconnection(MockAction(load_set_bus=[0, -1, 1])) #is True

def test_internal_no_disconnection(classifier_instance):
    action = MockAction()
    assert not classifier_instance._is_line_disconnection(action) #is False

def test_internal_line_disconnection_with_status(classifier_instance):
    action = MockAction(line_set_status=[-1])
    assert classifier_instance._is_line_disconnection(action) #is True

def test_internal_line_disconnection_with_origin_bus(classifier_instance):
    action = MockAction(line_or_set=[-1])
    assert classifier_instance._is_line_disconnection(action) #is True

def test_internal_line_disconnection_with_extremity_bus(classifier_instance):
    action = MockAction(line_ex_set=[-1])
    assert classifier_instance._is_line_disconnection(action) #is True

def test_internal_multiple_lines_disconnections_mixed(classifier_instance):
    assert classifier_instance._is_line_disconnection(MockAction(line_or_set=[1, 0, -1])) #is True

def test_internal_warning_on_unsupported_actions_line_disco(classifier_instance, capfd):
    classifier_instance._is_line_disconnection(MockAction(line_or_change=[1]))
    assert "WARNING" in capfd.readouterr()[0]

def test_internal_no_nodale_action(classifier_instance):
    result, subs, splits = classifier_instance._is_nodale_grid2op_action(MockAction(set_topo_vect=[0, 1, 0], topo_vect_to_sub=[0, 1, 2]))
    assert not result and not subs and not splits

def test_internal_single_substation_with_two_elements(classifier_instance):
    result, subs, splits = classifier_instance._is_nodale_grid2op_action(MockAction(set_topo_vect=[1, 1, 0], topo_vect_to_sub=[0, 0, 1]))
    assert result and subs == [0] and splits == [False]

def test_internal_substation_with_splitting(classifier_instance):
    result, subs, splits = classifier_instance._is_nodale_grid2op_action(MockAction(set_topo_vect=[1, 2, 0], topo_vect_to_sub=[0, 0, 1]))
    assert result and subs == [0] and splits == [True]

def test_internal_multiple_substations_mixed(classifier_instance):
    result, subs, splits = classifier_instance._is_nodale_grid2op_action(MockAction(set_topo_vect=[1, 1, 2, 1], topo_vect_to_sub=[0, 0, 1, 1]))
    assert result and set(subs) == {0, 1} and splits == [False, True]

## Tests for public identify_action_type method ##

def test_identify_action_type_by_description(classifier_instance):
    assert classifier_instance.identify_action_type({"description_unitaire": "Ouverture Ligne L1", "content": {"set_bus": {"lines_ex_id": {"L1":-1}}}}) == "open_line"
    assert classifier_instance.identify_action_type({"description_unitaire": "Fermeture COUPL S1", "content": {"set_bus": {"lines_or_id": {"L_coup":1}}}, "VoltageLevelId": "S1"}) == "close_coupling"
    assert classifier_instance.identify_action_type({"description_unitaire": "Ouverture ... Load1 Line1", "content": {"set_bus": {"lines_ex_id": {"Line1":-1}, "loads_id": {"Load1":-1}}}}) == "open_line_load"
    assert classifier_instance.identify_action_type({"description_unitaire": "Fermeture Line1", "content": {"set_bus": {"lines_or_id": {"Line1":1}, "lines_ex_id": {"Line1":1}}}}) == "close_line"
    # Test unknown
    assert classifier_instance.identify_action_type({"description_unitaire": "Unknown Action", "content": {"set_bus": {}}}) == "unknown"
    # Test KeyError resilience
    assert classifier_instance.identify_action_type({"description_unitaire": "Ouverture Test", "content": {}}) == "unknown" # Missing set_bus

## Tests for public identify_grid2op_action_type method ##

def test_identify_grid2op_action_type(classifier_instance):
    # Create MockAction instances representing different scenarios
    action_open_line = MockAction(line_ex_set=[-1])
    action_close_line = MockAction(line_or_set=[1], line_ex_set=[1])
    action_open_load = MockAction(load_set_bus=[-1])
    action_open_line_load = MockAction(line_ex_set=[-1], load_set_bus=[-1])
    action_open_coupling = MockAction(set_topo_vect=[1, 2], topo_vect_to_sub=[0, 0])
    action_close_coupling = MockAction(set_topo_vect=[1, 1], topo_vect_to_sub=[0, 0])
    action_unknown = MockAction()

    assert classifier_instance.identify_grid2op_action_type(action_open_line) == "open_line"
    assert classifier_instance.identify_grid2op_action_type(action_close_line) == "close_line"
    assert classifier_instance.identify_grid2op_action_type(action_open_load) == "open_load"
    assert classifier_instance.identify_grid2op_action_type(action_open_line_load) == "open_line_load"
    assert classifier_instance.identify_grid2op_action_type(action_open_coupling) == "open_coupling"
    assert classifier_instance.identify_grid2op_action_type(action_close_coupling) == "close_coupling"
    assert classifier_instance.identify_grid2op_action_type(action_unknown) == "unknown"

# --- Optional: Keep slow integration tests for Grid2Op object classification ---
@pytest.mark.slow
def test_grid2op_action_types_integration():
    """ Integration test using a real environment for object classification """
    env_folder = Path(__file__).parent.parent.resolve() / "data"
    env_name = "env_dijon_v2_assistant"
    if not (env_folder / env_name).exists(): pytest.skip("Env data not found")
    try: env = make_grid2op_training_env(env_folder, env_name)
    except Exception as e: pytest.skip(f"Could not create env: {e}")

    classifier = ActionClassifier(grid2op_action_space=env.action_space)

    # Test cases adapted from original file
    action_open_line_load = env.action_space({"set_bus": {"lines_ex_id": {"GEN.PY762": -1}, "loads_id": {"CHAV6L61GEN.P":-1}}})
    assert classifier.identify_grid2op_action_type(action_open_line_load) == "open_line_load"

    action_close_line = env.action_space({"set_bus": {"lines_or_id": {"CPVANL61PYMON": 1}, "lines_ex_id": {"CPVANL61PYMON": 1}}})
    assert classifier.identify_grid2op_action_type(action_close_line) == "close_line"

    action_open_coupling = env.action_space({"set_bus": {"lines_or_id": {"VOUGLY612": 1,"VOUGLY631": 1,"VOUGLY632": 2}, "lines_ex_id": {"FLEYRL61VOUGL": 2, "GEN.PL61VOUGL": 1, "PYMONL61VOUGL": 2}}})
    assert classifier.identify_grid2op_action_type(action_open_coupling) == "open_coupling"

    action_close_coupling = env.action_space({'set_bus': {'lines_or_id': {'CPVANL31RIBAU': 1}, 'lines_ex_id': {'BEON L31CPVAN': 1, 'CPVANY631': 1, 'CPVANY632': 1, 'CPVANY633': 1}, 'loads_id': {'ARBOIL31CPVAN': 1, 'BREVAL31CPVAN': 1, 'CPDIVL32CPVAN': 1, 'CPVANL31MESNA': 1, 'CPVANL31ZBRE6': 1, 'CPVAN3TR312': 1, 'CPVAN3TR311': 1}}})
    assert classifier.identify_grid2op_action_type(action_close_coupling) == "close_coupling"