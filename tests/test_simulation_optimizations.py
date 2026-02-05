# tests/test_simulation_optimizations.py
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Comprehensive tests for the optimized simulation functions:
- compute_baseline_simulation
- check_rho_reduction_with_baseline
- ActionRuleValidator caching optimizations
"""

import pytest
import os
import sys
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check if simulation module can be imported (doesn't have grid2op dependency issues)
def _check_simulation_import():
    """Check if simulation functions can be imported."""
    try:
        from expert_op4grid_recommender.utils.simulation import compute_baseline_simulation
        return True
    except Exception:
        return False


# Note: ActionRuleValidator tests require full grid2op environment.
# These are marked to skip in CI environments where grid2op has dependency issues.
# The tests will run properly in environments with correctly installed grid2op.
SKIP_RULES_TESTS = True  # Set to False when running in full environment
RULES_SKIP_REASON = "ActionRuleValidator requires full grid2op environment (set SKIP_RULES_TESTS=False to run)"


# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockAction:
    """Mock action object that supports addition and has all required attributes."""

    # Class attributes needed by aux_prevent_line_reconnection
    n_line = 3
    SUB_COL = 0
    grid_objects_types = np.array([[0], [0], [1]])  # Mock grid objects
    line_or_to_subid = np.array([0, 0, 1])
    line_ex_to_subid = np.array([1, 1, 0])

    @classmethod
    def get_line_info(cls, line_name=None, line_id=None):
        """Mock get_line_info class method."""
        # Return (line_id, sub_or, sub_ex, connected)
        return (0, 0, 1, True)

    def __init__(self, name: str = "mock_action"):
        self.name = name
        self.content = {}

        # Attributes needed by aux_prevent_asset_reconnection
        self.gen_set_bus = np.array([0])
        self.load_set_bus = np.array([0])

        # Attributes needed by aux_prevent_line_reconnection
        self.line_or_set_bus = np.array([0, 0, 0])
        self.line_ex_set_bus = np.array([0, 0, 0])
        self.line_or_change_bus = np.array([False, False, False])
        self.line_ex_change_bus = np.array([False, False, False])
        self.line_change_status = np.array([False, False, False])
        self.line_set_status = np.array([0, 0, 0])
        self.set_bus = np.array([0, 0, 0])
        self.set_line_status = np.array([0, 0, 0])

    def __add__(self, other):
        combined = MockAction(f"{self.name}+{other.name}")
        combined.content = {**self.content, **other.content}
        return combined

    def __repr__(self):
        return f"MockAction({self.name})"

    def as_dict(self):
        """Return action as dictionary (required by some code paths)."""
        return self.content

    def update(self, action_dict):
        """Update action with new values (required by aux_prevent_asset_reconnection)."""
        self.content.update(action_dict)

    def remove_line_status_from_topo(self, check_cooldown=False):
        """Mock method for line status removal."""
        pass


class MockActionSpace:
    """Mock action space that creates MockAction objects."""
    def __call__(self, action_dict: Dict) -> MockAction:
        action = MockAction("created_action")
        action.content = action_dict
        return action


class MockObservationForSimulation:
    """
    Mock observation with configurable simulation behavior.
    Allows testing different simulation outcomes.
    """
    def __init__(
        self,
        name_line: List[str],
        name_sub: List[str],
        rho: np.ndarray,
        line_status: np.ndarray = None,
        simulate_returns: List[Tuple] = None,
        simulate_exception: bool = False
    ):
        self.name_line = np.array(name_line)
        self.name_sub = np.array(name_sub)
        self.rho = np.array(rho)
        self.line_status = line_status if line_status is not None else np.ones(len(name_line), dtype=bool)

        # Add attributes needed by aux_prevent_asset_reconnection
        self.gen_bus = np.array([1])  # All generators connected to bus 1
        self.load_bus = np.array([1])  # All loads connected to bus 1

        # Track simulation calls for verification
        self._simulate_call_count = 0
        self._simulate_calls = []

        # Configure simulation returns
        self._simulate_returns = simulate_returns or []
        self._simulate_exception = simulate_exception

        # Default return if not configured
        self._default_return = (self, 0.0, False, {"exception": []})

    def simulate(self, action, time_step: int = 0):
        """Mock simulate method that tracks calls and returns configured results."""
        self._simulate_call_count += 1
        self._simulate_calls.append({
            'action': action,
            'time_step': time_step,
            'call_number': self._simulate_call_count
        })

        # Return exception if configured
        if self._simulate_exception:
            return (self, 0.0, False, {"exception": ["Simulation failed"]})

        # Return from configured returns if available
        if self._simulate_returns and self._simulate_call_count <= len(self._simulate_returns):
            return self._simulate_returns[self._simulate_call_count - 1]

        return self._default_return

    def sub_topology(self, sub_id: int) -> List[int]:
        """Return mock topology for a substation."""
        return [1, 1]  # Default: all on bus 1

    def reset_simulate_tracking(self):
        """Reset simulation call tracking."""
        self._simulate_call_count = 0
        self._simulate_calls = []


class MockObservationWithRhoChange(MockObservationForSimulation):
    """
    Mock observation that returns different rho values on simulation.
    Useful for testing rho reduction checks.

    When skip_baseline=False (default):
        - First simulate() call returns baseline_rho (for compute_baseline_simulation)
        - Subsequent calls return candidate_rho (for check_rho_reduction_with_baseline)

    When skip_baseline=True:
        - All simulate() calls return candidate_rho (for direct check_rho_reduction_with_baseline tests)
    """
    def __init__(
        self,
        name_line: List[str],
        name_sub: List[str],
        initial_rho: np.ndarray,
        baseline_rho: np.ndarray,
        candidate_rho: np.ndarray,
        baseline_exception: bool = False,
        candidate_exception: bool = False,
        skip_baseline: bool = False
    ):
        super().__init__(name_line, name_sub, initial_rho)
        self._initial_rho = np.array(initial_rho)
        self._baseline_rho = np.array(baseline_rho)
        self._candidate_rho = np.array(candidate_rho)
        self._baseline_exception = baseline_exception
        self._candidate_exception = candidate_exception
        self._skip_baseline = skip_baseline

    def simulate(self, action, time_step: int = 0):
        self._simulate_call_count += 1
        self._simulate_calls.append({
            'action': action,
            'time_step': time_step,
            'call_number': self._simulate_call_count
        })

        # Determine if this is a baseline or candidate simulation
        is_baseline_call = (self._simulate_call_count == 1) and not self._skip_baseline

        if is_baseline_call:
            if self._baseline_exception:
                return (self, 0.0, False, {"exception": ["Baseline simulation failed"]})
            # Return observation with baseline rho
            result_obs = MockObservationForSimulation(
                list(self.name_line), list(self.name_sub), self._baseline_rho
            )
            return (result_obs, 0.0, False, {"exception": []})
        else:
            # Candidate simulation
            if self._candidate_exception:
                return (self, 0.0, False, {"exception": ["Candidate simulation failed"]})
            # Return observation with candidate rho
            result_obs = MockObservationForSimulation(
                list(self.name_line), list(self.name_sub), self._candidate_rho
            )
            return (result_obs, 0.0, False, {"exception": []})


# ============================================================================
# Tests for compute_baseline_simulation
# ============================================================================

@pytest.mark.skipif(not _check_simulation_import(), reason="Cannot import simulation functions")
class TestComputeBaselineSimulation:
    """Tests for the compute_baseline_simulation function."""

    def test_successful_baseline_computation(self):
        """Test that baseline is computed correctly on success."""
        from expert_op4grid_recommender.utils.simulation import compute_baseline_simulation

        # Setup
        baseline_rho_values = np.array([0.8, 0.9, 0.7, 0.6])
        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2", "L3", "L4"],
            name_sub=["S1", "S2"],
            initial_rho=np.array([0.5, 0.5, 0.5, 0.5]),
            baseline_rho=baseline_rho_values,
            candidate_rho=np.array([0.6, 0.7, 0.5, 0.4])
        )
        act_defaut = MockAction("defaut")
        act_reco = MockAction("reco")
        overload_ids = [0, 1]  # Lines L1 and L2

        # Execute
        baseline_rho, obs_baseline = compute_baseline_simulation(
            obs, timestep=1, act_defaut=act_defaut,
            act_reco_maintenance=act_reco, overload_ids=overload_ids
        )

        # Verify
        assert baseline_rho is not None
        assert obs_baseline is not None
        np.testing.assert_array_almost_equal(baseline_rho, baseline_rho_values[overload_ids])
        assert obs._simulate_call_count == 1  # Only one simulation call

    def test_baseline_computation_with_exception(self):
        """Test that baseline returns None when simulation fails."""
        from expert_op4grid_recommender.utils.simulation import compute_baseline_simulation

        # Setup with exception
        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2"],
            name_sub=["S1"],
            initial_rho=np.array([0.5, 0.5]),
            baseline_rho=np.array([0.8, 0.9]),
            candidate_rho=np.array([0.6, 0.7]),
            baseline_exception=True
        )

        # Execute
        baseline_rho, obs_baseline = compute_baseline_simulation(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            act_reco_maintenance=MockAction("reco"), overload_ids=[0, 1]
        )

        # Verify
        assert baseline_rho is None
        assert obs_baseline is None

    def test_baseline_extracts_correct_overload_ids(self):
        """Test that baseline correctly extracts rho for specified overload IDs."""
        from expert_op4grid_recommender.utils.simulation import compute_baseline_simulation

        # Setup with 5 lines, only check 2 of them
        baseline_rho_all = np.array([0.5, 0.6, 0.95, 0.7, 0.85])
        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2", "L3", "L4", "L5"],
            name_sub=["S1", "S2"],
            initial_rho=np.zeros(5),
            baseline_rho=baseline_rho_all,
            candidate_rho=np.zeros(5)
        )
        overload_ids = [2, 4]  # L3 and L5 are overloaded

        # Execute
        baseline_rho, _ = compute_baseline_simulation(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            act_reco_maintenance=MockAction("reco"), overload_ids=overload_ids
        )

        # Verify - should only have rho for lines 2 and 4
        expected = np.array([0.95, 0.85])
        np.testing.assert_array_almost_equal(baseline_rho, expected)


# ============================================================================
# Tests for check_rho_reduction_with_baseline
# ============================================================================

@pytest.mark.skipif(not _check_simulation_import(), reason="Cannot import simulation functions")
class TestCheckRhoReductionWithBaseline:
    """Tests for the check_rho_reduction_with_baseline function."""

    def test_rho_reduction_detected(self):
        """Test that rho reduction is correctly detected."""
        from expert_op4grid_recommender.utils.simulation import check_rho_reduction_with_baseline

        # Setup - candidate rho is lower than baseline
        baseline_rho = np.array([0.95, 0.90])
        candidate_rho = np.array([0.80, 0.75])  # Reduced

        # Use skip_baseline=True since check_rho_reduction_with_baseline doesn't do baseline sim
        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2", "L3"],
            name_sub=["S1", "S2"],
            initial_rho=np.array([0.5, 0.5, 0.5]),
            baseline_rho=baseline_rho,
            candidate_rho=np.concatenate([candidate_rho, [0.5]]),  # Full array
            skip_baseline=True  # First simulate call returns candidate_rho
        )

        # Execute
        is_reduction, obs_result = check_rho_reduction_with_baseline(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            action=MockAction("candidate"), overload_ids=[0, 1],
            act_reco_maintenance=MockAction("reco"),
            baseline_rho=baseline_rho,
            verbose=False
        )

        # Verify
        assert is_reduction is True
        assert obs_result is not None

    def test_no_rho_reduction(self):
        """Test that no reduction is detected when rho doesn't decrease enough."""
        from expert_op4grid_recommender.utils.simulation import check_rho_reduction_with_baseline

        # Setup - candidate rho is same or higher
        baseline_rho = np.array([0.90, 0.85])
        candidate_rho = np.array([0.91, 0.86])  # Not reduced

        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2"],
            name_sub=["S1"],
            initial_rho=np.array([0.5, 0.5]),
            baseline_rho=baseline_rho,
            candidate_rho=candidate_rho,
            skip_baseline=True  # First simulate call returns candidate_rho
        )

        # Execute
        is_reduction, obs_result = check_rho_reduction_with_baseline(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            action=MockAction("candidate"), overload_ids=[0, 1],
            act_reco_maintenance=MockAction("reco"),
            baseline_rho=baseline_rho,
            verbose=False
        )

        # Verify
        assert is_reduction is False
        assert obs_result is not None

    def test_rho_tolerance_boundary(self):
        """Test rho reduction at tolerance boundary."""
        from expert_op4grid_recommender.utils.simulation import check_rho_reduction_with_baseline

        baseline_rho = np.array([1.0])
        tolerance = 0.01

        # Test just below tolerance - should NOT be detected as reduction
        candidate_rho_at_boundary = np.array([0.995])  # 0.5% reduction, less than 1%
        obs = MockObservationWithRhoChange(
            name_line=["L1"],
            name_sub=["S1"],
            initial_rho=np.array([0.5]),
            baseline_rho=baseline_rho,
            candidate_rho=candidate_rho_at_boundary,
            skip_baseline=True  # First simulate call returns candidate_rho
        )

        is_reduction, _ = check_rho_reduction_with_baseline(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            action=MockAction("candidate"), overload_ids=[0],
            act_reco_maintenance=MockAction("reco"),
            baseline_rho=baseline_rho,
            rho_tolerance=tolerance,
            verbose=False
        )
        assert is_reduction is False

        # Test just above tolerance - should be detected as reduction
        candidate_rho_above = np.array([0.98])  # 2% reduction, more than 1%
        obs2 = MockObservationWithRhoChange(
            name_line=["L1"],
            name_sub=["S1"],
            initial_rho=np.array([0.5]),
            baseline_rho=baseline_rho,
            candidate_rho=candidate_rho_above,
            skip_baseline=True  # First simulate call returns candidate_rho
        )

        is_reduction2, _ = check_rho_reduction_with_baseline(
            obs2, timestep=1, act_defaut=MockAction("defaut"),
            action=MockAction("candidate"), overload_ids=[0],
            act_reco_maintenance=MockAction("reco"),
            baseline_rho=baseline_rho,
            rho_tolerance=tolerance,
            verbose=False
        )
        assert is_reduction2 is True

    def test_candidate_simulation_exception(self):
        """Test handling of candidate simulation exception."""
        from expert_op4grid_recommender.utils.simulation import check_rho_reduction_with_baseline

        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2"],
            name_sub=["S1"],
            initial_rho=np.array([0.5, 0.5]),
            baseline_rho=np.array([0.9, 0.9]),
            candidate_rho=np.array([0.7, 0.7]),
            candidate_exception=True,
            skip_baseline=True  # First simulate call is candidate (which will fail)
        )

        is_reduction, obs_result = check_rho_reduction_with_baseline(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            action=MockAction("candidate"), overload_ids=[0, 1],
            act_reco_maintenance=MockAction("reco"),
            baseline_rho=np.array([0.9, 0.9]),
            verbose=False
        )

        assert is_reduction is False
        assert obs_result is not None  # Still returns observation even on failure

    def test_partial_reduction_fails(self):
        """Test that partial reduction (not all lines) returns False."""
        from expert_op4grid_recommender.utils.simulation import check_rho_reduction_with_baseline

        # Only one of two lines reduces
        baseline_rho = np.array([0.95, 0.90])
        candidate_rho = np.array([0.80, 0.92])  # L1 reduces, L2 doesn't

        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2"],
            name_sub=["S1"],
            initial_rho=np.array([0.5, 0.5]),
            baseline_rho=baseline_rho,
            candidate_rho=candidate_rho,
            skip_baseline=True  # First simulate call returns candidate_rho
        )

        is_reduction, _ = check_rho_reduction_with_baseline(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            action=MockAction("candidate"), overload_ids=[0, 1],
            act_reco_maintenance=MockAction("reco"),
            baseline_rho=baseline_rho,
            verbose=False
        )

        assert is_reduction is False


# ============================================================================
# Tests for check_rho_reduction (wrapper function)
# ============================================================================

@pytest.mark.skipif(not _check_simulation_import(), reason="Cannot import simulation functions")
class TestCheckRhoReduction:
    """Tests for the check_rho_reduction wrapper function."""

    def test_wrapper_calls_both_functions(self):
        """Test that wrapper correctly calls baseline and with_baseline functions."""
        from expert_op4grid_recommender.utils.simulation import check_rho_reduction

        # Setup
        baseline_rho = np.array([0.95, 0.90])
        candidate_rho = np.array([0.80, 0.75])

        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2"],
            name_sub=["S1"],
            initial_rho=np.array([0.5, 0.5]),
            baseline_rho=baseline_rho,
            candidate_rho=candidate_rho
        )

        is_reduction, obs_result = check_rho_reduction(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            action=MockAction("candidate"), overload_ids=[0, 1],
            act_reco_maintenance=MockAction("reco")
        )

        # Should have made 2 simulation calls (1 baseline + 1 candidate)
        assert obs._simulate_call_count == 2
        assert is_reduction is True

    def test_wrapper_returns_none_on_baseline_failure(self):
        """Test wrapper returns (False, None) when baseline fails."""
        from expert_op4grid_recommender.utils.simulation import check_rho_reduction

        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2"],
            name_sub=["S1"],
            initial_rho=np.array([0.5, 0.5]),
            baseline_rho=np.array([0.9, 0.9]),
            candidate_rho=np.array([0.7, 0.7]),
            baseline_exception=True
        )

        is_reduction, obs_result = check_rho_reduction(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            action=MockAction("candidate"), overload_ids=[0, 1],
            act_reco_maintenance=MockAction("reco")
        )

        assert is_reduction is False
        assert obs_result is None


# ============================================================================
# Tests for ActionRuleValidator Caching
# ============================================================================

@pytest.mark.skipif(SKIP_RULES_TESTS, reason=RULES_SKIP_REASON)
class TestActionRuleValidatorCaching:
    """Tests for the caching optimizations in ActionRuleValidator."""

    @pytest.fixture
    def mock_obs_for_validator(self):
        """Create a mock observation for validator tests."""
        return MockObservationForSimulation(
            name_line=["L1", "L2", "L3", "L4", "L5"],
            name_sub=["S1", "S2", "S3", "S4"],
            rho=np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            line_status=np.array([True, True, False, True, True])
        )

    def test_sets_are_precomputed(self, mock_obs_for_validator):
        """Test that path sets are pre-computed in __init__."""
        from expert_op4grid_recommender.action_evaluation.rules import ActionRuleValidator
        from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

        paths = (
            (["L1", "L2"], ["S1", "S2"]),  # Constrained
            (["L3", "L4"], ["S3"])          # Dispatch
        )

        validator = ActionRuleValidator(
            obs=mock_obs_for_validator,
            action_space=MockActionSpace(),
            classifier=ActionClassifier(MockActionSpace()),
            hubs=["S1"],
            paths=paths,
            by_description=True
        )

        # Verify sets are created
        assert hasattr(validator, '_lines_constrained_set')
        assert hasattr(validator, '_lines_dispatch_set')
        assert hasattr(validator, '_hubs_set')
        assert hasattr(validator, '_nodes_constrained_set')
        assert hasattr(validator, '_nodes_dispatch_set')

        # Verify set contents
        assert validator._lines_constrained_set == {"L1", "L2"}
        assert validator._lines_dispatch_set == {"L3", "L4"}
        assert validator._hubs_set == {"S1"}
        assert validator._nodes_constrained_set == {"S1", "S2"}
        assert validator._nodes_dispatch_set == {"S3"}

    def test_sub_name_to_idx_cache(self, mock_obs_for_validator):
        """Test that sub_name to index mapping is cached."""
        from expert_op4grid_recommender.action_evaluation.rules import ActionRuleValidator
        from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

        validator = ActionRuleValidator(
            obs=mock_obs_for_validator,
            action_space=MockActionSpace(),
            classifier=ActionClassifier(MockActionSpace()),
            hubs=["S1"],
            paths=(([], []), ([], [])),
            by_description=True
        )

        # Verify cache exists
        assert hasattr(validator, '_sub_name_to_idx')

        # Verify cache contents
        expected = {"S1": 0, "S2": 1, "S3": 2, "S4": 3}
        assert validator._sub_name_to_idx == expected

    def test_line_status_map_cache(self, mock_obs_for_validator):
        """Test that line status map is cached."""
        from expert_op4grid_recommender.action_evaluation.rules import ActionRuleValidator
        from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

        validator = ActionRuleValidator(
            obs=mock_obs_for_validator,
            action_space=MockActionSpace(),
            classifier=ActionClassifier(MockActionSpace()),
            hubs=[],
            paths=(([], []), ([], [])),
            by_description=True
        )

        # Verify cache exists
        assert hasattr(validator, '_line_status_map')

        # Verify cache contents match observation
        expected = {"L1": True, "L2": True, "L3": False, "L4": True, "L5": True}
        assert validator._line_status_map == expected

    def test_localize_line_uses_sets(self, mock_obs_for_validator):
        """Test that localize_line_action uses pre-computed sets."""
        from expert_op4grid_recommender.action_evaluation.rules import ActionRuleValidator
        from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

        paths = (
            (["L1", "L2"], ["S1"]),   # Constrained
            (["L3", "L4"], ["S2"])    # Dispatch
        )

        validator = ActionRuleValidator(
            obs=mock_obs_for_validator,
            action_space=MockActionSpace(),
            classifier=ActionClassifier(MockActionSpace()),
            hubs=["S1"],
            paths=paths,
            by_description=True
        )

        # Test localization
        assert validator.localize_line_action(["L1"]) == "constrained_path"
        assert validator.localize_line_action(["L2"]) == "constrained_path"
        assert validator.localize_line_action(["L3"]) == "dispatch_path"
        assert validator.localize_line_action(["L4"]) == "dispatch_path"
        assert validator.localize_line_action(["L5"]) == "out_of_graph"

        # Test with multiple lines - first match wins
        assert validator.localize_line_action(["L5", "L1"]) == "constrained_path"
        assert validator.localize_line_action(["L5", "L3"]) == "dispatch_path"

    def test_localize_coupling_uses_sets(self, mock_obs_for_validator):
        """Test that localize_coupling_action uses pre-computed sets."""
        from expert_op4grid_recommender.action_evaluation.rules import ActionRuleValidator
        from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

        paths = (
            (["L1"], ["S2"]),     # Constrained path includes S2
            (["L2"], ["S3"])      # Dispatch path includes S3
        )

        validator = ActionRuleValidator(
            obs=mock_obs_for_validator,
            action_space=MockActionSpace(),
            classifier=ActionClassifier(MockActionSpace()),
            hubs=["S1"],
            paths=paths,
            by_description=True
        )

        # Test localization - hubs first, then constrained, then dispatch
        assert validator.localize_coupling_action(["S1"]) == "hubs"
        assert validator.localize_coupling_action(["S2"]) == "constrained_path"
        assert validator.localize_coupling_action(["S3"]) == "dispatch_path"
        assert validator.localize_coupling_action(["S4"]) == "out_of_graph"


# ============================================================================
# Performance Tests (verify optimization actually helps)
# ============================================================================

@pytest.mark.skipif(SKIP_RULES_TESTS, reason=RULES_SKIP_REASON)
class TestPerformanceOptimizations:
    """Tests to verify performance optimizations work correctly."""

    def test_baseline_computed_once_in_categorize_actions(self):
        """Test that baseline is only computed once for multiple filtered actions."""
        from expert_op4grid_recommender.action_evaluation.rules import ActionRuleValidator
        from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

        # Create observation that tracks simulation calls
        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2", "L3"],
            name_sub=["S1", "S2"],
            initial_rho=np.array([0.5, 0.5, 0.5]),
            baseline_rho=np.array([0.9, 0.9, 0.9]),
            candidate_rho=np.array([0.7, 0.7, 0.7])
        )

        paths = (
            (["L1"], ["S1"]),   # Constrained
            (["L2"], ["S2"])    # Dispatch
        )

        validator = ActionRuleValidator(
            obs=obs,
            action_space=MockActionSpace(),
            classifier=ActionClassifier(MockActionSpace()),
            hubs=[],
            paths=paths,
            by_description=True
        )

        # Create multiple actions that will be filtered (dispatch path line opens)
        dict_action = {
            "action_1": {
                "description_unitaire": "Ouverture L2",
                "content": {"set_bus": {"lines_ex_id": {"L2": -1}}}
            },
            "action_2": {
                "description_unitaire": "Ouverture L2 variant",
                "content": {"set_bus": {"lines_ex_id": {"L2": -1}}}
            },
            "action_3": {
                "description_unitaire": "Ouverture L2 another",
                "content": {"set_bus": {"lines_ex_id": {"L2": -1}}}
            }
        }

        # Execute with simulation checks enabled
        obs.reset_simulate_tracking()
        actions_filtered, actions_unfiltered = validator.categorize_actions(
            dict_action=dict_action,
            timestep=1,
            defauts=["L1"],
            overload_ids=[0, 1],
            lines_reco_maintenance=[],
            do_simulation_checks=True
        )

        # With optimization: 1 baseline + N candidate simulations
        # Without optimization: N+1 baseline + N candidate simulations (2N+1 total)
        # Here N=3 filtered actions
        # Expected with optimization: 1 + 3 = 4 simulations
        # Without optimization would be: 4 + 3 = 7 simulations (or 3*2=6 if baseline in check_rho)

        # All 3 actions should be filtered
        assert len(actions_filtered) == 3

        # Verify we made significantly fewer simulation calls than naive approach
        # The exact number depends on implementation, but should be close to N+1
        # where N is number of filtered actions
        print(f"Simulation calls made: {obs._simulate_call_count}")
        assert obs._simulate_call_count <= 5  # Should be 1 baseline + up to 3 candidates + 1 (safety margin)

    def test_no_simulation_when_disabled(self):
        """Test that no simulations occur when simulation checks are disabled."""
        from expert_op4grid_recommender.action_evaluation.rules import ActionRuleValidator
        from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

        obs = MockObservationForSimulation(
            name_line=["L1", "L2"],
            name_sub=["S1"],
            rho=np.array([0.5, 0.5])
        )

        validator = ActionRuleValidator(
            obs=obs,
            action_space=MockActionSpace(),
            classifier=ActionClassifier(MockActionSpace()),
            hubs=[],
            paths=((["L1"], []), (["L2"], [])),
            by_description=True
        )

        dict_action = {
            "action_1": {
                "description_unitaire": "Ouverture L2",
                "content": {"set_bus": {"lines_ex_id": {"L2": -1}}}
            }
        }

        # Execute with simulation checks disabled
        obs.reset_simulate_tracking()
        validator.categorize_actions(
            dict_action=dict_action,
            timestep=1,
            defauts=["L1"],
            overload_ids=[0],
            lines_reco_maintenance=[],
            do_simulation_checks=False
        )

        # Should not have made any simulation calls
        assert obs._simulate_call_count == 0


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.skipif(not _check_simulation_import(), reason="Cannot import simulation functions")
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_overload_ids(self):
        """Test handling of empty overload IDs."""
        from expert_op4grid_recommender.utils.simulation import compute_baseline_simulation

        obs = MockObservationWithRhoChange(
            name_line=["L1", "L2"],
            name_sub=["S1"],
            initial_rho=np.array([0.5, 0.5]),
            baseline_rho=np.array([0.9, 0.9]),
            candidate_rho=np.array([0.7, 0.7])
        )

        baseline_rho, _ = compute_baseline_simulation(
            obs, timestep=1, act_defaut=MockAction("defaut"),
            act_reco_maintenance=MockAction("reco"), overload_ids=[]
        )

        # Should return empty array, not fail
        assert baseline_rho is not None
        assert len(baseline_rho) == 0

    @pytest.mark.skipif(SKIP_RULES_TESTS, reason=RULES_SKIP_REASON)
    def test_empty_paths_in_validator(self):
        """Test validator with empty paths."""

        obs = MockObservationForSimulation(
            name_line=["L1", "L2"],
            name_sub=["S1"],
            rho=np.array([0.5, 0.5])
        )

        # Empty paths
        paths = (([], []), ([], []))

        validator = ActionRuleValidator(
            obs=obs,
            action_space=MockActionSpace(),
            classifier=ActionClassifier(MockActionSpace()),
            hubs=[],
            paths=paths,
            by_description=True
        )

        # Everything should be out_of_graph
        assert validator.localize_line_action(["L1"]) == "out_of_graph"
        assert validator.localize_coupling_action(["S1"]) == "out_of_graph"

    @pytest.mark.skipif(SKIP_RULES_TESTS, reason=RULES_SKIP_REASON)
    def test_large_number_of_paths(self):
        """Test validator with large number of paths for performance."""

        # Create large lists
        n_lines = 1000
        n_subs = 200

        lines = [f"L{i}" for i in range(n_lines)]
        subs = [f"S{i}" for i in range(n_subs)]

        obs = MockObservationForSimulation(
            name_line=lines,
            name_sub=subs,
            rho=np.ones(n_lines) * 0.5
        )

        # Split lines between constrained and dispatch
        constrained_lines = lines[:500]
        dispatch_lines = lines[500:]
        constrained_nodes = subs[:100]
        dispatch_nodes = subs[100:]

        paths = (
            (constrained_lines, constrained_nodes),
            (dispatch_lines, dispatch_nodes)
        )

        validator = ActionRuleValidator(
            obs=obs,
            action_space=MockActionSpace(),
            classifier=ActionClassifier(MockActionSpace()),
            hubs=["S0"],
            paths=paths,
            by_description=True
        )

        # Verify lookups work correctly
        assert validator.localize_line_action(["L0"]) == "constrained_path"
        assert validator.localize_line_action(["L500"]) == "dispatch_path"
        assert validator.localize_coupling_action(["S0"]) == "hubs"
        assert validator.localize_coupling_action(["S50"]) == "constrained_path"
        assert validator.localize_coupling_action(["S150"]) == "dispatch_path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
