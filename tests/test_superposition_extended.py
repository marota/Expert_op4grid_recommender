
import pytest
import numpy as np
from unittest.mock import MagicMock
from expert_op4grid_recommender.utils.superposition import (
    compute_all_pairs_superposition, 
    compute_combined_pair_superposition
)

def test_compute_pair_no_elements():
    # Mock environment and observations
    obs_start = MagicMock()
    obs_act1 = MagicMock()
    obs_act2 = MagicMock()
    
    # Case 1: Both actions have no elements
    result = compute_combined_pair_superposition(
        obs_start, obs_act1, obs_act2,
        act1_line_idxs=[], act1_sub_idxs=[],
        act2_line_idxs=[], act2_sub_idxs=[]
    )
    assert "error" in result
    assert "Cannot identify elements" in result["error"]

    # Case 2: One action has no elements
    result = compute_combined_pair_superposition(
        obs_start, obs_act1, obs_act2,
        act1_line_idxs=[1], act1_sub_idxs=[],
        act2_line_idxs=[], act2_sub_idxs=[]
    )
    assert "error" in result
    assert "Cannot identify elements" in result["error"]

def test_compute_pair_multiple_elements():
    # Verify that it uses the first element when multiple are provided
    obs_start = MagicMock()
    obs_start.p_or = np.array([100.0, 50.0])
    # Line 0 disconnected (status False), line 1 connected (status True)
    obs_start.line_status = np.array([False, True])

    obs_act1 = MagicMock()
    obs_act1.p_or = np.array([90.0, 50.0])
    # act1 reconnects line 0: status flips to True
    obs_act1.line_status = np.array([True, True])

    obs_act2 = MagicMock()
    obs_act2.p_or = np.array([100.0, 40.0])
    # act2 acts on line 1 (disconnects it): status flips to False
    obs_act2.line_status = np.array([False, False])
    
    # We need to mock get_delta_theta_line and other helper functions 
    # if we want to test the full logic, but compute_combined_pair_superposition 
    # calls them. For a unit test, we might need to mock them in the module.
    
    from expert_op4grid_recommender.utils import superposition
    with patch('expert_op4grid_recommender.utils.superposition.get_delta_theta_line', return_value=0.1), \
         patch('expert_op4grid_recommender.utils.superposition.get_betas_coeff', return_value=np.array([0.5, 0.5])):
        
        result = compute_combined_pair_superposition(
            obs_start, obs_act1, obs_act2,
            act1_line_idxs=[0, 1], act1_sub_idxs=[], # Multiple elements
            act2_line_idxs=[1], act2_sub_idxs=[]
        )
        
        assert "error" not in result
        assert "betas" in result
        # Check that it didn't crash and used 2 elements for the beta system
        assert len(result["betas"]) == 2

def test_compute_all_pairs_singular_system():
    # Mock environment and observations
    env = MagicMock()
    env.name_line = ["LINE1"]
    
    obs_start = MagicMock()
    obs_start.rho = np.array([1.1])
    
    aid1 = "act1"
    aid2 = "act2"
    
    detailed_actions = {
        aid1: {"action": MagicMock(), "observation": MagicMock()},
        aid2: {"action": MagicMock(), "observation": MagicMock()}
    }
    
    classifier = MagicMock()
    from expert_op4grid_recommender.utils import superposition
    
    with patch('expert_op4grid_recommender.utils.superposition._identify_action_elements', return_value=([0], [])), \
         patch('expert_op4grid_recommender.utils.superposition.compute_combined_pair_superposition', return_value={"error": "Singular system", "betas": [np.nan, np.nan]}):
        
        results = compute_all_pairs_superposition(
            obs_start=obs_start,
            detailed_actions=detailed_actions,
            classifier=classifier,
            env=env,
            lines_overloaded_ids=[0],
            lines_we_care_about=["LINE1"],
            pre_existing_rho={0: 0.5}
        )
        
        pair_id = f"{aid1}+{aid2}"
        assert pair_id in results
        assert "error" in results[pair_id]
        assert results[pair_id]["error"] == "Singular system"

def test_compute_pair_out_of_range_betas_returns_error():
    """When get_betas_coeff returns out-of-range betas, compute_combined_pair_superposition
    must return an error instead of propagating physically nonsensical values.

    This reproduces the real-world failure mode where two strongly coupled actions
    (e.g. a coupling action + node_merging) produce betas like [0.89, -4.25]
    (individual beta < -2.0) which are outside the valid [-2, 3] range and
    indicate the A-matrix has become ill-conditioned.
    """
    obs_start = MagicMock()
    obs_start.p_or = np.array([100.0, 50.0])
    obs_start.line_status = np.array([True, False])   # line 0 connected, line 1 disconnected

    obs_act1 = MagicMock()
    obs_act1.p_or = np.array([80.0, 50.0])
    obs_act1.line_status = np.array([False, False])   # act1 disconnects line 0

    obs_act2 = MagicMock()
    obs_act2.p_or = np.array([100.0, 60.0])
    obs_act2.line_status = np.array([True, True])     # act2 reconnects line 1

    from expert_op4grid_recommender.utils import superposition

    # Inject out-of-range betas (simulates ill-conditioned A-matrix)
    # Valid range is [-2, 3] for individual betas
    for bad_betas in [
        np.array([0.89, -4.25]),  # betas[1]=-4.25 < -2.0 — matches node_merging case
        np.array([4.0, 4.0]),     # both > 3.0 — clearly out of range
        np.array([3.5, -2.5]),    # both out of range
        np.array([0.5, -3.0]),    # betas[1]=-3.0 < -2.0
    ]:
        with patch('expert_op4grid_recommender.utils.superposition.get_betas_coeff',
                   return_value=bad_betas), \
             patch('expert_op4grid_recommender.utils.superposition.get_delta_theta_line',
                   return_value=0.05):

            result = compute_combined_pair_superposition(
                obs_start, obs_act1, obs_act2,
                act1_line_idxs=[0], act1_sub_idxs=[],
                act2_line_idxs=[1], act2_sub_idxs=[],
            )

            assert "error" in result, (
                f"Expected error for out-of-range betas {bad_betas}, got: {result}"
            )
            assert "Unreliable" in result["error"], (
                f"Error message should mention 'Unreliable', got: {result['error']}"
            )
            assert "betas" in result, "Error result should still include betas for debugging"


def test_compute_pair_in_range_betas_no_error():
    """When betas are within [-2, 3], the result should be valid (no error)."""
    obs_start = MagicMock()
    obs_start.p_or = np.array([100.0, 50.0])
    obs_start.p_ex = np.array([-100.0, -50.0])
    obs_start.rho = np.array([0.8, 0.5])
    obs_start.line_status = np.array([True, False])

    obs_act1 = MagicMock()
    obs_act1.p_or = np.array([80.0, 50.0])
    obs_act1.p_ex = np.array([-80.0, -50.0])
    obs_act1.rho = np.array([0.7, 0.5])
    obs_act1.line_status = np.array([False, False])

    obs_act2 = MagicMock()
    obs_act2.p_or = np.array([100.0, 60.0])
    obs_act2.p_ex = np.array([-100.0, -60.0])
    obs_act2.rho = np.array([0.8, 0.6])
    obs_act2.line_status = np.array([True, True])

    from expert_op4grid_recommender.utils import superposition

    # Normal betas — within [-2, 3] range
    for good_betas in [
        np.array([0.5, 0.5]),
        np.array([0.9, 0.1]),
        np.array([-1.5, 0.5]),  # in range (>= -2)
        np.array([2.8, 0.1]),   # in range (<= 3)
        np.array([1.46, 0.60]), # realistic case from disco_BEON+reco_CHALOL
    ]:
        with patch('expert_op4grid_recommender.utils.superposition.get_betas_coeff',
                   return_value=good_betas), \
             patch('expert_op4grid_recommender.utils.superposition.get_delta_theta_line',
                   return_value=0.05):

            result = compute_combined_pair_superposition(
                obs_start, obs_act1, obs_act2,
                act1_line_idxs=[0], act1_sub_idxs=[],
                act2_line_idxs=[1], act2_sub_idxs=[],
            )

            assert "error" not in result, (
                f"Unexpected error for valid betas {good_betas}: {result.get('error')}"
            )
            assert "betas" in result


from unittest.mock import patch

if __name__ == "__main__":
    pytest.main([__file__])
