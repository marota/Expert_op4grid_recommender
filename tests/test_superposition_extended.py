
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

from unittest.mock import patch

if __name__ == "__main__":
    pytest.main([__file__])
