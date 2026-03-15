
import pytest
import numpy as np
from unittest.mock import MagicMock
from expert_op4grid_recommender.utils.superposition import compute_all_pairs_superposition
from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

def test_compute_all_pairs_superposition_simplified_dict():
    # Mock environment and observations
    env = MagicMock()
    env.name_line = ["LINE1", "LINE2", "LINE3"]

    obs_start = MagicMock()
    obs_start.rho = np.array([1.1, 0.5, 0.5]) # Overload on LINE1
    obs_start.p_or = np.array([100.0, 50.0, 50.0])

    # Mock unitary actions
    aid1 = "reco_LINE2"
    aid2 = "reco_LINE3"

    obs_act1 = MagicMock()
    obs_act1.rho = np.array([1.0, 0.5, 0.5])
    obs_act1.p_or = np.array([90.0, 50.0, 50.0])

    obs_act2 = MagicMock()
    obs_act2.rho = np.array([1.0, 0.5, 0.5])
    obs_act2.p_or = np.array([90.0, 50.0, 50.0])

    detailed_actions = {
        aid1: {
            "action": MagicMock(),
            "observation": obs_act1,
            "description_unitaire": "Close LINE2"
        },
        aid2: {
            "action": MagicMock(),
            "observation": obs_act2,
            "description_unitaire": "Close LINE3"
        }
    }

    classifier = MagicMock(spec=ActionClassifier)
    classifier.identify_action_type.return_value = "close_line"
    classifier._action_space = MagicMock()

    from expert_op4grid_recommender.utils import superposition

    def mock_identify(action, action_id, *args):
        if action_id == aid1: return ([1], [])
        if action_id == aid2: return ([2], [])
        return ([], [])

    # Save originals and restore them after the test so other tests are not affected
    orig_identify = superposition._identify_action_elements
    orig_compute = superposition.compute_combined_pair_superposition
    try:
        superposition._identify_action_elements = MagicMock(side_effect=mock_identify)
        # Mock compute_combined_pair_superposition to avoid real linear solving
        superposition.compute_combined_pair_superposition = MagicMock(return_value={
            "betas": [0.5, 0.5],
            "p_or_combined": [80.0, 50.0, 50.0]
        })

        results = compute_all_pairs_superposition(
            obs_start=obs_start,
            detailed_actions=detailed_actions,
            classifier=classifier,
            env=env,
            lines_overloaded_ids=[0],  # LINE1
            lines_we_care_about=["LINE1", "LINE2", "LINE3"],
            pre_existing_rho={i: 0.5 for i in range(3)},
            dict_action={}
        )

        assert len(results) == 1
        pair_id = f"{aid1}+{aid2}"
        assert pair_id in results

        res = results[pair_id]

        # Check present keys
        assert "betas" in res
        assert "max_rho" in res
        assert "max_rho_line" in res
        assert "is_rho_reduction" in res
        assert "description" in res
        assert "action1_id" in res
        assert "action2_id" in res

        # Check keys that MUST be removed
        assert "p_or_combined" not in res
        assert "rho_after" not in res
        assert "rho_before" not in res

        print("\n[Test] Simplified dictionary verified successfully.")
    finally:
        superposition._identify_action_elements = orig_identify
        superposition.compute_combined_pair_superposition = orig_compute


if __name__ == "__main__":
    test_compute_all_pairs_superposition_simplified_dict()
