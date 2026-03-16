
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
    obs_start.rho = np.array([1.1, 0.5, 0.5])  # Overload on LINE1
    obs_start.p_or = np.array([100.0, 50.0, 50.0])
    obs_start.p_ex = np.array([-100.0, -50.0, -50.0])
    # Per-extremity current and limit data for _estimate_rho_from_p
    obs_start.a_or = np.array([110.0, 50.0, 50.0])
    obs_start.a_ex = np.array([110.0, 50.0, 50.0])
    limit_or_mock = MagicMock()
    limit_or_mock.values = np.array([100.0, 100.0, 100.0])
    limit_ex_mock = MagicMock()
    limit_ex_mock.values = np.array([100.0, 100.0, 100.0])
    obs_start._limit_or = limit_or_mock
    obs_start._limit_ex = limit_ex_mock

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
        # Mock compute_combined_pair_superposition to avoid real linear solving.
        # Include p_ex_combined for completeness (needed when use_p_based_rho=True).
        superposition.compute_combined_pair_superposition = MagicMock(return_value={
            "betas": [0.5, 0.5],
            "p_or_combined": [80.0, 50.0, 50.0],
            "p_ex_combined": [-80.0, -50.0, -50.0],
            "is_islanded": False,
            "disconnected_mw": 0.0,
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

        # Check expected keys are present
        assert "betas" in res
        assert "max_rho" in res
        assert "max_rho_line" in res
        assert "is_rho_reduction" in res
        assert "description" in res
        assert "action1_id" in res
        assert "action2_id" in res
        assert "p_or_combined" in res
        assert "p_ex_combined" in res
        assert "rho_after" in res
        assert "rho_before" in res

        # Basic sanity checks
        assert res["action1_id"] == aid1
        assert res["action2_id"] == aid2
        assert "Close LINE2" in res["description"]
        assert "Close LINE3" in res["description"]
        # P-based rho: p_or_combined=[80,50,50], factor=rho_or_start/|p_or_start|=[1.1/100,...]
        # rho_or_est on LINE1 = 80 * (1.1/100) = 0.88 < 1.1 → rho reduction
        assert res["is_rho_reduction"] is True

        print("\n[Test] Simplified dictionary verified successfully.")
    finally:
        superposition._identify_action_elements = orig_identify
        superposition.compute_combined_pair_superposition = orig_compute


def test_compute_all_pairs_superposition_use_p_based_rho_false():
    """use_p_based_rho=False uses the approximate direct rho superposition."""
    env = MagicMock()
    env.name_line = ["LINE1", "LINE2", "LINE3"]

    obs_start = MagicMock()
    obs_start.rho = np.array([1.1, 0.5, 0.5])
    obs_start.p_or = np.array([100.0, 50.0, 50.0])
    obs_start.p_ex = np.array([-100.0, -50.0, -50.0])

    aid1 = "reco_LINE2"
    aid2 = "reco_LINE3"

    obs_act1 = MagicMock()
    obs_act1.rho = np.array([0.9, 0.5, 0.5])
    obs_act1.p_or = np.array([90.0, 50.0, 50.0])

    obs_act2 = MagicMock()
    obs_act2.rho = np.array([0.9, 0.5, 0.5])
    obs_act2.p_or = np.array([90.0, 50.0, 50.0])

    detailed_actions = {
        aid1: {"action": MagicMock(), "observation": obs_act1, "description_unitaire": "Close LINE2"},
        aid2: {"action": MagicMock(), "observation": obs_act2, "description_unitaire": "Close LINE3"},
    }

    classifier = MagicMock(spec=ActionClassifier)
    classifier.identify_action_type.return_value = "close_line"
    classifier._action_space = MagicMock()

    from expert_op4grid_recommender.utils import superposition

    def mock_identify(action, action_id, *args):
        if action_id == aid1: return ([1], [])
        if action_id == aid2: return ([2], [])
        return ([], [])

    orig_identify = superposition._identify_action_elements
    orig_compute = superposition.compute_combined_pair_superposition
    try:
        superposition._identify_action_elements = MagicMock(side_effect=mock_identify)
        betas = [0.5, 0.5]
        superposition.compute_combined_pair_superposition = MagicMock(return_value={
            "betas": betas,
            "p_or_combined": [80.0, 50.0, 50.0],
            "p_ex_combined": [-80.0, -50.0, -50.0],
            "is_islanded": False,
            "disconnected_mw": 0.0,
        })

        results = compute_all_pairs_superposition(
            obs_start=obs_start,
            detailed_actions=detailed_actions,
            classifier=classifier,
            env=env,
            lines_overloaded_ids=[0],
            lines_we_care_about=["LINE1", "LINE2", "LINE3"],
            pre_existing_rho={i: 0.5 for i in range(3)},
            dict_action={},
            use_p_based_rho=False,  # Use approximate method
        )

        pair_id = f"{aid1}+{aid2}"
        assert pair_id in results
        res = results[pair_id]

        # Approximate method: rho_combined = |w*rho_start + b1*rho_act1 + b2*rho_act2|
        # = |0 * 1.1 + 0.5 * 0.9 + 0.5 * 0.9| = |0.9| = 0.9 on LINE1
        expected_rho_line1 = abs(0 * 1.1 + 0.5 * 0.9 + 0.5 * 0.9)
        assert abs(res["max_rho"] - expected_rho_line1) < 0.01

    finally:
        superposition._identify_action_elements = orig_identify
        superposition.compute_combined_pair_superposition = orig_compute


if __name__ == "__main__":
    test_compute_all_pairs_superposition_simplified_dict()
    test_compute_all_pairs_superposition_use_p_based_rho_false()
