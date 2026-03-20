
import pytest
from unittest.mock import MagicMock
from expert_op4grid_recommender.utils.superposition import _identify_action_elements
from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier

@pytest.fixture
def env():
    env = MagicMock()
    env.name_line = ["LINE1", "LINE2", "ARKA TD 661", "PST_BRANCH_1_inc2"]
    env.name_sub = ["SUB1", "SUB2"]
    return env

@pytest.fixture
def classifier():
    return ActionClassifier()

def test_identify_pst_leading_dot(env, classifier):
    aid = "pst_tap_.ARKA TD 661_inc2"
    action = MagicMock()
    dict_action = {aid: {"description": "PST tap"}}
    
    line_idxs, sub_idxs = _identify_action_elements(
        action, aid, dict_action, classifier, env
    )
    # Extracted "ARKA TD 661" matches index 2
    assert line_idxs == [2]
    assert sub_idxs == []

def test_identify_pst_suffix_inc_dec(env, classifier):
    # Test inc2, inc1, dec5suffixes
    for suffix in ["_inc1", "_inc2", "_dec1", "_dec10"]:
        aid = f"pst_tap_LINE2{suffix}"
        action = MagicMock()
        dict_action = {aid: {"description": "PST tap"}}
        
        line_idxs, sub_idxs = _identify_action_elements(
            action, aid, dict_action, classifier, env
        )
        assert line_idxs == [1]

def test_identify_pst_exact_match_with_suffix_in_name_line(env, classifier):
    # If the line name in name_line ALREADY has the suffix, it should still match
    aid = "pst_tap_PST_BRANCH_1_inc2"
    action = MagicMock()
    dict_action = {aid: {"description": "PST tap"}}
    
    line_idxs, sub_idxs = _identify_action_elements(
        action, aid, dict_action, classifier, env
    )
    # Extracted is "PST_BRANCH_1" (after regex) but it might match "PST_BRANCH_1_inc2" via substring
    assert 3 in line_idxs

def test_identify_pst_nested_content(env, classifier):
    aid = "custom_id"
    action = MagicMock()
    dict_action = {aid: {
        "description": "Variation de slot",
        "content": {"pst_tap": {"LINE1": 10}}
    }}
    
    line_idxs, sub_idxs = _identify_action_elements(
        action, aid, dict_action, classifier, env
    )
    assert line_idxs == [0]

def test_identify_pst_flat_content(env, classifier):
    aid = "custom_id"
    action = MagicMock()
    # Some older or alternative structures might have pst_tap at the top level of the dict
    dict_action = {aid: {
        "description": "Variation de slot",
        "pst_tap": {"LINE2": 10}
    }}
    
    line_idxs, sub_idxs = _identify_action_elements(
        action, aid, dict_action, classifier, env
    )
    assert line_idxs == [1]

def test_identify_line_disco_reco(env, classifier):
    # Verify we didn't break normal line identification
    aid_disco = "disco_LINE1"
    line_idxs, _ = _identify_action_elements(MagicMock(), aid_disco, {}, classifier, env)
    assert line_idxs == [0]
    
    aid_reco = "reco_LINE2"
    line_idxs, _ = _identify_action_elements(MagicMock(), aid_reco, {}, classifier, env)
    assert line_idxs == [1]

def test_identify_sub_actions(env, classifier):
    aid = "open_coupling_SUB1"
    _, sub_idxs = _identify_action_elements(MagicMock(), aid, {}, classifier, env)
    assert sub_idxs == [0]
    
    aid2 = "node_merging_SUB2"
    _, sub_idxs = _identify_action_elements(MagicMock(), aid2, {}, classifier, env)
    assert sub_idxs == [1]
