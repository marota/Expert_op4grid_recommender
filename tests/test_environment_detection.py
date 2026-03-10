import pytest
from datetime import datetime
from expert_op4grid_recommender.environment_pypowsybl import setup_environment_configs_pypowsybl
from expert_op4grid_recommender import config
from pathlib import Path

def test_non_reconnectable_detection_with_date():
    """
    Verifies that non-reconnectable lines are detected even when an analysis_date is provided.
    This fixes a bug where detection was only active for bare environments (date=None).
    """
    # Force the small grid test environment
    original_env_name = config.ENV_NAME
    config.ENV_NAME = "bare_env_small_grid_test"
    
    # Use a dummy date - before the fix, this would skip topology-based detection
    dummy_date = datetime(2024, 1, 1)
    
    try:
        env, obs, env_path, chronic_name, layout, dict_actions, lines_non_reco, lines_care = \
            setup_environment_configs_pypowsybl(analysis_date=dummy_date)
        
        # Expected lines for bare_env_small_grid_test
        expected = {'CRENEL71VIELM', 'GEN.PL73VIELM', 'PYMONL61VOUGL', 'CPVANY632', 'PYMONY632'}
        
        detected_set = set(lines_non_reco)
        
        missing = expected - detected_set
        assert not missing, f"Non-reconnectable lines missing from detection: {missing}"
        
        print(f"Verified: {len(expected)} lines correctly detected with date={dummy_date}")
        
    finally:
        # Restore environment name
        config.ENV_NAME = original_env_name

if __name__ == "__main__":
    test_non_reconnectable_detection_with_date()
