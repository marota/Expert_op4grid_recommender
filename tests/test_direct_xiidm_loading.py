import pytest
import os
import shutil
from pathlib import Path
from expert_op4grid_recommender.environment_pypowsybl import get_env_first_obs_pypowsybl

def test_setup_environment_with_direct_xiidm_file():
    # Setup: Use existing data from bare_env_small_grid_test
    data_dir = Path(__file__).parent.parent / "data" / "bare_env_small_grid_test"
    xiidm_file = data_dir / "grid.xiidm"
    
    if not xiidm_file.exists():
        pytest.skip(f"Test data not found at {xiidm_file}")
        
    try:
        # 1. Test loading with direct file path as env_name
        env, obs, env_path = get_env_first_obs_pypowsybl(
            env_folder=str(data_dir),
            env_name="grid.xiidm"
        )
        
        # Verify the env is correctly set up
        assert env is not None
        assert env.network_manager.network is not None
        assert obs is not None
        
        # Verify env_path is the directory containing the .xiidm (due to the fix)
        assert Path(env_path).is_dir()
        assert (Path(env_path) / "grid.xiidm").exists()
        
        # 2. Test loading with absolute file path as env_name (frontend behavior)
        env2, obs2, env_path2 = get_env_first_obs_pypowsybl(
            env_folder="", # Empty env_folder
            env_name=str(xiidm_file)
        )
        
        assert env2 is not None
        assert Path(env_path2).is_dir()
        assert env_path2 == str(data_dir)
        
    except Exception as e:
        pytest.fail(f"Failed to load environment from direct .xiidm file: {e}")
