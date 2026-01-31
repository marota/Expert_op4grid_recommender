"""
Test to verify config override is working properly.
Run this test first to debug configuration issues.
"""
import sys


def test_config_is_test_version():
    """Verify we're using the test config, not the package config."""
    from expert_op4grid_recommender import config
    
    # Check the file path
    assert 'config_test' in config.__file__, \
        f"FAIL: Using {config.__file__} instead of config_test.py"
    
    print(f"\n✓ Config file: {config.__file__}")
    

def test_do_visualization_is_false():
    """Verify DO_VISUALIZATION is False in tests."""
    from expert_op4grid_recommender import config
    
    assert hasattr(config, 'DO_VISUALIZATION'), \
        "FAIL: DO_VISUALIZATION attribute not found"
    
    assert config.DO_VISUALIZATION == False, \
        f"FAIL: DO_VISUALIZATION should be False, got {config.DO_VISUALIZATION}"
    
    print(f"\n✓ DO_VISUALIZATION = {config.DO_VISUALIZATION}")


def test_config_in_sys_modules():
    """Verify the config module in sys.modules is the test version."""
    assert 'expert_op4grid_recommender.config' in sys.modules, \
        "FAIL: expert_op4grid_recommender.config not in sys.modules"
    
    config_module = sys.modules['expert_op4grid_recommender.config']
    
    assert 'config_test' in config_module.__file__, \
        f"FAIL: sys.modules has wrong config: {config_module.__file__}"
    
    print(f"\n✓ sys.modules['expert_op4grid_recommender.config'] = {config_module.__file__}")


def test_main_uses_test_config():
    """Verify that when we import main, it gets the test config."""
    # Import main (this will trigger its config import)
    from expert_op4grid_recommender import main
    
    # Now check what config main is using
    # Main imports config at the module level, so we need to check what it got
    from expert_op4grid_recommender import config
    
    assert config.DO_VISUALIZATION == False, \
        f"FAIL: After importing main, DO_VISUALIZATION = {config.DO_VISUALIZATION}"
    
    print(f"\n✓ After importing main, DO_VISUALIZATION = {config.DO_VISUALIZATION}")


def test_run_analysis_will_skip_visualization():
    """
    Verify that run_analysis function will skip visualization.
    This doesn't actually run the analysis, just checks the config it would use.
    """
    from expert_op4grid_recommender import config
    
    # Simulate what run_analysis does
    if config.DO_VISUALIZATION:
        result = "VISUALIZATION WOULD RUN"
    else:
        result = "VISUALIZATION WOULD BE SKIPPED"
    
    assert result == "VISUALIZATION WOULD BE SKIPPED", \
        f"FAIL: {result}"
    
    print(f"\n✓ run_analysis will skip visualization (DO_VISUALIZATION={config.DO_VISUALIZATION})")


if __name__ == "__main__":
    # Allow running this file directly for debugging
    print("Running config override verification...")
    print("=" * 70)
    
    test_config_is_test_version()
    test_do_visualization_is_false()
    test_config_in_sys_modules()
    test_main_uses_test_config()
    test_run_analysis_will_skip_visualization()
    
    print("=" * 70)
    print("✓ ALL VERIFICATION TESTS PASSED!")
    print("=" * 70)
