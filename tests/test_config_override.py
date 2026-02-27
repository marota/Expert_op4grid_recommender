"""
Test to verify config override is working properly.
Run this test first to debug configuration issues.
"""
import sys
import pytest


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


def test_monitoring_factor_thermal_limits_exists():
    """Verify MONITORING_FACTOR_THERMAL_LIMITS is present in config with default value."""
    from expert_op4grid_recommender import config

    assert hasattr(config, 'MONITORING_FACTOR_THERMAL_LIMITS'), \
        "FAIL: MONITORING_FACTOR_THERMAL_LIMITS attribute not found in config"

    assert config.MONITORING_FACTOR_THERMAL_LIMITS == 0.95, \
        f"FAIL: MONITORING_FACTOR_THERMAL_LIMITS should be 0.95, got {config.MONITORING_FACTOR_THERMAL_LIMITS}"

    print(f"\n✓ MONITORING_FACTOR_THERMAL_LIMITS = {config.MONITORING_FACTOR_THERMAL_LIMITS}")


def test_set_thermal_limits_uses_monitoring_factor():
    """Verify set_thermal_limits applies MONITORING_FACTOR_THERMAL_LIMITS correctly."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from unittest.mock import MagicMock
    from expert_op4grid_recommender import config
    from expert_op4grid_recommender.main import set_thermal_limits

    permanent_limit_value = 1000.0
    line_name = "LINE_A"

    # Build a mock operational limits DataFrame with a permanent_limit row
    limits_df = pd.DataFrame({
        "element_id": [line_name],
        "name": ["permanent_limit"],
        "value": [permanent_limit_value],
    })

    mock_network = MagicMock()
    mock_network.get_operational_limits.return_value = limits_df.set_index("element_id")
    mock_network.get_lines.return_value = pd.DataFrame(index=[line_name])
    mock_network.get_2_windings_transformers.return_value = pd.DataFrame(index=[])

    mock_env = MagicMock()
    mock_env.name_line = [line_name]
    mock_env.set_thermal_limit = MagicMock()

    set_thermal_limits(mock_network, mock_env, thresold_thermal_limit=config.MONITORING_FACTOR_THERMAL_LIMITS)

    applied_limits = mock_env.set_thermal_limit.call_args[0][0]
    expected = np.round(config.MONITORING_FACTOR_THERMAL_LIMITS * permanent_limit_value)
    assert applied_limits[0] == expected, \
        f"FAIL: expected thermal limit {expected}, got {applied_limits[0]}"

    print(f"\n✓ set_thermal_limits applied factor {config.MONITORING_FACTOR_THERMAL_LIMITS} correctly "
          f"({permanent_limit_value} -> {applied_limits[0]})")


def test_set_thermal_limits_respects_custom_factor():
    """Verify set_thermal_limits correctly uses an overridden factor value."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from unittest.mock import MagicMock
    from expert_op4grid_recommender.main import set_thermal_limits

    permanent_limit_value = 800.0
    custom_factor = 0.80
    line_name = "LINE_B"

    limits_df = pd.DataFrame({
        "element_id": [line_name],
        "name": ["permanent_limit"],
        "value": [permanent_limit_value],
    })

    mock_network = MagicMock()
    mock_network.get_operational_limits.return_value = limits_df.set_index("element_id")
    mock_network.get_lines.return_value = pd.DataFrame(index=[line_name])
    mock_network.get_2_windings_transformers.return_value = pd.DataFrame(index=[])

    mock_env = MagicMock()
    mock_env.name_line = [line_name]
    mock_env.set_thermal_limit = MagicMock()

    set_thermal_limits(mock_network, mock_env, thresold_thermal_limit=custom_factor)

    applied_limits = mock_env.set_thermal_limit.call_args[0][0]
    expected = np.round(custom_factor * permanent_limit_value)
    assert applied_limits[0] == expected, \
        f"FAIL: expected thermal limit {expected} for factor {custom_factor}, got {applied_limits[0]}"

    print(f"\n✓ set_thermal_limits with custom factor {custom_factor}: "
          f"{permanent_limit_value} -> {applied_limits[0]}")


if __name__ == "__main__":
    # Allow running this file directly for debugging
    print("Running config override verification...")
    print("=" * 70)

    test_config_is_test_version()
    test_do_visualization_is_false()
    test_config_in_sys_modules()
    test_main_uses_test_config()
    test_run_analysis_will_skip_visualization()
    test_monitoring_factor_thermal_limits_exists()
    test_set_thermal_limits_uses_monitoring_factor()
    test_set_thermal_limits_respects_custom_factor()

    print("=" * 70)
    print("✓ ALL VERIFICATION TESTS PASSED!")
    print("=" * 70)
