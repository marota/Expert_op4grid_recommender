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


def test_min_action_count_params_exist_with_defaults():
    """Verify all MIN_* action count parameters exist in config with default value 0."""
    from expert_op4grid_recommender import config

    for param in ('MIN_LINE_RECONNECTIONS', 'MIN_CLOSE_COUPLING', 'MIN_OPEN_COUPLING', 'MIN_LINE_DISCONNECTIONS'):
        assert hasattr(config, param), \
            f"FAIL: {param} attribute not found in config"
        value = getattr(config, param)
        assert value == 0, \
            f"FAIL: {param} should be 0, got {value}"

    print("\n✓ All MIN_* parameters exist and default to 0")


def test_lines_monitoring_file_defaults_to_none():
    """Verify LINES_MONITORING_FILE is absent or None in test config (getattr fallback behaviour)."""
    from expert_op4grid_recommender import config

    # The environment.py code uses getattr(config, 'LINES_MONITORING_FILE', None)
    # So whether the attribute is absent or explicitly None, the result must be None.
    value = getattr(config, 'LINES_MONITORING_FILE', None)
    assert value is None, \
        f"FAIL: LINES_MONITORING_FILE should be None (or absent), got {value!r}"

    print(f"\n✓ LINES_MONITORING_FILE resolves to None as expected")


def test_ignore_lines_monitoring_exists_in_test_config():
    """Verify IGNORE_LINES_MONITORING is present in the test config."""
    from expert_op4grid_recommender import config

    assert hasattr(config, 'IGNORE_LINES_MONITORING'), \
        "FAIL: IGNORE_LINES_MONITORING attribute not found in config"

    # The test config deliberately sets this to True so that tests do not need
    # an actual monitoring CSV file on disk.
    assert isinstance(config.IGNORE_LINES_MONITORING, bool), \
        f"FAIL: IGNORE_LINES_MONITORING should be bool, got {type(config.IGNORE_LINES_MONITORING)}"

    print(f"\n✓ IGNORE_LINES_MONITORING = {config.IGNORE_LINES_MONITORING}")


def test_sum_min_actions_does_not_exceed_n_prioritized_actions():
    """Verify the default MIN_* values don't exceed N_PRIORITIZED_ACTIONS."""
    from expert_op4grid_recommender import config

    sum_min = (config.MIN_LINE_RECONNECTIONS +
               config.MIN_CLOSE_COUPLING +
               config.MIN_OPEN_COUPLING +
               config.MIN_LINE_DISCONNECTIONS)

    assert sum_min <= config.N_PRIORITIZED_ACTIONS, (
        f"FAIL: sum of MIN_* ({sum_min}) exceeds N_PRIORITIZED_ACTIONS "
        f"({config.N_PRIORITIZED_ACTIONS}) in test config"
    )

    print(f"\n✓ sum(MIN_*) = {sum_min} <= N_PRIORITIZED_ACTIONS = {config.N_PRIORITIZED_ACTIONS}")


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
    test_min_action_count_params_exist_with_defaults()
    test_lines_monitoring_file_defaults_to_none()
    test_ignore_lines_monitoring_exists_in_test_config()
    test_sum_min_actions_does_not_exceed_n_prioritized_actions()

    print("=" * 70)
    print("✓ ALL VERIFICATION TESTS PASSED!")
    print("=" * 70)
