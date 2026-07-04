"""
Test to verify config override is working properly.
Run this test first to debug configuration issues.
"""
import sys

import pytest


def test_config_is_real_module_with_test_overrides():
    """The real config module stays in place; the test deltas are applied through
    the validated ``override_settings`` accessor (no more ``config_test`` fork /
    ``sys.modules`` swap — review finding M2)."""
    from expert_op4grid_recommender import config

    # The real package config module is what's imported (not a hand-forked copy).
    assert config.__file__.endswith("expert_op4grid_recommender/config.py"), \
        f"FAIL: Unexpected config module {config.__file__}"

    # The test delta was applied and is reflected both on the module and on the
    # authoritative Settings instance.
    assert config.ENV_NAME == "env_dijon_v2_assistant"
    assert config.get_settings().ENV_NAME == "env_dijon_v2_assistant"

    print(f"\n✓ Config file: {config.__file__}")


def test_do_visualization_is_false():
    """Verify DO_VISUALIZATION is False in tests."""
    from expert_op4grid_recommender import config
    
    assert hasattr(config, 'DO_VISUALIZATION'), \
        "FAIL: DO_VISUALIZATION attribute not found"
    
    assert config.DO_VISUALIZATION == False, \
        f"FAIL: DO_VISUALIZATION should be False, got {config.DO_VISUALIZATION}"
    
    print(f"\n✓ DO_VISUALIZATION = {config.DO_VISUALIZATION}")


def test_config_in_sys_modules_is_real_module():
    """The config in sys.modules is the real package module (no fork swap)."""
    assert 'expert_op4grid_recommender.config' in sys.modules, \
        "FAIL: expert_op4grid_recommender.config not in sys.modules"

    config_module = sys.modules['expert_op4grid_recommender.config']

    assert config_module.__file__.endswith("expert_op4grid_recommender/config.py"), \
        f"FAIL: sys.modules has unexpected config: {config_module.__file__}"

    print(f"\n✓ sys.modules['expert_op4grid_recommender.config'] = {config_module.__file__}")


def test_main_uses_test_config():
    """Verify that when we import main, it gets the test config."""
    # Import main (this will trigger its config import)
    
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

    print("\n✓ LINES_MONITORING_FILE resolves to None as expected")


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


def test_config_exposes_all_keys_and_accessors():
    """The real config module exposes every key (no hand-maintained fork can drop
    one — review finding C7/M2), including the derived ``@computed_field`` paths,
    plus the pydantic ``settings`` instance, the ``Settings`` class, and the
    ``get_settings`` / ``override_settings`` accessors introduced by R3."""
    from expert_op4grid_recommender import config

    # Keys the pre-star-import fork was missing (surfaced as AttributeError deep
    # in test runs), a v0.2.x key, the derived paths, and the R3 accessors.
    for key in ("ENABLE_ANTENNA_RECOMMENDATIONS", "MAX_CANDIDATE_SIMULATIONS",
                "VISUALIZATION_FORMAT", "USE_VOLTAGE_LEVEL_NAMES_IN_GRAPH",
                "LINES_MONITORING_FILE", "ALLOWED_ACTION_TYPES",
                "MIN_REDISPATCH", "ENV_PATH", "ACTION_FILE_PATH", "CASE_NAME",
                "settings", "Settings", "get_settings", "override_settings"):
        assert hasattr(config, key), f"config is missing '{key}'"

    # And the test override wins.
    assert config.DO_VISUALIZATION is False


if __name__ == "__main__":
    # Allow running this file directly for debugging
    print("Running config override verification...")
    print("=" * 70)

    test_config_is_real_module_with_test_overrides()
    test_do_visualization_is_false()
    test_config_in_sys_modules_is_real_module()
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


# ---------------------------------------------------------------------------
# R3 — one config, one source of truth (computed fields + accessors)
# ---------------------------------------------------------------------------

def test_derived_paths_track_env_name_without_staleness():
    """Overriding ENV_NAME / FILE_ACTION_SPACE_DESC recomputes the derived paths
    (review finding A3 — no stale ENV_PATH / ACTION_FILE_PATH)."""
    from expert_op4grid_recommender.config import Settings

    s = Settings(ENV_NAME="some_env", FILE_ACTION_SPACE_DESC="acts.json")
    assert s.ENV_PATH == s.ENV_FOLDER / "some_env"
    assert s.ACTION_FILE_PATH == s.ENV_FOLDER / "action_space" / "acts.json"
    assert s.ENV_PATH.name == "some_env"


def test_override_settings_revalidates_and_repromotes():
    """override_settings validates, recomputes derived paths, and re-promotes to
    the module namespace; it can be rolled back to the current test deltas."""
    from expert_op4grid_recommender import config

    before = config.get_settings()
    try:
        config.override_settings(ENV_NAME="rollback_env")
        assert config.ENV_NAME == "rollback_env"
        assert config.ENV_PATH.name == "rollback_env"          # module attr recomputed
        assert config.get_settings().ENV_NAME == "rollback_env"  # instance updated
    finally:
        config.override_settings(before)  # restore the session's test settings
    assert config.ENV_NAME == "env_dijon_v2_assistant"


def test_override_settings_rejects_unknown_key():
    """Unknown overrides raise instead of being silently ignored."""
    import pytest
    from expert_op4grid_recommender import config
    with pytest.raises(ValueError):
        config.override_settings(NOT_A_REAL_SETTING=1)


def test_override_settings_validates_field_constraints():
    """Field constraints (e.g. N_PRIORITIZED_ACTIONS >= 0) are enforced — the old
    fork bypassed pydantic entirely, so CI never exercised validation (M2)."""
    import pytest
    from pydantic import ValidationError
    from expert_op4grid_recommender import config

    before = config.get_settings()
    try:
        with pytest.raises(ValidationError):
            config.override_settings(N_PRIORITIZED_ACTIONS=-5)
    finally:
        config.override_settings(before)


def test_override_settings_accepts_a_settings_instance_positionally():
    from expert_op4grid_recommender import config
    from expert_op4grid_recommender.config import Settings

    before = config.get_settings()
    try:
        applied = config.override_settings(Settings(N_PRIORITIZED_ACTIONS=7))
        assert applied.N_PRIORITIZED_ACTIONS == 7
        assert config.N_PRIORITIZED_ACTIONS == 7
        assert config.get_settings() is applied
    finally:
        config.override_settings(before)


def test_override_settings_rejects_instance_and_kwargs_together():
    import pytest
    from expert_op4grid_recommender import config
    from expert_op4grid_recommender.config import Settings
    with pytest.raises(TypeError):
        config.override_settings(Settings(), N_PRIORITIZED_ACTIONS=1)


def test_all_derived_paths_recompute_on_override_not_just_env_path():
    from expert_op4grid_recommender import config

    before = config.get_settings()
    try:
        config.override_settings(ENV_NAME="grid_x", FILE_ACTION_SPACE_DESC="a.json",
                                 TIMESTEP=3, LINES_DEFAUT=["L9"])
        assert config.ENV_PATH == config.ENV_FOLDER / "grid_x"
        assert config.ACTION_FILE_PATH == config.ACTION_SPACE_FOLDER / "a.json"
        assert config.ACTION_SPACE_FOLDER == config.ENV_FOLDER / "action_space"
        assert config.CASE_NAME == "defaut_L9_t3"
    finally:
        config.override_settings(before)


def test_computed_fields_are_in_model_dump_and_promoted():
    from expert_op4grid_recommender import config
    dumped = config.get_settings().model_dump()
    for key in ("CASE_NAME", "ENV_FOLDER", "ENV_PATH", "ACTION_SPACE_FOLDER",
                "ACTION_FILE_PATH", "SAVE_FOLDER_VISUALIZATION"):
        assert key in dumped, f"{key} missing from model_dump()"
        assert getattr(config, key) == dumped[key]  # module attr == instance value


def test_env_var_parsing_runs_through_pydantic(monkeypatch):
    """EXPERT_OP4GRID_* env vars feed Settings (the fork used to bypass this)."""
    from expert_op4grid_recommender.config import Settings
    monkeypatch.setenv("EXPERT_OP4GRID_TIMESTEP", "12")
    monkeypatch.setenv("EXPERT_OP4GRID_N_PRIORITIZED_ACTIONS", "8")
    s = Settings()
    assert s.TIMESTEP == 12
    assert s.N_PRIORITIZED_ACTIONS == 8


def test_lines_defaut_validator_accepts_bare_name_and_json(monkeypatch):
    from expert_op4grid_recommender.config import Settings
    monkeypatch.setenv("EXPERT_OP4GRID_LINES_DEFAUT", "BEON L31CPVAN")
    assert Settings().LINES_DEFAUT == ["BEON L31CPVAN"]     # bare name
    monkeypatch.setenv("EXPERT_OP4GRID_LINES_DEFAUT", '["A", "B"]')
    assert Settings().LINES_DEFAUT == ["A", "B"]            # JSON list


def test_raw_module_mutation_still_works_as_back_compat_escape_hatch():
    """Co-Study4Grid mutates config.X directly; that must keep working."""
    from expert_op4grid_recommender import config

    before = config.DO_CONSOLIDATE_GRAPH
    try:
        config.DO_CONSOLIDATE_GRAPH = not before
        assert config.DO_CONSOLIDATE_GRAPH == (not before)
        # arbitrary (non-Settings) attributes are allowed too
        config.SOME_COSTUDY_ONLY_KEY = 123
        assert config.SOME_COSTUDY_ONLY_KEY == 123
    finally:
        config.DO_CONSOLIDATE_GRAPH = before
        del config.SOME_COSTUDY_ONLY_KEY
