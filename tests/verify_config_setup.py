#!/usr/bin/env python3
"""
Quick verification script to check if the test config override is working.

Run this script to verify your test setup:
    python tests/verify_config_setup.py

Since R3 the test suite no longer forks the config module: ``conftest.py``
applies the test deltas through ``config.override_settings(...)`` (validated by
pydantic) instead of swapping a hand-forked ``config_test.py`` in via
``sys.modules``. This script verifies that path.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 70)
    print("Verifying Test Configuration Setup")
    print("=" * 70)
    print()

    # Check 1: conftest.py exists
    print("✓ Check 1: conftest.py exists")
    conftest_path = project_root / "tests" / "conftest.py"
    if conftest_path.exists():
        print(f"  ✓ Found: {conftest_path}")
    else:
        print(f"  ✗ Missing: {conftest_path}")
        return False
    print()

    # Check 2: the real config module + R3 accessors are importable
    print("✓ Check 2: config module exposes the R3 accessors")
    try:
        from expert_op4grid_recommender import config
        for accessor in ("get_settings", "override_settings", "reset_settings"):
            assert hasattr(config, accessor), accessor
        print("  ✓ get_settings / override_settings / reset_settings present")
        print(f"  ✓ Config file: {config.__file__}")
    except Exception as e:  # noqa: BLE001 - diagnostic script
        print(f"  ✗ Failed: {e}")
        return False
    print()

    # Check 3: applying the test deltas through override_settings works and
    #          recomputes the derived paths (no staleness).
    print("✓ Check 3: Simulating pytest config override")
    try:
        from tests.conftest import TEST_CONFIG_DELTAS

        config.override_settings(**TEST_CONFIG_DELTAS)
        if config.ENV_NAME == "env_dijon_v2_assistant" and config.DO_VISUALIZATION is False:
            print("  ✓ Config override working correctly!")
            print(f"  ✓ ENV_NAME  = {config.ENV_NAME}")
            print(f"  ✓ ENV_PATH  = {config.ENV_PATH}  (recomputed)")
            print(f"  ✓ DO_VISUALIZATION = {config.DO_VISUALIZATION}")
        else:
            print(f"  ✗ Override not working - ENV_NAME={config.ENV_NAME}")
            return False
    except Exception as e:  # noqa: BLE001 - diagnostic script
        print(f"  ✗ Error during override simulation: {e}")
        return False
    print()

    # Check 4: Verify config attributes
    print("✓ Check 4: Verifying config attributes")
    required_attrs = [
        'DATE', 'TIMESTEP', 'LINES_DEFAUT', 'ENV_FOLDER', 'ENV_NAME', 'ENV_PATH',
        'ACTION_FILE_PATH', 'USE_DC_LOAD_FLOW', 'DO_CONSOLIDATE_GRAPH',
        'DO_VISUALIZATION', 'PARAM_OPTIONS_EXPERT_OP',
    ]

    all_present = True
    for attr in required_attrs:
        if hasattr(config, attr):
            print(f"  ✓ {attr}: {getattr(config, attr)}")
        else:
            print(f"  ✗ Missing: {attr}")
            all_present = False

    if not all_present:
        return False
    print()

    # Summary
    print("=" * 70)
    print("✓ ALL CHECKS PASSED!")
    print("=" * 70)
    print()
    print("Your test configuration is set up correctly!")
    print("When you run pytest, conftest.py applies the test deltas via")
    print("config.override_settings(**TEST_CONFIG_DELTAS).")
    print()
    print("To verify with the dedicated test:")
    print("  pytest tests/test_config_override.py -v")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
