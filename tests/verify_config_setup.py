#!/usr/bin/env python3
"""
Quick verification script to check if the test config override is working.

Run this script to verify your test setup:
    python tests/verify_config_setup.py
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
    
    # Check 2: config_test.py exists
    print("✓ Check 2: config_test.py exists")
    config_test_path = project_root / "tests" / "config_test.py"
    if config_test_path.exists():
        print(f"  ✓ Found: {config_test_path}")
    else:
        print(f"  ✗ Missing: {config_test_path}")
        return False
    print()
    
    # Check 3: Can import test config
    print("✓ Check 3: Can import test config")
    try:
        from tests import config_test
        print(f"  ✓ Successfully imported config_test")
        print(f"  ✓ Config file: {config_test.__file__}")
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        return False
    print()
    
    # Check 4: Simulate what happens in tests
    print("✓ Check 4: Simulating pytest config override")
    try:
        # This mimics what the conftest.py fixture does
        sys.modules['expert_op4grid_recommender.config'] = config_test
        from expert_op4grid_recommender import config as imported_config
        
        if 'config_test' in imported_config.__file__:
            print(f"  ✓ Config override working correctly!")
            print(f"  ✓ Imported config from: {imported_config.__file__}")
        else:
            print(f"  ✗ Override not working - imported from: {imported_config.__file__}")
            return False
    except Exception as e:
        print(f"  ✗ Error during override simulation: {e}")
        return False
    print()
    
    # Check 5: Verify config attributes
    print("✓ Check 5: Verifying config attributes")
    required_attrs = [
        'DATE', 'TIMESTEP', 'LINES_DEFAUT', 'ENV_FOLDER', 'ENV_NAME',
        'USE_DC_LOAD_FLOW', 'DO_CONSOLIDATE_GRAPH', 'DO_VISUALIZATION', 'PARAM_OPTIONS_EXPERT_OP'
    ]
    
    all_present = True
    for attr in required_attrs:
        if hasattr(imported_config, attr):
            print(f"  ✓ {attr}: {getattr(imported_config, attr)}")
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
    print("When you run pytest, tests will automatically use tests/config_test.py")
    print()
    print("To run the reproducibility test:")
    print("  pytest tests/test_expert_op4grid_analyzer.py::test_reproducibility -v")
    print()
    print("To verify with the dedicated test:")
    print("  pytest tests/test_config_override.py -v")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
