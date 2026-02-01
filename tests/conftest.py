"""
Pytest configuration file for test suite.

This file ensures that tests use tests/config_test.py instead of the package config.

CRITICAL: This module-level code runs IMMEDIATELY when conftest.py is imported,
which happens BEFORE any test modules are loaded. This ensures the config
replacement happens before main.py or any other module imports config.
"""
import sys
import os
from pathlib import Path

# Get the tests directory
tests_dir = Path(__file__).parent.resolve()
# Get the project root directory
project_root = tests_dir.parent.resolve()

# Add project root to path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# === CRITICAL: Replace config IMMEDIATELY ===
# DO NOT move this into a fixture - it must run at module import time!

# First, import the test config
from tests import config_test

# Then IMMEDIATELY replace it in sys.modules BEFORE anything else can import it
sys.modules['expert_op4grid_recommender.config'] = config_test
sys.modules['config'] = config_test

# Print confirmation (this helps debug if the override isn't working)
print(f"\n{'='*70}")
print(f"✓ CONFIG OVERRIDE INSTALLED")
print(f"{'='*70}")
print(f"  expert_op4grid_recommender.config -> {config_test.__file__}")
print(f"  DO_VISUALIZATION = {config_test.DO_VISUALIZATION}")
print(f"{'='*70}\n")

# Now import pytest
import pytest
import numpy as np

# === PATCH: Fix pypowsybl2grid backend issue ===
# This patches the update_integer_value method to handle zero values correctly
# See: https://github.com/powsybl/pypowsybl-grid2op/issues/XXX
def _apply_pypowsybl2grid_patch():
    """
    Patches PyPowSyBlBackend.update_integer_value to fix zero-value handling.
    
    The original code sets changed[value == 0] = False, which causes issues.
    This patch changes value[value==0] = -1 instead.
    """
    try:
        from pypowsybl2grid.backend import PyPowSyBlBackend
        import pypowsybl._pypowsybl as _pypowsybl
        from pypowsybl2grid.backend import Grid2opUpdateIntegerValueType
        
        # Store original method for reference
        _original_update_integer_value = PyPowSyBlBackend.update_integer_value
        
        def patched_update_integer_value(self, value_type: Grid2opUpdateIntegerValueType, 
                                          value: np.ndarray, changed: np.ndarray) -> None:
            """Patched version: converts 0 to -1 instead of marking as unchanged."""
            # PATCH: Replace zeros with -1 instead of setting changed=False
            value = value.copy()  # Don't modify original array
            value[value == 0] = -1
            _pypowsybl.update_grid2op_integer_value(self._handle, value_type, value, changed)
        
        # Apply the patch
        PyPowSyBlBackend.update_integer_value = patched_update_integer_value
        print(f"✓ PYPOWSYBL2GRID PATCH APPLIED")
        print(f"  PyPowSyBlBackend.update_integer_value patched for zero-value handling")
        return True
        
    except ImportError as e:
        print(f"⚠ Could not apply pypowsybl2grid patch: {e}")
        return False
    except Exception as e:
        print(f"⚠ Error applying pypowsybl2grid patch: {e}")
        return False

# Apply patch at module load time (before any tests import the backend)
_apply_pypowsybl2grid_patch()

# Add a verification fixture
@pytest.fixture(scope="session", autouse=True)
def verify_pypowsybl2grid_patch():
    """
    Verify the pypowsybl2grid patch is applied correctly.
    """
    try:
        from pypowsybl2grid.backend import PyPowSyBlBackend
        
        # Check if our patched method is in place
        method_doc = PyPowSyBlBackend.update_integer_value.__doc__ or ""
        if "Patched version" in method_doc:
            print(f"\n✓ pypowsybl2grid patch verified")
        else:
            print(f"\n⚠ pypowsybl2grid patch may not be applied (doc: {method_doc[:50]}...)")
    except ImportError:
        print(f"\n⚠ pypowsybl2grid not installed, patch not needed")
    
    yield


@pytest.fixture(scope="session", autouse=True)
def verify_config_override():
    """
    Verify the config override is working.
    This runs after module imports but before tests.
    """
    # Import and check
    from expert_op4grid_recommender import config
    
    # Verify
    assert 'config_test' in config.__file__, \
        f"❌ Config override FAILED! Using {config.__file__}"
    
    assert config.DO_VISUALIZATION == False, \
        f"❌ DO_VISUALIZATION should be False in tests, got {config.DO_VISUALIZATION}"
    
    print(f"\n✓ Config override verified in fixture")
    print(f"  File: {config.__file__}")
    print(f"  DO_VISUALIZATION: {config.DO_VISUALIZATION}\n")
    
    yield
