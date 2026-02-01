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

# NOTE: pypowsybl2grid backend patch is applied via scripts/patch_pypowsybl2grid_file.py
# This must be run BEFORE tests (see .circleci/config.yml)
# Runtime monkey-patching doesn't work because modules are imported before conftest runs
#
# For local development, run:
#   python scripts/patch_pypowsybl2grid_file.py
# before running tests.


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
