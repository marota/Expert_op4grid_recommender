"""
Pytest configuration for the test suite.

Historically this file swapped a hand-forked ``tests/config_test.py`` in for
``expert_op4grid_recommender.config`` via ``sys.modules`` (and the package
attribute). That fork bypassed pydantic entirely — ``Settings`` validation and
env parsing were never exercised in CI, and every new config key had to be added
twice (review findings M2 / C7).

Now the real config module stays in place and we apply the handful of test
deltas through the sanctioned :func:`config.override_settings` accessor. That
runs full pydantic validation (so CI now exercises it), recomputes the derived
paths (``ENV_PATH`` / ``ACTION_FILE_PATH`` follow ``ENV_NAME`` /
``FILE_ACTION_SPACE_DESC``), and re-promotes the values so ``config.X`` reads
observe them.

CRITICAL: this runs at conftest import time — BEFORE any test module is
collected — so the override is in place before the pipeline or any library
module reads a config value.
"""
import sys
from pathlib import Path

# Get the tests directory / project root and make the package importable.
tests_dir = Path(__file__).parent.resolve()
project_root = tests_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# === Apply the test config deltas through the validated accessor ===
# DO NOT move this into a fixture — it must run at module import time so the
# override is installed before anything imports/reads config.
from expert_op4grid_recommender import config

#: Only the values that genuinely differ from the shipped production config.
#: Everything else (DATE, TIMESTEP, LINES_DEFAUT, the load-shedding / curtailment
#: / redispatch tunables, PARAM_OPTIONS_EXPERT_OP, …) inherits the real defaults,
#: so this list can never drift out of sync with config.py the way the old fork
#: did. The dijon environment + 5-action output mirrors config_basic.py, with
#: visualization disabled for faster test runs.
TEST_CONFIG_DELTAS = dict(
    ENV_NAME="env_dijon_v2_assistant",
    FILE_ACTION_SPACE_DESC="reduced_model_actions.json",
    CHECK_ACTION_SIMULATION=True,
    N_PRIORITIZED_ACTIONS=5,
    IGNORE_LINES_MONITORING=False,
    DO_VISUALIZATION=False,  # Skip visualization in tests for faster execution
    MAX_RHO_BOTH_EXTREMITIES=False,  # only possible for now with pypowsybl backend
    MIN_LINE_RECONNECTIONS=0,
    MIN_CLOSE_COUPLING=0,
    MIN_OPEN_COUPLING=0,
    MIN_LINE_DISCONNECTIONS=0,
    MIN_PST=0,
    MIN_LOAD_SHEDDING=0,
    MIN_RENEWABLE_CURTAILMENT=0,
    MIN_REDISPATCH=0,
)

config.override_settings(**TEST_CONFIG_DELTAS)

# Print confirmation (this helps debug if the override isn't working)
print(f"\n{'='*70}")
print("✓ TEST CONFIG OVERRIDE INSTALLED (via config.override_settings)")
print(f"{'='*70}")
print(f"  expert_op4grid_recommender.config -> {config.__file__}")
print(f"  ENV_NAME = {config.ENV_NAME}")
print(f"  DO_VISUALIZATION = {config.DO_VISUALIZATION}")
print(f"{'='*70}\n")

# Now import pytest
import pytest

# NOTE: pypowsybl2grid backend patch is applied via scripts/patch_pypowsybl2grid_file.py
# This must be run BEFORE tests (see .github/workflows/ci.yml)
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
    from expert_op4grid_recommender import config

    # The real config module is still in place (no more sys.modules fork) …
    assert config.__file__.endswith("expert_op4grid_recommender/config.py"), \
        f"❌ Unexpected config module: {config.__file__}"

    # … and the test delta is applied and validated by pydantic.
    assert config.DO_VISUALIZATION is False, \
        f"❌ DO_VISUALIZATION should be False in tests, got {config.DO_VISUALIZATION}"
    assert config.get_settings().DO_VISUALIZATION is False, \
        "❌ Settings instance not overridden"

    print("\n✓ Config override verified in fixture")
    print(f"  File: {config.__file__}")
    print(f"  DO_VISUALIZATION: {config.DO_VISUALIZATION}\n")

    yield
