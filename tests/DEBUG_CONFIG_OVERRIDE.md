# Debugging Config Override Issues

## Problem

When running `pytest tests/test_expert_op4grid_analyzer.py::test_reproducibility`, visualization is still running even though `DO_VISUALIZATION` should be `False`.

## Root Cause

The test file imports `expert_op4grid_recommender.main` at the module level (line 45), which causes `main.py` to import `config` before the test config override can be applied.

## Solution

The `conftest.py` file now does the config replacement at **module import time** (not in a fixture), which happens before pytest loads test files.

## Verification Steps

### Step 1: Run the verification test

```bash
pytest tests/test_config_override.py -v -s
```

Expected output:
```
✓ CONFIG OVERRIDE INSTALLED
  expert_op4grid_recommender.config -> /path/to/tests/config_test.py
  DO_VISUALIZATION = False

test_config_is_test_version PASSED
test_do_visualization_is_false PASSED
test_config_in_sys_modules PASSED
test_main_uses_test_config PASSED
test_run_analysis_will_skip_visualization PASSED
```

### Step 2: Run a single reproducibility test with verbose output

```bash
pytest tests/test_expert_op4grid_analyzer.py::test_reproducibility[Case_CPVANY633_T22] -v -s
```

Look for these indicators in the output:
- `✓ CONFIG OVERRIDE INSTALLED` at the start
- `DO_VISUALIZATION = False` in the override message
- `Skipping visualization (DO_VISUALIZATION=False)` during test execution
- NO "Visualization" timer in the output

### Step 3: Add temporary debug print in test

Add this at the start of `test_reproducibility`:

```python
def test_reproducibility(test_id, date_str, timestep, lines_defaut, expected_keys_set):
    # DEBUG: Check config
    from expert_op4grid_recommender import config
    print(f"\nDEBUG: config.__file__ = {config.__file__}")
    print(f"DEBUG: DO_VISUALIZATION = {config.DO_VISUALIZATION}")
    
    # Rest of test...
    prioritized_actions=run_analysis(...)
```

## Common Issues

### Issue 1: Config override not installing

**Symptom:** Don't see "✓ CONFIG OVERRIDE INSTALLED" message

**Fix:** Make sure `conftest.py` exists in the `tests/` directory

### Issue 2: Wrong config being used

**Symptom:** See `config.py` instead of `config_test.py` in paths

**Fix:** 
1. Delete `tests/__pycache__` directory
2. Delete any `.pyc` files
3. Run: `pytest --cache-clear`

### Issue 3: Visualization still running

**Symptom:** See "Visualization" timer in output

**Fix:**
1. Verify `tests/config_test.py` has `DO_VISUALIZATION = False`
2. Check that `expert_op4grid_recommender/main.py` has the `if config.DO_VISUALIZATION:` check
3. Try running with: `pytest tests/test_expert_op4grid_analyzer.py::test_reproducibility -v -s --tb=short`

### Issue 4: Import order problems

**Symptom:** Config override verification passes but visualization still runs

**Fix:** Check if `test_expert_op4grid_analyzer.py` has any module-level code that imports and caches config values

## Manual Testing

Run this Python code to manually verify:

```python
import sys
sys.path.insert(0, '/path/to/Expert_op4grid_recommender')

# Simulate what conftest does
from tests import config_test
sys.modules['expert_op4grid_recommender.config'] = config_test

# Now import and check
from expert_op4grid_recommender import config
print(f"Config file: {config.__file__}")
print(f"DO_VISUALIZATION: {config.DO_VISUALIZATION}")

# Import main and check again
from expert_op4grid_recommender.main import run_analysis
from expert_op4grid_recommender import config as config2
print(f"After importing main:")
print(f"  Config file: {config2.__file__}")
print(f"  DO_VISUALIZATION: {config2.DO_VISUALIZATION}")
```

Expected output:
```
Config file: .../tests/config_test.py
DO_VISUALIZATION: False
After importing main:
  Config file: .../tests/config_test.py
  DO_VISUALIZATION: False
```

## If All Else Fails

If the config override still isn't working, you can use an environment variable as a fallback:

1. Add to `expert_op4grid_recommender/config.py`:
```python
import os
DO_VISUALIZATION = os.environ.get('EXPERT_DISABLE_VIZ', 'false').lower() != 'true' if os.environ.get('EXPERT_DISABLE_VIZ') else True
```

2. Run tests with:
```bash
EXPERT_DISABLE_VIZ=true pytest tests/test_expert_op4grid_analyzer.py::test_reproducibility
```

But this should NOT be necessary if the conftest.py is working correctly!
