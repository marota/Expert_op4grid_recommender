# How to Skip Visualization in Tests

## Quick Answer

To skip the visualization step when running `test_reproducibility`, the test configuration has been updated to automatically set `DO_VISUALIZATION = False`.

Since your tests use `tests/config_test.py` (thanks to the `conftest.py` setup), visualization is already disabled for all tests!

## What Was Changed

### 1. Added `DO_VISUALIZATION` Config Parameter

**In `expert_op4grid_recommender/config.py`:**
```python
DO_VISUALIZATION = True  # Set to False to skip graph visualization (useful for tests)
```

**In `tests/config_test.py`:**
```python
DO_VISUALIZATION = False  # Skip visualization in tests for faster execution
```

### 2. Updated `run_analysis()` Function

The visualization step in `main.py` now checks the config before running:

```python
# Visualize graph (only if enabled in config)
if config.DO_VISUALIZATION:
    with Timer("Visualization"):
        # ... visualization code ...
else:
    print("Skipping visualization (DO_VISUALIZATION=False)")
```

## How It Works

1. When you run tests, `conftest.py` automatically replaces the package config with `tests/config_test.py`
2. The test config has `DO_VISUALIZATION = False`
3. When `run_analysis()` is called, it reads `config.DO_VISUALIZATION` and skips the visualization step
4. Your tests run faster without creating graph files!

## Running Tests

Just run your tests normally - visualization is automatically skipped:

```bash
# This will skip visualization automatically
pytest tests/test_expert_op4grid_analyzer.py::test_reproducibility -v
```

## Running Main Script (With Visualization)

When you run the main script directly, it uses the package config which has `DO_VISUALIZATION = True`:

```bash
# This WILL create visualizations
python -m expert_op4grid_recommender.main --date 2024-12-07 --timestep 9 --lines-defaut CHALOL61CPVAN
```

## Benefits

✅ **Faster tests** - No time spent creating visualizations  
✅ **No test artifacts** - No graph files cluttering your test runs  
✅ **Automatic** - No code changes needed in tests  
✅ **Flexible** - Easy to toggle if you need visualizations in specific tests  

## Other Config Parameters You Can Control

The same approach works for other config parameters! Some useful ones to adjust in tests:

```python
# In tests/config_test.py
DO_CONSOLIDATE_GRAPH = False      # Skip graph consolidation
DO_SAVE_DATA_FOR_TEST = False     # Skip saving test data
CHECK_ACTION_SIMULATION = True    # Keep action simulation checks
USE_DC_LOAD_FLOW = False          # Use AC load flow
N_PRIORITIZED_ACTIONS = 5         # Number of actions to prioritize
```

Simply edit `tests/config_test.py` to change any parameter for your tests!
