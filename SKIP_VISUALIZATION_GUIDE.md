# Skipping Visualization in Tests - Complete Guide

## Summary

Visualization is now automatically skipped when running tests! This was done by:

1. Adding a new `DO_VISUALIZATION` config parameter
2. Setting it to `False` in the test config
3. Wrapping the visualization code in a conditional check

## Changes Made

### 1. Config Files Updated

**`expert_op4grid_recommender/config.py` (package config):**
```python
DO_VISUALIZATION = True  # Set to False to skip graph visualization (useful for tests)
```

**`tests/config_test.py` (test config):**
```python
DO_VISUALIZATION = False  # Skip visualization in tests for faster execution
```

### 2. Main Analysis Function Updated

**`expert_op4grid_recommender/main.py`:**

The visualization section now checks the config before running:

```python
# Visualize graph (only if enabled in config)
if config.DO_VISUALIZATION:
    with Timer("Visualization"):
        graph_file_name = get_graph_file_name(current_lines_defaut, chronic_name, current_timestep, use_dc)
        save_folder = config.SAVE_FOLDER_VISUALIZATION
        lines_swapped = list(df_of_g[df_of_g.new_flows_swapped].line_name)
        make_overflow_graph_visualization(
            env, overflow_sim, g_overflow, hubs, obs_simu_defaut, save_folder, graph_file_name, lines_swapped,
            custom_layout
        )
else:
    print("Skipping visualization (DO_VISUALIZATION=False)")
```

## How It Works

### When Running Tests

1. `conftest.py` replaces the package config with test config
2. Test config has `DO_VISUALIZATION = False`
3. Visualization step is skipped
4. Tests run faster!

```bash
# Visualization automatically skipped
pytest tests/test_expert_op4grid_analyzer.py::test_reproducibility -v
```

**Output will show:**
```
Skipping visualization (DO_VISUALIZATION=False)
```

### When Running Main Script

1. Package config is used (not test config)
2. Package config has `DO_VISUALIZATION = True`  
3. Visualization runs normally

```bash
# Visualization runs normally
python -m expert_op4grid_recommender.main --date 2024-12-07 --timestep 9
```

## Benefits

| Aspect | Tests | Main Script |
|--------|-------|-------------|
| **Visualization** | Skipped | Created |
| **Speed** | Faster | Normal |
| **Artifacts** | None | Graph files |
| **Config** | `tests/config_test.py` | `expert_op4grid_recommender/config.py` |

## Customization

### To Enable Visualization in a Specific Test

If you need visualization in a specific test, you can temporarily override the config:

```python
def test_with_visualization():
    import expert_op4grid_recommender.config as config
    
    # Temporarily enable visualization
    original_value = config.DO_VISUALIZATION
    config.DO_VISUALIZATION = True
    
    try:
        # Run your test that needs visualization
        result = run_analysis(...)
        # ... assertions ...
    finally:
        # Restore original value
        config.DO_VISUALIZATION = original_value
```

### To Change Other Test Behaviors

Edit `tests/config_test.py` to control other aspects:

```python
# Speed up tests
DO_CONSOLIDATE_GRAPH = False      # Skip graph consolidation  
DO_SAVE_DATA_FOR_TEST = False     # Don't save test data files
N_PRIORITIZED_ACTIONS = 3         # Reduce number of actions to test

# Maintain test quality  
CHECK_ACTION_SIMULATION = True    # Keep simulation checks
USE_DC_LOAD_FLOW = False          # Use accurate AC load flow
```

## Verification

Run the verification script to confirm everything is set up correctly:

```bash
python tests/verify_config_setup.py
```

This will check that:
- Config override is working
- `DO_VISUALIZATION` is present in both configs
- Test config has the correct value

## Troubleshooting

**Problem:** Visualization still runs in tests

**Solution:** Make sure `conftest.py` is in the `tests/` directory and pytest is finding it:

```bash
pytest --co  # Show what tests pytest found
```

**Problem:** Want to see what config value is being used

**Solution:** Add a print statement at the start of your test:

```python
def test_reproducibility(...):
    from expert_op4grid_recommender import config
    print(f"DO_VISUALIZATION = {config.DO_VISUALIZATION}")
    # ... rest of test
```

## Files Modified

1. ✅ `expert_op4grid_recommender/config.py` - Added `DO_VISUALIZATION = True`
2. ✅ `tests/config_test.py` - Added `DO_VISUALIZATION = False`  
3. ✅ `expert_op4grid_recommender/main.py` - Wrapped visualization in conditional
4. ✅ `tests/verify_config_setup.py` - Added check for new parameter
5. ✅ `SETUP_SUMMARY.md` - Updated with new info
6. ✅ `tests/HOW_TO_SKIP_VISUALIZATION.md` - This file!

## Summary

🎉 **Visualization is now automatically skipped in tests!**

- **No code changes needed in your tests**
- **Tests run faster**
- **No unwanted graph files**
- **Main script still works normally**
- **Easy to customize if needed**

Just run your tests as usual and enjoy the speed boost!
