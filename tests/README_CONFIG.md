# Test Configuration

## Overview

This test suite uses a separate configuration file (`config_test.py`) instead of the package's main configuration file (`expert_op4grid_recommender/config.py`).

## How It Works

The `conftest.py` file in this directory contains a pytest fixture that automatically runs before any tests. This fixture replaces the package config module with the test config module by modifying `sys.modules`.

### What This Means

When any code in your tests (or code imported by your tests) does:

```python
from expert_op4grid_recommender import config
```

or

```python
from expert_op4grid_recommender.config import SOME_VARIABLE
```

Python will actually import from `tests/config_test.py` instead of `expert_op4grid_recommender/config.py`.

## Benefits

1. **Test Isolation**: Tests use consistent configuration values without affecting the main package configuration
2. **Reproducibility**: Test results are reproducible because they always use the same config
3. **Flexibility**: You can easily modify `config_test.py` to set up specific test scenarios
4. **Transparency**: No need to modify the main config file or pass config parameters around

## Files

- `config_test.py`: The configuration file used during testing
- `conftest.py`: pytest configuration that sets up the config override
- `test_config_override.py`: Simple tests to verify the override is working correctly

## Running Tests

Simply run pytest as normal:

```bash
pytest tests/test_expert_op4grid_analyzer.py::test_reproducibility
```

The config override happens automatically - no special flags or setup required!

## Verifying the Override

Run the verification test to ensure the override is working:

```bash
pytest tests/test_config_override.py -v
```

This will confirm that the test configuration is being used instead of the package configuration.

## Modifying Test Configuration

To change configuration values for tests, simply edit `tests/config_test.py`. The changes will automatically be picked up by all tests.

## Important Notes

- The config override is **session-scoped**, meaning it applies to all tests in a test run
- The original package config remains unchanged - only the import behavior is modified
- This approach works because pytest runs tests in a separate process
