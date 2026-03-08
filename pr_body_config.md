# fix: configuration updates for testing and fast mode

This PR updates the default configuration to simplify testing and ensure consistency across environments.

## Changes
- **PYPOWSYBL_FAST_MODE**: Set to `False` by default to ensure high-fidelity simulations during testing.
- **Environment & Action Space**: Updated `ENV_NAME` and `FILE_ACTION_SPACE_DESC` to point to the small grid test environment, ensuring faster developer iteration and reliable automated tests.
