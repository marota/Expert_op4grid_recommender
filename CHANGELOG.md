# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.5] - 2026-03-07

### Added

- **Dynamic Action Content Computation**: Implemented `LazyActionDict` to compute action `content` (bus assignments) on-demand from switch states, significantly reducing action JSON file sizes.
- **Prioritization of Direct Overload Disconnections**: Added a +1.0 score boost for actions that disconnect currently overloaded lines in unconstrained regimes.
- **Thermal Limit Monitoring Factor**: Added support for rescaling thermal limits in overflow graph visualizations via `monitoring_factor_thermal_limits`.
- **Minimum Action Count Enforcement**: Introduced `MIN_*` configuration parameters to guarantee a minimum number of actions per type (reconnection, disconnection, coupling).
- **Flexible Monitoring File Routing**: Improved configuration for `LINES_MONITORING_FILE`.

### Changed

- **Two-Step Analysis Refactor**: Split `run_analysis` into `run_analysis_step1` and `run_analysis_step2` for better decoupling.
- **Improved Parameter Propagation**: Safely propagate `fast_mode` down to all simulation sub-components.

### Fixed

- **Monkey-patching Bug**: Fixed `AttributeError` in `main.py` where `_check_rho_reduction` was incorrectly accessed.

---

## [0.1.4] - 2026-03-04

### Added

- **Pre-Existing Overload Filtering**: Pre-existing overloads (already overloaded in N state) are excluded from N-1 analysis results and `max_rho` prioritization, unless worsened by a configurable threshold. Controlled by `PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD` (default 0.02).
- **Pypowsybl Backend Optimizations**:
    - **Incremental Simulation Branching**: Remedial actions now branch directly from converged N-1 contingency states, leveraging "hot starts" and ensuring state consistency.
    - **Simulation Fast Mode**: Introduced `PYPOWSYBL_FAST_MODE` (default: true) which disables voltage control for shunts and transformers during variants to significantly boost speed.
    - **Automatic Fallback Mechanism**: Simulations in "fast" mode automatically fallback and retry in standard "slow" mode if they fail to converge or diverge.
    - **Vectorized Observation Creation**: Over 80% reduction in observation initialization time via NumPy-based state extraction.
    - **Batched Topological Changes**: Multiple switch and bus changes are now applied in fewer pypowsybl update calls.
- **Robustness Improvements**:
    - **Flexible Switch ID Matching**: Improved ID matching supporting substation-prefixed switch names.
    - **Unified Initialization Fallback**: Consistent fallback from `PREVIOUS_VALUES` to `DC_VALUES` initialization in `network_manager`.
    - **Consistent Simulation Tuning**: Applied "fast" mode logic consistently across `observation.simulate` and `overflow_analysis` (PTDF-based) passes.

### Fixed

- **Switch Action Test Fix**: Corrected mock registry in `test_switch_action_apply_to_network` to properly verify batch switch updates.

### Tests

- Added `verify_incremental_branching.py` script for end-to-end variant state validation.
- Enhanced `tests/test_pypowsybl_backend.py` with specific test cases for incremental branching and fast mode logic.
- Add `test_pre_existing_overloads_excluded_from_analysis` and `test_pre_existing_overloads_excluded_from_max_rho`.
- Regression test for `run_analysis` to handle pre-existing overloads correctly when all lines are monitored.

---
## [0.1.3] - 2026-02-25

### Added

- **Configurable Line Extremity Loading**: Added `MAX_RHO_BOTH_EXTREMITIES` flag in `config.py` (default: false). When true, the pypowsybl backend evaluates the maximum loading rate (`rho`) from both extremities of a line using potentially distinct thermal limits.
- **Improved Limit Parsing**: `network_manager` `get_thermal_limits` returns a struct that can support separate limits for line origins and extremities.
- **Launch Options for Action Filtering**: Added `MIN_LINE_RECONNECTIONS`, `MIN_CLOSE_COUPLING`, `MIN_OPEN_COUPLING`, and `MIN_LINE_DISCONNECTIONS` to `config.py` to ensure minimum counts of each action type are considered. The `main.py` pipeline is updated to enforce these minimums by pulling up relevant actions if they aren't met naturally.
- **Ignore Monitoring Flag**: Added `IGNORE_LINES_MONITORING` flag to optionally bypass lines monitoring limits under specific configurations.
- Explicit test `test_max_rho_both_extremities` added to the test suite to verify loading calculation bounds behavior.

### Fixed

- **Improved Disconnection Scoring Constraints**: Fixed issue #30 where disconnection constraint formulas used incorrect bounding states. Upgraded `compute_line_disconnection_action_score` to properly evaluate redispatch limits between N-1 baseline (`obs_defaut`) and N-2 (`obs_linecut`) utilizing the actual line capacities.
- **Unconstrained Disconnection Regime**: Scoring logic simplified to use direct flow ratio (`capacity * (1 - rho_before) / (rho_after - rho_before)`) when no new overloads are instantiated.
- **CI Dependency formatting**: Cleaned up trailing commas and spaces in `requirements.txt`.

---

## [0.1.2] - 2026-02-20

### Added

- **Action scores dictionary**: `run_analysis()` now returns an `action_scores` dict alongside `prioritized_actions`. It has four keys — `"line_reconnection"`, `"line_disconnection"`, `"open_coupling"`, `"close_coupling"` — each containing:
  - `"scores"`: `{action_id: float}` sorted by descending score.
  - `"params"`: underlying scoring hypotheses (thresholds, flow bounds, etc.).
- **Line disconnection scoring**: asymmetric bell curve (alpha=3, beta=1.5) centred between the minimum required redispatch and the maximum tolerable redispatch; score is positive inside the acceptable window and negative outside.
- **Node merging scoring**: delta-phase score (`theta2 − theta1`) based on voltage angle difference between the two buses being merged; the red-loop bus (carrying more positive dispatch flow) is used as the reference.
- **Node splitting — per-action details**: `compute_node_splitting_action_score_value` now returns a `(score, details)` tuple. `details` contains `node_type`, `bus_of_interest`, and the four flow components (`in_negative_flows`, `out_negative_flows`, `in_positive_flows`, `out_positive_flows`) for the selected bus; these are stored per-action in `params_splits_dict` and exposed through `action_scores["open_coupling"]["params"]`.
- **Per-action assets for coupling actions**: `action_scores["open_coupling"]["params"]` and `action_scores["close_coupling"]["params"]` now include per-action `"assets"` dictionaries listing the lines, loads, and generators connected to the scored bus.
- **Unconstrained disconnection scoring**: when the overflow graph produces no new overloads after redispatch (i.e. `max_redispatch = ∞`), a linear ramp replaces the bell curve — score = 1 at `max_overload_flow`, linearly decreasing to 0 at `min_redispatch`, and negative quadratic tail below. The `params` field includes a `"regime"` indicator (`"constrained"` or `"unconstrained"`).
- **Score rounding**: all float values in `action_scores` (both scores and params) are rounded to 2 decimal places.

### Fixed

- **Red loop bus identification** in `compute_node_merging_score`: the bus connected to the red loop is now correctly identified as the one with the **most positive** dispatch flow on its overflow graph edges (previously used negative flow, which was inverted).
- **Test tuple unpacking**: `test_integration_full_scoring_pipeline` now correctly unpacks the `(score, details)` tuple returned by `compute_node_splitting_action_score_value`.

### Tests

- `TestNodeSplittingScoreValueReturn` (5 tests): verifies the `(score, details)` tuple return format, required keys in `details`, flow values matching the selected `bus_of_interest`, `node_type` propagation, and the empty-buses edge case.
- Backward compatibility tests: `compute_node_splitting_action_score` wraps a plain-float return as `(float, {})` and passes a tuple through unchanged.
- `TestDiscoveryParamsStorage` (4 tests): verifies that `params_reconnections`, `params_disconnections`, `params_splits_dict`, and `params_merges` are correctly populated after each discovery method.
- `TestActionScoresStructureAndRounding` (7 tests): verifies the assembled `action_scores` structure, descending sort order, 2-decimal rounding for flat and nested params, and graceful handling of empty categories.
- `TestUnconstrainedLinearScore` (7 tests): verifies the linear ramp scoring for the unconstrained disconnection regime — score at min/max/midpoint, capping at 1 above max, zero at min, negative quadratic tail below min, and increasingly negative further below.

---

## [0.1.1.post4] - 2026-02-17

### Changed

- Made `grid2op` fully optional across `make_env_utils` and related modules; importing the package no longer fails when `grid2op` is not installed.

---

## [0.1.1] - 2026-01-xx

### Added

- `run_analysis()` now returns a detailed output dictionary including per-action metadata (type, substation, lines involved, simulation results).

### Fixed

- Variant ID collision in `pypowsybl` `simulate(keep_variant=True)` caused incorrect results when the same variant was reused across simulations.

---

## [0.1.0.post1] - 2025-xx-xx

### Added

- `reco_deco` (reconnect-then-disconnect) composite actions are now included in the default action space.

### Changed

- Optimised reconnectable line detection: uses `expertop4grid` new methods, collapses graph to key components, and retains only the main overflow components.

---

## [0.1.0] - 2025-xx-xx

### Added

- Initial PyPI release.
- Modular package structure (`action_evaluation`, `graph_analysis`, `pypowsybl_backend`, `utils`).
- Pure pypowsybl backend (`--backend pypowsybl`) as an alternative to Grid2Op.
- Expert rule engine for filtering topology actions.
- Action prioritisation: line reconnections, disconnections, node splitting, and node merging.
- Overflow graph construction and visualisation via `alphaDeesp` and `networkx`.
- CLI entry point with `--date`, `--timestep`, `--lines-defaut`, `--backend`, `--rebuild-actions` flags.
