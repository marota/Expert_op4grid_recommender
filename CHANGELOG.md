# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
