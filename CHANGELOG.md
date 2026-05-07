# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.1.post1] - 2026-05-07

### Added

- **`extra_lines_to_cut_ids` plumbing** (`graph_analysis/builder.py`, `pypowsybl_backend/overflow_analysis.py`, `graph_analysis/visualization.py`, `main.py`, PR #89): `build_overflow_graph` (and its grid2op / pypowsybl wrappers) now accept an optional `extra_lines_to_cut_ids` parameter. Operator-supplied indices are appended to `Grid2opSimulation.ltc` / `AlphaDeespAdapter.ltc` so the cut still happens, and forwarded as `extra_lines_to_cut=…` to `OverFlowGraph` so the new `is_extra_cut` tag flows through (and the visualization keeps these edges out of the Overloads / Monitored layers). Implements ExpertAgent's `additionalLinesToCut` semantic. `run_analysis_step2_graph` reads `context["extra_lines_to_cut_ids"]` (default `[]`); `make_overflow_graph_visualization` accepts the parameter for plumbing completeness.

### Compatibility

- Defaults to an empty list / `None` everywhere — existing callers see no behaviour change. Step1 populates `context["extra_lines_to_cut_ids"] = []`; step2 callers (e.g. CoStudy4Grid) can override before invoking `run_analysis_step2_graph`. Extras already present in `overloaded_line_ids` are silently de-duplicated.

---

## [0.2.1] - 2026-05-05

### Added

- **Overflow-graph tagger wiring** (`graph_analysis/visualization.py`, `main.py`, PR #88): `make_overflow_graph_visualization` now accepts optional `lines_constrained_path` / `nodes_constrained_path` / `red_loop_lines` / `red_loop_nodes` / `lines_overloaded` parameters and forwards them to the new `OverflowGraph.tag_constrained_path` and `OverflowGraph.tag_red_loops` taggers (alongside the existing `highlight_significant_line_loading`). The pipeline computes these lists right after the distribution-graph pass and passes them into the three call sites of `make_overflow_graph_visualization`. Result: the serialised overflow graph now carries explicit `is_hub` / `in_red_loop` / `on_constrained_path` / `is_monitored` / `is_overload` boolean flags driving the upstream alphaDeesp interactive viewer's semantic layer toggles.

### Compatibility

- All new parameters default to `None`. Existing callers see no behaviour change — the taggers are no-ops when the recommender does not pass any list. Requires `ExpertOp4Grid >= 0.3.2` to consume the flags in the interactive HTML viewer; older versions still serialise the same numerical / colour content unchanged.

---

## [0.2.0] - 2026-04-14

### Added

- **`PowerReductionAction`** (`pypowsybl_backend/action_space.py`, PR #74): New action class that modifies active power setpoints (`target_p`) for loads and generators without electrically disconnecting them. Enables partial load shedding and renewable curtailment with maintained grid connectivity and voltage support. Integrated via `set_load_p` and `set_gen_p` action dictionary keys with `update_loads()` / `update_generators()` batch calls.
- **Renewable curtailment discovery fully integrated** (`action_evaluation/discovery/`, PR #73): `find_relevant_renewable_curtailment` is now part of the main analysis pipeline. Candidates are identified on upstream nodes of the constrained path among wind/solar generators. Controlled by `ENABLE_RENEWABLE_CURTAILMENT`, `RENEWABLE_CURTAILMENT_MARGIN`, `RENEWABLE_CURTAILMENT_MIN_MW`, and `RENEWABLE_ENERGY_SOURCES` configuration flags.
- **`ENABLE_RENEWABLE_CURTAILMENT` / `ENABLE_LOAD_SHEDDING` config flags** (PR #73): Explicit boolean switches to include or exclude heuristic action types from the analysis without touching `MIN_*` counts.
- **Pydantic-based configuration** (PR #84): `config.py` and `config_basic.py` define a `Settings(BaseSettings)` class with type validation, range/bound checking, and `EXPERT_OP4GRID_*` environment variable overrides. Module-level attribute publishing is preserved, so existing `config.DATE = ...` mutation and `from ... import DATE` call sites continue to work unchanged.
- **`quality` optional dependency group** (PR #84): `pip install -e .[quality]` installs `radon>=6.0`, `vulture>=2.10`, `interrogate>=1.5`, and `ruff>=0.5` for static analysis.
- **Comprehensive discovery caching** (`action_evaluation/discovery/_base.py`, PR #76): Six cache helpers that eliminate repeated expensive traversals on large networks — `_get_edge_data_cache()`, `_get_blue_edge_names_set()`, `_get_subs_with_loads()`, `_get_subs_with_renewable_gens()`, `_build_line_capacity_map()`, and `_build_node_flow_cache()`.
- **Baseline simulation hoisted outside action loops** (PR #76): For load shedding and renewable curtailment, the N-1 baseline rho is computed once per scenario and reused across all candidate actions.
- **`SimulationEnvironment` caching** (PR #72): Avoids redundant environment initialisation on repeated analysis calls.
- **`skip_enrichment` parameter** on the detection phase (PR #72): Bypasses redundant action enrichment during the initial overload detection step.
- **New tests**: `test_graph_analysis.py` (PR #78) for graph analysis helpers and `test_environment_pypowsybl.py` (PR #78) for pypowsybl environment setup logic.
- **Design and quality documents** (PR #71, PR #77): `docs/renewable_curtailment_design.md` (algorithm, scoring, data requirements) and `docs/code-quality-analysis.md` (static analysis snapshot: god-module inventory, testing gaps, TODO/FIXME catalogue).
- **Type hints and docstrings** back-filled on `load_training_data.py`, `load_evaluation_data.py`, `repas.py`, `make_env_utils.py`, `make_assistant_env.py`, and `make_training_env.py` (PR #84).

### Changed

- **Discovery module refactored to mixin architecture** (PR #78): The monolithic `discovery.py` (3001 lines, 42+ methods) is split into `action_evaluation/discovery/` with nine focused mixin modules:
  - `_base.py` — `DiscovererBase` with shared state, caches, and simulation plumbing
  - `_line_reconnection.py` — line reconnection discovery
  - `_line_disconnection.py` — line disconnection scoring
  - `_node_merging.py` — bus merge discovery and delta-theta scoring
  - `_node_splitting.py` — bus split discovery (AlphaDeesp)
  - `_load_shedding.py` — load shedding candidate identification
  - `_renewable_curtailment.py` — renewable curtailment candidate identification
  - `_pst.py` — phase-shifter transformer tap discovery
  - `_orchestrator.py` — top-level pipeline orchestration and scoring assembly
- **Load shedding and curtailment emit `PowerReductionAction`** (PR #74): Both discovery methods now produce partial setpoint reductions (`set_load_p` / `set_gen_p`) instead of `set_bus` disconnections. Action metadata includes `action_mode`, `target_p_MW`, and `reduction_MW`.
- **`ActionClassifier` enhanced** (PR #73, PR #74): Now supports `open_load`, `open_gen`, `load_power_reduction`, and `gen_power_reduction` action types; handles `None` description input without raising `AttributeError`.
- **Superposition theorem filtering** (PR #73): `curtail_*` and `load_shedding_*` action IDs are excluded from the beta-coefficient linear solver, which assumes standard topological coupling not applicable to power-setpoint actions.
- **Vectorised topology cache** (PR #72): `NetworkTopologyCache` construction uses vectorised operations instead of per-element Python loops — faster initialisation and update.
- **Environment variable for training data path** (PR #77): `load_training_data.py` reads `EXPERT_OP4GRID_TRAINING_OBS_DIR` instead of a hardcoded developer path.
- **`sys.path` manipulation removed from `main.py`** (PR #77, PR #84): The package now relies on proper editable installation (`pip install -e .`) rather than runtime path hacking.

### Fixed

- **`ActionClassifier` robustness** (PR #73): `None` description no longer raises `AttributeError` during type identification.
- **`NoneType` and `AttributeError` regressions** (PR #73): Fixed during integration of renewable curtailment in `discovery.py` and `classifier.py`.
- **Topology reconstruction for mixed actions** (PR #78): `_build_action_entry_from_topology` robustified for combined topology/switch action formats.
- **Duplicate config definitions** (PR #77): Removed second (silent last-write-wins) definitions of `RENEWABLE_CURTAILMENT_MARGIN`, `RENEWABLE_CURTAILMENT_MIN_MW`, `RENEWABLE_ENERGY_SOURCES`, and `PYPOWSYBL_FAST_MODE`.

### Removed

- **`observation_timers.py`** — 1052-line stale fork of `observation.py` with zero importers; deleted (PR #77).
- **`conversion_actions_repas_original.py`** — 274-line superseded stub with zero importers; deleted (PR #77).

### Dependencies

- Added `pydantic>=2.0` and `pydantic-settings>=2.0` as core runtime dependencies (PR #84).

---

## [0.1.9] - 2026-03-25

### Added

- **Load Shedding Actions**: Automated discovery and scoring of load shedding candidates on downstream nodes of constrained paths to alleviate overloads when topological actions are insufficient.
- **Improved Action Prioritization**: Introduced `MIN_LOAD_SHEDDING` and `MIN_PST` configuration parameters to guarantee a minimum number of prioritized actions for these types.
- **Integrated Pipeline Support**: Load shedding is now fully integrated into the two-step analysis pipeline, with detailed scoring hypotheses included in `action_scores`.

### Changed

- **Path Management Refactor**: Switched to `pathlib.Path` for all base directories and file paths in `config.py`, improving reliability for relative execution and cross-platform compatibility.
- **Enhanced Instrumentation**: Added comprehensive timing blocks for Load Shedding discovery and prioritization steps.

---

## [0.1.8_post1] - 2026-03-20

### Added

- **PST Support in Superposition Theorem**: Added `act1_is_pst` and `act2_is_pst` flags to `compute_combined_pair_superposition` to correctly quantify impacts for phase-shifter actions.
- **Direct XIIDM Loading**: Enhanced `main.py` entry point to allow loading a grid case directly from an `.xiidm` file path, rather than requiring it to reside within a specific directory structure.

### Fixed

- **Robust PST Asset Identification**: Improved ID-based identification logic to handle REPAS-style PST IDs (stripping leading dots and discovery-added suffixes like `_inc1`/`_dec2`).
- **PST Affected Line Detection**: Correctly propagates the `affected_line` (PST branch ID) in PST action details, ensuring branch highlighting in the UI results.

### Documentation

- Added detailed technical documentation for the Superposition Theorem implementation and its application to topological and PST-based remedial actions.

---

## [0.1.8] - 2026-03-16

### Added

- **Superposition Theorem Integration**: Implemented impact quantification for topological actions using the superposition theorem.
- **Islanding Impact Quantization**: Enhanced islanding detection to report disconnected MW, providing better visibility into the severity of grid splits.
- **Superposition Results in Analysis**: Integrated virtual flow and delta-theta computations into the analysis results dictionary.

### Changed

- **Improved Non-Reconnectable Detection**: Switched to OR logic for line isolation detection — a line is now considered non-reconnectable if at least one of its extremities is isolated (all breakers/disconnectors open).

### Fixed

- **Superposition Data Integrity**: Fixed missing data fields in superposition results and resolved `NameError` bugs in calculation modules.
- **Virtual Flow Computations**: Corrected delta-theta and virtual flow logic for more accurate impact estimation.

---

## [0.1.7] - 2026-03-11

### Added

- **Phase Shifter Transformer (PST) Support**: Integrated PST tap variations and atomized PST actions from REPAS JSON.
- **PST Support in Grid2Op conversion**: Added handling for atomized PST actions in Grid2Op format.

### Fixed

- **Analyzer Stability**: Resolved analyzer test failures and improved environment creation robustness.

---

## [0.1.6] - 2026-03-10

### Added

- **Pypowsybl Format for Rebuild Actions**: Added `--pypowsybl-format` option to `--rebuild-actions` for switch-based output.
- **Asset Identification Enhancement**: Inferred `has_line`/`has_load` from switch names in the pypowsybl backend.

### Changed

- **Network Cache Optimization**: Optimized `NetworkTopologyCache` to eliminate O(all_elements) cost per action.

### Fixed

- **Switch Operation Diffing**: Corrected `set_bus` logic to only include assets changed by the switch operation.

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
