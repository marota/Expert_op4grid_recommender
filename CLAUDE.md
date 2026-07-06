# CLAUDE.md - Expert Op4Grid Recommender

> **Purpose**: Quick context for AI assistants (Claude, etc.) to understand this codebase for code improvements, development, or testing tasks.

## Project Overview

**ExpertOp4Grid Recommender** is an expert system analyzer for power grid contingencies. It analyzes N-1 contingencies in Grid2Op/pypowsybl environments, builds overflow graphs, applies expert rules to filter potential actions, and identifies corrective measures to alleviate line overloads.

**License**: Mozilla Public License 2.0 (MPL 2.0)  
**Python**: ≥3.10  
**Domain**: Power systems / Grid operations / RTE (French TSO)

---

## Documentation

A full, categorized documentation index lives in
[`docs/README.md`](docs/README.md). Quick map:

- **Architecture**: `docs/architecture/` — overview, simulation pipeline, recommender-model contract, maneuver plugin phases
- **Recommender action designs**: `docs/recommender/` — antenna graph, load shedding, renewable curtailment, superposition theorem
- **Maneuver module**: `docs/manoeuvre/` — module, rules, IHM, N-busbar, optimisations, plan
- **RTE-7000 dataset**: `docs/manoeuvre/dataset_rte7000/` — campaign, guide, plan, handoff
- **Archive**: `docs/archive/` — migration plan, code-quality analysis, setup summary
- **Release notes**: `docs/release-notes/`

---

## Quick Start Commands

```bash
# Install (editable mode for development)
pip install -e .
pip install -e .[test]  # with test dependencies

# Run analysis (default parameters from config)
python expert_op4grid_recommender/main.py

# Run with custom parameters
python expert_op4grid_recommender/main.py --date 2024-08-28 --timestep 36 --lines-defaut P.SAOL31RONCI

# Use bare environment (no date/chronics)
python expert_op4grid_recommender/main.py --date None

# Use pypowsybl backend instead of grid2op
python expert_op4grid_recommender/main.py --backend pypowsybl

# Ignore lines monitoring limits
python expert_op4grid_recommender/main.py --ignore-lines-monitoring

# Rebuild action space from REPAS files
python expert_op4grid_recommender/main.py --rebuild-actions --repas-file data/action_space/allLogics.json

# Rebuild actions in pypowsybl (switch-based) format
python expert_op4grid_recommender/main.py --rebuild-actions --pypowsybl-format

# Enable pypowsybl fast mode (no voltage control, faster variants)
python expert_op4grid_recommender/main.py --backend pypowsybl --fast-mode

# Run tests
pytest
pytest -v tests/test_ActionClassifier.py  # specific test file
pytest -k "test_name"  # specific test by name
```

---

## Architecture Overview

```
expert_op4grid_recommender/
├── main.py                    # Thin backward-compat facade re-exporting the
│                              #   pipeline/CLI/backends API (kept so every
│                              #   `from …main import …` keeps resolving)
├── cli.py                     # Command-line entry point (argparse + main())
├── pipeline.py                # Analysis pipeline core: run_analysis[_step1/2*],
│                              #   AnalysisContext / AnalysisResult dataclasses
├── backends.py                # SimulationBackend protocol + Grid2opBackend /
│                              #   PypowsyblBackend (fast_mode as ctor state);
│                              #   replaces the old *_grid2op/*_pypowsybl wrappers,
│                              #   the context function-pointers and the is_pypowsybl forks
├── config.py                  # Main configuration (pydantic Settings, re-exported as module attrs)
├── config_basic.py            # One alternative config variant (Settings(...) with overrides)
├── exceptions.py              # Domain exceptions (LoadFlowDivergedError)
├── environment.py             # Grid2Op environment setup
├── environment_pypowsybl.py   # Pure pypowsybl environment setup
├── data_loader.py             # Load action dictionaries from JSON
│
├── models/                    # Pluggable recommender-model contract
│   ├── base.py                # RecommenderModel ABC, RecommenderInputs/Output DTOs,
│   │                          #   SimulatedAction card, DictCompatMixin
│   ├── expert.py              # ExpertRecommender (the default expert-system model)
│   └── _expert_discovery.py   # _run_expert_discovery + _run_expert_action_filter
│                              #   (moved out of main.py to break the import cycle)
│
├── action_evaluation/         # Action analysis module
│   ├── classifier.py          # ActionClassifier: categorize action types (now
│   │                          #   table-driven via action_types.classify_by_description)
│   ├── action_types.py        # ActionType enum (values == the historical type
│   │                          #   strings) + declarative keyword→type table (R5)
│   ├── rules.py               # ActionRuleValidator: apply expert rules
│   │                          #   (backend-agnostic coupling localization, C7 fix)
│   └── discovery/             # ActionDiscoverer PACKAGE (split from the old ~3000-line
│       ├── _base.py           #   discovery.py). DiscovererBase + one mixin per family.
│       ├── _results.py        #   R5 data model: FamilyResult (per-family outcome) +
│       │                      #   FAMILY_SPECS registry + generated legacy @property
│       │                      #   bridges + ACTION_SCORES_ORDER / MIN_PHASE_ORDER /
│       │                      #   FILL_PHASE_ORDER tables + DisconnectionBounds
│       ├── _injection_base.py #   InjectionDiscoveryBase: shared overload preamble +
│       │                      #   influence factor for the 3 injection families
│       ├── _protocols.py      #   DiscovererProtocol: the shared self surface mixins need
│       ├── _orchestrator.py   #   discover_and_prioritize() — data-driven action_scores
│       │                      #   assembly + two-pass prioritization over the tables
│       ├── _line_reconnection.py, _line_disconnection.py, _node_splitting.py,
│       ├── _node_merging.py, _pst.py, _load_shedding.py,
│       └── _renewable_curtailment.py, _redispatch.py
│
├── graph_analysis/            # Overflow graph module
│   ├── builder.py             # build_overflow_graph() using alphaDeesp
│   ├── processor.py           # Graph connectivity, path analysis, antenna context
│   ├── antenna_graph.py       # Antenna (islanded-pocket) synthetic overflow graph
│   └── visualization.py       # Graph visualization (Graphviz/pydot)
│
├── pypowsybl_backend/         # Pure pypowsybl implementation (no grid2op)
│   ├── simulation_env.py      # SimulationEnvironment: main interface
│   ├── network_manager.py     # Network loading, variants, load flow, caches
│   ├── observation.py         # Grid2op-compatible observation
│   ├── action_space.py        # Action creation (topology, switching)
│   ├── topology.py            # Topology vector management (legacy; unused by pipeline)
│   └── overflow_analysis.py   # Overflow graph without alphaDeesp
│
├── manoeuvre/                 # Detailed-topology maneuver module (~9k LOC, self-contained;
│   ├── algo/                  #   zero imports to/from the rest of the package). Sequencing,
│   ├── dataset/               #   targets, placement; RTE-7000 dataset; plugin architecture;
│   ├── plugins/               #   Flask IHM lives in scripts/manoeuvre_ihm.py. See
│   └── ...                    #   expert_op4grid_recommender/manoeuvre/CLAUDE.md + docs/manoeuvre/.
│
└── utils/                     # Utility modules
    ├── simulation.py          # Grid2Op simulation helpers
    ├── simulation_pypowsybl.py # pypowsybl simulation helpers
    ├── reassessment.py        # Per-action reassessment (parallelised) + combined-pair estimation
    ├── helpers.py             # Timer, sorting, test data saving
    ├── helpers_pypowsybl.py   # pypowsybl-specific helpers
    ├── action_rebuilder.py    # Rebuild actions from REPAS format
    ├── conversion_actions_repas.py  # REPAS action conversion
    ├── superposition.py       # Superposition theorem for impact estimation
    ├── repas.py               # REPAS-specific utilities
    ├── data_utils.py          # StateInfo and data structures
    └── make_*_env.py          # Environment factory functions
```
(The test configuration lives in `tests/config_test.py` — it star-imports
`config.py` and overrides only the test deltas; `tests/conftest.py` swaps it in
via `sys.modules` + the package attribute.)

---

## Key Classes & Functions

### Main Entry Point
- **`run_analysis(analysis_date, current_timestep, current_lines_defaut, env_path=None, env_name=None, backend=Backend.GRID2OP, fast_mode=None)`** — lives in `pipeline.py`, re-exported from `main.py`
  - Orchestrates the full analysis pipeline
  - **Two-step pipeline** (v0.1.5+): internally delegates to
    - `run_analysis_step1(...)` → detects overloads; returns an
      `AnalysisContext` to continue, or an `AnalysisResult` short-circuit when
      there is no actionable overload (was a `(Optional, Optional)` tuple)
    - `run_analysis_step2_graph(context)` → builds the overflow graph
    - `run_analysis_step2_discovery(context)` → discovers, scores, prioritizes actions
  - The two-step split lets external callers (e.g. UI, notebooks) intervene between steps.
  - Returns an **`AnalysisResult`** (a dataclass with a dict-compatible view, so
    `result["prioritized_actions"]` / `result.get(...)` still work) carrying:
    - `"lines_overloaded_names"`: `List[str]`
    - `"prioritized_actions"`: `{action_id: {action, description_unitaire, rho_before, rho_after, max_rho, max_rho_line, is_rho_reduction, observation, ...}}`
    - `"action_scores"`: per-type scoring dict (see Data Structures below)
    - `"reassessment_parallelism"`: `{parallel, workers, cores_available, n_actions}` — how the per-action reassessment was parallelised (v0.2.6+)
    - `"prediction_time"` / `"assessment_time"`: per-stage execution times
    - Superposition theorem fields (`virtual_flows`, `delta_theta`, etc.) when available

### Action Evaluation (`action_evaluation/`)
- **`ActionClassifier`**: Determines action type (line open/close, nodal split/merge, load disconnect)
  - `identify_action_type(action_desc, by_description=True) -> str` — returns a
    type string (e.g. `"open_line"`, `"close_coupling"`). The description cascade
    is now data-driven via `action_types.classify_by_description` (R5).
  - **`ActionType` enum** (`action_evaluation/action_types.py`, R5): the 14 type
    tokens with `.value` **byte-identical** to the historical strings, plus
    category predicates (`involves_line`, `involves_coupling`, `is_open`,
    `is_topological`, …) that replace the scattered `"x" in action_type`
    substring checks. `classify_by_description(desc, has_line_load)` is the
    ordered keyword table.
  - `_is_nodale_grid2op_action(act) -> (is_nodale, subs, is_splitting)`

- **`ActionRuleValidator`**: Filters actions based on expert rules
  - `categorize_actions(dict_action, ...) -> (filtered_out, unfiltered)`
  - `check_rules(action_type, localization, subs_topology) -> (do_filter, reason)`
  - `localize_line_action(lines)`, `localize_coupling_action(subs)`
  - `_resolve_coupling_subs(action_desc)` (C7 fix): resolves a coupling's
    substation(s) + pre-action topology **backend-agnostically** — from either
    the pypowsybl `VoltageLevelId` or the grid2op `content['set_bus']
    ['substations_id']`. Previously grid2op couplings localized to `"unknown"`
    and escaped every expert rule.

- **`ActionDiscoverer`** (`action_evaluation/discovery/` **package** — `DiscovererBase` in `_base.py` + one mixin per family; the old ~3000-line `discovery.py` monolith is gone): Discovers and scores candidate actions across **eight** action types (the seven below + redispatch)
  - **Data model (R5, `_results.py`)**: the per-family outcome lives in one typed
    `FamilyResult` per family under `self.results[family]` (identified / effective
    / ineffective / scores / params / non_convergence), described by the
    declarative `FAMILY_SPECS` registry that also generates the back-compat
    `@property` bridges for the legacy attribute names
    (`identified_reconnections`, `scores_splits_dict`, `scores_pst_actions`, …).
    The `action_scores` assembly and the two-pass prioritization are data-driven
    loops over `ACTION_SCORES_ORDER` / `MIN_PHASE_ORDER` / `FILL_PHASE_ORDER`
    (one entry per family). Disconnection/PST scoring share the memoised
    `_get_disconnection_bounds()` (frozen `DisconnectionBounds`). The three
    injection families share `InjectionDiscoveryBase`.
  - `discover_and_prioritize(n_action_max) -> (Dict[action_id, action], action_scores)` — returns a **tuple** (despite an older `-> Dict` hint)
  - Action types discovered:
    1. `verify_relevant_reconnections` — line reconnections
    2. `find_relevant_disconnections` — line disconnections (asymmetric bell / linear scoring)
    3. `find_relevant_node_splitting` — node splits (via AlphaDeesp)
    4. `find_relevant_node_merging` — node merges (delta-theta scoring)
    5. `find_relevant_pst_actions` — phase-shifter transformer tap changes (v0.1.7+)
    6. `find_relevant_load_shedding` — load shedding on downstream constrained nodes (v0.1.9+)
    7. `find_relevant_renewable_curtailment` — wind/solar curtailment (v0.1.9+)
  - Returns a tuple; `action_scores` dict has per-type scores and params (see Data Structures)
  - Enforces minimum action counts from `config.MIN_LINE_RECONNECTIONS`, `MIN_CLOSE_COUPLING`, `MIN_OPEN_COUPLING`, `MIN_LINE_DISCONNECTIONS`, `MIN_PST`, `MIN_LOAD_SHEDDING`, `MIN_RENEWABLE_CURTAILMENT`
  - Uses AlphaDeesp for node splitting analysis and superposition theorem (`utils/superposition.py`) for virtual-flow impact estimation

### Graph Analysis (`graph_analysis/`)
- **`build_overflow_graph(env, obs, overload_ids, ...) -> (df, sim, graph, hubs, dist_graph, mapping)`**
  - Uses `alphaDeesp.Grid2opSimulation` to compute flow changes
  
- **`get_constrained_and_dispatch_paths(g_distribution, obs, ...) -> (lines_constrained, nodes_constrained, lines_dispatch, nodes_dispatch)`**

- **`identify_overload_lines_to_keep_overflow_graph_connected(obs, overload_ids, ...) -> (ids_kept, islanded_subs)`**

### Environment Setup
- **`setup_environment_configs(analysis_date) -> (env, obs, path, name, layout, actions, non_reco_lines, care_lines)`**
- **`get_env_first_obs(folder, name, use_eval, date, is_DC) -> (env, obs, path)`**

### pypowsybl Backend (`pypowsybl_backend/`)
- **`SimulationEnvironment`**: Drop-in replacement for Grid2Op env
  - `get_obs() -> PypowsyblObservation`
  - `action_space(dict) -> Action`
- **`PypowsyblObservation`**: Grid2Op-compatible observation
  - `simulate(action) -> (obs_simu, reward, done, info)`

---

## Key Configuration (`config.py`)

```python
# Case parameters
DATE = datetime(2024, 12, 7)
TIMESTEP = 9
LINES_DEFAUT = ["CHALOL61CPVAN"]

# Paths (relative to PROJECT_ROOT)
ENV_NAME = "env_dijon_v2_assistant"
ACTION_FILE_PATH = PROJECT_ROOT / "data/action_space/reduced_model_actions.json"

# Feature flags
USE_DC_LOAD_FLOW = False
DO_VISUALIZATION = True
CHECK_ACTION_SIMULATION = True
IGNORE_RECONNECTIONS = False
N_PRIORITIZED_ACTIONS = 5
USE_VOLTAGE_LEVEL_NAMES_IN_GRAPH = True  # readable VL names as overflow-graph node labels (v0.2.4+)

# Lines monitoring flags (v0.1.3+)
IGNORE_LINES_MONITORING = False     # bypass lines monitoring limits
LINES_MONITORING_FILE = None        # path to monitoring file (None = use env default)

# Loading rate flags (v0.1.3+, pypowsybl only)
MAX_RHO_BOTH_EXTREMITIES = False    # evaluate max rho from both line extremities
MONITORING_FACTOR_THERMAL_LIMITS = 0.95  # rescale permanent thermal limits (v0.1.5+)

# Pre-existing overload filtering (v0.1.4+)
PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD = 0.02  # exclude unless worsened by 2%

# Pypowsybl simulation tuning (v0.1.4+)
PYPOWSYBL_FAST_MODE = False         # disable voltage control for speed

# Reassessment parallelism (v0.2.9+) — the per-action reassessment clones a
# full pypowsybl network per worker, so on a CPU-limited container it over-
# subscribes the CPU and is slower than serial. Detection is container-aware
# (cgroup CPU quota + affinity), not os.cpu_count().
REASSESSMENT_PARALLEL = None        # None=auto, True=force parallel, False=force serial
REASSESSMENT_MIN_PARALLEL_CORES = 4 # auto mode: min effective cores to parallelise

# Minimum prioritized actions per type (v0.1.3+ / v0.1.9+ / v0.2.x)
MIN_LINE_RECONNECTIONS = 0
MIN_CLOSE_COUPLING = 0
MIN_OPEN_COUPLING = 0
MIN_LINE_DISCONNECTIONS = 0
MIN_PST = 0                         # v0.1.7+
MIN_LOAD_SHEDDING = 0               # v0.1.9+
MIN_RENEWABLE_CURTAILMENT = 0       # v0.1.9+
MIN_REDISPATCH = 0                  # v0.2.x

# Recommendation restriction (v0.2.x): empty list = all families allowed;
# otherwise only the listed family tokens are discovered
# ("reco","close","open","split","disco","pst","ls","rc","redispatch").
ALLOWED_ACTION_TYPES = []

# Antenna (islanded-pocket) recommendations (v0.2.x): when an overload islands a
# radial pocket, build a synthetic downstream overflow graph and restrict
# recommendations to injection actions.
ENABLE_ANTENNA_RECOMMENDATIONS = True

# Load shedding parameters (v0.1.9+)
LOAD_SHEDDING_MARGIN = 0.05         # 5% safety margin on required shedding
LOAD_SHEDDING_MIN_MW = 1.0          # ignore trivial shedding

# Renewable curtailment parameters (v0.1.9+)
RENEWABLE_CURTAILMENT_MARGIN = 0.05
RENEWABLE_CURTAILMENT_MIN_MW = 1.0
RENEWABLE_ENERGY_SOURCES = ["WIND", "SOLAR"]

# Redispatching parameters (v0.2.x)
REDISPATCH_DEFAULT_DELTA_MW = 10.0
REDISPATCH_MARGIN = 0.05
REDISPATCH_MIN_MW = 1.0

# Expert system parameters
PARAM_OPTIONS_EXPERT_OP = {
    "ThresholdReportOfLine": 0.05,
    "ThersholdMinPowerOfLoop": 0.1,
    "ratioToKeepLoop": 0.25,
    "ratioToReconsiderFlowDirection": 0.75,
    "maxUnusedLines": 3,
    "totalnumberofsimulatedtopos": 30,
    "numberofsimulatedtopospernode": 10
}
```

---

## Data Structures

### Action Dictionary (JSON format)
```json
{
  "action_id": {
    "description_unitaire": "Human readable description",
    "action_space_change": {...},  // Grid2Op action dict
    "lines": ["LINE_NAME"],
    "subs": ["SUB_NAME"],
    "type": "open_line|close_line|nodal_split|..."
  }
}
```

### Observation Attributes (Grid2Op compatible)
```python
obs.rho           # Line loadings (ratio to thermal limit)
obs.line_status   # Boolean array of line connection status
obs.name_line     # Array of line names
obs.name_sub      # Array of substation names
obs.a_or          # Current at origin side
obs.topo_vect     # Topology vector (bus assignments)
```

### Action Scores Dictionary (v0.1.2+)
```python
action_scores = {
    "line_reconnection": {
        "scores": {action_id: float, ...},   # sorted descending
        "params": {"percentage_threshold_min_dispatch_flow": float, "max_dispatch_flow": float}
    },
    "line_disconnection": {
        "scores": {action_id: float, ...},
        "params": {
            "min_redispatch": float,
            "max_redispatch": float,
            "peak_redispatch": float,
            "regime": "constrained" | "unconstrained"  # scoring regime used
        }
    },
    "open_coupling": {
        "scores": {action_id: float, ...},
        "params": {
            action_id: {
                "node_type": str,
                "bus_of_interest": int,
                "in_negative_flows": float,
                "out_negative_flows": float,
                "in_positive_flows": float,
                "out_positive_flows": float,
                "assets": {"lines": [...], "loads": [...], "generators": [...]}
            }, ...
        }
    },
    "close_coupling": {
        "scores": {action_id: float, ...},
        "params": {"percentage_threshold_min_dispatch_flow": float, "max_dispatch_flow": float,
                   action_id: {"assets": {"lines": [...], "loads": [...], "generators": [...]}}, ...}
    },
}
# All float values are rounded to 2 decimal places.
```

---

## Testing

### Test Structure
```
tests/
├── conftest.py                          # Config override (DO_VISUALIZATION=False)
├── config_test.py                       # Test-specific configuration
├── test_ActionClassifier.py             # Unit tests for classifier
├── test_ActionDiscoverer.py             # Unit tests for discoverer (incl. action_scores)
├── test_ActionRuleValidator.py          # Unit tests for rules
├── test_NodeSplittingDiscovery.py       # Node splitting scoring unit tests
├── test_expert_op4grid_analyzer.py      # Integration tests
├── test_pypowsybl_backend.py            # Backend compatibility tests
├── test_conversion_actions_repas.py     # REPAS conversion tests
├── test_action_rebuilder.py             # Action rebuilder tests
├── test_config_override.py              # Config override mechanism tests
├── test_simulation_optimizations.py     # Simulation performance tests
├── test_switch_action_and_substation_extraction.py  # Switch/substation tests
├── test_pst_actions.py                  # PST action unit tests (v0.1.7+)
├── test_superposition.py                # Superposition theorem core tests (v0.1.8+)
├── test_superposition_extended.py       # Superposition extended scenarios
├── test_superposition_action_types.py   # Superposition per action type
├── test_superposition_identification.py # Superposition target identification
├── test_superposition_rho_estimation.py # Virtual-flow rho estimation
├── test_lazy_action_dict.py             # LazyActionDict (v0.1.5+)
├── test_min_action_counts.py            # MIN_* enforcement
├── test_islanding_mw.py                 # Islanding MW quantification (v0.1.8+)
├── test_environment_detection.py        # Env detection logic
├── test_antenna_graph.py                # Antenna (islanded-pocket) overflow graph
│                                        # (see docs/recommender/antenna_overflow_graph.md)
├── test_visualization_filtering.py      # Visualization filters
├── test_typed_pipeline_spine.py         # R1/R2 contracts: AnalysisContext/Result +
│                                        # SimulatedAction dict-compat, SimulationBackend
│                                        # flags, shared-baseline routing, import-cycle
│                                        # dissolution, main facade re-exports
├── test_discovery_package_structure.py  # discovery package layout invariants + the
│                                        # DiscovererProtocol conformance check (A5)
├── test_discovery_results_model.py      # R5 FamilyResult / FAMILY_SPECS registry /
│                                        # property bridge / phase-order no-dup invariant
│                                        # + rc double-add behavioural regression
├── test_injection_base.py              # InjectionDiscoveryBase (overload preamble +
│                                        # influence factor) + memoised _get_disconnection_bounds
├── test_action_types_enum.py           # ActionType enum + declarative classify_by_description
└── test_data_modules.py                # first tests for utils/load_{training,evaluation}_data
                                         # (import smoke, load_interesting_lines, C6 guards)
```

### Test Configuration Override
Tests automatically use `tests/config_test.py` via `conftest.py` which:
- Sets `DO_VISUALIZATION = False`
- Uses smaller test environments
- Adjusts paths for test fixtures

### Running Tests
```bash
pytest                              # All tests
pytest -v                           # Verbose
pytest --tb=short                   # Short traceback
pytest tests/test_ActionClassifier.py::test_specific  # Single test
```

---

## Dependencies

**Core:**
- `numpy >= 2.0.0`, `scipy >= 1.13.0`, `pandas`, `networkx`
- `pypowsybl >= 1.13.0`, `pypowsybl2grid >= 0.2.1`
- `expertop4grid >= 0.2.8` (contains alphaDeesp)
- `matplotlib >= 3.8.0`

**Test:**
- `pytest`

---

## Current Development Status

**Current version**: `0.2.9` (see `CHANGELOG.md` for full history)

> **v0.2.9 highlights** (deep revisions R5 + A5 + R6-partial from the 2026-07 review, plus a
> container-aware reassessment fix): discovery restructured around data — one typed
> `FamilyResult` per family in `self.results` via a declarative `FAMILY_SPECS` registry (with
> generated back-compat `@property` bridges), data-driven `action_scores` assembly +
> prioritization loops (`MIN_PHASE_ORDER` / `FILL_PHASE_ORDER`) that make the old latent
> `renewable_curtailment` double-add impossible, a memoised `_get_disconnection_bounds()`
> replacing the PST/disconnection `_disco_bounds` temporal coupling, a shared
> `InjectionDiscoveryBase`, and a `DiscovererProtocol` declaring the mixin surface. `ActionType`
> enum + declarative keyword classifier (byte-identical string values) with the **C7** grid2op-
> coupling rule-bypass fixed. **Container-aware reassessment parallelism**: CPU detection now
> reads cgroup quota + scheduler affinity (not `os.cpu_count()`), so a 2-vCPU container no longer
> over-subscribes; new `REASSESSMENT_PARALLEL` / `REASSESSMENT_MIN_PARALLEL_CORES` config knobs
> (env-overridable). Behaviour-preserving (mock discovery suite green; byte-identical
> `action_scores`). Skips `0.2.8` (published to PyPI outside this repo). See
> `docs/release-notes/v0.2.9.md`.
>
> **v0.2.7 highlights** (deep revisions R1 + R2 from the 2026-07 review): typed pipeline
> spine — `AnalysisContext` / `AnalysisResult` dataclasses replace the ~41-key context dict
> and the untyped result dict (dict-compatible via `DictCompatMixin`); a `SimulationBackend`
> protocol (`backends.py`) replaces the 18 delegation wrappers, the 8 context function
> pointers, the `is_pypowsybl` forks and the discoverer monkey-patching; `main.py` split into
> `cli.py` + `pipeline.py` with `_run_expert_discovery` moved under `models/`, dissolving the
> three import cycles. Behaviour-preserving (byte-identical prioritized-action output).
>
> **v0.2.6 highlights**: parallelised per-action reassessment (worker threads on private
> pypowsybl network copies, `min(10, cores, n_actions)`) + cheaper observation construction;
> maneuver-IHM path-traversal fix; `sys.exit(0)` → `LoadFlowDivergedError`; shared discovery
> baseline; overflow-graph edge cache keyed by `(u, v, key)`; CI migrated CircleCI → GitHub
> Actions. See `docs/release-notes/v0.2.6.md` and `docs/reviews/2026-07_full_code_review.md`.

### Active Migration: Grid2Op → Pure pypowsybl
See `docs/archive/MIGRATION_PLAN.md` for details. The goal is to remove `grid2op` dependency.

**Completed:**
- `pypowsybl_backend/` module with grid2op-compatible interfaces
- `environment_pypowsybl.py` for pure pypowsybl environment setup
- `--backend pypowsybl` CLI option
- `grid2op` is now fully optional (PR #26): importing the package no longer fails if `grid2op` is not installed

**Pending:**
- Full integration testing with all scenarios
- alphaDeesp adaptation for direct pypowsybl support
- Removal of grid2op from dependencies

### Recent Major Features (v0.1.4 – v0.1.9)

- **Load Shedding & Renewable Curtailment** (`v0.1.9`): `find_relevant_load_shedding` and `find_relevant_renewable_curtailment` in `action_evaluation/discovery.py` identify candidates on downstream nodes of constrained paths. Controlled by `MIN_LOAD_SHEDDING`, `MIN_RENEWABLE_CURTAILMENT`, `LOAD_SHEDDING_MARGIN`, `RENEWABLE_CURTAILMENT_MARGIN`, `RENEWABLE_ENERGY_SOURCES`. Deeply optimized for large networks (#76).
- **Pathlib migration** (`v0.1.9`): all base directories and file paths use `pathlib.Path` for cross-platform robustness.
- **Superposition Theorem** (`v0.1.8`): `utils/superposition.py` quantifies topological and PST action impacts using virtual flows and delta-theta. Integrated into analysis results. **Generalized Superposition Theorem (GST)** (unreleased): `compute_combined_pair_gst` + `is_injection_action` extend pair estimation to load shedding / curtailment / redispatch (injection changes), reported with `beta=1.0` so the existing reconstruction is unchanged. See `docs/recommender/superposition_module.md` §10.
- **Islanding MW impact** (`v0.1.8`): islanding detection now reports disconnected MW.
- **PST Support** (`v0.1.7+`): phase-shifter transformer tap variations, atomized PST actions from REPAS JSON. `find_relevant_pst_actions` in discovery. Grid2Op conversion support. PST asset-ID matching handles REPAS quirks (leading dots, `_inc1`/`_dec2` suffixes).
- **Direct XIIDM loading** (`v0.1.8_post1`): `main.py` accepts a direct `.xiidm` file path.
- **Pypowsybl format for rebuild** (`v0.1.6`): `--pypowsybl-format` option produces switch-based action JSONs with dedup.
- **NetworkTopologyCache optimization** (`v0.1.6`): eliminates O(all_elements) cost per action.
- **LazyActionDict** (`v0.1.5`): lazily computes action `content` (bus assignments) from switch states, dramatically reducing action JSON size.
- **Two-step `run_analysis` refactor** (`v0.1.5`): `run_analysis_step1` + `run_analysis_step2_graph` + `run_analysis_step2_discovery`, with `fast_mode` propagated to sub-components.
- **Direct-overload disconnection boost** (`v0.1.5`): +1.0 score boost for actions disconnecting currently overloaded lines in unconstrained regime.
- **Thermal limit monitoring factor** (`v0.1.5`): `MONITORING_FACTOR_THERMAL_LIMITS` rescales permanent limits.
- **Pre-existing overload filtering** (`v0.1.4`): excludes pre-existing overloads from N-1 analysis unless worsened by `PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD`.
- **Pypowsybl perf wave** (`v0.1.4`):
    - Incremental simulation branching from converged N-1 states (hot start).
    - `PYPOWSYBL_FAST_MODE` (default true) disables voltage control on shunts/transformers during variants.
    - Automatic fallback to "slow" mode on divergence.
    - Vectorized observation creation (>80% faster init).
    - Batched topological changes in fewer pypowsybl update calls.

### Historical Features (v0.1.2 – v0.1.3)

- **Action scores dictionary** (`v0.1.2`): `run_analysis()` now returns `action_scores` alongside `prioritized_actions`. Four scoring categories: `line_reconnection`, `line_disconnection`, `open_coupling`, `close_coupling`, each with `scores` (descending) and `params`.
- **Line disconnection scoring** (`v0.1.2`): asymmetric bell curve (alpha=3, beta=1.5). Separate `"constrained"` and `"unconstrained"` regimes depending on whether the overflow graph produces new overloads.
- **Node merging scoring** (`v0.1.2`): delta-phase score based on voltage angle difference between buses being merged.
- **Node splitting detail tuple** (`v0.1.2`): `compute_node_splitting_action_score_value` returns `(score, details)` with flow components and bus info.
- **Per-action assets** (`v0.1.2`): `open_coupling` and `close_coupling` params include `assets` dict (lines, loads, generators per action).
- **Configurable minimum action counts** (`v0.1.3`): `MIN_LINE_RECONNECTIONS`, `MIN_CLOSE_COUPLING`, `MIN_OPEN_COUPLING`, `MIN_LINE_DISCONNECTIONS` in `config.py` ensure at least N actions of each type are returned.
- **Lines monitoring from config** (`v0.1.3`): `LINES_MONITORING_FILE` and `IGNORE_LINES_MONITORING` flags; `--ignore-lines-monitoring` CLI flag. Unmonitored lines are excluded from disconnection flow bounds.
- **Max rho at both extremities** (`v0.1.3`, pypowsybl only): `MAX_RHO_BOTH_EXTREMITIES` flag evaluates loading from both line ends using potentially distinct thermal limits.
- **Swapped flows fix** (`v0.1.3+`): `_inhibit_swapped_flows` restored in `overflow_analysis.py` to correctly render swapped blue edges in overflow graphs.

### Key Files for Development
| Task | Primary Files |
|------|---------------|
| Add new action type | `action_evaluation/action_types.py` (add an `ActionType` + keyword rule), `classifier.py`, `rules.py`; add its `FamilySpec` to `discovery/_results.py` if it is a discovery family |
| Modify expert rules | `action_evaluation/rules.py` |
| Add a discovery family / per-family result | `action_evaluation/discovery/_results.py` (`FamilySpec` + phase-order tables) + the family mixin |
| Modify action scoring | `action_evaluation/discovery/` (per-family mixins); injection families share `_injection_base.py` |
| Change graph analysis | `graph_analysis/builder.py`, `processor.py` |
| Antenna (islanded-pocket) graph | `graph_analysis/antenna_graph.py`, `processor.py` (`extract_antenna_context`, `pre_process_antenna_graph`); see `docs/recommender/antenna_overflow_graph.md` |
| pypowsybl migration | `pypowsybl_backend/*`, `environment_pypowsybl.py` |
| Adjust rho calculation | `pypowsybl_backend/observation.py`, `overflow_analysis.py` |
| Configure monitoring | `config.py` (`LINES_MONITORING_FILE`, `IGNORE_LINES_MONITORING`) |
| Add load shedding logic | `action_evaluation/discovery/_load_shedding.py` |
| Add curtailment logic | `action_evaluation/discovery/_renewable_curtailment.py` |
| PST actions | `action_evaluation/discovery/_pst.py`, `utils/repas.py` |
| Superposition theorem | `utils/superposition.py` |
| Add new test | `tests/test_*.py`, update `conftest.py` if needed |

---

## Common Patterns

### Backend Abstraction Pattern (`backends.py`)
The old per-call `if backend == GRID2OP: fn = fn_grid2op else fn = fn_pypowsybl`
forks (and the function-pointers stashed in the context dict) are gone. The
pipeline builds one `SimulationBackend` and calls methods on it; `fast_mode` is
constructor state, so call sites never thread it through:
```python
from expert_op4grid_recommender.backends import make_backend, Backend

backend = make_backend(Backend.PYPOWSYBL, fast_mode=True)  # or Backend.GRID2OP
obs_simu_defaut, converged = backend.simulate_contingency(env, obs, lines, act_reco, t)
df, sim, g, hubs, dist, mapping = backend.build_overflow_graph(...)
# Discovery is configured from backend flags (branch_candidates_from_baseline,
# use_shared_baseline_for_topological) instead of monkey-patching the discoverer.
```

### Timer Context Manager
```python
from expert_op4grid_recommender.utils.helpers import Timer

with Timer("Operation Name"):
    # ... code to time
```

### Action Creation
```python
# Line disconnection
action = env.action_space({"set_line_status": [("LINE_NAME", -1)]})

# Topology change
action = env.action_space({
    "set_bus": {
        "lines_or_id": {"LINE_NAME": 1},
        "lines_ex_id": {"LINE_NAME": 2}
    }
})

# Combine actions
combined = action1 + action2
```

---

## Known Issues / Gotchas

1. **Thermal limits**: If `env.get_thermal_limit()` returns very high values (≥10⁴), the code auto-loads limits from `n_grid.get_operational_limits()`.

2. **Config override in tests**: Always import config as a module
   (`from expert_op4grid_recommender import config`; read `config.X`) rather than
   binding values at import time (`from ...config import X`), so the test override
   applies. `tests/conftest.py` swaps `tests/config_test.py` in for the package
   config via **both** `sys.modules` and the package attribute
   (`expert_op4grid_recommender.config = config_test`). `config_test.py`
   **star-imports** the real `config.py` (so `Settings` validation runs and no key
   goes missing) and overrides only the test deltas — do not re-fork it.

3. **Path handling**: Use `Path` objects and `PROJECT_ROOT` from config, not relative string paths.

4. **DC load flow fallback**: If AC simulation fails, the code automatically switches to DC. Results are marked as "more approximate".

5. **alphaDeesp dependency**: `expertop4grid >= 0.2.8` is required for `AlphaDeesp_warmStart`.

6. **pypowsybl2grid backend patch**: The `PyPowSyBlBackend.update_integer_value` method has an issue with zero-value handling. The original code `changed[value == 0] = False` must be replaced with `value[value == 0] = -1`. **This patch must be applied to the installed package file before running tests**:
   ```bash
   python scripts/patch_pypowsybl2grid_file.py
   ```
   In CI, this is done automatically in the `build-and-test` job of `.github/workflows/ci.yml`. Runtime monkey-patching doesn't work because modules are imported before conftest.py runs.

---

## Contact / License

**Author**: RTE (https://www.rte-france.com)  
**License**: MPL 2.0
