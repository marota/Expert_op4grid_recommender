# CLAUDE.md - Expert Op4Grid Recommender

> **Purpose**: Quick context for AI assistants (Claude, etc.) to understand this codebase for code improvements, development, or testing tasks.

## Project Overview

**ExpertOp4Grid Recommender** is an expert system analyzer for power grid contingencies. It analyzes N-1 contingencies in Grid2Op/pypowsybl environments, builds overflow graphs, applies expert rules to filter potential actions, and identifies corrective measures to alleviate line overloads.

**License**: Mozilla Public License 2.0 (MPL 2.0)  
**Python**: ≥3.10  
**Domain**: Power systems / Grid operations / RTE (French TSO)

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

# Run tests
pytest
pytest -v tests/test_ActionClassifier.py  # specific test file
pytest -k "test_name"  # specific test by name
```

---

## Architecture Overview

```
expert_op4grid_recommender/
├── main.py                    # Entry point, CLI, run_analysis() orchestration
├── config.py                  # Main configuration (paths, parameters, flags)
├── config_*.py                # Alternative configs (RTE, basic, test)
├── environment.py             # Grid2Op environment setup
├── environment_pypowsybl.py   # Pure pypowsybl environment setup
├── data_loader.py             # Load action dictionaries from JSON
│
├── action_evaluation/         # Action analysis module
│   ├── classifier.py          # ActionClassifier: categorize action types
│   ├── rules.py               # ActionRuleValidator: apply expert rules
│   └── discovery.py           # ActionDiscoverer: find & prioritize actions
│
├── graph_analysis/            # Overflow graph module
│   ├── builder.py             # build_overflow_graph() using alphaDeesp
│   ├── processor.py           # Graph connectivity, path analysis
│   └── visualization.py       # Graph visualization with matplotlib
│
├── pypowsybl_backend/         # Pure pypowsybl implementation (no grid2op)
│   ├── simulation_env.py      # SimulationEnvironment: main interface
│   ├── network_manager.py     # Network loading, variants, load flow
│   ├── observation.py         # Grid2op-compatible observation
│   ├── action_space.py        # Action creation (topology, switching)
│   ├── topology.py            # Topology vector management
│   └── overflow_analysis.py   # Overflow graph without alphaDeesp
│
└── utils/                     # Utility modules
    ├── simulation.py          # Grid2Op simulation helpers
    ├── simulation_pypowsybl.py # pypowsybl simulation helpers
    ├── helpers.py             # Timer, sorting, test data saving
    ├── helpers_pypowsybl.py   # pypowsybl-specific helpers
    ├── action_rebuilder.py    # Rebuild actions from REPAS format
    ├── conversion_actions_repas.py  # REPAS action conversion
    ├── data_utils.py          # StateInfo and data structures
    └── make_*_env.py          # Environment factory functions
```

---

## Key Classes & Functions

### Main Entry Point
- **`run_analysis(analysis_date, current_timestep, current_lines_defaut, env_path, env_name, backend)`** in `main.py`
  - Orchestrates the full analysis pipeline
  - Returns `Dict[str, Any]` with keys:
    - `"lines_overloaded_names"`: `List[str]`
    - `"prioritized_actions"`: `{action_id: {action, description_unitaire, rho_before, rho_after, max_rho, max_rho_line, is_rho_reduction, observation}}`
    - `"action_scores"`: per-type scoring dict (see Data Structures below)

### Action Evaluation (`action_evaluation/`)
- **`ActionClassifier`**: Determines action type (line open/close, nodal split/merge, load disconnect)
  - `identify_action_type(action_desc, by_description=True) -> ActionType`
  - `_is_nodale_grid2op_action(act) -> (is_nodale, subs, is_splitting)`

- **`ActionRuleValidator`**: Filters actions based on expert rules
  - `categorize_actions(dict_action, ...) -> (filtered_out, unfiltered)`
  - `check_rules(action_type, localization, subs_topology) -> (do_filter, reason)`
  - `localize_line_action(lines)`, `localize_coupling_action(subs)`

- **`ActionDiscoverer`**: Discovers and scores candidate actions
  - `discover_and_prioritize(n_action_max) -> (Dict[action_id, action], action_scores)`
  - Returns a tuple; `action_scores` dict has per-type scores and params (see Data Structures)
  - Enforces minimum action counts from `config.MIN_LINE_RECONNECTIONS`, `MIN_CLOSE_COUPLING`, `MIN_OPEN_COUPLING`, `MIN_LINE_DISCONNECTIONS`
  - Uses AlphaDeesp for node splitting analysis

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

# Lines monitoring flags (v0.1.3+)
IGNORE_LINES_MONITORING = False     # bypass lines monitoring limits
LINES_MONITORING_FILE = None        # path to monitoring file (None = use env default)

# Loading rate flags (v0.1.3+, pypowsybl only)
MAX_RHO_BOTH_EXTREMITIES = False    # evaluate max rho from both line extremities

# Minimum prioritized actions per type (v0.1.3+)
MIN_LINE_RECONNECTIONS = 0
MIN_CLOSE_COUPLING = 0
MIN_OPEN_COUPLING = 0
MIN_LINE_DISCONNECTIONS = 0

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
└── test_switch_action_and_substation_extraction.py  # Switch/substation tests
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

**Current version**: `0.1.3`

### Active Migration: Grid2Op → Pure pypowsybl
See `MIGRATION_PLAN.md` for details. The goal is to remove `grid2op` dependency.

**Completed:**
- `pypowsybl_backend/` module with grid2op-compatible interfaces
- `environment_pypowsybl.py` for pure pypowsybl environment setup
- `--backend pypowsybl` CLI option
- `grid2op` is now fully optional (PR #26): importing the package no longer fails if `grid2op` is not installed

**Pending:**
- Full integration testing with all scenarios
- alphaDeesp adaptation for direct pypowsybl support
- Removal of grid2op from dependencies

### Recent Major Features (v0.1.2 – v0.1.3)

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
| Add new action type | `action_evaluation/classifier.py`, `rules.py` |
| Modify expert rules | `action_evaluation/rules.py` |
| Modify action scoring | `action_evaluation/discovery.py` |
| Change graph analysis | `graph_analysis/builder.py`, `processor.py` |
| pypowsybl migration | `pypowsybl_backend/*`, `environment_pypowsybl.py` |
| Adjust rho calculation | `pypowsybl_backend/observation.py`, `overflow_analysis.py` |
| Configure monitoring | `config.py` (`LINES_MONITORING_FILE`, `IGNORE_LINES_MONITORING`) |
| Add new test | `tests/test_*.py`, update `conftest.py` if needed |

---

## Common Patterns

### Backend Abstraction Pattern (main.py)
```python
if backend == Backend.GRID2OP:
    setup_environment = setup_environment_grid2op
    simulate_contingency = simulate_contingency_grid2op
    # ... other functions
elif backend == Backend.PYPOWSYBL:
    setup_environment = setup_environment_pypowsybl
    simulate_contingency = simulate_contingency_pypowsybl
    # ... other functions
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

2. **Config override in tests**: Always import config through `expert_op4grid_recommender.config` not directly, to ensure test overrides work.

3. **Path handling**: Use `Path` objects and `PROJECT_ROOT` from config, not relative string paths.

4. **DC load flow fallback**: If AC simulation fails, the code automatically switches to DC. Results are marked as "more approximate".

5. **alphaDeesp dependency**: `expertop4grid >= 0.2.8` is required for `AlphaDeesp_warmStart`.

6. **pypowsybl2grid backend patch**: The `PyPowSyBlBackend.update_integer_value` method has an issue with zero-value handling. The original code `changed[value == 0] = False` must be replaced with `value[value == 0] = -1`. **This patch must be applied to the installed package file before running tests**:
   ```bash
   python scripts/patch_pypowsybl2grid_file.py
   ```
   In CircleCI, this is done automatically in `.circleci/config.yml`. Runtime monkey-patching doesn't work because modules are imported before conftest.py runs.

---

## Contact / License

**Author**: RTE (https://www.rte-france.com)  
**License**: MPL 2.0
