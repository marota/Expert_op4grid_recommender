# Code Quality & Maintainability Analysis

**Repository**: `expert_op4grid_recommender`
**Version analyzed**: `0.1.9`
**Date**: 2026-04-12
**Scope**: Static diagnostic across the `expert_op4grid_recommender/` package, tests, config, and repository root.

This document captures a snapshot of code-quality and maintainability findings. Items marked **P0** are low-risk immediate cleanups, **P1** are structural improvements, **P2** are quality-of-life upgrades.

> **Status**: All P0 items have been completed — see [Cleanup log](#cleanup-log) at the bottom of this document. Findings below reflect the state **before** P0 cleanup.

---

## 1. Dead / duplicate code

| Finding | Location | Evidence |
|---|---|---|
| Stale fork of `observation.py` | `expert_op4grid_recommender/pypowsybl_backend/observation_timers.py` (1052 lines) | `diff` vs `observation.py` (1108 lines) shows ~589 differing lines. The "timers" copy is missing the `variant_id` parameter and the pre-computed thermal-limit arrays added later to `observation.py`. Repo-wide grep finds **zero importers**. |
| Pre-rewrite REPAS conversion module | `expert_op4grid_recommender/utils/conversion_actions_repas_original.py` (273 lines) | Superseded by `conversion_actions_repas.py` (1309 lines). Grep finds **zero importers**. Still contains a stub TODO on line 103. |
| Stray validation script at repo root | `verify_incremental_branching.py` | One-off end-to-end validation helper. Belongs under `scripts/` (which already exists). |

**Action (P0)**: Delete the two dead modules; move the validation script into `scripts/`.

---

## 2. God modules / complexity hotspots

| File | LOC | Structure | Concern |
|---|---|---|---|
| `action_evaluation/discovery.py` | **3001** | 1 class, **42+ methods** | `ActionDiscoverer` handles all seven action families in one class. `discover_and_prioritize`, `find_relevant_node_splitting`, `find_relevant_load_shedding`, `find_relevant_renewable_curtailment` are each ~200+ lines. |
| `utils/conversion_actions_repas.py` | 1309 | Flat module | REPAS parsing, Union-Find topology, and vectorized pandas transforms jumbled together. |
| `utils/superposition.py` | 1107 | 15 functions | Analytical flow calculations with no intermediate abstractions. |
| `pypowsybl_backend/observation.py` | 1108 | 1 class | Mixes constructor, vectorized init, `simulate`, thermal-limit handling. |
| `pypowsybl_backend/overflow_analysis.py` | 987 | 2 classes | `build_graph` ~100 lines; large parameter objects. |
| `main.py` | 1026 | Mixed | `run_analysis_step1` ~250 lines; backend-abstraction branching by `if/else`. |

`discovery.py` is the single biggest ongoing maintenance tax in the repo.

**Action (P1)**: Split `ActionDiscoverer` along action-type axes — one discoverer per family (`LineActionsDiscoverer`, `CouplingDiscoverer`, `PSTDiscoverer`, `LoadSheddingDiscoverer`, `RenewableCurtailmentDiscoverer`, `NodeSplittingDiscoverer`, `NodeMergingDiscoverer`) with a shared cache/context object via composition.

---

## 3. Testing gaps

- **23 test files** cover **~60 source modules**.
- Well covered: action evaluation (classifier, rules, discoverer, node splitting), PST, superposition (5 files), pypowsybl backend, LazyActionDict, min-action counts, islanding MW, config override, switch/substation extraction, visualization filtering.
- **No dedicated tests** for:
  - `graph_analysis/builder.py`
  - `graph_analysis/processor.py`
  - `graph_analysis/visualization.py`
  - `environment_pypowsybl.py`
  - `data_loader.py`
  - `utils/simulation.py`, `utils/simulation_pypowsybl.py`
  - `utils/repas.py`

Graph analysis being untested is notable given it sits on the critical path between overflow detection and action discovery.

**Action (P1)**: Add unit tests for `graph_analysis/*` and `environment_pypowsybl.py`; smoke tests for `utils/simulation*.py`.

---

## 4. Error-handling smells

### `expert_op4grid_recommender/__init__.py:33-38`

```python
except (ImportError, Exception) as e:
    try:
        import sys
        print(f"DEBUG: Failed to patch grid2op: {e}", file=sys.stderr)
    except:
        pass
```

- `except (ImportError, Exception)` is equivalent to `except Exception` — the `ImportError` is redundant.
- The nested bare `except: pass` silently swallows any error inside the error handler.
- Runtime monkey-patching `grid2op.Backend.Backend.get_shunt_setpoint` at package import time is a code smell; it also becomes dead weight once the `grid2op → pypowsybl` migration (see `MIGRATION_PLAN.md`) lands.

**Action (P1)**: Narrow the exception type, route through `logging`, and reassess whether the monkey-patch is still needed.

---

## 5. TODO / FIXME inventory

**14 markers across 5 files**:

| File | Count | Context |
|---|---|---|
| `utils/repas.py` | 6 | attribute handling |
| `utils/make_env_utils.py` | 3 | config consistency |
| `action_evaluation/discovery.py` | 3 | flow-analysis heuristics |
| `graph_analysis/processor.py` | 1 | topology simplification |
| `utils/conversion_actions_repas_original.py` | 1 | irrelevant — delete file |

**Action (P2)**: Promote to GitHub issues, reference issue numbers in the code.

---

## 6. Hardcoded paths / magic numbers

| Finding | Location |
|---|---|
| **Developer-specific absolute path committed to source** | `utils/load_training_data.py:341` — `"/home/donnotben/Documents/assistflux/read_history/20250228_livraison_LJN/time_series/20250527"` |
| Default thermal-limit sentinel `9999.0` | `main.py::set_thermal_limits` |
| ~7 lines of commented-out `datetime(2024, …)` scenario snippets | `config.py` header |
| Default action-space path literal | `main.py` argparse defaults |

**Action (P0)**: Remove the hardcoded `/home/donnotben/...` path (make it a function argument or config value). Move scenario snippets into a `docs/scenarios.md` or a dict.

---

## 7. Config sprawl

Three near-parallel module-level configs:

- `expert_op4grid_recommender/config.py` (122 lines) — active configuration with hardcoded `DATE`, `TIMESTEP`, `LINES_DEFAUT`, `ENV_NAME`.
- `expert_op4grid_recommender/config_basic.py` (90 lines) — alternative snapshot config.
- `tests/config_test.py` — test overrides applied via attribute assignment (`config.DO_VISUALIZATION = False`) from `conftest.py`.

**Duplicate definitions** inside `config.py` (silent-win behavior):

- `RENEWABLE_CURTAILMENT_MARGIN`
- `RENEWABLE_CURTAILMENT_MIN_MW`
- `RENEWABLE_ENERGY_SOURCES`
- `PYPOWSYBL_FAST_MODE`

Each appears twice in the file — whichever definition comes last wins, silently.

**Action (P0)**: Remove duplicate definitions.
**Action (P2)**: Migrate to `pydantic.BaseSettings` or `python-dotenv`; add schema validation and environment-variable support.

---

## 8. Root-level cleanliness

Files at the repository root that should live elsewhere:

| File | Recommended destination |
|---|---|
| `verify_incremental_branching.py` | `scripts/` |
| `SETUP_SUMMARY.md` | `docs/` |
| `SKIP_VISUALIZATION_GUIDE.md` | `docs/` |
| `pr_body_config.md` | `docs/` or delete if obsolete |
| `MIGRATION_PLAN.md` | Keep at root while migration is active |

**Action (P0)**: Reorganize into `docs/` and `scripts/`.

---

## 9. Import hygiene

- No wildcard imports detected.
- No obvious circular dependencies. Layering: `utils/` → `pypowsybl_backend/` → `action_evaluation/` → `main.py`.
- `main.py:22-26` manually inserts `project_root` into `sys.path`. Since `pyproject.toml` installs the package, this is unnecessary after `pip install -e .` and is a carry-over from script-style execution.

**Action (P2)**: Drop the manual `sys.path` insertion once all entry points use the installed package.

---

## 10. Type hints & docstrings

Rough coverage: **~40%** of functions have type hints.

- **Newer modules** (post-v0.1.5 `discovery.py` additions, `superposition.py`, `pypowsybl_backend/*`) have hints and docstrings.
- **Older utils** (`load_training_data.py`, `make_env_utils.py`, `repas.py`, `load_evaluation_data.py`, `make_assistant_env.py`) have minimal-to-zero docstrings and sparse hints.

**Action (P2)**: Back-fill type hints and docstrings on utils modules; prioritize public APIs consumed by `main.py` and tests.

---

## Priority-ordered action list

### P0 — low-risk, immediate wins ✅ done
1. ✅ Delete `expert_op4grid_recommender/pypowsybl_backend/observation_timers.py`.
2. ✅ Delete `expert_op4grid_recommender/utils/conversion_actions_repas_original.py`.
3. ✅ Remove the hardcoded developer path in `utils/load_training_data.py:341`.
4. ✅ Remove duplicate definitions in `config.py` (`RENEWABLE_*`, `PYPOWSYBL_FAST_MODE`).
5. ✅ Move root-level `.md` guides and `verify_incremental_branching.py` into `docs/` and `scripts/`.

### P1 — structural
6. Split `action_evaluation/discovery.py` into one module per action-type family, sharing common caches via composition.
7. Add tests for `graph_analysis/*` and `environment_pypowsybl.py`.
8. Replace the bare `except` nest in `expert_op4grid_recommender/__init__.py`; narrow exception types and route through `logging`.

### P2 — quality-of-life
9. Migrate `config.py` / `config_basic.py` to `pydantic.BaseSettings` with env-var support and validation.
10. Back-fill type hints and docstrings on `utils/` modules.
11. Convert TODO markers to GitHub issues and reference issue numbers from the code.
12. Drop the manual `sys.path` insertion in `main.py`.

---

## Appendix: module line-count inventory

| Module | LOC |
|---|---|
| `action_evaluation/discovery.py` | 3001 |
| `utils/conversion_actions_repas.py` | 1309 |
| `pypowsybl_backend/observation.py` | 1108 |
| `utils/superposition.py` | 1107 |
| ~~`pypowsybl_backend/observation_timers.py`~~ | ~~1052~~ *(deleted in P0 cleanup)* |
| `main.py` | 1026 |
| `pypowsybl_backend/overflow_analysis.py` | 987 |
| `pypowsybl_backend/network_manager.py` | 690 |
| `utils/action_rebuilder.py` | 514 |
| `action_evaluation/rules.py` | 424 |
| `utils/helpers_pypowsybl.py` | 415 |
| `pypowsybl_backend/action_space.py` | 398 |
| `utils/load_training_data.py` | 389 |
| `action_evaluation/classifier.py` | 367 |
| `environment_pypowsybl.py` | 338 |
| `pypowsybl_backend/simulation_env.py` | 339 |
| `pypowsybl_backend/topology.py` | 333 |
| `utils/helpers.py` | 332 |
| `utils/simulation.py` | 308 |
| `graph_analysis/processor.py` | 303 |
| `utils/simulation_pypowsybl.py` | 291 |
| ~~`utils/conversion_actions_repas_original.py`~~ | ~~273~~ *(deleted in P0 cleanup)* |
| `data_loader.py` | 245 |
| `environment.py` | 236 |
| `graph_analysis/visualization.py` | 222 |
| `utils/load_evaluation_data.py` | 198 |
| `utils/make_env_utils.py` | 142 |
| `config.py` | 122 |
| `graph_analysis/builder.py` | 111 |
| `data_utils.py` | 104 |
| `config_basic.py` | 90 |
| `utils/make_assistant_env.py` | 85 |
| `utils/make_training_env.py` | 68 |
| **Package total (pre-P0)** | **17439** |
| **Package total (post-P0)** | **16114** *(-1325 LOC: dead-code removal)* |

---

## Cleanup log

### P0 cleanup — completed

All five P0 items have been applied to the codebase on branch `claude/code-quality-analysis-9x1tq`.

#### 1. Deleted dead modules
- **`expert_op4grid_recommender/pypowsybl_backend/observation_timers.py`** (1052 LOC).
  Stale fork of `observation.py`; repo-wide grep confirmed zero importers.
- **`expert_op4grid_recommender/utils/conversion_actions_repas_original.py`** (273 LOC).
  Pre-rewrite REPAS converter superseded by `conversion_actions_repas.py`; zero importers.

**Net impact**: -1325 LOC from the package.

#### 2. Removed hardcoded developer path
- **`expert_op4grid_recommender/utils/load_training_data.py`**: the `__main__` demo block
  previously used a literal path `/home/donnotben/Documents/assistflux/…`. Replaced with
  a read from the `EXPERT_OP4GRID_TRAINING_OBS_DIR` environment variable, raising a
  clear `RuntimeError` if unset. `os` was already imported, so no new imports were added.

  ```python
  training_obs_dir = os.environ.get("EXPERT_OP4GRID_TRAINING_OBS_DIR")
  if not training_obs_dir:
      raise RuntimeError(
          "EXPERT_OP4GRID_TRAINING_OBS_DIR is not set. "
          "Point it at a directory of gzipped training observations before running this script."
      )
  all_obs_files = list_all_obs_files(training_obs_dir, sort_results=True, ext=".gz")
  ```

#### 3. Removed duplicate definitions in `config.py`
The following keys were each defined twice in `expert_op4grid_recommender/config.py`
(silent last-wins behavior). Removed the second definition in every case:

- `RENEWABLE_CURTAILMENT_MARGIN`
- `RENEWABLE_CURTAILMENT_MIN_MW`
- `RENEWABLE_ENERGY_SOURCES`
- `PYPOWSYBL_FAST_MODE`

Verified post-cleanup with `grep`: each key now appears exactly once.

#### 4. Root-level file reorganization
Moved the following files out of the repository root using `git mv` (preserving history):

| From (root) | To |
|---|---|
| `verify_incremental_branching.py` | `scripts/verify_incremental_branching.py` |
| `SETUP_SUMMARY.md` | `docs/SETUP_SUMMARY.md` |
| `SKIP_VISUALIZATION_GUIDE.md` | `docs/SKIP_VISUALIZATION_GUIDE.md` |
| `pr_body_config.md` | `docs/pr_body_config.md` |

`MIGRATION_PLAN.md` was left at the repo root — it is still active while the
grid2op → pypowsybl migration is ongoing.

The moved script uses absolute package imports (`from expert_op4grid_recommender.pypowsybl_backend import …`), so it continues to work from its new location under an editable install (`pip install -e .`).

### Verification
- `python -c "import expert_op4grid_recommender.config as c"` imports cleanly and
  reports the expected values for `PYPOWSYBL_FAST_MODE`, `RENEWABLE_CURTAILMENT_MARGIN`,
  and `ENV_NAME`.
- Repo-wide grep confirms no remaining references to the deleted modules.

### Remaining work
P1 and P2 items (split `discovery.py`, add graph-analysis tests, fix the bare-`except`
nest in `__init__.py`, migrate configs to `pydantic.BaseSettings`, back-fill type hints)
have not been touched yet — those are follow-up tasks.
