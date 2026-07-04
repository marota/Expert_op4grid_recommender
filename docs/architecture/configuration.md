# Configuration — the pydantic `Settings` single source of truth

Since **R3** (v0.2.8) the runtime configuration has one authoritative owner: the
pydantic `Settings` instance in `expert_op4grid_recommender/config.py`. This note
describes how it is read, overridden and derived, and how tests override it.

---

## 1. Object model

```
config.py
 ├── Settings(BaseSettings)          ← the authoritative, validated config
 │    ├── primary fields             ← DATE, TIMESTEP, LINES_DEFAUT, ENV_NAME, …
 │    │                                 (env-var overridable: EXPERT_OP4GRID_*)
 │    └── @computed_field derived    ← CASE_NAME, ENV_FOLDER, ENV_PATH,
 │                                      ACTION_SPACE_FOLDER, ACTION_FILE_PATH,
 │                                      SAVE_FOLDER_VISUALIZATION
 ├── settings: Settings              ← the process-wide instance
 └── module attributes              ← every field (incl. derived) promoted to
                                       config.X for the many `config.X` readers
```

- **Primary fields** carry validation (`ge=0` on counts, `Literal` on
  `VISUALIZATION_FORMAT`, a `LINES_DEFAUT` validator that accepts a bare name or
  a JSON list). Each is overridable at process start via an
  `EXPERT_OP4GRID_<NAME>` environment variable.
- **Derived paths are `@computed_field` properties** — they recompute from the
  primary fields, so they can never go stale. `ENV_PATH = ENV_FOLDER / ENV_NAME`,
  `ACTION_FILE_PATH = ACTION_SPACE_FOLDER / FILE_ACTION_SPACE_DESC`, etc.
- `model_dump()` includes computed fields, so `apply_settings_to_namespace`
  promotes **all** of them to module attributes. Legacy code that reads
  `config.ENV_PATH` (or `from …config import DATE`) keeps working unchanged.

Before R3 the derived paths were recomputed by hand *after* the instance was
built, so overriding `ENV_NAME` left a stale `ENV_PATH` behind — review finding
**A3**. Expressing them as computed fields fixes that at the source.

---

## 2. Reading and changing config

| Need | Do this |
|---|---|
| Read a value | `config.X` (module attribute) or `config.get_settings().X` |
| Change values (validated, staleness-free) | `config.override_settings(**overrides)` |
| Reset to defaults + env | `config.reset_settings()` |
| Legacy escape hatch | `config.X = y` (still works; skips validation + derivation) |

`override_settings(**overrides)`:

1. Rejects unknown keys (`ValueError`).
2. Rebuilds a `Settings` from the current field values + the overrides, so full
   pydantic validation runs.
3. Recomputes the derived paths (overriding `ENV_NAME` recomputes `ENV_PATH` /
   `ACTION_FILE_PATH`).
4. Re-promotes every field to the module namespace, so subsequent `config.X`
   reads observe the change.

It also accepts a ready `Settings` instance positionally
(`override_settings(some_settings)`), which the test rollback pattern uses.

```python
from expert_op4grid_recommender import config

before = config.get_settings()
config.override_settings(ENV_NAME="pypsa_eur_fr400")
assert config.ENV_PATH.name == "pypsa_eur_fr400"   # recomputed, no staleness
config.override_settings(before)                    # roll back
```

The pipeline routes its `ENV_NAME` override through this accessor instead of
mutating module attributes directly.

### Deliberately *not* locked down

`config` stays a plain, mutable module: `config.X = y` writes and *arbitrary*
extra attributes still work. This is load-bearing for the Co-Study4Grid backend,
which drives the recommender by mutating `config.ENV_PATH` / `config.ENV_FOLDER`
and its own keys (`LAYOUT_FILE_PATH`, `MONITORED_LINES_COUNT`, …) directly.
"Authoritative" means *a validated, staleness-free accessor path* — not a frozen
module.

---

## 3. Test override (no more fork)

Before R3 the test suite swapped a hand-forked `tests/config_test.py` in for the
real module via `sys.modules`. That fork bypassed pydantic (validation never ran
in CI) and every new key had to be added twice — review findings **M2 / C7**.

Now `tests/conftest.py` keeps the real module in place and applies only the test
deltas through the accessor, at import time (before any test module is
collected):

```python
from expert_op4grid_recommender import config

TEST_CONFIG_DELTAS = dict(
    ENV_NAME="env_dijon_v2_assistant",
    FILE_ACTION_SPACE_DESC="reduced_model_actions.json",
    CHECK_ACTION_SIMULATION=True,
    N_PRIORITIZED_ACTIONS=5,
    DO_VISUALIZATION=False,
    # … only the values that differ from production
)
config.override_settings(**TEST_CONFIG_DELTAS)
```

Everything the deltas don't mention inherits the real defaults, so this list can
never drift out of sync with `config.py`, and pydantic validation now runs in CI.
See `tests/test_config_override.py` for the mechanism + computed-field + accessor
tests.
