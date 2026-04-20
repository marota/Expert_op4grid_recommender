# `pypowsybl_backend/`

Pure [pypowsybl](https://www.powsybl.org/pages/documentation/developer/pypowsybl.html)
backend for **Expert Op4Grid Recommender**. It provides a grid2op-free
interface to power-system simulation, using pypowsybl's *network variants*
for efficient what-if analysis on N-1 contingencies and topological
remedial actions.

The package is designed as a drop-in replacement for the grid2op
environment used elsewhere in the codebase: the public classes expose
grid2op-compatible properties (`rho`, `line_status`, `topo_vect`,
`name_line`, `name_sub`, ...) and a `simulate(action)` entry point, so
downstream analysis code (`action_evaluation/`, `graph_analysis/`) runs
unchanged when `--backend pypowsybl` is selected.

---

## Quick Start

```python
from expert_op4grid_recommender.pypowsybl_backend import SimulationEnvironment

env = SimulationEnvironment(
    network_path="/path/to/network.xiidm",
    thermal_limits_path="/path/to/thermal_limits.json",
    threshold_thermal_limit=0.95,
)

obs = env.get_obs()

# Disconnect a line and re-run the load flow on a variant
action = env.action_space({"set_line_status": [("LINE_NAME", -1)]})
obs_simu, reward, done, info = obs.simulate(action)

print("Overloaded lines:", obs_simu.name_line[obs_simu.rho >= 1.0])
```

See `migration_guide.py` for a side-by-side comparison with the grid2op
API.

---

## File Layout

```
pypowsybl_backend/
├── __init__.py            # Public exports (see below)
├── simulation_env.py      # SimulationEnvironment: high-level entry point
├── network_manager.py     # NetworkManager: pypowsybl network + variants + LF
├── observation.py         # PypowsyblObservation / PypowsyblAction base class
├── action_space.py        # ActionSpace + concrete action classes
├── topology.py            # TopologyManager: topo_vect & bus assignments
├── overflow_analysis.py   # Overflow graph (alphaDeesp-compatible)
└── migration_guide.py     # Grid2Op → pypowsybl migration notes
```

---

## Module Reference

### `__init__.py` — public API

Re-exports the user-facing symbols:

| Symbol | Kind | Purpose |
|--------|------|---------|
| `SimulationEnvironment` | class | Main entry point (grid2op-like env) |
| `make_simulation_env` | function | Factory that looks up network/limits in a folder |
| `NetworkManager` | class | Low-level pypowsybl network wrapper |
| `PypowsyblObservation` | class | Read-only state + `simulate()` |
| `PypowsyblAction` | class | Base class for all actions |
| `ActionSpace` | class | Builds actions from grid2op-style dicts |
| `LineStatusAction`, `BusAction` | class | Concrete action types |
| `TopologyManager` | class | `topo_vect` / `sub_topology` management |
| `OverflowSimulator` | class | Flow-change computation after disconnect |
| `OverflowGraphBuilder` | class | Builds overflow `networkx` graph |
| `AlphaDeespAdapter` | class | Shim for legacy alphaDeesp consumers |
| `build_overflow_graph_pypowsybl` | function | Top-level overflow graph builder |

---

### `simulation_env.py` — `SimulationEnvironment`

High-level facade. Bundles a `NetworkManager`, an `ActionSpace`, and
thermal limits, and exposes the grid2op-style attributes used by the
rest of the recommender.

Key methods / properties:

- `__init__(network_path=None, network=None, thermal_limits_path=None,
  thermal_limits=None, lf_parameters=None, threshold_thermal_limit=1.0)`
  Loads the network, runs an initial load flow, applies a multiplier
  on thermal limits (e.g. `0.95` → 95% of permanent limit).
- `get_obs() -> PypowsyblObservation` — observation of the current
  working variant.
- `reset() -> PypowsyblObservation` — restore the base variant.
- `get_thermal_limit()` / `set_thermal_limit(...)` — array-based access
  aligned with `name_line`.
- `action_space(dict) -> PypowsyblAction` — callable action factory.
- Properties: `name_line`, `name_sub`, `name_gen`, `name_load`,
  `n_line`, `n_sub`, `backend` (grid2op-compat `env.backend._grid.network`).

Helpers:

- `make_simulation_env(env_folder, env_name, thermal_limits_file=None,
  threshold_thermal_limit=0.95)` — looks for `*.xiidm|*.iidm|*.xml`
  (optionally under `grid/`) and a `thermal_limits.json` / `limits.json`
  and returns a fully initialized `SimulationEnvironment`.
- `BackendWrapper`, `GridWrapper`, `ChronicsHandlerPlaceholder`,
  `RealDataPlaceholder` — compatibility shims for code that reads
  `env.backend._grid.network` or `env.chronics_handler`.

---

### `network_manager.py` — `NetworkManager`

Thin, cache-friendly wrapper around `pypowsybl.network.Network`.
Handles:

- **Loading**: from file path or a pre-built network.
- **Variants**: keeps a `base` variant and lets callers clone / switch /
  remove variants for N-1 or topology hypotheticals.
- **Load flow**: RTE-style default `Parameters` (AC, with fallback),
  `_run_ac_with_init_fallback`, a `fast` mode that disables
  shunt/transformer voltage control for speed.
- **Element caches**: lines, transformers, buses, elements-per-substation,
  PST info, thermal limits — populated once at init.

Representative methods:

| Method | Purpose |
|--------|---------|
| `run_load_flow(dc=None, fast=False)` | AC (with fallback) or DC solve |
| `get_line_flows(columns=...)` | `DataFrame` of active/current/MVA flows |
| `get_line_p1_array()` / `get_line_currents_array()` | vectorized flow arrays |
| `get_thermal_limits()` / `get_thermal_limits_arrays()` | limits from the network |
| `disconnect_line(line_id)` / `reconnect_line(...)` | single-line toggles |
| `disconnect_lines_batch(line_ids)` | batched topology update |
| `detect_non_reconnectable_lines()` | lines with open side switches |
| `get_pst_ids()`, `get_pst_tap_info(pst_id)`, `update_pst_tap_step(...)` | PST support |
| `create_variant(...)`, `set_working_variant(...)`, `remove_variant(...)`, `reset_to_base()` | variant lifecycle |

Name properties mirror grid2op: `name_line`, `name_sub`, `name_gen`,
`name_load`, `n_line`, `n_sub`, plus `get_line_or_subid()`,
`get_line_ex_subid()`, `get_sub_idx(name)`, `get_line_idx(name)`.

---

### `observation.py` — `PypowsyblObservation`, `PypowsyblAction`

`PypowsyblObservation` is the grid2op-compatible read-only view of the
network. `_refresh_state()` materializes caches once at construction
(flows, voltages, angles, impedances, per-element bus assignments) so
that repeated property reads are array lookups.

Exposes the full grid2op observation surface used by the analyzer:

- Flows & loading: `rho`, `a_or`, `a_ex`, `p_or`, `p_ex`.
- Voltages & angles: `v_or`, `v_ex`, `theta_or`, `theta_ex`.
- Line parameters: `line_r`, `line_x`, `line_cos_phi`.
- Status & topology: `line_status`, `line_or_bus`, `line_ex_bus`,
  `topo_vect`, `sub_topology(sub_id)`, `sub_info`,
  `line_or_to_subid`, `line_ex_to_subid`.
- Injections: `load_p`, `load_q`, `gen_p`, `gen_q`, `gen_type`,
  `gen_energy_source`, `gen_renewable`.
- Naming: `name_line`, `name_sub`, `name_gen`, `name_load`, `n_line`,
  `n_sub`.
- Connectivity: `n_components`, `main_component_load_mw` (for
  islanding MW quantification).
- Introspection: `get_obj_connect_to(sub_id)`,
  `topo_vect_element(pos)`, `get_time_stamp()`.

The key method is `simulate(action, ...)` which:
1. Clones the working variant,
2. Applies the action via `action.apply(network_manager)`,
3. Runs the load flow (with DC fallback on divergence),
4. Returns `(obs_simu, reward, done, info)` Grid2Op-style.

`ObservationWithTopologyOverride` is a lightweight sub-observation
returned by `obs + action` that reports the post-action `topo_vect` /
`sub_topology` without running a load flow (used when callers only need
the topological preview).

`PypowsyblAction` (bottom of file) is the abstract base class: defines
`apply(network_manager)` and `__add__` for composing actions with
`action1 + action2`.

---

### `action_space.py` — `ActionSpace` and action classes

`ActionSpace(network_manager)` is callable: it turns a grid2op-style
action dictionary into a concrete `PypowsyblAction`. Supported keys
(parsed in `__call__`): `set_line_status`, `set_bus`, `switch_states`,
`pst_tap`, `power_reduction`.

Concrete action classes:

| Class | Action |
|-------|--------|
| `LineStatusAction` | Open/close one or more lines (`[(name, ±1), ...]`) |
| `BusAction` | Bus reassignments for lines / loads / generators |
| `SwitchAction` | Directly set pypowsybl switch states |
| `PhaseShifterAction` | Change tap step on PSTs |
| `PowerReductionAction` | Load shedding / renewable curtailment |

All action classes inherit from `PypowsyblAction` (defined in
`observation.py`) and implement `apply(network_manager)`.

`ActionSpace.get_do_nothing_action()` returns a no-op
`PypowsyblAction`; `n_line` / `n_sub` proxy to the underlying network.

---

### `topology.py` — `TopologyManager`

Owns the grid2op-style *topology vector* abstraction on top of
pypowsybl's bus/switch model.

- `_build_element_mapping()` and `_build_topo_vect_structure()` are run
  at init to assign each line-end / load / generator a deterministic
  position in the `topo_vect`.
- `get_topo_vect()` returns the full vector (values in
  `{-1, 1, 2, ...}` — disconnected or bus number).
- `get_sub_topology(sub_id)` returns the slice for a single substation.
- `sub_info` / `topo_vect_to_sub` mirror grid2op attributes.
- `get_element_at_topo_pos(pos)` / `get_objects_in_substation(sub_id)`
  provide reverse lookups.
- `apply_topology_change(...)` mutates bus assignments to realize a
  topology change on the current variant.

---

### `overflow_analysis.py` — overflow graph & flow deltas

Replaces the alphaDeesp-based overflow graph construction for the pure
pypowsybl backend. Three levels of API:

- **`OverflowSimulator`** — core flow-sensitivity engine.
  - `compute_ptdf_for_line(line_id)` — per-line PTDF after disconnect.
  - `compute_flow_changes_after_disconnection(line_id)` — full Δflow
    vector using variant branching.
  - `compute_flow_changes_and_rho(...)` — Δflow + new `rho` values,
    used by the discoverer to rank disconnections.
  - `get_dataframe()` — tabular summary consumed by the graph builder.

- **`OverflowGraphBuilder`** — turns an `OverflowSimulator` result into
  a `networkx.MultiDiGraph` of the overflow pattern.
  - `build_graph() -> (graph, df)`
  - `get_topology()` exports a dict compatible with alphaDeesp.

- **`build_overflow_graph_pypowsybl(env, obs, overload_ids, ...)`** —
  top-level convenience wrapper called from `graph_analysis.builder`.
  Returns the same `(df, sim, graph, hubs, dist_graph, mapping)` tuple
  as the grid2op version, so `graph_analysis/` consumes both backends
  uniformly.

Internal helpers:

- `_inhibit_swapped_flows(df)` — restores correct rendering of blue
  edges where flow direction swaps (v0.1.3+ fix).
- `_find_hubs_simple(graph)` — fallback hub detection.
- `_ObsLineCut` — minimal observation stub used for AlphaDeesp-style
  signatures.

**`AlphaDeespAdapter`** exposes the pypowsybl overflow data through the
legacy alphaDeesp interface (`get_dataframe()`, `get_substation_elements()`,
`get_substation_to_node_mapping()`, `get_internal_to_external_mapping()`)
so AlphaDeesp-based node-splitting code can run on the pypowsybl
backend without modification.

---

### `migration_guide.py`

Not imported at runtime — it contains documented, copy-pasteable
before/after snippets showing how to port grid2op usage
(`grid2op.make`, `env.reset`, `obs.simulate`, action dicts) to
`SimulationEnvironment`. Read it first when migrating a script.

---

## How the Pieces Fit Together

```
                ┌────────────────────────────┐
   user code →  │   SimulationEnvironment    │   ← main.py (--backend pypowsybl)
                └────────────┬───────────────┘
                             │ owns
        ┌────────────────────┼────────────────────────┐
        ▼                    ▼                        ▼
 NetworkManager         ActionSpace          PypowsyblObservation
 (pypowsybl +          (grid2op-style         (grid2op-style
  variants + LF)        dict → Action)         read-only state)
        │                    │                        │
        │                    ▼                        ▼
        │            LineStatusAction,         simulate(action)
        │            BusAction, Switch/         → clones variant,
        │            PST / PowerReduction       applies action,
        │                                       runs LF, returns obs
        ▼
 TopologyManager   ──▶ topo_vect / sub_topology
        │
        ▼
 OverflowSimulator ──▶ OverflowGraphBuilder ──▶ build_overflow_graph_pypowsybl
                                                 (consumed by graph_analysis/)
```

The boundary with the rest of the codebase is intentionally narrow:
only `SimulationEnvironment` / `PypowsyblObservation` /
`build_overflow_graph_pypowsybl` are consumed from outside the package,
which keeps the grid2op ↔ pypowsybl swap at the `main.py` backend-dispatch
level (see the "Backend Abstraction Pattern" in the root `CLAUDE.md`).

---

## Notes & Gotchas

- **Variants, not deep copies**: `simulate()` clones a pypowsybl variant
  rather than cloning the whole network. This is cheap but means
  caller code must not hold onto the working variant across calls.
- **Fast mode**: `NetworkManager.run_load_flow(fast=True)` and the
  `PYPOWSYBL_FAST_MODE` config flag disable transformer/shunt voltage
  control. The env falls back to slow mode automatically on divergence.
- **Thermal limits**: when the XIIDM file has placeholder limits
  (≥ 10⁴), prefer passing an explicit `thermal_limits_path`. The
  `threshold_thermal_limit` multiplier (e.g. `0.95`) rescales permanent
  limits (`MONITORING_FACTOR_THERMAL_LIMITS`).
- **Both-extremity `rho`**: controlled by `MAX_RHO_BOTH_EXTREMITIES` in
  `config.py`; implemented in `observation._compute_rho`.
- **pypowsybl2grid patch**: only relevant for the grid2op backend — the
  pure-pypowsybl backend bypasses `pypowsybl2grid` entirely.
