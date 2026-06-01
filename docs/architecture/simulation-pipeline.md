# Simulation pipeline — pypowsybl backend

This document explains how `expert_op4grid_recommender` actually
**simulates** a contingency, an action, or a (contingency + action)
combination on a pypowsybl-backed network. It covers the mechanism, the
parameters, the hypotheses, and every mode the pipeline exposes.

The library has two interchangeable backends — **grid2op** (legacy) and
**pypowsybl** (current default). This document focuses on the pypowsybl
backend because it is the one used in production (Co-Study4Grid + every
recent CLI / pytest workflow). The grid2op path is preserved for parity
tests; its simulation semantics are externally defined and not repeated
here.

## 1. Object model (one-paragraph summary)

```
SimulationEnvironment           ← top-level container, grid2op-shaped
 ├── NetworkManager             ← wraps pypowsybl.network.Network
 │    ├── lf_parameters         ← pypowsybl.loadflow.Parameters
 │    ├── base_variant          ← clean snapshot of the network
 │    └── variants (named)      ← cheap copies for what-if branches
 ├── ActionSpace                ← turns dict payloads into PypowsyblAction
 ├── thermal_limits             ← {line_id → A_max} after threshold mult.
 └── current_obs                ← PypowsyblObservation lazily built
                                    from network state
```

`PypowsyblObservation.simulate(action)` is the main entry point. It is
shaped to be a drop-in for grid2op's `obs.simulate(action)`. Under the
hood:

1. clone the current variant → a fresh named variant ;
2. apply the action ;
3. run a load flow ;
4. wrap the resulting network state in a new `PypowsyblObservation` ;
5. either keep or destroy the variant.

Steps 3–5 are described in detail below.

## 2. Variants — the unit of "what if"

pypowsybl networks are stateful (current values on every bus / branch /
generator are mutable). To avoid corrupting the base state, every
hypothetical scenario lives in its own **variant**:

- `NetworkManager.BASE_VARIANT = "base"` — the clean, post-load,
  post-initial-LF state. Reset target.
- `nm.create_variant(name, from_variant=...)` — `O(1)` clone, copies
  on write.
- `nm.set_working_variant(name)` — switches the "active" variant. All
  subsequent reads and writes target it.
- `nm.remove_variant(name)` — drops the storage. Cannot remove `base`.

Lifecycle rules enforced by the codebase:

| Operation | Variant rule |
|---|---|
| `SimulationEnvironment.__init__` | LF on base variant; result cached. |
| `env.reset()` | restore base + re-run initial LF with `DC_VALUES`. |
| `obs.simulate(action)` | clone *current* obs variant → temp variant → apply + LF → discard (or keep if `keep_variant=True`). |
| `simulate_contingency_pypowsybl(...)` | clone from base → disconnect contingency lines → LF → return obs (variant kept on `obs._variant_id` so callers can branch off). |
| Manual cleanup | every keep-variant caller is responsible for `nm.remove_variant(obs._variant_id)`. |

`obs.simulate` always restores the working variant to `base` in its
`finally` block — so a caller can chain simulations freely without
having to remember to reset.

## 3. Load flow — modes, parameters, default values

### 3.1 The four orthogonal knobs

| Knob | Values | Meaning |
|---|---|---|
| **AC vs DC** | `dc=True` / `dc=False` | full non-linear Newton-Raphson (`run_ac`) vs linear DC approximation (`run_dc`). DC has no reactive power and assumes \|V\|≈1 p.u.. |
| **Fast vs Slow** | `fast=True` / `fast=False` | when AC, *fast* disables transformer voltage control and shunt compensator voltage control (removes two outer loops). Slow = full physics. |
| **Voltage init** | `PREVIOUS_VALUES` / `DC_VALUES` / `UNIFORM_VALUES` | seed for Newton-Raphson. `PREVIOUS_VALUES` warm-starts from the last converged state; `DC_VALUES` runs a DC LF first and uses its angles as seed. |
| **Provider parameters** | OpenLoadFlow-specific | iteration caps, solver scaling mode, slack-bus selector, etc. |

These are independent — every `run_load_flow(dc=, fast=, voltage_init_mode=)`
call selects one point in the four-dimensional space.

### 3.2 Default `lf.Parameters` (RTE-style)

Built in `NetworkManager._create_default_lf_parameters`:

```python
lf.Parameters(
    read_slack_bus=False,
    write_slack_bus=False,
    voltage_init_mode=lf.VoltageInitMode.PREVIOUS_VALUES,
    transformer_voltage_control_on=True,
    use_reactive_limits=True,
    shunt_compensator_voltage_control_on=True,
    phase_shifter_regulation_on=True,
    distributed_slack=True,
    dc_use_transformer_ratio=False,
    twt_split_shunt_admittance=True,
    provider_parameters={
        "useActiveLimits": "true",
        "svcVoltageMonitoring": "false",
        "voltageRemoteControl": "false",
        "writeReferenceTerminals": "false",
        "slackBusSelectionMode": "MOST_MESHED",
        "maxOuterLoopIterations": "100",   # raised from stock 20 — see § 3.6
    },
)
```

Hypotheses baked in:

- **Slack distribution**: `distributed_slack=True` with default
  `BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX`. The slack mismatch is
  spread across generators proportionally to their `pmax`, not absorbed
  by a single slack bus.
- **Slack selection**: `slackBusSelectionMode=MOST_MESHED`. The slack
  bus is picked dynamically as the most-connected bus in the largest
  synchronous component, no need for `read_slack_bus`.
- **Reactive limits enforced**: `use_reactive_limits=True` — generators
  hit their `min_q` / `max_q` envelope, with PV → PQ switching.
- **Transformer / shunt / PST regulation ON**: outer loops adjust taps
  and compensator sections to match `target_v` setpoints.
- **Active limits ON**: `useActiveLimits=true` — generators clipped to
  `[pmin, pmax]` before slack distribution.
- **Active SVC voltage monitoring OFF**: avoids classifying SVCs as
  controlled buses when the data isn't reliable.
- **Voltage remote control OFF**: a generator regulating a remote bus
  only acts locally.
- **Two-winding transformer**: shunt admittance split between the two
  sides (`twt_split_shunt_admittance=True`).
- **DC transformer ratio**: `dc_use_transformer_ratio=False` — in DC LF
  the tap ratio is ignored (canonical DC approximation).

### 3.3 Fast vs slow — what gets disabled

`run_load_flow(fast=True)` shallow-copies the params and toggles two
booleans before calling `run_ac`:

```python
params.transformer_voltage_control_on = False
params.shunt_compensator_voltage_control_on = False
```

These two outer loops are the heaviest non-linear parts of the OpenLoadFlow
solver:

- **`IncrementalTransformerVoltageControl`**: re-positions tap changers
  to satisfy `target_v` on the controlled side.
- **`IncrementalShuntVoltageControl`**: switches reactive shunt sections
  in / out to match controlled-bus setpoints.

Disabling them gives a 1.5×–3× speed-up on large grids but produces
**physically less accurate** voltage and reactive power solutions. The
network is still a valid AC solution under the assumption that tap
positions and shunt sections are **fixed at their input values**.

### 3.4 Voltage init modes — when each is right

- `PREVIOUS_VALUES` (default for **warm starts**): use \|V\| / θ from the
  current variant state as Newton-Raphson seed. Converges in 2–4 iter
  on small perturbations. **Fragile** when the previous state is far
  from the next solution (e.g. after a topology change that moves a
  bus angle by 8°).
- `DC_VALUES` (default for **cold starts**, used in `_ensure_valid_state`
  and as fallback): run a fast DC LF first, use its bus angles as θ
  seed and \|V\|=1 p.u. as voltage seed. **Robust** to topology
  perturbations and to networks with no prior voltage state — at the
  cost of 1 extra DC LF (~50–100 ms on the French grid).
- `UNIFORM_VALUES`: \|V\|=1 p.u., θ=0. Worst seed, only kept for
  pathological cases.

### 3.5 The dual fallback in `_run_ac_with_init_fallback`

If the initial seed is `PREVIOUS_VALUES`, the helper retries with
`DC_VALUES` on **two distinct failure modes**:

1. **Synchronous `PowsyblException`** — e.g.
   `"Voltage magnitude is undefined for bus '.A.ZA 6_0'"` when no
   previous values exist on that bus (commit `22e8a39e`, v0.2.0).
2. **Non-converged returned status** — `ComponentResult.status` is
   `FAILED` ("Unrealistic state" reached by OpenLoadFlow's voltage-control
   consistency check), `MAX_ITERATION_REACHED`, `SOLVER_FAILED`, etc.
   The Java side does **not** raise in this case — it returns a result
   object with a bad status (commit added in v0.2.2.post2).

Both fall back to a one-shot retry with `DC_VALUES`. If the retry also
fails, the bad result is propagated up.

When the initial seed is **already** `DC_VALUES`, the fallback does
nothing (there's no further seed to try) and the status / exception
propagates as-is. This prevents infinite fallback loops.

### 3.6 Outer-loop cap and why we raised it

Default OpenLoadFlow caps:

| Parameter | Stock default | Our default |
|---|---|---|
| `maxNewtonRaphsonIterations` | 15 | 15 (unchanged) |
| `maxOuterLoopIterations` | 20 | **100** (since v0.2.2.post2) |

The outer loops (transformer VC, shunt VC, phase-shifter regulation,
incremental shunt control, area interchange control, reactive limits)
iterate **on top of** the Newton-Raphson inner loop. On the French
400 kV grid after a `node_merging` action, the
`IncrementalTransformerVoltageControl` outer loop needs ~40–50
iterations to settle (because two tap-changing 220/63 kV transformers
in parallel must rebalance their Q after the coupler closes). The stock
20-iteration cap was triggering `MAX_ITERATION_REACHED` even after a
correct DC_VALUES seed. 100 leaves a comfortable margin with no
measurable cost on the normal warm-start path (which converges in
single-digit outer iterations).

### 3.7 Sequence inside `run_load_flow`

```text
run_load_flow(dc=False, fast=True, voltage_init_mode=None)
│
├── if dc → lf.run_dc(network) and return (single linear LF, no retry).
│
└── AC path
    ├── if fast: copy params; transformer_voltage_control_on=False;
    │            shunt_compensator_voltage_control_on=False
    │
    ├── attempt #1: _run_ac_with_init_fallback(params)
    │    ├── try lf.run_ac(network, parameters=params)
    │    │    └── if Exception and init=PREVIOUS_VALUES → retry DC_VALUES
    │    └── on success: if status != CONVERGED and init=PREVIOUS_VALUES
    │                    → retry DC_VALUES
    │
    └── if attempt #1 still not CONVERGED → retry slow mode with the
         original (unmuted) lf_parameters, same fallback logic
```

So the **full retry tree from a fast / PREVIOUS_VALUES call** has up to
**four pypowsybl calls** in the worst case:

```text
1. fast PREVIOUS_VALUES        ← initial
2. fast DC_VALUES              ← if exception OR bad status
3. slow PREVIOUS_VALUES        ← if (1)+(2) still bad
4. slow DC_VALUES              ← if exception/bad status on (3)
```

In practice the warm-start case stops at (1) in a single AC call.

## 4. `obs.simulate(action)` — the canonical action simulation

Signature:

```python
obs.simulate(
    action,           # PypowsyblAction or combined action
    time_step=0,      # accepted for grid2op compat; ignored
    keep_variant=False,
    fast_mode=True,   # ← drives the `fast=` knob of run_load_flow
) -> (new_obs, reward, done, info)
```

Algorithm (`pypowsybl_backend/observation.py:simulate`):

1. **Pick variant name**: deterministic when `keep_variant=False`
   (so concurrent simulations can't collide), monotonic counter when
   `keep_variant=True` (so the caller's reference doesn't get reused).
2. **Clone** the current observation's variant
   (`from_variant=self._variant_id`, falling back to base when `None`).
3. **Switch** the working variant to the new variant.
4. **Apply** the action via `action.apply(nm)` — this calls into
   pypowsybl topology mutators (`set_bus`, `connect`, `disconnect`,
   `set_gen_p`, `set_load_p`, tap-changer position, …).
5. **Run** `nm.run_load_flow(fast=fast_mode)` — defaults to `fast=True`.
6. **Status check** — if status != CONVERGED, push the failure into
   `info["exception"]` and set `done=True`. The returned observation
   is still built (with NaN where the LF didn't write a value), so the
   caller can inspect partial state.
7. **Build a new observation** — vectorised refresh of branch / bus /
   gen / load arrays from the post-LF network state.
8. **Restore** the working variant to base in `finally` — and remove
   the temp variant unless `keep_variant=True`.

Return tuple:

- `new_obs` — fresh `PypowsyblObservation` (or NaN-filled when LF
  failed).
- `reward` — always `0.0`. Reserved for grid2op API compat.
- `done` — `True` only if the LF didn't converge or an exception was
  thrown during apply / LF.
- `info` — `{"exception": [<list of exceptions / strings>]}`. Empty
  list on success.

### 4.1 `keep_variant` semantics

Default is `keep_variant=False`. The temp variant is removed in the
`finally` block.

`keep_variant=True` leaves the variant alive and stamps its id on the
returned observation as `obs._variant_id`. Used by:

- `simulate_contingency_pypowsybl` — the N-1 baseline used by the
  analysis pipeline (step 1 stores it on
  `context["obs_simu_defaut"]`).
- `discoverer` candidate evaluation, where each candidate action gets
  its own kept variant during the optimisation loop.

Caller is responsible for cleanup — `nm.remove_variant(obs._variant_id)`
when done. Leaks are detectable via `nm.network.get_variant_ids()`.

## 5. Contingency simulation (`simulate_contingency_pypowsybl`)

Wraps `obs.simulate` with the contingency model:

1. Build an action that disconnects every line in `lines_defaut`
   (the contingency) plus the maintenance reconnection action
   `act_reco_maintenance` if provided.
2. Run `obs.simulate(combined, keep_variant=True, fast_mode=fast_mode)`
   from the **N-state** observation.
3. Return the resulting **N-1 (or N-K)** observation.

The `act_reco_maintenance` action is built by
`get_maintenance_timestep_pypowsybl`. It re-connects any line that was
intentionally disconnected at the snapshot time (planned outage) and
that should be **available** for the recommender to suggest as a
reconnection action. Passed as part of the same combined action so a
single LF computes both effects.

`fast_mode` defaults to `True` for performance; the
`PYPOWSYBL_FAST_MODE` global config is consulted in
`expert_op4grid_recommender.main.run_analysis` to set this default.

## 6. Action simulation in the analysis pipeline

The analysis flow uses three layered entry points:

| Function | Role |
|---|---|
| `simulate_contingency_pypowsybl(env, obs, lines_defaut, ...)` | Build the N-1 observation that is the **baseline** for every candidate action. Called once per (timestep, contingency). |
| `_pypowsybl_compute_baseline(...)` | Re-uses the N-1 baseline; injects an action's `obs` into the recommender's scoring math. |
| `_pypowsybl_check_with_baseline(...)` / `_check_rho_reduction` | Simulate an action against the N-1 baseline, return `max_rho`, `rho_after`, impacted-line set. Used both by the discoverer (during candidate filtering) and by `reassess_prioritized_actions` (after the model recommends). |

All three call `obs.simulate(action, fast_mode=...)` and pass the
`actual_fast_mode` resolved at the top of `run_analysis`:

```python
if fast_mode:                             # explicit CLI/API override
    actual_fast_mode = True
else:
    actual_fast_mode = (
        config.PYPOWSYBL_FAST_MODE if fast_mode is None else fast_mode
    )
```

So a caller can force fast mode for an entire analysis run, or rely
on the global default (`config.PYPOWSYBL_FAST_MODE`, default `False`).

## 7. Thermal limits & overload calculation

### 7.1 Sources

`SimulationEnvironment._load_thermal_limits` resolves thermal limits in
this order:

1. Caller-supplied `thermal_limits` dict.
2. Caller-supplied `thermal_limits_path` JSON.
3. `NetworkManager.get_thermal_limits()` — extracted from the
   pypowsybl Network's `permanent_limit` on each branch's `apparent`
   or `current` limits.

The dict is then **multiplied by the threshold** `threshold_thermal_limit`
(default 1.0; the analysis pipeline passes
`config.MONITORING_FACTOR_THERMAL_LIMITS`, default `0.95`).

### 7.2 Overload detection (rho)

`PypowsyblObservation.rho` = `I / I_max` per branch, where
`I = max(\|I_or\|, \|I_ex\|)` and `I_max = thermal_limits[branch_id]`.

A branch is considered **overloaded** when `rho > 1` (after monitoring
factor).

### 7.3 Pre-existing overload filter

`config.PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD` (default `0.02`,
i.e. 2 %) governs when a remedial action's impact on a pre-existing
overload is counted as "worsening". An action that lifts
`rho_civauy711` from 4.34 to 4.345 is **not** flagged as worsening.

## 8. Hypotheses that the pipeline does NOT model

- **Dynamic / transient stability**: every LF is **steady-state**. No
  rotor angle dynamics, no fault clearing time, no protection
  responses.
- **Frequency response**: distributed slack absorbs imbalances
  instantaneously and proportionally to `pmax`. No primary frequency
  control timeline.
- **HVDC dynamics**: `acDcNetwork`-style emulation is supported (see
  `provider_parameters`) but every HVDC link is at steady-state
  setpoint within a single LF.
- **Probabilistic outages**: contingencies are deterministic — the
  caller specifies the set of lines to open.
- **Time evolution**: a single LF is one timestep. Multi-period
  analysis is achieved by re-loading separate snapshots.
- **Operational constraints beyond limits**: bus voltages outside
  nominal ranges, reactive reserves below regulatory floors, etc.,
  are not enforced. The pipeline only flags **branch rho > 1**.

## 9. Available "modes" — cheat-sheet

| Mode | Knobs | When | Trade-off |
|---|---|---|---|
| **Default AC fast** | `dc=False, fast=True, init=PREVIOUS_VALUES` | every `obs.simulate(...)` and every analysis call by default | quickest AC; ignores tap / shunt re-regulation. |
| **AC slow** | `dc=False, fast=False, init=PREVIOUS_VALUES` | falls back here automatically when fast didn't converge | full physics; ~2× slower; converges more cases. |
| **AC + DC seed** | `dc=False, init=DC_VALUES` | falls back here when PREVIOUS_VALUES failed (exception OR bad status); also used for the initial LF in `_ensure_valid_state` | robust to topology perturbations and cold starts; +1 internal DC LF. |
| **DC LF** | `dc=True` | direct `nm.run_load_flow(dc=True)` calls in the recommender for screening, or via the analysis CLI `--use-dc`. | linear, no reactive, ignores tap ratios; converges almost always; not suitable for thermal overloads close to limits. |
| **Initial LF** | `dc=False, init=DC_VALUES` (forced) | every `__init__` / `reset()` of `SimulationEnvironment._ensure_valid_state` | avoids the spurious "Voltage magnitude is undefined" warning + retry. |

## 10. Cross-references

- `expert_op4grid_recommender/pypowsybl_backend/network_manager.py` —
  `run_load_flow`, `_run_ac_with_init_fallback`,
  `_create_default_lf_parameters`.
- `expert_op4grid_recommender/pypowsybl_backend/observation.py:simulate`
  — the `obs.simulate` entry point.
- `expert_op4grid_recommender/pypowsybl_backend/simulation_env.py:_ensure_valid_state`
  — the initial-LF policy.
- `expert_op4grid_recommender/config.py` — `PYPOWSYBL_FAST_MODE`,
  `MONITORING_FACTOR_THERMAL_LIMITS`,
  `PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD`.
- `expert_op4grid_recommender/main.py:run_analysis` — `actual_fast_mode`
  resolution; pipeline-level `fast_mode` propagation.
- `docs/release-notes/v0.2.2.post2.md` — rationale for the
  outer-loop cap bump and the non-converged-status fallback.
- `tests/test_initial_lf_voltage_init_mode.py` — guards
  `_ensure_valid_state` seed.
- `tests/test_lf_fallback_non_converged.py` — guards the four retry
  branches and the bumped outer-loop default.
