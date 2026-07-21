# Bug: overflow-graph build corrupts a reused pypowsybl env (step2 → later step1 "vanishes" overloads)

**Severity:** correctness (silent) — affects any workflow that reuses one
`SimulationEnvironment` across **multiple** analyses (batch classifiers,
long-lived services, the Co-Study4Grid game backend serving several
contingencies from one loaded network).

**Status:** root-caused with a minimal reproduction; verified workaround below.
No in-place code fix landed yet (needs care not to change the graph-build flow
math — see "Fix directions").

## Symptom

Run a full analysis (`step1` → `step2_graph` → `step2_discovery`) for one
contingency on a pypowsybl env, then run `step1` for a **different** contingency
on the **same** env: the second `step1` no longer detects the overloads it
detects on a freshly built env — it returns an `AnalysisResult` short-circuit
("Overload breaks the grid apart") with an empty overload set. The analysis
silently reports *no actionable overload* where there is one.

Only a **full env rebuild** (reload the network from file) restores correct
detection. `reset_to_base()`, `sweep_kept_variants()`, removing every non-base
variant, re-cloning `base` from the pristine `InitialState` variant, and a cold
AC load flow on `base` **all fail to recover**.

## Root cause

`run_analysis_step2_graph` builds the overflow graph via
`pypowsybl_backend/overflow_analysis.py`. Its flow-change analysis clones a
working variant, disconnects the overloaded lines, and runs a load flow (DC for
the graph transfer). On a stressed operating point disconnecting the overloads
can island part of the grid and that load flow does **not** converge
(`MAX_ITERATION_REACHED`).

After the graph build returns, the **base variant's AC voltage state is
destroyed**: on the France THT `hiver_pic_2021` snapshot the base variant goes
from 35 NaN `v_mag` buses to **1701** (≈1666 buses left de-energized), while line
connection flags, switch open/closed states, load and generation totals, and the
cached thermal limits are all **unchanged**. The next `step1` calls
`reset_to_base()` (which only switches the working variant — it does not re-solve)
and then `obs.simulate(contingency)`, which warm-starts (`PREVIOUS_VALUES`) from
those NaN voltages, diverges, and yields a degenerate near-zero-flow observation
→ every `rho < 1` → "no overload".

The damage lives **below the variant layer** (in pypowsybl network-global solver
state): it survives variant purge and a re-clone of `base` from `InitialState`,
and is only cleared by reloading the network object. That is why the workaround
must rebuild the whole env.

## Minimal reproduction

```python
# env built with the default config (IGNORE_LINES_MONITORING=True,
# MONITORING_FACTOR_THERMAL_LIMITS=0.95); ctx0 = prebuilt_env_context
def probe(cid):
    out = run_analysis_step1(analysis_date=None, current_timestep=0,
        current_lines_defaut=[cid], backend=Backend.PYPOWSYBL,
        dict_action={}, prebuilt_env_context=ctx0)
    return "CONTEXT" if isinstance(out, AnalysisContext) else "no-context"

print(probe("AVALLL61VNOL"))                      # CONTEXT (overload detected)
out = run_analysis_step1(..., current_lines_defaut=["AVALLL61J.VIL"],
                         dict_action=full, prebuilt_env_context=ctx0)
run_analysis_step2_graph(out)                     # builds overflow graph
print(probe("AVALLL61VNOL"))                      # no-context  <-- BUG

# base variant voltages are now NaN even though topology/loads are identical:
nm.set_working_variant(nm.base_variant_id)
np.isnan(net.get_buses()["v_mag"].values).sum()   # 35 fresh -> 1701 after
```

Contingency-specific: contingencies whose overload disconnection keeps the grid
solvable (e.g. an "easy" case) do not trip it; a stressed one whose flow-change
LF hits `MAX_ITERATION` does.

## Workaround (used by the RTE7000 game-mode classifier)

Rebuild the env after every analysis that ran `step2` (env rebuild is only
~4 s). Screening with `step1` alone never corrupts, so only grading needs it:

```python
rec = process(cid, ctx0, dict_action)     # step1 (+ step2 for candidates)
if rec["kind"] in ("candidate", "error"): # step2 ran or state is unknown
    del env; env, ctx0, dict_action, net = build_env(lab)
```

## Fix directions (not yet applied)

1. **Isolate the flow-change load flow** on a network object that is not the one
   reused across analyses (e.g. a dedicated clone loaded once), so a divergent
   graph-transfer LF can never pollute the shared network's global solver state.
2. **Restore a valid AC base state after `step2_graph`** — but note a plain
   `run_ac(DC_VALUES)` on `base` did *not* recover here, so this must reload /
   re-import the network, not just re-solve.
3. **Guard the flow-change LF divergence**: when disconnecting the overloads
   islands the grid, skip/flag it instead of leaving the shared network in a
   half-solved state.

Any fix must keep the overflow-graph flow numbers byte-identical on the
non-divergent path (the existing pipeline output contract).
