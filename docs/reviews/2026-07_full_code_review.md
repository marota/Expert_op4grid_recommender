# ExpertOp4Grid Recommender — Comprehensive Code Review

**Scope**: whole repository at v0.2.5 (branch point `401f25a`, July 2026) — architecture, interfaces & interactions, performance, documentation, maintainability.
**Method**: six parallel deep-dive reviews (architecture/orchestration, `action_evaluation`, `pypowsybl_backend` + performance, `graph_analysis` + `utils`, `manoeuvre` + IHM, tests/docs/CI/packaging), cross-checked against quantitative tooling (radon, ruff, grep-based audits). Every finding cited below was verified against the code at the referenced lines; the highest-severity findings were independently re-verified.

---

## 1. Executive summary

This is a codebase with **two clearly different generations of code living side by side**, and the newer generation is markedly better. The recent work — the `models/` recommender contract, the pydantic `Settings`, the three-step pipeline split, the `manoeuvre` module and its plugin architecture, the superposition estimator, the CI lint ratchet — shows a deliberate architectural direction and unusually good engineering discipline (measured optimizations with benchmark numbers in comments, golden tests with a regeneration protocol, an isolation invariant codified in `pyproject.toml`). The liability is concentrated in the **transitional glue**: a 34–41-key untyped context dict carrying function pointers, module-level mutable config used as a parameter channel, a hand-forked test config, `main.py` acting simultaneously as CLI, pipeline core and import hub, and a `utils/` package that mixes excellent engineering with genuinely broken modules.

> **Status note (revert).** The four discovery-pipeline changes originally landed on this
> branch — the C-diag rho-check contract fix, C3 (MultiDiGraph edge cache), C4/P1 (shared
> baseline) — were **reverted** after they regressed the recommendation set on the
> `config_pypsa_eur_*` / pypowsybl configuration (5 fewer actions: the non-topological
> injection actions and one parallel-line disconnection went missing). The discovery
> pipeline is now byte-identical to `main`. Any "Fixed in this branch" wording below for
> C-diag / C3 / C4 / P1 refers to those reverted commits and **no longer applies** — those
> findings are open again and need a safe re-introduction validated against that config.
> The non-discovery fixes remain in place: **C2** (IHM path traversal), **C5** (`sys.exit`),
> **C6** (utils bugs), plus the CI migration and this document.

**Highest-priority findings** (detail in §4). One finding originally headlined here — a pypowsybl rho-check comparing the candidate against the wrong reference state — was **downgraded to cosmetic** after tracing the data flow (discovery ranks from the overflow graph, not from that simulation; the rho-check output is diagnostic-only and gated off by default) and **fixed in this branch**; it is kept in the table's last row and written up as C-diag in §4.1 as a worked example of review principle #4 (verify the failure scenario before assigning severity).

| # | Finding | Where | Severity |
|---|---------|-------|----------|
| 1 | **Path traversal** on `/api/load_scenario` (name unsanitized → arbitrary `*.json` read), reachable on the public HF Space. **Fixed in this branch** (sanitize + resolved-parent check; `/api/config` dirs constrained). | `scripts/manoeuvre_ihm.py` | Security |
| 2 | **MultiDiGraph edge cache keyed by 2-tuples** — parallel circuits (ubiquitous on the RTE grid) silently overwrote each other in flow-influence caches. **Fixed in this branch** (keyed by `(u, v, k)`; regression test added). | `action_evaluation/discovery/_base.py:775-800` | Correctness |
| 3 | **Per-candidate pypowsybl baseline-variant leak** (topological discovery) — `compute_baseline_simulation` was called once per topological candidate, each minting a `keep_variant=True` copy of the N-1 state that nobody released (~30/analysis, duplicates of the kept N-1 variant). *Not* the intentionally-retained N / N-1 / prioritized-action variants. **Fixed in this branch** (shared baseline → 1). | `utils/simulation_pypowsybl.py:177` via `check_rho_reduction` | Resource leak |
| 4 | Library code calls **`sys.exit(0)`** on load-flow divergence — terminates the host process (UI/HTTP callers) with a *success* code. **Fixed in this branch** (`raise LoadFlowDivergedError`). | `environment.py`, `environment_pypowsybl.py` | Robustness |
| 5 | `utils/load_training_data.py` / `load_evaluation_data.py` contain **hard bugs** (indexing a `StateInfo`, `NameError` on a `__main__`-only global, `raise("string")`, a "reconnect" action that disconnects). **Fixed in this branch.** | see §4.4 | Correctness |
| — | *(downgraded — see C-diag)* pypowsybl discovery rho-check compared the candidate against the wrong reference state; **diagnostic-only** (does not affect scores, ranking, returned actions, or reassessment numbers), gated off by default. **Fixed** in this branch. | `utils/simulation_pypowsybl.py:221-233` | Cosmetic |

**Biggest performance lever**: the five topology-discovery mixins run **2 AC load flows per candidate** (baseline recomputed every time) while the fix already exists in-tree and is used by the three injection mixins (`_get_simulation_baseline` + `_check_rho_with_baseline`). Routing all mixins through the shared baseline roughly **halves the discovery load-flow budget** — the dominant cost of an analysis.

**Biggest maintainability lever**: retiring the three "shadow systems" that coexist with their modern replacements — the context dict vs. the `RecommenderInputs`-style dataclasses, the module-attribute config vs. the pydantic `Settings`, the hand-forked `tests/config_test.py` vs. `Settings(...)` instantiation. In each case the codebase already contains the good pattern; the revision is finishing migrations already started.

---

## 2. Review principles and method

The review was structured around principles worth stating explicitly, both to justify the findings and as a reusable template.

1. **Review the system, not the diff.** A whole-repo review asks different questions than a PR review: where do dependencies point? Where does state live? What happens to a new contributor's first change? We mapped module boundaries and import direction first, then descended.
2. **Divide by concern, not by directory, and review in parallel.** Six independent deep dives (architecture, core business logic, backend/perf, supporting libraries, the separate `manoeuvre` product, and the meta-level: tests/docs/CI/packaging). Independent reviewers don't anchor on each other's framing; overlapping findings (e.g. the config-override fragility surfaced from both the architecture and the testing angle, print-vs-logging from four angles) are corroboration, not duplication.
3. **Quantify before judging.** Radon complexity, ruff rule statistics, line counts, print/logging/except counts, test-function counts, data-directory sizes. Numbers turn "this file feels big" into "38 functions rate D or worse on cyclomatic complexity; the worst is CC 165". They also reveal *strengths* that impressions miss (1,248 test functions; a working TODO gate: zero TODOs in the package).
4. **Every claim needs a file:line and a failure scenario.** A weakness that can't name concrete inputs and a concrete wrong outcome is an opinion. The headline findings above were re-verified by a second reader against the source before being reported.
5. **Weigh by blast radius × likelihood, not by aesthetics.** A stylistic issue repeated 400 times (missing type hints) matters less than one 2-tuple cache key on a multigraph. Findings are ranked accordingly, and lint-level noise is aggregated rather than itemized.
6. **Credit deliberate engineering.** A review that only lists defects teaches nothing about what to preserve. This codebase contains patterns that should be *spread* (the golden-test protocol, benchmark-annotated optimizations, the ratchet lint baseline, the plugin verification layer) — identifying them is as important as the bug list.
7. **Distinguish deep revisions from quick wins.** Every proposal is classified by cost and risk. Quick wins are mechanical, low-risk, high-signal (often one line); deep revisions restructure and need the safety nets (goldens, tests) that — fortunately — this repo already has.
8. **Read the history.** Version drift (CLAUDE.md at 0.1.9 vs. code at 0.2.5), half-finished migrations (`applied()` context manager, pydantic shim) and issue-referenced TODOs (#81, #82) tell you which debt is *known* and which is invisible to the team.

---

## 3. Repo snapshot and metrics

- **Size**: ~28.4k lines of package code across 81 files; ~29.8k lines of tests (~1,248 test functions, >1:1 test:code ratio). `data/` is a lean 6.1 MB.
- **Two products in one repo**: the recommender (English-first, ~19k lines) and `manoeuvre` (French-first, ~9.1k lines + a 2,131-line Flask app in `scripts/` + a 1,799-line inline-JS frontend). `manoeuvre` has **zero** imports to/from the rest of the package — the isolation is codified as an invariant in `pyproject.toml`.
- **Complexity**: 38 functions at radon grade D or worse. Worst offenders: `determiner_manoeuvres_avec_sections` (CC **165**, ~600 lines), `_aggressive_impl` (64), `_identify_action_elements` (52), `_placement_decompose` (49), `_rejeu_securite` (46), `OrchestratorMixin.discover_and_prioritize` (43), `compute_combined_pair_superposition` (42), `run_analysis_step1` (41).
- **Observability**: 191 `print()` calls vs. 11 modules using `logging`. 22 broad/bare `except` clauses; five sites use the self-defeating `except (ImportError, Exception)`.
- **Lint**: `ruff check .` passes — because a per-file-ignore **ratchet baseline** grandfathers legacy violations while keeping `manoeuvre` strictly clean. Running ruff with ALL rules shows the shape of the debt: 404 missing parameter annotations, 424 long lines, 186 print statements, 166 private-member accesses.
- **Docstring coverage** is gated (interrogate, fail-under 80) — but only for `manoeuvre`.
- **CI**: CircleCI runs `pytest -m "not slow"` + the lint gate; GitHub Actions runs a code-quality report that *fails* on bare TODOs and hardcoded home paths (and demonstrably works: zero TODOs in the package). No coverage measurement, no Python-version matrix (CI on 3.12, Dockerfile on 3.11, support claimed for 3.10–3.12), and no CI leg without grid2op installed.

---

## 4. Weaknesses

Ordered by severity class. Every item verified at the cited location.

### 4.1 Correctness & security (fix first)

**C-diag — pypowsybl rho-check compared against the wrong state (diagnostic-only; FIXED in this branch).**
*Originally reported as a top correctness finding; downgraded to cosmetic after tracing the data flow — recorded here in full because the mechanism is real and the fix has been applied.*

The mechanism: the pypowsybl `check_rho_reduction_with_baseline` (`utils/simulation_pypowsybl.py:110-116`) simulates *only* the candidate `action` from the observation it receives, whereas the grid2op variant (`utils/simulation.py:110-113`) simulates `action + act_defaut + act_reco_maintenance`. So the two backends have **opposite first-argument contracts**: grid2op expects the healthy N-state (and re-applies the contingency itself); pypowsybl expects the already-contingency-applied observation (and applies nothing). The composing `check_rho_reduction` (`:221-233`) passed the healthy `obs` on the pypowsybl path, and the five topological mixins pass `self.obs` = the N-state (`_line_reconnection.py:179`, `_line_disconnection.py:144`, `_node_merging.py:170`, `_node_splitting.py:526`, `_pst.py:149`). Under pypowsybl the candidate was therefore simulated **on the healthy grid** and compared to an **N-1 baseline** — mismatched physics.

Why the impact is cosmetic, not results-affecting (the correction to the original write-up): discovery ranks and selects entirely from the overflow graph. `prioritized_actions` is assembled from the `identified_*` dicts and their overflow-graph scores (`_orchestrator.py:261-357`); the `is_rho_reduction` result is used *only* to append an action to an `effective`/`ineffective` list and print a line (verified at every call site — none gate inclusion), and those lists are consumed nowhere except a docstring. The final `rho_before`/`rho_after` a caller reads come from `reassess_prioritized_actions`, which correctly re-branches from `obs_simu_defaut` (`reassessment.py:163-167`). The whole path is additionally gated by `CHECK_ACTION_SIMULATION`, which is `False` in the shipped `config.py`. So the only wrong output was the printed "Effective/Ineffective" verdict and the unused `effective_*`/`ineffective_*` attributes under the pypowsybl backend.

**Fix (applied):** `check_rho_reduction` now passes `obs_baseline` — the contingency-applied kept-variant observation already computed by `compute_baseline_simulation` — into the with-baseline call, so the candidate branches from the N-1 state and the function's own "branching from the already contingency-applied observation" comment becomes true. This covers all five topological mixins with no call-site changes. **Not yet fixed:** the three injection mixins (load shedding / curtailment / redispatch) reach the same pypowsybl function via `_get_simulation_baseline` (which discards `obs_baseline`) + `_check_rho_with_baseline`, passing `self.obs`; correcting them cleanly is coupled to the backend-contract divergence and is folded into revision **R4** (a shared `BaselineContext` that takes the baseline observation explicitly). Same cosmetic-only impact in the meantime.

**C2 — Path traversal in the manoeuvre IHM (publicly deployed) — FIXED in this branch.**
`/api/load_scenario` took `name` from the request body with no sanitization and `load_scenario` read `(SCEN_DIR / f"{name}.json").read_text()` (the route re-read meta the same way). A name like `../../../<path>` read any `.json` file on the server. The save path *was* sanitized (`_safe_name`) — the asymmetry indicated oversight. The app is deployed on a public Hugging Face Space. Related: `/api/config` POST let any client repoint `SCEN_DIR`/`SEQ_DIR` to arbitrary paths, and `/api/scenarios_archive` then zips every `*.json` under it — an arbitrary-directory JSON exfiltration primitive in non-hosted mode.
**Fix (applied):** a `_stored_json_path(base, name)` helper sanitizes via `_safe_name` and requires the resolved path's parent to equal `base` (rejecting traversal with `ValueError`); it is used on both the `load_scenario` read and the meta re-read, and `api_load_scenario` now returns HTTP 400/404 instead of reading an arbitrary file or 500-ing. `/api/config` now validates any requested directory against an allowed-root check (`_dir_within_allowed`: process cwd or `MANOEUVRE_DATA_DIR`/`DGITT_CACHE_DIR`), logging accepted/refused changes — neutralising the exfiltration (zip) and arbitrary-write (`/api/save`) primitives. Verified in isolation: `../secret`, `../../etc/passwd`, `/etc/passwd`, `foo/../secret` are all contained to a flat in-directory name or rejected. **Remaining (out of scope here):** no `Host`-header validation (DNS-rebinding vector for the localhost tool) and the still-open `SESSION is None` route guards (W3).

**C3 — Parallel circuits silently collapsed in discovery flow caches — FIXED in this branch.**
The overflow graph is a multigraph (code unpacks `(u, v, key)` and passes `keys=True` in `_node_splitting.py:95-124`, `_node_merging.py:62`, `_line_reconnection.py:88`). But `_get_edge_data_cache` (`discovery/_base.py`) keyed `_cached_edge_names`/`_cached_edge_labels` by `(u, v)` 2-tuples, so twin circuits between the same substations — ubiquitous on the RTE grid (L61/L62 pairs) — overwrote each other. `_build_node_flow_cache` used the collapsed cache *when it existed*, else fell back to 3-tuple-keyed `nx.get_edge_attributes` — so load-shedding/curtailment/redispatch influence flows differed depending on call order, and parallel-edge flows were dropped.
**Fix (applied):** `_get_edge_data_cache` now keys both lookup dicts by the full edge id `(u, v, k)` on a multigraph (falling back to `(u, v)` on a plain graph), built in a single pass — matching the 3-tuple keys the fallback path already produced. `_build_node_flow_cache` reads only `edge[0]`/`edge[1]`, so it is unaffected by the wider key. Regression test added (`tests/test_parallel_circuit_flow_cache.py`): two parallel −50 MW circuits now sum to 100 MW at the shared node instead of collapsing to 50.

**C4 — Per-candidate pypowsybl baseline-variant leak (topological discovery) — FIXED in this branch.**
*Original wording ("kept variants leak unboundedly") over-generalized. Corrected after enumerating the three `keep_variant=True` sites and their intent.*

There are exactly three `keep_variant=True` sites, and two are **intentional and bounded** — the N-1 contingency observation (`simulation_pypowsybl.py:68`, one per analysis) and the reassessed-action state per prioritized action (`reassessment.py:165`, ≤ `N_PRIORITIZED_ACTIONS`). These are the states the operator wants to retain; they are not a leak.

The actual defect was the third site, `compute_baseline_simulation` (`simulation_pypowsybl.py:177`), reached through `check_rho_reduction`. The injection passes reach it once per run (cached via `_get_simulation_baseline`), but the five topological passes called the full `check_rho_reduction` **per candidate**, so each topological candidate minted a fresh `keep_variant=True` copy of the N-1 baseline that nobody released — ~30 per analysis, each a redundant duplicate of the N-1 variant already retained at site `:68`. The function's own docstring even says "use `compute_baseline_simulation()` once … for multiple actions" (`:203`); the topological path violated exactly that. Gated by `CHECK_ACTION_SIMULATION` (off in the shipped `config.py`), and bounded within one analysis — the "unbounded" framing applies only to a long-running service that reuses the `NetworkManager` across analyses (nothing sweeps kept variants between runs; `reset_to_base` doesn't purge them).

**Fix (applied):** the topological passes now share the single cached baseline via a wrapper in `main.py` that routes them through `_get_simulation_baseline` + `check_rho_reduction_with_baseline` — one kept baseline variant per run instead of ~30 (same change that resolves P1). The three injection passes now also branch candidates from that shared baseline observation, closing the C-diag injection remnant in the same edit. **Remaining (folded into R4):** a cross-run variant sweep (the single per-run baseline variant, like the intended N-1/prioritized ones, is still not explicitly released — bounded per run, but a `NetworkManager` variant registry with an LRU/`max_variants` backstop would bound a long-running service across analyses); and reusing the already-kept N-1 variant (`obs_defaut`) as the baseline branch point to eliminate even that one duplicate.

**C5 — `sys.exit(0)` in library code — FIXED in this branch.** On DC-fallback non-convergence, `environment.py` and `environment_pypowsybl.py` terminated the process — deadly for the documented UI/notebook callers, and with a **success** exit code. **Fix (applied):** both sites now `raise LoadFlowDivergedError(...)` (new `expert_op4grid_recommender/exceptions.py`, subclassing `RuntimeError` so the CLI's existing `except (ValueError, RuntimeError, TypeError)` still maps it to exit 1, while library/UI callers can catch and recover). The now-unused `import sys` was removed from both modules.

**C6 — Broken modules in `utils/` — FIXED in this branch** (each a 1–5-line fix; they had made the modules unusable as a library):
- `load_training_data.py` — `state = action_path[0]` indexed a `StateInfo`, which has no `__getitem__` → `TypeError`. Now `state = action_path`.
- `load_training_data.py` — `filter_out_non_reproductible_observation` asserted against `line_we_disconnect`, defined only in the `__main__` block → `NameError` when imported. Now a required parameter (caller updated).
- `load_evaluation_data.py` — `raise("no chronic is found...")` → `TypeError: exceptions must derive from BaseException`, masking the real error. Now `raise ValueError(...)`.
- `load_evaluation_data.py` — the "reconnect maintenance lines" action set buses to **-1** (disconnecting them; the sibling `run_contingency_on_scenario` did it correctly). Now `set_line_status: 1`, and the loop collects the timesteps that *still* overload into a fresh list so the return value matches the docstring and the "Success" print fires only when nothing remains overloaded.

**C7 — Latent crashes and silent misbehavior** (secondary but real):
- `main.py:269-296`: passing `prebuilt_env_context` without `dict_action` → `NameError` on `raw_dict_action`; conversely, the grid2op path silently discards a caller-supplied `dict_action`.
- `ObservationWithTopologyOverride`: `n_components`/`main_component_load_mw` read a never-set attribute; `q_or`/`q_ex` delegate to properties that don't exist on `PypowsyblObservation` (`observation.py:987-995`, `:1013-1019`) → `AttributeError` on access.
- `processor.py:58-66`: overload edges recovered by **float equality on rho** — two lines with identical loading (parallel circuits again) match the wrong edge; matching by the `name` attribute already stored on edges would be exact.
- `_node_merging.py:115,134`: unguarded `max()` on possibly-empty dicts → `ValueError` (the reconnection mixin guards the identical expression).
- `rules.py:294-298`: only couplings carrying `"VoltageLevelId"` get localized; grid2op-format couplings become `"unknown"`, matching none of the five expert rules — a silent rule bypass for one backend's action format.
- `tests/config_test.py` is missing ≥6 attributes present in `config.py` (`ENABLE_ANTENNA_RECOMMENDATIONS`, `MAX_CANDIDATE_SIMULATIONS`, `VISUALIZATION_FORMAT`, …); `main.py:384` accesses one directly → latent `AttributeError` for any test driving that branch.
- Ambiguous physics left in-tree: three competing `max_redispatch` formulas (live code vs. commented code vs. docstring) at `discovery/_base.py:884-890`; reconnection candidates ranked by **highest** delta-theta with an in-comment acknowledgment that the sort may need reversing (`_line_reconnection.py:154-161`); an in/out flow-orientation asymmetry between the node-flow cache and the node-splitting convention (`_base.py:508-513`).

### 4.2 Architecture & interface design

**A1 — The pipeline state is a stringly-typed god object.** The `context` built in `run_analysis_step1` (`main.py:406-441`) has 34 keys, **8 of which are backend-selected function pointers**; step2_graph adds 7 more (~41 total). There is no `AnalysisContext` type; consumers hand-unpack 15–20 keys (`main.py:543-565`, `:754-779`; `reassessment.py:107-120`). Typos become runtime `KeyError`s; the shape is undiscoverable; `RecommenderInputs._context` (`models/base.py:150`) leaks the whole blob into the otherwise-clean DTO, and `ExpertRecommender` hard-fails without it — the plugin contract has a trapdoor back into the god object.

**A2 — Inverted dependencies, cycles held together by deferred imports.** The entry-point module *is* the library core: `models/expert.py:91` imports `main._run_expert_discovery` inside `recommend()`; `reassessment.py:122-124` imports from `main` inside a function; `main.py:870-876` imports both back. Lower layers importing upward is the root cause; splitting `main.py` into `cli.py` + `pipeline.py` breaks all three cycles.

**A3 — Config: two desynchronized sources of truth.** The pydantic `Settings` is validated once, dumped into module attributes for backward compatibility (`config.py:252-263`), and thereafter the pipeline **mutates the module attributes** as its parameter-passing mechanism (`main.py:263-266`, `:997`) — bypassing validation (`validate_assignment=True` never fires) and leaving derived values stale: overriding `ENV_NAME` does not recompute `ENV_PATH`/`ACTION_FILE_PATH`. The codebase copes with 29 defensive `getattr(config, 'X', default)` calls coexisting with direct attribute access — two idioms chosen per call site by whether someone got burned there.

**A4 — The shipped defaults are the test fixture.** `config.py` defaults to `bare_env_small_grid_test`, `IGNORE_LINES_MONITORING=True`, `CHECK_ACTION_SIMULATION=False`, `N_PRIORITIZED_ACTIONS=20` (`config.py:89-151`) while the documented production values (dijon snapshot, N=5) live in the "alternative" `config_basic.py`. `python main.py` out of the box analyzes the toy grid, contradicting CLAUDE.md.

**A5 — The discovery mixin split is a file split, not a modularization.** `ActionDiscoverer.__init__` sets **65 instance attributes**, 16 more appear lazily, and 5 PST holders are never initialized (forcing `getattr` in the orchestrator). No mixin declares its dependencies (no Protocol); there is cross-file temporal coupling — `PSTMixin` lazily creates `_disco_bounds` which `LineDisconnectionMixin` *deletes* at entry and exit, so PST scoring correctness depends on orchestrator call order across three files (`_pst.py:46-49` vs. `_line_disconnection.py:103-105,208-210`). The per-family result quintuplet (`identified_/effective_/ineffective_/scores_/params_`) is hand-repeated 8×, and the `action_scores` assembly is ~100 lines of copy-paste (`_orchestrator.py:382-485`) — which is exactly where a duplicated `add_prioritized_actions` call slipped in twice (`:294-299`/`:312-317` and `:344-348`/`:355-359`).

**A6 — Interface bloat.** `run_analysis_step1`: 10 params, returns an `(Optional, Optional)` sentinel tuple. `setup_environment_configs`: positional 8-tuples. `ActionDiscoverer`: constructed with 23 keyword arguments (`main.py:804-828`). `fast_mode` has three conflicting defaults (wrapper `True`, config `False`, CLI `store_true` making the config knob unreachable) and is injected by **monkey-patching private methods of a live object** (`main.py:830-838`). `SimulatedAction` is defined, exported and unit-tested — and never instantiated in production (`reassessment.py:212-222` returns plain dicts).

**A7 — Leaky backend abstraction.** External code routinely reaches into privates: `env.backend._grid.network` (`main.py:284`), `obs._network_manager._default_dc` read *and written* (`reassessment.py:129-134`), `obs._limit_or` (`superposition.py:471-480`). Quiet parity gaps: `theta_or` docstring says degrees, code produces radians (`observation.py:199-200`); two different definitions of rho feed different pipeline stages (`observation.py:159-180` uses `i_or/limit_or`; `overflow_analysis.py:330-337` uses `max(i1,i2)` for lines).

**A8 — `utils/` is a dumping ground** mixing five concerns: a REPAS subsystem (~2,250 lines), simulation orchestration, environment factories, core physics (`superposition.py`, 1,358 lines), and two script-style pipelines with `__main__` blocks reading files from the CWD. Three-way duplication across backends: `get_theta_node`/`get_delta_theta_line` implemented **three times** (helpers, helpers_pypowsybl, superposition); `simulation.py` vs. `simulation_pypowsybl.py` ~85% identical (~290 lines each) — and the C-diag rho-check bug bred precisely in that duplicated seam (opposite first-argument contracts hidden behind identical-looking signatures); `_inhibit_swapped_flows` implemented three times (builder, antenna_graph, overflow_analysis).

### 4.3 Performance

Ranked by expected wall-clock impact on a full analysis (AC load flows dominate).

**P1 — 2 load flows per topology candidate instead of 1 — FIXED in this branch.** The five topological mixins called the full `check_rho_reduction`, which recomputed the baseline (1 LF + 1 leaked variant) for **every candidate**; the shared-baseline pattern (`_get_simulation_baseline` + `_check_rho_with_baseline`) already existed and was used by the three injection mixins. For ~30 simulated candidates that was ~60 LFs where ~31 suffice. The pypowsybl backend now routes the topological passes through the shared cached baseline (one baseline LF per run), which also resolves the C4 variant leak and subsumes the C-diag rho-check contract cleanup. The routing lives in the `main.py` pypowsybl-wiring block so the grid2op path and the mock-based discovery tests are unchanged.

**P2 — ~11 native DataFrame fetches per observation, one observation per simulate.** `_refresh_state` + `_compute_line_impedances` + first `sub_topology()` call trigger ~11 cross-JNI pandas materializations per candidate (`observation.py:95-143`, `:229-231`, `:566-575`). R/X are variant-invariant (belongs in `NetworkManager`); three `get_buses` calls should be one. Target ≤3 fetches per candidate.

**P3 — Per-element full-table fetches in action application.** `disconnect_line`/`reconnect_line` fetch full `get_lines()`/`get_2_windings_transformers()` frames just for membership tests (`network_manager.py:468-491`) — up to 2k fetches for a k-line action — while cached `_lines_set`/`_trafos_set` and `disconnect_lines_batch` sit unused. `SwitchAction.apply` fetches the entire switch table (~85k rows at RTE scale) **per switch ID** (`action_space.py:72-91`), in the primary action format of `--pypowsybl-format` mode.

**P4 — The `name_*` property trap, third occurrence live.** `name_line`/`name_sub`/`name_gen` rebuild a numpy array on every access (`network_manager.py:276-298`). The team has already diagnosed and fixed two O(n²) regressions from this (comments at `discovery/_base.py:397-400`, `_load_shedding.py:31-37`); a third is live: `set_thermal_limit` indexes `self.name_line[i]` in a loop (`simulation_env.py:150-152`) ≈ 49M ops on 7,000 lines. Cache the arrays once (read-only) and the whole trap class disappears.

**P5 — Per-candidate full-graph recomputation in scoring.** `computing_buses_values_of_interest` rebuilds `nx.get_edge_attributes` dicts over the entire graph for every node-splitting candidate (`_node_splitting.py:86-89`), while `_cached_edge_labels` sits unused (blocked on the C3 fix); node merging does the same (`_node_merging.py:46-58`) and recreates `act_defaut` inside the loop (`:167`).

**P6 — Miscellaneous**: JSON round-trip of LF parameters on every fast-mode run (`network_manager.py:423-433`); slow-mode retry that re-runs identical parameters; defensive `.copy()` on every observation property combined with per-element `sum(obs.p_or[i] for i in ...)` patterns; dead `topology.py` containing the worst complexity in the package (O(n_sub·n_line) ≈ 28M iterations) exported but unused — a trap for future callers.

Credit where due: the architecture-level performance choices are **right** — variant hot-start from converged N-1 states, single-LF graph+rho computation, vectorized observation construction, and the superposition estimator that prices N·(N−1)/2 action pairs at **zero** load flows.

### 4.4 Maintainability, documentation, tests, packaging

**M1 — CLAUDE.md is one full minor version stale** (says 0.1.9; code is 0.2.5): points to the deleted `discovery.py` monolith, omits `manoeuvre/` (a third of the package), `models/`, `antenna_graph.py`; documents an `ActionType` enum that doesn't exist and config keys that no longer match. For a repo explicitly designed for AI-assisted development, a misleading CLAUDE.md is compounding debt: it misdirects every future session.

**M2 — The test config-override is fragile by construction.** `tests/conftest.py` swaps `sys.modules` at import time; `tests/config_test.py` is a **hand-maintained fork** of `config.py` (every new key must be added twice — and six are already missing, see C7); the mechanism bypasses pydantic entirely, so `Settings` validation and env parsing are never exercised in CI.

**M3 — No coverage measurement** anywhere, despite 1,248 tests. Concrete blind spots found manually: zero test references for `utils/load_training_data.py` (which contains bug C6 — no test would have caught it), `pypowsybl_backend/migration_guide.py`, `simulation_env.py` (only transitive); near-zero for `data_loader.py`, `make_*_env.py`. The `slow` marker is unregistered (no `[tool.pytest.ini_options]` at all) — a typo like `@pytest.mark.slwo` silently *runs as fast in CI*. The ruff baseline grandfathers `F811` (shadowed, i.e. **never-running** tests) in two files and an `F821` undefined name in a third.

**M4 — Dependency declarations have drifted in both directions.** `requirements.txt` pins `grid2op==1.12.1`+`LightSim2Grid` (absent from `pyproject.toml`) — so CI always tests the grid2op-installed path and **never** the "grid2op optional" contract of PR #26; it omits `numpy` (CircleCI compensates with an ad-hoc `pip install numpy==2.3.0`), `pydantic`, `pydantic-settings`. Floors disagree (`expertop4grid>=0.2.8` vs `==0.3.2.post3`). No lockfile. Meanwhile `pyproject.toml` still hard-depends on `pypowsybl2grid`, which transitively pulls grid2op — "grid2op optional" is currently true at import time only, not at packaging time.

**M5 — The pypowsybl2grid site-packages file patch is load-bearing.** `scripts/patch_pypowsybl2grid_file.py` regex-edits an installed third-party file, mutating a shared venv, with no upstream-version guard — if upstream refactors `update_integer_value`, the patch silently stops matching or becomes semantically wrong. A vendored `PatchedPyPowSyBlBackend` subclass overriding the one method would be importable, testable and version-guardable; in parallel the fix should be upstreamed.

**M6 — Observability**: 191 `print()` vs. 11 logging modules; failures printed-and-swallowed so callers can't distinguish success from failure (`action_rebuilder.py:511-515` catches everything and returns the input dict as if converted; failed simulation checks counted as "ineffective", `_load_shedding.py:182-184`, while the `non_convergence` field in `action_scores` stays always-empty); a library factory **reconfigures the root logger to ERROR** as a side effect (`make_assistant_env.py:40-41`), silencing the host application's own logs.

**M7 — Bilingual codebase without a declared policy.** `manoeuvre` is consistently French (fine — domain-appropriate and internally uniform); the recommender mixes `obs_defaut`/`amont`/`aval` with English docstrings, and the CHANGELOG's Unreleased section switched to French mid-file. The cost is real for onboarding; the fix is a one-paragraph declared policy, not a rewrite.

**M8 — manoeuvre-specific structural debt**: the 2,131-line single-file Flask app in `scripts/` (outside the package — 8+ test files each re-implement `importlib` file loading to import it); a ~950-line `Session` god class; module-level globals making the app strictly mono-user; the production Space runs the **Flask dev server** single-threaded; a 1,799-line/153-function inline-JS frontend with no lint or JS tests; ~30 underscore-private symbols re-exported as the de-facto public API of `algo/` (`algo/__init__.py:86-149`), making the privacy convention meaningless; 1.1 MB of map JSON shipped in every wheel including recommender-only installs.

---

## 5. Strengths (patterns to preserve and spread)

1. **The `models/` recommender contract** (`models/base.py`) — a textbook strategy pattern: ABC with capability flags, documented `RecommenderInputs` DTO with field-level semantics (None-vs-empty-list spelled out), `ParamSpec` for UI parameter introspection. This is the template the context dict should converge to.
2. **The manoeuvre plugin architecture** — three-phase decomposition with PEP 544 Protocols (no inheritance required), a serializable pivot type with graph round-trip converters, an entry-point registry with collision protection and broken-plugin isolation, and — the standout — **independent verification of every plugin result** (`pipeline.py:67-120`): third-party algorithms "n'ont pas à être crus sur parole", down to flagging when a plugin's self-declared feasibility contradicts the replay.
3. **The golden-test protocol** (`tests/manoeuvre/test_golden_sequences.py`) — frozen outputs with a conscious `UPDATE_GOLDENS=1` regeneration path, canonicalized set-derived fields, plus 33 real-substation fixtures that rebuild networkx graphs **without pypowsybl at runtime**. This is behavior-level characterization done right, and it is exactly what makes the proposed refactor of the CC-165 function *cheap and safe*.
4. **The lint ratchet** — `ruff check .` green over the whole repo via a grandfathered per-file baseline, with a codified invariant that `manoeuvre` never enters the baseline. "Strict for new code, tolerated for legacy" is the correct way to introduce linting into a legacy codebase; the TODO/hardcoded-path gate demonstrably works (zero TODOs in the package).
5. **Benchmark-annotated optimization** — comments carrying real measured numbers next to the optimization they justify (`3 239 ms → 683 ms, identical output` in `helpers_pypowsybl.py:296-303`; `6 100 ms → 80 ms` in `conversion_actions_repas.py:330-334`; `374 ms → 48 ms` for the `NetworkTopologyCache` groupby replacement). The `NetworkTopologyCache` itself (analytical Union-Find replacing variant cloning, with a documented invalidation rule) is the best pure-performance engineering in the repo.
6. **The right simulation architecture** — variant-based what-if with hot-start branching from converged N-1 states, single-LF combined graph+rho computation, a robust LF fallback ladder with empirically-justified parameters, and the superposition/GST estimator pricing all action pairs at zero load flows (with sanity bounds on beta and no-op detection).
7. **The three-step pipeline split** with the extension points treated as API (a test asserts the `prebuilt_obs_simu_defaut` signature contract) — external callers can interpose between overload detection, graph building and discovery without forking.
8. **Deliberate failure isolation for decorative stages** — visualization and superposition failures cannot abort an analysis (`main.py:623-644`; `reassessment.py:269-273` with `exc_info=True`); `reassessment.py` overall is the observability model the rest should copy.
9. **grid2op made genuinely import-optional** without landmines (guarded imports, deferred per-backend imports, a Dockerfile exploiting it for a lean image), and dependency injection in `ActionDiscoverer` (simulation functions injected, enabling the 2,555-line pure-mock unit suite).
10. **Docs infrastructure with a quarantine** — a curated `docs/README.md` index, an explicit `archive/` for finished/transient notes, and a stated canonical-history rule. Most projects never make that distinction.

---

## 6. Deep revision proposals

Ordered as a coherent roadmap; each step is enabled by the previous and by safety nets that already exist.

**R1 — Typed pipeline spine** (highest leverage, mostly mechanical).
Introduce `AnalysisContext` and `AnalysisResult` dataclasses replacing the 41-key dict and the untyped result dict; make step1 return `AnalysisResult | AnalysisContext` (or raise a typed outcome) instead of the `(Optional, Optional)` sentinel. Introduce a `SimulationBackend` protocol with the 9 backend operations plus `fast_mode` as constructor state (`Grid2opBackend`, `PypowsyblBackend`), deleting the 18 delegation wrappers (~120 lines), the 8 function pointers in the context, the call-site `if is_pypowsybl:` forks and the `discoverer._*` monkey-patching in one move. The codebase already proves it can do this well — `RecommenderInputs` is the template. Actually use `SimulatedAction`.

**R2 — Split `main.py` into `cli.py` + `pipeline.py`**; move `_run_expert_discovery` under `models/`. Dependency direction becomes `cli → pipeline → models → action_evaluation/graph_analysis → utils`, dissolving all three import cycles (A2) and making `print` acceptable exactly where it remains (the CLI).

**R3 — One config, one source of truth.** Make the pydantic `Settings` instance authoritative: derived paths as `@computed_field` (fixes staleness), an explicit `get_settings()/override_settings()` accessor, pipeline overrides passed as parameters instead of module mutation. Tests override via a `Settings(...)` fixture — deleting `tests/config_test.py` and the `sys.modules` swap, so validation actually runs in CI and the 29 defensive `getattr` sites collapse. (Interim 1-hour step: make `config_test.py` star-import the real config and override only deltas.)

**R4 — Unify the per-backend simulation seam.** One `BaselineContext` (holds `act_defaut`, baseline rho, the kept baseline observation, and a `release()`) created once per discovery/reassessment run; `check_rho_reduction_with_baseline` takes the baseline observation **explicitly**. This deletes the ~85%-duplicated `simulation.py`/`simulation_pypowsybl.py` pair, structurally prevents the C-diag class of bug (and completes the fix for the three injection mixins left open by the surgical patch), routes all eight discovery families through one shared baseline (P1: halves LF count), and gives variants a lifecycle (fixes C4 — add a `NetworkManager` variant registry with an LRU/`max_variants` guard as a backstop).

**R5 — Restructure discovery around data, not mixins.** A `FamilyResult` dataclass (`identified/effective/ineffective/scores/params/non_convergence`) stored as `self.results: Dict[family, FamilyResult]` kills ~40 lines of `__init__`, the 8× hand-written `action_scores` assembly, the PST `getattr` special-casing, and makes prioritization a data-driven loop over an ordered `(family, min_key, cap)` table — the duplicate-call slip becomes impossible. Extract a shared `InjectionDiscoveryBase` for the three injection families (~120 duplicated preamble lines, already drifting: curtailment weighs 4 flow components, shedding 2, undocumented). Replace substring action-type matching with an `ActionType` enum and a declarative keyword→type table (fixes the rules bypass in C7).

**R6 — Split `utils/` by intent**: `repas/` (parser, converter — itself split into `_compat`/`topology_cache`/`convert` —, rebuilder), environment factories merged with the top-level `environment*.py`, `superposition` and `reassessment` promoted to the pipeline layer, `__main__` script bodies moved to `scripts/`. Fix the C6 bugs as part of the move — and give the two data modules their first tests.

**R7 — manoeuvre: promote the IHM into the package** (`manoeuvre/ihm/`): `create_app()` factory replacing module globals (enables `test_client()`, multi-session later), `Session` split into `NetworkView`/`SequenceEditor`/`ScenarioStore`, routes as blueprints, `scripts/manoeuvre_ihm.py` reduced to a shim; run the Space under waitress/gunicorn (workers=1, threads=1 to preserve the documented serialization invariant) with a `/healthz`. Decompose `determiner_manoeuvres_avec_sections` (CC 165) into phase functions over a `SequencingContext` dataclass — the golden suite makes this near risk-free, which is precisely what it was built for. Define a real public API for `algo/` and retire the 30 re-exported underscore names over one release.

**R8 — CI/packaging convergence.** One CI platform; Python matrix 3.10–3.12; `pytest --cov` with a coverage ratchet (same philosophy as the ruff baseline); **one matrix leg installing without grid2op** to enforce the optionality contract; move `pypowsybl2grid` behind a `[grid2op]` extra; a single dependency source (`pyproject.toml` + lockfile) feeding both CI configs. Replace the site-packages file patch with a vendored backend subclass and upstream the fix.

**R9 — Documentation reset.** Regenerate CLAUDE.md against 0.2.5 (tree with `manoeuvre/`, `models/`, `discovery/`-as-package; current config keys; a declared language policy); add a release-checklist item so it can't drift a full minor version again. Extend the golden-test pattern to the recommender: freeze `action_scores`/`prioritized_actions` for 2–3 canonical contingencies on the dijon fixture — the scoring code changes every minor version and currently has no characterization net.

**R10 — Consider the product split.** `manoeuvre` already has zero coupling, its own CLAUDE.md, docs tree, test tree and deploy pipeline. Either extract it as its own distribution or formalize the monorepo (independent versioning, CI path filters). Right now it pays a shared-repo tax (wheel size — including 1.1 MB of map JSON in recommender-only installs —, entangled release notes, a bilingual changelog).

---

## 7. Quick wins

Low-risk, mostly mechanical; the first block should land immediately.

**Correctness/security (hours, not days):**
1. ~~C-diag~~ **Done in this branch**: the pypowsybl topological passes now share the cached baseline and branch candidates from the contingency observation (`main.py` wiring + `_get_simulation_baseline`); the three injection passes branch from the same shared baseline. `check_rho_reduction` also passes `obs_baseline` for any residual direct callers.
2. ~~C2~~ **Done in this branch**: `_stored_json_path` (sanitize + resolved-parent check) on load + meta read, 400/404 on bad names; `/api/config` dirs validated against an allowed root.
3. ~~C3~~ **Done in this branch**: edge caches keyed by `(u, v, key)` + parallel-lines regression test. Remaining: pass the cached labels into `computing_buses_values_of_interest` to also close P5.
4. ~~C4~~ **Done in this branch** (topological baseline-variant leak: ~30/run → 1, via the shared baseline). Remaining: a cross-run `NetworkManager` variant sweep and reusing `obs_defaut` as the baseline branch — folded into R4.
5. ~~C5~~ **Done in this branch**: `sys.exit(0)` → `raise LoadFlowDivergedError` (2 sites, new `exceptions.py`).
6. ~~C6~~ **Done in this branch**: the four fixes in `load_training_data.py` / `load_evaluation_data.py`.
7. C7: initialize `raw_dict_action`; add the missing keys to `config_test.py` (or star-import); guard the two `max()` calls; match overload edges by name; fix the two `ObservationWithTopologyOverride` property groups.

**Performance (each independently measurable):**
8. ~~Route the five topology mixins through `_get_simulation_baseline`~~ **Done in this branch** (P1 — ~2× fewer LFs in discovery, and closes the C4 baseline-variant leak).
9. Cache `name_line`/`name_sub`/`name_gen` arrays once, read-only (P4 — kills the O(n²) trap class including `set_thermal_limit`).
10. Use `_lines_set`/`_trafos_set` in `disconnect_line`/`reconnect_line`; fetch `get_switches()` once per `SwitchAction.apply`; one `get_buses()` per `_refresh_state`; cache R/X per network; pre-build the fast-mode LF `Parameters` once.
11. Hoist `act_defaut` out of the node-merging loop.

**Hygiene:**
12. `[tool.pytest.ini_options]`: register the `slow` marker, `testpaths`, error on unknown marks (~10 lines; kills silent marker typos).
13. Sync `requirements.txt` with `pyproject.toml` (add numpy/pydantic/pydantic-settings; drop the ad-hoc CircleCI numpy install); pick one floor per dependency.
14. Fix the grandfathered `F811` shadowed tests (dead coverage) and the `F821` undefined name in tests.
15. `except (ImportError, Exception)` → `except ImportError` (5 sites); stop reconfiguring the root logger in `make_assistant_env.py`.
16. Delete dead code: `topology.py` (or quarantine it), the string `UnionFind` family, `_edge_names_buses_dict_new`, the always-true `len(...) >= 0` guard, the duplicated curtailment `add_prioritized_actions` calls, `network_path` in `switch_to_dc_load_flow_pypowsybl`, the deprecated runtime-patch script.
17. Include all 8 `MIN_*` knobs in the CLI sanity sum; give `--fast-mode` a real tri-state; unify the empty `action_scores` schema with the full 8-category list.
18. CLAUDE.md: fix the two most misleading pointers today (discovery path, version) pending the full R9 regeneration; move `docs/manoeuvre/dataset_rte7000/handoff.md` to `archive/`; move the `.pptx` out of the import package.
19. manoeuvre: `_session_requise()` helper for the ~12 unguarded routes (400 instead of 500); finish the `applied()` migration (3 sites); factor `_load_ihm()` into a conftest fixture (deletes 8 copies); gzip the packaged layout JSONs (~85% smaller).
20. Mutable default args (3 sites), builtin shadowing (`id`, `open(file) as file`), the wrong copy-pasted docstrings (2), the `"Priorization"`/`"clonging"`/`"ThersholdMinPowerOfLoop"` typos (the last needs a deprecation alias — it's a public config key).

---

## 8. Closing assessment

The trajectory of this codebase is **good**: each generation of code is better than the last, migrations are real (pydantic, discovery package, models contract, lint ratchet) even where unfinished, and the team demonstrably measures before optimizing and tests before refactoring. The dominant risk is not decay but **seams**: the correctness findings in this review (C3's parallel-circuit cache collapse and the C-diag rho-check contract mismatch — both fixed in this branch — and the still-open C7 formula ambiguities) each live in a duplicated or half-migrated seam between two systems that both still run. The deep revisions above are therefore less about redesign than about **finishing**: pick the winner at each seam (Settings over module attributes, dataclasses over the context dict, shared baseline over per-candidate recomputation, one simulation module over two), delete the loser, and let the existing golden/characterization infrastructure absorb the risk.
