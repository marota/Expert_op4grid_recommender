# Documentation — ExpertOp4Grid Recommender

Index of all project documentation. For the project overview, installation and
usage see the [root README](../README.md); for the full release history see
[CHANGELOG.md](../CHANGELOG.md).

```
docs/
├── architecture/      simulation pipeline & backend internals
├── design/            feature / algorithm design specs
├── manoeuvre/         detailed-topology maneuver module (+ IHM, assets)
├── dataset_rte7000/   historical RTE-7000 topology/maneuver benchmark (+ data)
├── notes/             transient planning & analysis notes
└── release-notes/     per-version notes (canonical history: CHANGELOG.md)
```

## Architecture & pipeline
- [`architecture/simulation-pipeline.md`](architecture/simulation-pipeline.md) —
  the pypowsybl simulation pipeline: AC/DC & fast/slow load-flow modes, voltage
  initialisation, variant lifecycle, thermal-limit hypotheses, retry branches.

## Feature & algorithm design (`design/`)
- [`design/recommender_models.md`](design/recommender_models.md) — the pluggable
  `RecommenderModel` contract: DTO fields, capability flags, reusable pipeline
  phases, integration point, and a minimal new-model example.
- [`design/superposition_module.md`](design/superposition_module.md) —
  superposition theorem and the Generalized Superposition Theorem (§10) used to
  estimate combined action impact via virtual flows / delta-theta.
- [`design/antenna_overflow_graph.md`](design/antenna_overflow_graph.md) —
  overflow-graph handling for islanded-pocket ("antenna") contingencies.
- [`design/load_shedding_design.md`](design/load_shedding_design.md) — load
  shedding (downstream consumption reduction) as a corrective action type.
- [`design/renewable_curtailment_design.md`](design/renewable_curtailment_design.md) —
  renewable curtailment, the upstream counterpart to load shedding.

## Maneuver module (`manoeuvre/`)
Detailed-topology (node-breaker) maneuver planning — its rules, plugins, and web IHM.
- [`manoeuvre/module.md`](manoeuvre/module.md) — module reference: objectives,
  pipeline, node-breaker topology analysis.
- [`manoeuvre/regles.md`](manoeuvre/regles.md) — business rules (R1–R16) with
  rule-to-code traceability.
- [`manoeuvre/plugins.md`](manoeuvre/plugins.md) — pluggable calculation phases
  (identification / sequencing / end-to-end planning).
- [`manoeuvre/ihm.md`](manoeuvre/ihm.md) — the interactive web IHM (topology
  editor + sequence animation) and the hosted HuggingFace Space.
- [`manoeuvre/postes_n_jeux_de_barres.md`](manoeuvre/postes_n_jeux_de_barres.md) —
  extension from 2-busbar to N-busbar substations.
- [`manoeuvre/optimisations.md`](manoeuvre/optimisations.md) — performance/quality
  review and measured results.
- [`manoeuvre/implementation_plan.md`](manoeuvre/implementation_plan.md) — port
  roadmap from C++ libTOPO to Python.

Developer conventions for the module live in
[`../expert_op4grid_recommender/manoeuvre/CLAUDE.md`](../expert_op4grid_recommender/manoeuvre/CLAUDE.md).

## RTE-7000 dataset (`dataset_rte7000/`)
Historical topology/maneuver benchmark built from the public D-GITT RTE-7000 dataset.
- [`dataset_rte7000/README.md`](dataset_rte7000/README.md) — campaign table
  (7 days over 2021–2023) with statistics.
- [`dataset_rte7000/GUIDE.md`](dataset_rte7000/GUIDE.md) — dataset contents,
  schema, and local reproduction.
- [`dataset_rte7000/plan.md`](dataset_rte7000/plan.md) — the 6-phase build &
  publication plan.
- [`dataset_rte7000/HANDOFF.md`](dataset_rte7000/HANDOFF.md) — session handoff:
  status, results, and next tasks.

## Notes & analysis (`notes/`)
Transient planning / analysis documents (kept for context, not living reference).
- [`notes/migration_plan.md`](notes/migration_plan.md) — grid2op → pure-pypowsybl
  migration strategy.
- [`notes/code-quality-analysis.md`](notes/code-quality-analysis.md) —
  static-analysis snapshot and cleanup log.

## Release notes (`release-notes/`)
Per-version notes (v0.2.2 → v0.2.5). The canonical, continuously updated history
is [CHANGELOG.md](../CHANGELOG.md).

## Deployment & backend docs (co-located with code)
These live next to the code they document:
- [`../deploy/huggingface/README.md`](../deploy/huggingface/README.md) ·
  [`../deploy/huggingface/SETUP.md`](../deploy/huggingface/SETUP.md) —
  HuggingFace Space deployment of the maneuver IHM.
- [`../expert_op4grid_recommender/pypowsybl_backend/README.md`](../expert_op4grid_recommender/pypowsybl_backend/README.md) —
  pure-pypowsybl backend quick-start & API.
- [`../expert_op4grid_recommender/pypowsybl_backend/methodologie_developpement_backend_pypowsybl.md`](../expert_op4grid_recommender/pypowsybl_backend/methodologie_developpement_backend_pypowsybl.md) —
  backend development & validation methodology.
- [`../tests/README_CONFIG.md`](../tests/README_CONFIG.md) — test configuration
  and the `conftest.py` override mechanism (incl. skipping visualization &
  troubleshooting).
