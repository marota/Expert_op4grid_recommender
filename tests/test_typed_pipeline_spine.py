# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Contract tests for the typed pipeline spine (R1) + main.py split (R2).

Locks in the invariants the refactor promised:

* ``AnalysisContext`` / ``AnalysisResult`` / ``SimulatedAction`` are typed
  dataclasses that still behave like the dicts they replaced
  (``DictCompatMixin`` — item access, ``.get``/``.pop``/``.update``/``in``,
  iteration, and dynamic keys).
* ``SimulationBackend`` exposes the right per-backend flags, and ``make_backend``
  builds the right implementation with ``fast_mode`` as constructor state.
* ``ActionDiscoverer`` routes the topological candidate rho-check through the
  shared baseline exactly when the backend asks it to (replacing the old
  ``main.py`` monkey-patch) — and per-candidate otherwise.
* The three import cycles are dissolved: ``models`` / ``utils.reassessment`` no
  longer import the pipeline entry point at module load.
* The ``main`` facade keeps re-exporting the whole historical surface.
"""
from __future__ import annotations

import ast
import pathlib
from unittest.mock import MagicMock

import numpy as np
import pytest

from expert_op4grid_recommender.backends import (
    Backend,
    Grid2opBackend,
    PypowsyblBackend,
    SimulationBackend,
    make_backend,
)
from expert_op4grid_recommender.pipeline import AnalysisContext, AnalysisResult
from expert_op4grid_recommender.models.base import DictCompatMixin, SimulatedAction


# ---------------------------------------------------------------------------
# DictCompatMixin — the dataclasses must be drop-in for the old dicts
# ---------------------------------------------------------------------------

class TestDictCompat:
    def test_item_access_and_attribute_access_are_the_same_slot(self):
        ctx = AnalysisContext(env="E", obs="O")
        assert ctx["env"] == "E" == ctx.env
        ctx["env"] = "E2"
        assert ctx.env == "E2"
        ctx.obs = "O2"
        assert ctx["obs"] == "O2"

    def test_get_contains_and_missing(self):
        ctx = AnalysisContext(env="E")
        assert "env" in ctx
        assert ctx.get("env") == "E"
        assert ctx.get("nope") is None
        assert ctx.get("nope", 7) == 7
        assert "nope" not in ctx

    def test_dynamic_keys_not_declared_as_fields(self):
        # Co-Study adds undeclared keys (e.g. ``lines_overloaded``).
        ctx = AnalysisContext()
        ctx["lines_overloaded"] = [1, 2]
        assert ctx["lines_overloaded"] == [1, 2]
        assert ctx.get("lines_overloaded") == [1, 2]

    def test_pop_update_setdefault(self):
        res = AnalysisResult(lines_overloaded_names=["L1"])
        assert res.pop("prediction_time", "d") is None       # declared, default None
        assert "prediction_time" not in res
        assert res.pop("prediction_time", "d") == "d"         # now genuinely gone
        with pytest.raises(KeyError):
            res.pop("prediction_time")
        res.update({"combined_actions": {"a+b": {}}})
        assert res["combined_actions"] == {"a+b": {}}
        assert res.setdefault("action_scores", "x") == {}     # already present
        assert res.setdefault("brand_new", "x") == "x"

    def test_iteration_and_keys_items_values(self):
        ctx = AnalysisContext(env="E", obs="O")
        keys = set(ctx.keys())
        assert "env" in keys and "obs" in keys
        assert dict(ctx.items())["env"] == "E"
        assert "E" in list(ctx.values())
        assert set(iter(ctx)) == keys

    def test_simulated_action_is_dict_compatible(self):
        sa = SimulatedAction(
            action="a", description_unitaire="d",
            rho_before=[0.5], rho_after=[0.4], max_rho=0.4,
            max_rho_line="L1", is_rho_reduction=True, observation="o",
        )
        assert isinstance(sa, DictCompatMixin)
        assert sa["rho_before"] == [0.5]
        assert sa.get("observation") == "o"
        assert sa.get("lines_overloaded_after") is None       # never set → None
        assert sa["non_convergence"] is None

    def test_no_array_equality_landmine(self):
        # eq=False keeps object identity, so a numpy-array field never triggers
        # "ambiguous truth value" on ==.
        a = AnalysisContext(obs=np.array([1, 2, 3]))
        b = AnalysisContext(obs=np.array([1, 2, 3]))
        assert a != b          # identity, not element-wise
        assert a == a


# ---------------------------------------------------------------------------
# SimulationBackend protocol + make_backend
# ---------------------------------------------------------------------------

class TestBackends:
    def test_grid2op_backend_flags(self):
        b = make_backend(Backend.GRID2OP, fast_mode=True)
        assert isinstance(b, Grid2opBackend) and isinstance(b, SimulationBackend)
        assert b.is_pypowsybl is False
        assert b.branch_candidates_from_baseline is False
        assert b.use_shared_baseline_for_topological is False
        # fast_mode is inert for grid2op but still stored.
        assert b.fast_mode is True

    def test_pypowsybl_backend_flags_and_fast_mode(self):
        b = make_backend(Backend.PYPOWSYBL, fast_mode=True)
        assert isinstance(b, PypowsyblBackend)
        assert b.is_pypowsybl is True
        assert b.branch_candidates_from_baseline is True
        assert b.use_shared_baseline_for_topological is True
        assert b.fast_mode is True
        assert make_backend(Backend.PYPOWSYBL, fast_mode=False).fast_mode is False

    def test_make_backend_rejects_unknown(self):
        with pytest.raises(ValueError):
            make_backend("not-a-backend")

    def test_backend_exposes_every_pipeline_operation(self):
        for name in (
            "setup_environment", "get_network", "get_env_first_obs", "switch_to_dc",
            "simulate_contingency", "check_simu_overloads", "create_default_action",
            "check_rho_reduction", "compute_baseline", "check_rho_with_baseline",
            "build_overflow_graph",
        ):
            assert callable(getattr(make_backend(Backend.PYPOWSYBL), name))

    def test_pypowsybl_get_network_reads_network_manager(self):
        class _Env:
            class network_manager:  # noqa: N801
                network = "THE_NET"
        assert make_backend(Backend.PYPOWSYBL).get_network(_Env()) == "THE_NET"

    def test_grid2op_get_network_reads_backend_grid(self):
        class _Grid:
            network = "G_NET"
        class _Backend:
            _grid = _Grid()
        class _Env:
            backend = _Backend()
        assert make_backend(Backend.GRID2OP).get_network(_Env()) == "G_NET"


# ---------------------------------------------------------------------------
# Discovery shared-baseline routing (replaces the main.py monkey-patch)
# ---------------------------------------------------------------------------

def _make_discoverer(**overrides):
    """Bare ActionDiscoverer with __init__ patched — we only exercise the
    baseline-routing wiring set up from the constructor flags."""
    from unittest.mock import MagicMock

    from expert_op4grid_recommender.action_evaluation.discovery import ActionDiscoverer
    from expert_op4grid_recommender.action_evaluation.discovery._base import DiscovererBase

    d = ActionDiscoverer.__new__(ActionDiscoverer)
    # Minimal state _get_simulation_baseline / _shared_baseline_check need.
    d.action_space = MagicMock()
    d.lines_defaut = ["LX"]
    d.timestep = 0
    d.act_reco_maintenance = MagicMock()
    d.lines_overloaded_ids = [0]
    d.obs = "N_STATE_OBS"
    # Normally initialised in __init__ (bypassed here via __new__): the lazily
    # built shared BaselineContext cache.
    d._cached_simulation_baseline = None
    d._create_default_action = lambda space, defauts: "ACT_DEFAUT"
    d._compute_baseline = MagicMock(return_value=("BASELINE_RHO", "OBS_BASELINE"))
    d._check_rho_with_baseline = MagicMock(return_value=(True, "OBS_AFTER"))
    d._per_candidate_check = MagicMock(return_value=(False, "OBS_PC"))
    d._branch_candidates_from_baseline = overrides.get("branch", False)
    # Re-run the __init__ tail that wires _check_rho_reduction.
    if overrides.get("use_shared", False):
        d._check_rho_reduction = DiscovererBase._shared_baseline_check.__get__(d)
    else:
        d._check_rho_reduction = d._per_candidate_check
    return d


class TestSharedBaselineRouting:
    def test_pypowsybl_routes_through_shared_baseline(self):
        d = _make_discoverer(use_shared=True, branch=True)
        ok, obs = d._check_rho_reduction(
            "ignored_obs", 0, "ignored_act", "CANDIDATE", [0], d.act_reco_maintenance,
        )
        assert ok is True and obs == "OBS_AFTER"
        # Baseline computed once (cached); candidate branched from OBS_BASELINE.
        d._compute_baseline.assert_called_once()
        args = d._check_rho_with_baseline.call_args.args
        assert args[0] == "OBS_BASELINE"          # branch_obs, not the N-state
        assert args[3] == "CANDIDATE"
        assert args[6] == "BASELINE_RHO"
        d._per_candidate_check.assert_not_called()

    def test_grid2op_uses_per_candidate_check(self):
        d = _make_discoverer(use_shared=False, branch=False)
        ok, obs = d._check_rho_reduction(
            "N_STATE_OBS", 0, "ACT_DEFAUT", "CANDIDATE", [0], d.act_reco_maintenance,
        )
        assert ok is False and obs == "OBS_PC"
        d._per_candidate_check.assert_called_once()
        d._check_rho_with_baseline.assert_not_called()

    def test_shared_baseline_short_circuits_when_baseline_none(self):
        d = _make_discoverer(use_shared=True, branch=True)
        d._compute_baseline = MagicMock(return_value=(None, None))
        assert d._check_rho_reduction("o", 0, "a", "c", [0], d.act_reco_maintenance) == (False, None)


# ---------------------------------------------------------------------------
# R2 — import cycles dissolved + facade preserved
# ---------------------------------------------------------------------------

_PKG = pathlib.Path(__file__).resolve().parents[1] / "expert_op4grid_recommender"


def _module_level_imports(path: pathlib.Path) -> set[str]:
    """Fully-qualified names imported at MODULE level (not inside functions)."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:  # module body only → excludes deferred/in-function imports
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


class TestImportLayering:
    def test_models_expert_does_not_import_pipeline_entrypoint_at_module_level(self):
        imports = _module_level_imports(_PKG / "models" / "expert.py")
        assert not any(m.endswith((".main", ".pipeline", ".cli")) for m in imports), imports

    def test_reassessment_does_not_import_main_at_module_level(self):
        imports = _module_level_imports(_PKG / "utils" / "reassessment.py")
        assert not any(m.endswith((".main", ".pipeline", ".cli")) for m in imports), imports

    def test_expert_discovery_does_not_import_pipeline_at_module_level(self):
        imports = _module_level_imports(_PKG / "models" / "_expert_discovery.py")
        assert not any(m.endswith((".main", ".pipeline", ".cli")) for m in imports), imports

    def test_backends_module_has_no_top_level_grid2op_import(self):
        # grid2op stays optional: no grid2op import at module load in backends.py.
        imports = _module_level_imports(_PKG / "backends.py")
        assert not any("grid2op" in m or "environment" in m or "simulation" in m
                       for m in imports), imports

    def test_main_facade_reexports_the_historical_surface(self):
        import expert_op4grid_recommender.main as m
        for name in (
            "Backend", "SimulationBackend", "Grid2opBackend", "PypowsyblBackend",
            "make_backend", "AnalysisContext", "AnalysisResult", "set_thermal_limits",
            "run_analysis", "run_analysis_step1", "run_analysis_step2",
            "run_analysis_step2_graph", "run_analysis_step2_discovery",
            "simulate_contingency_grid2op", "simulate_contingency_pypowsybl",
            "_run_expert_discovery", "_run_expert_action_filter", "main",
        ):
            assert hasattr(m, name), f"main facade lost {name}"


# ---------------------------------------------------------------------------
# step1 union return (short-circuit) is routed by run_analysis
# ---------------------------------------------------------------------------

class TestStep1UnionReturn:
    def test_run_analysis_returns_early_result_without_step2(self, monkeypatch):
        """When step1 short-circuits with an AnalysisResult, run_analysis
        returns it directly and never runs step2 (was: ``if res_step1 is not
        None: return res_step1``)."""
        import expert_op4grid_recommender.pipeline as pl

        early = AnalysisResult(lines_overloaded_names=["L1"])
        monkeypatch.setattr(pl, "run_analysis_step1", lambda **kw: early)

        def _boom(*a, **k):  # step2 must NOT run on the short-circuit path
            raise AssertionError("step2 must not run after an AnalysisResult short-circuit")

        monkeypatch.setattr(pl, "run_analysis_step2_graph", _boom)
        out = pl.run_analysis(None, 0, ["LX"], backend=Backend.PYPOWSYBL)
        assert out is early
        assert out["lines_overloaded_names"] == ["L1"]

    def test_run_analysis_proceeds_to_step2_on_context(self, monkeypatch):
        import expert_op4grid_recommender.pipeline as pl

        ctx = AnalysisContext(env="E")
        monkeypatch.setattr(pl, "run_analysis_step1", lambda **kw: ctx)
        seen = {}
        monkeypatch.setattr(pl, "run_analysis_step2_graph", lambda c: seen.setdefault("g", c) or c)
        monkeypatch.setattr(pl, "run_analysis_step2_discovery",
                            lambda c: AnalysisResult(lines_overloaded_names=["done"]))
        out = pl.run_analysis(None, 0, ["LX"], backend=Backend.PYPOWSYBL)
        assert seen["g"] is ctx
        assert out["lines_overloaded_names"] == ["done"]
