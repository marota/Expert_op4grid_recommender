# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# SPDX-License-Identifier: MPL-2.0
"""R4 — unified simulation seam.

Covers the two structural pieces introduced by revision R4:

- :class:`BaselineContext` — the shared per-run baseline (holds the contingency
  action, baseline rho, the kept baseline observation, an explicit branch
  observation, and a ``release()``), and the fact that
  ``check_rho_reduction_with_baseline`` now takes the baseline/branch observation
  and the backend contract (``reapply_contingency``) explicitly instead of via
  two forked modules with opposite first-argument contracts.
- The ``NetworkManager`` kept-variant registry + LRU backstop (review C4).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from expert_op4grid_recommender.utils.simulation import (
    BaselineContext,
    check_rho_reduction_with_baseline,
)


# ---------------------------------------------------------------------------
# BaselineContext
# ---------------------------------------------------------------------------

def test_baseline_context_unpacks_like_the_old_tuple():
    ctx = BaselineContext(act_defaut="ACT", baseline_rho=np.array([0.9]),
                          obs_baseline="OBS_BASE", branch_obs="BRANCH")
    act_defaut, baseline_rho, branch_obs = ctx
    assert act_defaut == "ACT"
    assert branch_obs == "BRANCH"
    assert baseline_rho[0] == 0.9


def test_baseline_context_branch_obs_defaults_to_obs_baseline():
    ctx = BaselineContext(act_defaut="ACT", baseline_rho=None, obs_baseline="OBS_BASE")
    assert ctx.branch_obs == "OBS_BASE"


def test_baseline_context_release_removes_pypowsybl_variant():
    nm = MagicMock()
    obs = MagicMock()
    obs._variant_id = "simulate_kept_1_0"
    obs._network_manager = nm
    ctx = BaselineContext(act_defaut="ACT", baseline_rho=None, obs_baseline=obs)

    ctx.release()

    nm.remove_variant.assert_called_once_with("simulate_kept_1_0")
    assert ctx.obs_baseline is None
    assert ctx.branch_obs is None


def test_baseline_context_release_is_noop_without_variant():
    # A grid2op observation has no _variant_id / _network_manager — release() is
    # a no-op and must not raise.
    ctx = BaselineContext(act_defaut="ACT", baseline_rho=None, obs_baseline=object())
    ctx.release()  # should not raise
    assert ctx.obs_baseline is None


class _RecordingObs:
    """Minimal observation whose simulate records the action it was handed."""

    def __init__(self, rho):
        self.rho = np.asarray(rho)
        self.name_line = np.array([f"L{i}" for i in range(len(self.rho))])
        self.last_action = None

    def simulate(self, action, time_step=0, **kwargs):
        self.last_action = action
        info = {"exception": []}
        return self, 0.0, False, info


def test_check_with_baseline_reapplies_contingency_when_requested():
    """grid2op contract: candidate is simulated as action + act_defaut + reco."""
    obs = _RecordingObs([0.5, 0.5])
    ok, _ = check_rho_reduction_with_baseline(
        obs, 0, act_defaut="D", action="A", overload_ids=[0, 1],
        act_reco_maintenance="M", baseline_rho=np.array([0.9, 0.9]),
        reapply_contingency=True,
    )
    assert ok is True                      # 0.5 < 0.9 for all overloads
    assert obs.last_action == "A" + "D" + "M"  # string concat stands in for action add


def test_check_with_baseline_candidate_only_when_not_reapplying():
    """pypowsybl contract: only the candidate action is simulated (branch obs is
    already contingency-applied)."""
    obs = _RecordingObs([0.5, 0.5])
    ok, _ = check_rho_reduction_with_baseline(
        obs, 0, act_defaut="D", action="A", overload_ids=[0, 1],
        act_reco_maintenance="M", baseline_rho=np.array([0.9, 0.9]),
        reapply_contingency=False,
    )
    assert ok is True
    assert obs.last_action == "A"           # only the candidate


def test_check_with_baseline_forwards_simulate_kwargs():
    captured = {}

    class _KwObs(_RecordingObs):
        def simulate(self, action, time_step=0, **kwargs):
            captured.update(kwargs)
            return super().simulate(action, time_step=time_step)

    obs = _KwObs([0.5])
    check_rho_reduction_with_baseline(
        obs, 0, "D", "A", [0], "M", np.array([0.9]),
        reapply_contingency=False, simulate_kwargs={"fast_mode": True},
    )
    assert captured == {"fast_mode": True}


# ---------------------------------------------------------------------------
# NetworkManager kept-variant registry + LRU backstop (review C4)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_network_manager():
    pp = pytest.importorskip("pypowsybl")
    from expert_op4grid_recommender.pypowsybl_backend.network_manager import NetworkManager

    net = pp.network.create_ieee14()
    return NetworkManager(network=net, max_kept_variants=2)


def test_register_kept_variant_tracks_and_evicts_lru(small_network_manager):
    nm = small_network_manager
    for vid in ("v1", "v2"):
        nm.create_variant(vid)
        nm.register_kept_variant(vid)

    assert list(nm._kept_variants.keys()) == ["v1", "v2"]

    # Third registration exceeds max_kept_variants=2 → evict the oldest (v1).
    nm.create_variant("v3")
    nm.register_kept_variant("v3")

    assert "v1" not in nm._kept_variants
    assert list(nm._kept_variants.keys()) == ["v2", "v3"]
    assert "v1" not in nm.network.get_variant_ids()   # actually removed
    assert "v3" in nm.network.get_variant_ids()


def test_register_kept_variant_never_evicts_base_or_working(small_network_manager):
    nm = small_network_manager
    # Make a kept variant the working one; it must survive eviction pressure.
    nm.create_variant("keep_me")
    nm.set_working_variant("keep_me")
    nm.register_kept_variant("keep_me")
    for vid in ("a", "b", "c", "d"):
        nm.create_variant(vid)
        nm.register_kept_variant(vid)

    assert "keep_me" in nm.network.get_variant_ids()   # working variant preserved
    assert nm.base_variant_id in nm.network.get_variant_ids()


def test_sweep_kept_variants_clears_registry(small_network_manager):
    nm = small_network_manager
    nm.set_working_variant(nm.base_variant_id)
    for vid in ("s1", "s2"):
        nm.create_variant(vid)
        nm.register_kept_variant(vid)

    nm.sweep_kept_variants()

    assert len(nm._kept_variants) == 0
    assert "s1" not in nm.network.get_variant_ids()
    assert "s2" not in nm.network.get_variant_ids()
    assert nm.base_variant_id in nm.network.get_variant_ids()


def test_remove_variant_deregisters(small_network_manager):
    nm = small_network_manager
    nm.create_variant("r1")
    nm.register_kept_variant("r1")
    assert "r1" in nm._kept_variants

    nm.remove_variant("r1")
    assert "r1" not in nm._kept_variants
    assert "r1" not in nm.network.get_variant_ids()


# ---------------------------------------------------------------------------
# check_rho_reduction composing function — explicit backend contract
# ---------------------------------------------------------------------------

class _TwoPhaseObs:
    """Observation returning a baseline rho on the first simulate call and a
    candidate rho on the second, recording every action it was handed. It
    returns ``self`` so the candidate branch (whether ``obs`` or the kept
    ``obs_baseline``) resolves to the same recorder."""

    def __init__(self, baseline_rho, candidate_rho):
        self._rhos = [np.asarray(baseline_rho), np.asarray(candidate_rho)]
        self._i = 0
        self.actions = []
        self.name_line = np.array([f"L{i}" for i in range(len(self._rhos[0]))])
        self.rho = self._rhos[0]

    def simulate(self, action, time_step=0, **kwargs):
        self.actions.append(action)
        self.rho = self._rhos[min(self._i, len(self._rhos) - 1)]
        self._i += 1
        return self, 0.0, False, {"exception": []}


def test_check_rho_reduction_grid2op_contract_two_sims_from_n_state():
    from expert_op4grid_recommender.utils.simulation import check_rho_reduction
    obs = _TwoPhaseObs(baseline_rho=[0.95, 0.9], candidate_rho=[0.8, 0.75])
    ok, _ = check_rho_reduction(
        obs, 0, act_defaut="D", action="A", overload_ids=[0, 1],
        act_reco_maintenance="M",
    )  # grid2op defaults: reapply_contingency=True
    assert ok is True
    # 2 sim calls: baseline (D+M) then candidate (A+D+M) — both from the N-state.
    assert obs.actions == ["D" + "M", "A" + "D" + "M"]


def test_check_rho_reduction_pypowsybl_contract_candidate_from_baseline():
    from expert_op4grid_recommender.utils.simulation import check_rho_reduction
    obs = _TwoPhaseObs(baseline_rho=[0.95, 0.9], candidate_rho=[0.8, 0.75])
    ok, _ = check_rho_reduction(
        obs, 0, act_defaut="D", action="A", overload_ids=[0, 1],
        act_reco_maintenance="M", reapply_contingency=False,
    )
    assert ok is True
    # baseline (D+M) then candidate (A alone, branched from the kept baseline obs).
    assert obs.actions == ["D" + "M", "A"]


def test_check_rho_reduction_returns_false_none_on_baseline_failure():
    from expert_op4grid_recommender.utils.simulation import check_rho_reduction

    class _FailObs:
        name_line = np.array(["L0"])
        rho = np.array([0.5])

        def simulate(self, action, time_step=0, **kwargs):
            return self, 0.0, True, {"exception": [Exception("diverged")]}

    ok, obs_after = check_rho_reduction(
        _FailObs(), 0, "D", "A", [0], "M",
    )
    assert ok is False and obs_after is None


def test_simulate_contingency_and_check_simu_overloads_forward_simulate_kwargs():
    from expert_op4grid_recommender.utils.simulation import (
        simulate_contingency, check_simu_overloads,
    )
    captured = []

    class _Obs:
        name_line = np.array(["L0", "L1"])
        load_p = np.array([10.0])

        def simulate(self, action, time_step=0, **kwargs):
            captured.append(kwargs)
            return self, 0.0, False, {"exception": []}

    class _Env:
        def action_space(self, d):
            return "ACT"

    env = _Env()
    simulate_contingency(env, _Obs(), ["L0"], "M", 0,
                         simulate_kwargs={"keep_variant": True, "fast_mode": True})
    check_simu_overloads(_Obs(), _Obs(), env.action_space, 0, ["L0"], [0], [],
                         simulate_kwargs={"fast_mode": True})
    assert captured[0] == {"keep_variant": True, "fast_mode": True}
    assert captured[1] == {"fast_mode": True}


# ---------------------------------------------------------------------------
# Discovery baseline release lifecycle
# ---------------------------------------------------------------------------

def _bare_discoverer():
    from expert_op4grid_recommender.action_evaluation.discovery import ActionDiscoverer
    d = ActionDiscoverer.__new__(ActionDiscoverer)
    d._cached_simulation_baseline = None
    return d


def test_release_simulation_baseline_is_noop_when_none_built():
    d = _bare_discoverer()
    d._release_simulation_baseline()  # must not raise
    assert d._cached_simulation_baseline is None


def test_release_simulation_baseline_releases_and_clears():
    d = _bare_discoverer()
    nm = MagicMock()
    obs = MagicMock()
    obs._variant_id = "simulate_kept_9_0"
    obs._network_manager = nm
    d._cached_simulation_baseline = BaselineContext(
        act_defaut="A", baseline_rho=np.array([0.9]), obs_baseline=obs,
    )
    d._release_simulation_baseline()
    nm.remove_variant.assert_called_once_with("simulate_kept_9_0")
    assert d._cached_simulation_baseline is None


# ---------------------------------------------------------------------------
# Real network: simulate(keep_variant=True) registers, release() frees
# ---------------------------------------------------------------------------

def test_real_kept_variant_is_registered_and_released_via_baseline_context():
    pp = pytest.importorskip("pypowsybl")
    from expert_op4grid_recommender.pypowsybl_backend import (
        NetworkManager, ActionSpace, PypowsyblObservation,
    )

    nm = NetworkManager(network=pp.network.create_ieee9())
    action_space = ActionSpace(nm)
    nm.run_load_flow()
    obs = PypowsyblObservation(nm, action_space)

    first_line = nm.name_line[0]
    action = action_space({"set_line_status": [(first_line, -1)]})

    n_variants_before = len(nm.network.get_variant_ids())
    obs_kept, _, _, info = obs.simulate(action, keep_variant=True)
    assert not info["exception"]

    # The retained variant is registered and actually present on the network.
    assert obs_kept._variant_id in nm._kept_variants
    assert obs_kept._variant_id in nm.network.get_variant_ids()
    assert len(nm.network.get_variant_ids()) == n_variants_before + 1

    # BaselineContext.release() frees it, restoring the variant count.
    ctx = BaselineContext(act_defaut=action, baseline_rho=None, obs_baseline=obs_kept)
    ctx.release()
    assert obs_kept._variant_id not in nm._kept_variants
    assert obs_kept._variant_id not in nm.network.get_variant_ids()
    assert len(nm.network.get_variant_ids()) == n_variants_before


def test_real_transient_variant_is_not_registered():
    pp = pytest.importorskip("pypowsybl")
    from expert_op4grid_recommender.pypowsybl_backend import (
        NetworkManager, ActionSpace, PypowsyblObservation,
    )

    nm = NetworkManager(network=pp.network.create_ieee9())
    action_space = ActionSpace(nm)
    nm.run_load_flow()
    obs = PypowsyblObservation(nm, action_space)

    n_variants_before = len(nm.network.get_variant_ids())
    action = action_space({"set_line_status": [(nm.name_line[0], -1)]})
    obs.simulate(action, keep_variant=False)  # transient — dropped in finally

    assert len(nm._kept_variants) == 0
    assert len(nm.network.get_variant_ids()) == n_variants_before
