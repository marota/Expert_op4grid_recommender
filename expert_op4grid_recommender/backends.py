# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Simulation-backend abstraction for the analysis pipeline.

Historically ``main.py`` carried two parallel families of thin delegation
wrappers (``*_grid2op`` / ``*_pypowsybl``), selected them into eight function
pointers stashed in the context dict, forked on ``if is_pypowsybl:`` at every
call site, and monkey-patched private methods of the live ``ActionDiscoverer``
to route the pypowsybl shared-baseline path. This module replaces all of that
with a single :class:`SimulationBackend` protocol and two implementations that
hold ``fast_mode`` as constructor state:

- :class:`Grid2opBackend`   — the grid2op / lightsim2grid backend
- :class:`PypowsyblBackend` — the pure pypowsybl backend

Every backend-specific operation the pipeline needs is a method here, so the
pipeline calls ``ctx.backend.<op>(...)`` uniformly and the discovery pass is
configured from the backend's flags instead of being monkey-patched.

The grid2op-dependent imports stay deferred inside the ``Grid2opBackend``
methods (mirroring the previous wrappers) so importing this module never
requires grid2op — the package keeps grid2op fully optional.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Tuple


class Backend(Enum):
    """Enumeration of available simulation backends."""
    GRID2OP = "grid2op"
    PYPOWSYBL = "pypowsybl"


class SimulationBackend(ABC):
    """Protocol for the backend operations the analysis pipeline needs.

    A backend bundles the nine physics operations (environment setup, first
    observation, DC switch, contingency simulation, overload check, default
    action, per-candidate rho check, baseline computation, overflow-graph
    build) plus the candidate rho-check-with-baseline used by the shared
    discovery baseline. ``fast_mode`` is held as constructor state so call
    sites never thread it through, and the discovery-configuration flags
    (:attr:`branch_candidates_from_baseline`,
    :attr:`use_shared_baseline_for_topological`) let the discoverer be
    parameterised instead of monkey-patched.
    """

    #: Whether this is the pure-pypowsybl backend.
    is_pypowsybl: bool = False
    #: Candidates are simulated on top of the contingency-applied baseline
    #: observation rather than re-applying the contingency from the N-state.
    branch_candidates_from_baseline: bool = False
    #: Route the topological discovery passes through the single shared
    #: baseline (``compute_baseline`` once + ``check_rho_with_baseline`` per
    #: candidate) instead of the per-candidate ``check_rho_reduction`` that
    #: recomputes the baseline load flow for every candidate.
    use_shared_baseline_for_topological: bool = False

    def __init__(self, fast_mode: bool = False):
        #: pypowsybl fast mode (no voltage control). Ignored by grid2op.
        self.fast_mode = fast_mode

    # --- environment lifecycle --------------------------------------------
    @abstractmethod
    def setup_environment(self, analysis_date: Optional[datetime]) -> Tuple:
        """Load the environment + first observation + metadata (8-tuple)."""

    @abstractmethod
    def get_network(self, env: Any) -> Any:
        """Return the underlying pypowsybl ``Network`` for ``env``."""

    @abstractmethod
    def get_env_first_obs(self, env_folder, env_name, use_evaluation_config,
                          date, is_DC):
        """Reload a fresh ``(env, obs, path_chronic)`` in the given LF mode."""

    @abstractmethod
    def switch_to_dc(self, env, analysis_date, current_timestep,
                     current_lines_defaut, lines_overloaded_ids_kept,
                     maintenance_to_reco_at_t):
        """Re-run the contingency under DC; returns ``(env, obs, obs_defaut)``."""

    # --- simulation -------------------------------------------------------
    @abstractmethod
    def simulate_contingency(self, env, obs, lines_defaut,
                             act_reco_maintenance, timestep) -> Tuple[Any, bool]:
        """Simulate the N-1 contingency (+ maintenance). ``(obs_simu, converged)``."""

    @abstractmethod
    def check_simu_overloads(self, obs, obs_defaut, action_space, timestep,
                             lines_defaut, lines_overloaded_ids,
                             maintenance_to_reco_at_t) -> Tuple[bool, bool]:
        """Disconnect all kept overloads; returns ``(converged, lost_load)``."""

    @abstractmethod
    def create_default_action(self, action_space, defauts) -> Any:
        """Build the action that disconnects the contingency lines."""

    @abstractmethod
    def check_rho_reduction(self, obs, timestep, act_defaut, action,
                            overload_ids, act_reco_maintenance,
                            lines_we_care_about=None) -> Tuple[bool, Optional[Any]]:
        """Per-candidate rho-reduction check (recomputes the baseline)."""

    @abstractmethod
    def compute_baseline(self, obs, timestep, act_defaut, act_reco_maintenance,
                         overload_ids) -> Tuple[Optional[Any], Optional[Any]]:
        """Compute the shared baseline once. ``(baseline_rho, obs_baseline)``."""

    @abstractmethod
    def check_rho_with_baseline(self, obs, timestep, act_defaut, action,
                                overload_ids, act_reco_maintenance, baseline_rho,
                                lines_we_care_about=None) -> Tuple[bool, Optional[Any]]:
        """Candidate rho check against a pre-computed baseline."""

    @abstractmethod
    def build_overflow_graph(self, env, obs_simu_defaut, lines_overloaded_ids_kept,
                             non_connected_reconnectable_lines, lines_non_reconnectable,
                             timestep, do_consolidate_graph, use_dc=False,
                             extra_lines_to_cut_ids=None):
        """Build the overflow graph (6-tuple)."""


class Grid2opBackend(SimulationBackend):
    """Grid2Op / lightsim2grid backend.

    ``fast_mode`` is inert here (grid2op has no fast-mode knob); candidates
    are re-simulated from the healthy N-state because the grid2op simulation
    helpers re-apply the contingency themselves.
    """

    is_pypowsybl = False
    branch_candidates_from_baseline = False
    use_shared_baseline_for_topological = False

    def setup_environment(self, analysis_date):
        from expert_op4grid_recommender.environment import setup_environment_configs
        return setup_environment_configs(analysis_date)

    def get_network(self, env):
        return env.backend._grid.network

    def get_env_first_obs(self, env_folder, env_name, use_evaluation_config,
                          date, is_DC):
        from expert_op4grid_recommender.environment import get_env_first_obs
        return get_env_first_obs(env_folder, env_name, use_evaluation_config,
                                 date, is_DC)

    def switch_to_dc(self, env, analysis_date, current_timestep,
                     current_lines_defaut, lines_overloaded_ids_kept,
                     maintenance_to_reco_at_t):
        from expert_op4grid_recommender.environment import switch_to_dc_load_flow
        return switch_to_dc_load_flow(env, analysis_date, current_timestep,
                                      current_lines_defaut, lines_overloaded_ids_kept,
                                      maintenance_to_reco_at_t)

    def simulate_contingency(self, env, obs, lines_defaut, act_reco_maintenance,
                             timestep):
        from expert_op4grid_recommender.utils.simulation import simulate_contingency
        return simulate_contingency(env, obs, lines_defaut, act_reco_maintenance,
                                    timestep)

    def check_simu_overloads(self, obs, obs_defaut, action_space, timestep,
                             lines_defaut, lines_overloaded_ids,
                             maintenance_to_reco_at_t):
        from expert_op4grid_recommender.utils.simulation import check_simu_overloads
        return check_simu_overloads(obs, obs_defaut, action_space, timestep,
                                    lines_defaut, lines_overloaded_ids,
                                    maintenance_to_reco_at_t)

    def create_default_action(self, action_space, defauts):
        from expert_op4grid_recommender.utils.simulation import create_default_action
        return create_default_action(action_space, defauts)

    def check_rho_reduction(self, obs, timestep, act_defaut, action, overload_ids,
                            act_reco_maintenance, lines_we_care_about=None):
        from expert_op4grid_recommender.utils.simulation import check_rho_reduction
        return check_rho_reduction(obs, timestep, act_defaut, action, overload_ids,
                                   act_reco_maintenance, lines_we_care_about)

    def compute_baseline(self, obs, timestep, act_defaut, act_reco_maintenance,
                         overload_ids):
        from expert_op4grid_recommender.utils.simulation import compute_baseline_simulation
        return compute_baseline_simulation(obs, timestep, act_defaut,
                                           act_reco_maintenance, overload_ids)

    def check_rho_with_baseline(self, obs, timestep, act_defaut, action,
                                overload_ids, act_reco_maintenance, baseline_rho,
                                lines_we_care_about=None):
        from expert_op4grid_recommender.utils.simulation import (
            check_rho_reduction_with_baseline,
        )
        return check_rho_reduction_with_baseline(
            obs, timestep, act_defaut, action, overload_ids, act_reco_maintenance,
            baseline_rho, lines_we_care_about,
        )

    def build_overflow_graph(self, env, obs_simu_defaut, lines_overloaded_ids_kept,
                             non_connected_reconnectable_lines, lines_non_reconnectable,
                             timestep, do_consolidate_graph, use_dc=False,
                             extra_lines_to_cut_ids=None):
        from expert_op4grid_recommender.graph_analysis.builder import build_overflow_graph
        return build_overflow_graph(
            env, obs_simu_defaut, lines_overloaded_ids_kept,
            non_connected_reconnectable_lines, lines_non_reconnectable,
            timestep, do_consolidate_graph=do_consolidate_graph, use_dc=use_dc,
            extra_lines_to_cut_ids=extra_lines_to_cut_ids,
        )


class PypowsyblBackend(SimulationBackend):
    """Pure pypowsybl backend.

    Candidates branch from the contingency-applied kept variant
    (``obs_baseline``), so :attr:`branch_candidates_from_baseline` is set, and
    the topological passes share the single cached baseline
    (:attr:`use_shared_baseline_for_topological`) — one baseline load flow per
    run instead of one per candidate. ``fast_mode`` is threaded into every
    pypowsybl simulation call from constructor state.
    """

    is_pypowsybl = True
    branch_candidates_from_baseline = True
    use_shared_baseline_for_topological = True

    def setup_environment(self, analysis_date):
        from expert_op4grid_recommender.environment_pypowsybl import (
            setup_environment_configs_pypowsybl,
        )
        return setup_environment_configs_pypowsybl(analysis_date)

    def get_network(self, env):
        return env.network_manager.network

    def get_env_first_obs(self, env_folder, env_name, use_evaluation_config,
                          date, is_DC):
        from expert_op4grid_recommender.environment_pypowsybl import (
            get_env_first_obs_pypowsybl,
        )
        return get_env_first_obs_pypowsybl(env_folder, env_name, is_DC=is_DC)

    def switch_to_dc(self, env, analysis_date, current_timestep,
                     current_lines_defaut, lines_overloaded_ids_kept,
                     maintenance_to_reco_at_t):
        from expert_op4grid_recommender.environment_pypowsybl import (
            switch_to_dc_load_flow_pypowsybl,
        )
        return switch_to_dc_load_flow_pypowsybl(
            env, current_lines_defaut, lines_overloaded_ids_kept,
            maintenance_to_reco_at_t,
        )

    # pypowsybl-only ``obs.simulate`` kwargs. The baseline / contingency calls
    # retain their variant (``keep_variant``) so candidates can branch from it;
    # the per-candidate calls only need ``fast_mode``.
    @property
    def _sk_kept(self):
        return {"keep_variant": True, "fast_mode": self.fast_mode}

    @property
    def _sk(self):
        return {"fast_mode": self.fast_mode}

    def simulate_contingency(self, env, obs, lines_defaut, act_reco_maintenance,
                             timestep):
        from expert_op4grid_recommender.utils.simulation import simulate_contingency
        return simulate_contingency(env, obs, lines_defaut, act_reco_maintenance,
                                    timestep, simulate_kwargs=self._sk_kept)

    def check_simu_overloads(self, obs, obs_defaut, action_space, timestep,
                             lines_defaut, lines_overloaded_ids,
                             maintenance_to_reco_at_t):
        from expert_op4grid_recommender.utils.simulation import check_simu_overloads
        return check_simu_overloads(obs, obs_defaut, action_space, timestep,
                                    lines_defaut, lines_overloaded_ids,
                                    maintenance_to_reco_at_t, simulate_kwargs=self._sk)

    def create_default_action(self, action_space, defauts):
        from expert_op4grid_recommender.utils.simulation import create_default_action
        return create_default_action(action_space, defauts)

    def check_rho_reduction(self, obs, timestep, act_defaut, action, overload_ids,
                            act_reco_maintenance, lines_we_care_about=None):
        from expert_op4grid_recommender.utils.simulation import check_rho_reduction
        # Branch candidates from the contingency-applied kept variant (no
        # re-application of the contingency) — see BaselineContext / C-diag.
        return check_rho_reduction(obs, timestep, act_defaut, action, overload_ids,
                                   act_reco_maintenance, lines_we_care_about,
                                   reapply_contingency=False,
                                   baseline_simulate_kwargs=self._sk_kept,
                                   candidate_simulate_kwargs=self._sk)

    def compute_baseline(self, obs, timestep, act_defaut, act_reco_maintenance,
                         overload_ids):
        from expert_op4grid_recommender.utils.simulation import compute_baseline_simulation
        return compute_baseline_simulation(obs, timestep, act_defaut,
                                           act_reco_maintenance, overload_ids,
                                           simulate_kwargs=self._sk_kept)

    def check_rho_with_baseline(self, obs, timestep, act_defaut, action,
                                overload_ids, act_reco_maintenance, baseline_rho,
                                lines_we_care_about=None):
        from expert_op4grid_recommender.utils.simulation import (
            check_rho_reduction_with_baseline,
        )
        return check_rho_reduction_with_baseline(
            obs, timestep, act_defaut, action, overload_ids, act_reco_maintenance,
            baseline_rho, lines_we_care_about,
            reapply_contingency=False, simulate_kwargs=self._sk,
        )

    def build_overflow_graph(self, env, obs_simu_defaut, lines_overloaded_ids_kept,
                             non_connected_reconnectable_lines, lines_non_reconnectable,
                             timestep, do_consolidate_graph, use_dc=False,
                             extra_lines_to_cut_ids=None):
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import (
            build_overflow_graph_pypowsybl,
        )
        return build_overflow_graph_pypowsybl(
            env, obs_simu_defaut, lines_overloaded_ids_kept,
            non_connected_reconnectable_lines, lines_non_reconnectable,
            timestep, do_consolidate_graph=do_consolidate_graph, use_dc=use_dc,
            param_options={"fast_mode": self.fast_mode},
            extra_lines_to_cut_ids=extra_lines_to_cut_ids,
        )


def make_backend(backend: Backend, fast_mode: bool = False) -> SimulationBackend:
    """Instantiate the :class:`SimulationBackend` for ``backend``."""
    if backend == Backend.GRID2OP:
        return Grid2opBackend(fast_mode=fast_mode)
    if backend == Backend.PYPOWSYBL:
        return PypowsyblBackend(fast_mode=fast_mode)
    raise ValueError(f"Unknown backend: {backend}")
