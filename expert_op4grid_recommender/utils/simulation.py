# expert_op4grid_recommender/utils/simulation.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.
"""Backend-agnostic contingency / candidate-simulation helpers.

Historically this logic was duplicated across ``simulation.py`` (grid2op) and
``simulation_pypowsybl.py`` (pure pypowsybl) — two ~300-line modules that were
~85% identical and whose ``check_rho_reduction_with_baseline`` had **opposite**
first-argument contracts hidden behind identical-looking signatures (review
findings A8 / C-diag). This single module replaces the pair (revision R4). The
two backends differ in only two ways, both now expressed as explicit parameters
rather than a forked module:

- ``simulate_kwargs`` — extra keyword args forwarded to ``obs.simulate()``.
  grid2op accepts none; pypowsybl accepts ``keep_variant`` and ``fast_mode``.
- ``reapply_contingency`` — in the candidate rho-check, whether to re-apply the
  contingency. grid2op branches a candidate from the **healthy N-state** and
  re-applies ``act_defaut`` + maintenance; pypowsybl branches from the
  already-contingency-applied kept variant and applies **only** the candidate.

:class:`BaselineContext` bundles the once-per-run baseline (contingency action,
baseline rho, the kept baseline observation each candidate branches from) with a
:meth:`BaselineContext.release` that frees the retained pypowsybl variant.

The defaults reproduce the grid2op contract exactly, so existing callers and
tests that import these functions positionally are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BaselineContext:
    """Shared per-run baseline for candidate rho-reduction checks.

    Created once per discovery / reassessment run and reused for every candidate,
    so the baseline load flow runs once instead of once per candidate (review
    P1). Fields:

    - ``act_defaut``   — the contingency (line-disconnection) action.
    - ``baseline_rho`` — rho of the monitored overloads in the contingency
      (N-1) state, the reference every candidate must beat.
    - ``obs_baseline`` — the kept baseline observation (a retained pypowsybl
      variant; ``None``/plain observation for grid2op). :meth:`release` frees it.
    - ``branch_obs``   — the observation a candidate action is simulated on top
      of: the healthy N-state for grid2op (which re-applies the contingency) or
      the contingency-applied kept variant for pypowsybl. Defaults to
      ``obs_baseline``.

    Iterating a context yields ``(act_defaut, baseline_rho, branch_obs)`` so the
    discovery call sites can keep unpacking it like the previous tuple.
    """

    act_defaut: Any
    baseline_rho: Optional[np.ndarray]
    obs_baseline: Optional[Any]
    branch_obs: Any = None

    def __post_init__(self) -> None:
        if self.branch_obs is None:
            self.branch_obs = self.obs_baseline

    def __iter__(self):
        return iter((self.act_defaut, self.baseline_rho, self.branch_obs))

    def release(self) -> None:
        """Free the retained pypowsybl baseline variant (no-op for grid2op).

        The kept baseline observation holds a ``_variant_id`` on its
        ``_network_manager``; removing it bounds the per-run variant that would
        otherwise linger until the ``NetworkManager`` is discarded (review C4).
        Best-effort: a failure to remove must never abort an analysis.
        """
        variant_id = getattr(self.obs_baseline, "_variant_id", None)
        nm = getattr(self.obs_baseline, "_network_manager", None)
        if variant_id is not None and nm is not None:
            try:
                nm.remove_variant(variant_id)
            except Exception as exc:  # best-effort cleanup
                print(f"Warning: could not release baseline variant {variant_id}: {exc}")
        self.obs_baseline = None
        self.branch_obs = None


def create_default_action(action_space: Callable, defauts: List[str]) -> Any:
    """
    Creates an action object that disconnects a specified list of lines.

    This is typically used to represent the initial N-1 contingency by setting
    both the origin and extremity buses of the fault lines (`defauts`) to -1.

    Args:
        action_space (Callable): The action space object, used to create
                                 the action from a dictionary definition.
        defauts (List[str]): A list of line names to be disconnected in the action.

    Returns:
        Any: The action object representing the disconnection of the specified lines.
    """
    return action_space({
        "set_bus": {
            # Set extremity bus to -1 (disconnected) for each default line
            "lines_ex_id": {defaut: -1 for defaut in defauts},
            # Set origin bus to -1 (disconnected) for each default line
            "lines_or_id": {defaut: -1 for defaut in defauts}
        }
    })


def simulate_contingency(env: Any, obs: Any, lines_defaut: List[str],
                         act_reco_maintenance: Any, timestep: int,
                         simulate_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Any, bool]:
    """
    Simulates the application of an initial N-1 contingency and any maintenance reconnections.

    Creates an action to disconnect the contingency lines (`lines_defaut`),
    combines it with any provided maintenance reconnection action
    (`act_reco_maintenance`), and simulates the result via ``obs.simulate``.

    Args:
        env: The environment object, providing ``env.action_space``.
        obs: The initial observation object *before* the contingency.
        lines_defaut: Line names representing the N-1 contingency to simulate.
        act_reco_maintenance: Action reconnecting lines from maintenance
                              simultaneously with the contingency.
        timestep: The simulation timestep index.
        simulate_kwargs: Extra keyword args forwarded to ``obs.simulate`` — the
                         pypowsybl backend passes ``{"keep_variant": True,
                         "fast_mode": ...}`` (grid2op passes none).

    Returns:
        Tuple[Any, bool]: ``(obs_simu, has_converged)`` — the post-contingency
        observation and whether the simulation converged.
    """
    simulate_kwargs = simulate_kwargs or {}
    # Create the action to disconnect the contingency lines
    act_deco_defaut = env.action_space({"set_line_status": [(line, -1) for line in lines_defaut]})
    # Combine contingency disconnection with maintenance reconnection and simulate
    obs_simu, _, _, info = obs.simulate(
        act_deco_defaut + act_reco_maintenance, time_step=timestep, **simulate_kwargs
    )

    # Check if the simulation raised any exceptions
    if info["exception"]:
        print(f"ERROR: Simulation of contingency {lines_defaut} failed: {info['exception']}")
        return obs_simu, False  # Return the (potentially invalid) observation and False

    # Simulation successful
    return obs_simu, True


def compute_baseline_simulation(obs: Any, timestep: int, act_defaut: Any,
                                 act_reco_maintenance: Any,
                                 overload_ids: List[int],
                                 simulate_kwargs: Optional[Dict[str, Any]] = None,
                                 ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """
    Computes the baseline simulation once for use with multiple candidate actions.

    Args:
        obs: The initial observation object *before* any actions are applied.
        timestep: The simulation timestep index.
        act_defaut: The baseline action (e.g., N-1 contingency disconnection).
        act_reco_maintenance: An action object for maintenance reconnections.
        overload_ids: Line indices whose rho values should be extracted.
        simulate_kwargs: Extra keyword args forwarded to ``obs.simulate`` — the
                         pypowsybl backend passes ``{"keep_variant": True,
                         "fast_mode": ...}`` so the baseline variant is retained
                         for candidates to branch from (grid2op passes none).

    Returns:
        Tuple[Optional[np.ndarray], Optional[Any]]:
            - baseline_rho: rho values for ``overload_ids``, or None if failed.
            - obs_baseline: the baseline observation, or None if failed.
    """
    simulate_kwargs = simulate_kwargs or {}
    obs_baseline, _, _, info_baseline = obs.simulate(
        act_defaut + act_reco_maintenance, time_step=timestep, **simulate_kwargs
    )

    if info_baseline["exception"]:
        print(f"ERROR: Baseline simulation failed: {info_baseline['exception']}")
        return None, None

    baseline_rho = obs_baseline.rho[overload_ids]
    return baseline_rho, obs_baseline


def check_rho_reduction_with_baseline(obs: Any, timestep: int, act_defaut: Any, action: Any,
                                       overload_ids: List[int], act_reco_maintenance: Any,
                                       baseline_rho: np.ndarray,
                                       lines_we_care_about: Optional[np.ndarray] = None,
                                       rho_tolerance: float = 0.01,
                                       verbose: bool = True,
                                       *,
                                       reapply_contingency: bool = True,
                                       simulate_kwargs: Optional[Dict[str, Any]] = None,
                                       ) -> Tuple[bool, Optional[Any]]:
    """
    Checks if applying a candidate action reduces line loadings (rho) below a pre-computed baseline.

    Takes a pre-computed ``baseline_rho`` array, avoiding redundant baseline
    simulations when checking many candidate actions.

    ``obs`` is the observation the candidate is simulated *on top of* (the
    baseline branch point). ``reapply_contingency`` captures the backend
    contract explicitly (review C-diag): with it True (grid2op) the candidate is
    simulated as ``action + act_defaut + act_reco_maintenance`` from the healthy
    N-state; with it False (pypowsybl) only ``action`` is simulated on top of the
    already-contingency-applied kept variant. ``simulate_kwargs`` forwards the
    pypowsybl ``fast_mode`` flag (grid2op passes none).

    Args:
        obs: The observation the candidate branches from (see above).
        timestep: The simulation timestep index.
        act_defaut: The baseline action (N-1 contingency disconnection).
        action: The candidate action whose effectiveness is being tested.
        overload_ids: Line indices whose rho values should be checked.
        act_reco_maintenance: Action for maintenance reconnections.
        baseline_rho: Pre-computed rho values from the baseline simulation.
        lines_we_care_about: Array of line names to monitor (for reporting).
        rho_tolerance: Minimum required reduction. Defaults to 0.01.
        verbose: Whether to print success messages. Defaults to True.
        reapply_contingency: Whether to re-apply the contingency (grid2op) or
            simulate the candidate alone (pypowsybl).
        simulate_kwargs: Extra keyword args forwarded to ``obs.simulate``.

    Returns:
        Tuple[bool, Optional[Any]]:
            - is_rho_reduction: True if all rho values decreased by > tolerance.
            - obs_simu_action: The observation after applying the candidate.
    """
    simulate_kwargs = simulate_kwargs or {}
    # grid2op re-applies the contingency + maintenance (candidate branches from
    # the healthy N-state); pypowsybl simulates only the candidate on top of the
    # already-contingency-applied observation it was handed.
    to_simulate = action + act_defaut + act_reco_maintenance if reapply_contingency else action
    obs_simu_action, _, _, info_action = obs.simulate(
        to_simulate, time_step=timestep, **simulate_kwargs
    )

    # If candidate simulation fails, return False but still return the observation
    if info_action["exception"]:
        print(f"ERROR: Candidate action simulation failed in check_rho_reduction: {info_action['exception']}")
        return False, obs_simu_action

    # Get final rho values from the candidate state
    rho_final = obs_simu_action.rho[overload_ids]

    # Check if *all* specified rho values decreased by more than the tolerance
    if np.all(rho_final + rho_tolerance < baseline_rho):
        if verbose:
            max_rho_line = "N/A"
            max_rho = 0.0

            # Find the maximum rho specifically among 'lines_we_care_about' if provided
            if lines_we_care_about is not None and len(lines_we_care_about) > 0:
                # Create a mask for lines we care about
                care_mask = np.isin(obs_simu_action.name_line, lines_we_care_about)
                if np.any(care_mask):
                    # Filter rho values and find the maximum
                    rhos_of_interest = obs_simu_action.rho[care_mask]
                    max_rho = np.max(rhos_of_interest)
                    # Find the name of the line corresponding to that max_rho
                    max_rho_line_idx = np.where(obs_simu_action.rho == max_rho)[0]
                    # Ensure index is valid before accessing name_line
                    if max_rho_line_idx.size > 0 and max_rho_line_idx[0] < len(obs.name_line):
                        max_rho_line = obs.name_line[max_rho_line_idx[0]]

            # If lines_we_care_about is not specified, find the overall maximum rho
            else:
                if obs_simu_action.rho.size > 0:
                    max_rho_idx = np.argmax(obs_simu_action.rho)
                    max_rho = obs_simu_action.rho[max_rho_idx]
                    # Ensure index is valid before accessing name_line
                    if max_rho_idx < len(obs.name_line):
                        max_rho_line = obs.name_line[max_rho_idx]

            print(
                f"✅ Rho reduction from {np.round(baseline_rho, 2)} to {np.round(rho_final, 2)}. "
                f"New max rho is {max_rho:.2f} on line {max_rho_line}."
            )
        return True, obs_simu_action

    # If rho reduction condition is not met
    return False, obs_simu_action


def check_rho_reduction(obs: Any, timestep: int, act_defaut: Any, action: Any, overload_ids: List[int],
                        act_reco_maintenance: Any, lines_we_care_about: Optional[np.ndarray] = None,
                        rho_tolerance: float = 0.01,
                        *,
                        reapply_contingency: bool = True,
                        baseline_simulate_kwargs: Optional[Dict[str, Any]] = None,
                        candidate_simulate_kwargs: Optional[Dict[str, Any]] = None,
                        ) -> Tuple[bool, Optional[Any]]:
    """
    Checks if applying a candidate action reduces line loadings (rho) below a baseline.

    Simulates two scenarios: the baseline (contingency + maintenance) and the
    candidate (baseline + candidate action), and compares rho for ``overload_ids``.
    The candidate is effective if *all* rho values decrease by > ``rho_tolerance``.

    Note: For checking multiple actions, use :func:`compute_baseline_simulation`
    once and :func:`check_rho_reduction_with_baseline` per action to avoid
    redundant baseline simulations (this is what the shared
    :class:`BaselineContext` does).

    ``reapply_contingency`` / the two ``*_simulate_kwargs`` capture the backend
    contract: grid2op branches the candidate from the healthy N-state (``obs``)
    and re-applies the contingency; pypowsybl retains a kept baseline variant and
    branches the candidate from it, applying only the candidate action.

    Returns:
        Tuple[bool, Optional[Any]]: ``(is_rho_reduction, obs_simu_action)``.
        ``obs_simu_action`` is None if the baseline simulation failed.
    """
    # Compute baseline
    baseline_rho, obs_baseline = compute_baseline_simulation(
        obs, timestep, act_defaut, act_reco_maintenance, overload_ids,
        simulate_kwargs=baseline_simulate_kwargs,
    )

    if baseline_rho is None:
        return False, None

    # Branch the candidate from the healthy N-state (grid2op, re-applies the
    # contingency) or from the contingency-applied kept variant (pypowsybl).
    branch_obs = obs if reapply_contingency else obs_baseline

    return check_rho_reduction_with_baseline(
        branch_obs, timestep, act_defaut, action, overload_ids, act_reco_maintenance,
        baseline_rho, lines_we_care_about, rho_tolerance,
        reapply_contingency=reapply_contingency,
        simulate_kwargs=candidate_simulate_kwargs,
    )


def check_simu_overloads(obs: Any, obs_defaut: Any, action_space: Callable, timestep: int,
                         lines_defaut: List[str], lines_overloaded_ids: List[int],
                         lines_reco_maintenance: List[str],
                         simulate_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[bool, bool]:
    """
    Simulates disconnecting all specified overloaded lines simultaneously along with contingencies.

    Checks for two failure conditions:
    1. Simulation exception (power flow fails to converge).
    2. Load shedding (total load served drops significantly below ``obs_defaut``).

    Args:
        obs: The initial observation object *before* any actions.
        obs_defaut: The baseline (post-contingency) observation, for load comparison.
        action_space: The action space object.
        timestep: The simulation timestep index.
        lines_defaut: Line names for the initial contingency.
        lines_overloaded_ids: Indices of *all* overloaded lines to disconnect.
        lines_reco_maintenance: Line names to reconnect from maintenance.
        simulate_kwargs: Extra keyword args forwarded to ``obs.simulate`` — the
                         pypowsybl backend passes ``{"fast_mode": ...}``.

    Returns:
        Tuple[bool, bool]: ``(has_converged, has_lost_load)``.
    """
    simulate_kwargs = simulate_kwargs or {}
    # Create action to disconnect all specified overloaded lines
    # Ensure line_id is valid before accessing name_line
    valid_overload_ids = [line_id for line_id in lines_overloaded_ids if line_id < len(obs.name_line)]
    act_deco_overloads = action_space(
        {"set_line_status": [(obs.name_line[line_id], -1) for line_id in valid_overload_ids]}
    )
    # Create action for initial contingency
    act_deco_defaut = action_space({"set_line_status": [(line, -1) for line in lines_defaut]})
    # Create action for maintenance reconnections
    act_reco_maintenance_obj = action_space({"set_line_status": [(line_reco, 1) for line_reco in lines_reco_maintenance]})

    # Simulate the combined action
    obs_simu, _, _, info = obs.simulate(
        act_deco_overloads + act_deco_defaut + act_reco_maintenance_obj,
        time_step=timestep, **simulate_kwargs
    )

    # Check for simulation failure
    if info["exception"]:
        print(f"ERROR: Simulation failed when disconnecting all specified overloads ({[obs.name_line[i] for i in valid_overload_ids]}): {info['exception']}")
        return False, False  # Cannot determine load loss if simulation failed

    # Check for load shedding by comparing total load before and after
    # Add a small tolerance (e.g., 1 MW) to avoid floating point issues
    if obs_simu.load_p.sum() + 1.0 < obs_defaut.load_p.sum():
        print(f"WARNING: Load shedding occurred when simulating disconnection of all specified overloads. "
              f"Load before: {obs_defaut.load_p.sum():.2f}, Load after: {obs_simu.load_p.sum():.2f}")
        return True, True  # Converged, but load was lost

    # Simulation converged without significant load loss
    return True, False
