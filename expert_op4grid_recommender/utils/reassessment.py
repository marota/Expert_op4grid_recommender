# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Per-action reassessment + combined-pair estimation.

This is the "action-card preparation" stage. It runs AFTER a
recommendation model has produced a flat
``{action_id: action_object}`` mapping and turns each entry into the
rich :class:`SimulatedAction` payload the frontend renders as a card.

Extracted from the historical tail of ``run_analysis_step2_discovery``
so ANY pluggable :class:`RecommenderModel` can reuse it instead of
re-implementing simulation, rho-evolution math, and superposition
combinations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from expert_op4grid_recommender import config
from expert_op4grid_recommender.utils.helpers import Timer

logger = logging.getLogger(__name__)


def _extract_pypowsybl_network(env: Any) -> Any:
    """Best-effort extraction of the underlying pypowsybl ``Network``
    from a simulation environment.

    The two supported backends expose it in different places:

    - pypowsybl backend: ``env.network_manager.network``
    - grid2op backend:   ``env.backend._grid.network``

    Returns ``None`` when neither path is present.
    """
    if env is None:
        return None
    nm = getattr(env, "network_manager", None)
    if nm is not None:
        net = getattr(nm, "network", None)
        if net is not None:
            return net
    backend = getattr(env, "backend", None)
    if backend is not None:
        grid = getattr(backend, "_grid", None)
        if grid is not None:
            return getattr(grid, "network", None)
    return None


def _extract_pypowsybl_network_from_obs(obs: Any) -> Any:
    """Best-effort extraction of the pypowsybl ``Network`` from an
    observation. Grid2Op observations return ``None``.
    """
    if obs is None:
        return None
    nm = getattr(obs, "_network_manager", None)
    if nm is not None:
        return getattr(nm, "network", None)
    return None


def _extract_overloaded_rho(
    obs_defaut: Any, lines_overloaded_ids: List[int],
) -> Optional[List[float]]:
    """Pre-extract the loading rate (rho) of each constrained line under
    the N-K state, paired with ``lines_overloaded_ids``.

    Returns a plain Python ``list[float]`` for serialisation friendliness
    (the underlying ``obs_defaut.rho`` may be a numpy array). Returns
    ``None`` when either the observation has no ``rho`` attribute or the
    indices cannot be resolved — callers treat that as "data not
    pre-computed".
    """
    if obs_defaut is None or not lines_overloaded_ids:
        return None
    rho = getattr(obs_defaut, "rho", None)
    if rho is None:
        return None
    try:
        return [float(rho[i]) for i in lines_overloaded_ids]
    except (TypeError, IndexError, KeyError):
        return None


def reassess_prioritized_actions(
    prioritized_actions: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[Dict[str, dict], List[dict]]:
    """Simulate each prioritised action and compute rho metrics.

    Args:
        prioritized_actions: ``{action_id: action_obj}`` from a recommender.
        context: pipeline context built by :func:`run_analysis_step1` and
            updated by :func:`run_analysis_step2_graph`.

    Returns:
        ``(detailed_actions, pre_existing_info)``.
    """
    backend = context["backend"]
    env = context["env"]
    obs = context["obs"]
    obs_simu_defaut = context["obs_simu_defaut"]
    current_timestep = context["current_timestep"]
    current_lines_defaut = context["current_lines_defaut"]
    lines_overloaded_ids = context["lines_overloaded_ids"]
    act_reco_maintenance = context["act_reco_maintenance"]
    lines_we_care_about = context["lines_we_care_about"]
    pre_existing_rho = context["pre_existing_rho"]
    dict_action = context["dict_action"]
    is_pypowsybl = context["is_pypowsybl"]
    actual_fast_mode = context["actual_fast_mode"]
    create_default_action = context["create_default_action"]

    from expert_op4grid_recommender.main import (
        simulate_contingency_grid2op, simulate_contingency_pypowsybl,
    )

    with Timer("Reassessment"):
        act_defaut = create_default_action(env.action_space, current_lines_defaut)

        if is_pypowsybl and obs_simu_defaut._network_manager._default_dc:
            obs_simu_defaut, _ = simulate_contingency_pypowsybl(
                env, obs, current_lines_defaut, act_reco_maintenance,
                current_timestep, fast_mode=actual_fast_mode,
            )
            obs_simu_defaut._network_manager._default_dc = False
        elif not is_pypowsybl:
            obs_simu_defaut, _ = simulate_contingency_grid2op(
                env, obs, current_lines_defaut, act_reco_maintenance, current_timestep,
            )
        baseline_rho = obs_simu_defaut.rho[lines_overloaded_ids]

        num_lines = len(obs.name_line)
        pre_existing_baseline = np.zeros(num_lines)
        is_pre_existing = np.zeros(num_lines, dtype=bool)
        for idx, rho_val in pre_existing_rho.items():
            pre_existing_baseline[idx] = rho_val
            is_pre_existing[idx] = True

        if lines_we_care_about is not None and len(lines_we_care_about) > 0:
            care_mask = np.isin(obs.name_line, list(lines_we_care_about))
        else:
            care_mask = np.ones(num_lines, dtype=bool)

        worsening_threshold = getattr(
            config, "PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD", 0.02,
        )

        detailed_actions: Dict[str, dict] = {}
        for action_id, action in prioritized_actions.items():
            description_unitaire = None
            if action_id in dict_action:
                description_unitaire = dict_action[action_id].get("description_unitaire")

            if is_pypowsybl:
                obs_simu_action, _, _, info_action = obs_simu_defaut.simulate(
                    action, time_step=current_timestep, keep_variant=True,
                    fast_mode=actual_fast_mode,
                )
            else:
                obs_simu_action, _, _, info_action = obs.simulate(
                    action + act_defaut + act_reco_maintenance,
                    time_step=current_timestep,
                )

            rho_before = baseline_rho
            rho_after = None
            max_rho = 0.0
            max_rho_line = "N/A"
            is_rho_reduction = False

            if not info_action["exception"]:
                rho_after = obs_simu_action.rho[lines_overloaded_ids]
                if rho_before is not None:
                    is_rho_reduction = bool(np.all(rho_after + 0.01 < rho_before))

                action_rho = obs_simu_action.rho
                worsened_mask = action_rho > pre_existing_baseline * (1 + worsening_threshold)
                eligible_mask = care_mask & (~is_pre_existing | worsened_mask)

                if np.any(eligible_mask):
                    masked_rho = action_rho[eligible_mask]
                    max_idx_in_masked = int(np.argmax(masked_rho))
                    max_rho = float(masked_rho[max_idx_in_masked])
                    max_rho_line = obs_simu_action.name_line[
                        np.where(eligible_mask)[0][max_idx_in_masked]
                    ]

            sim_exception = info_action.get("exception")
            non_convergence = None
            if sim_exception:
                if isinstance(sim_exception, list):
                    non_convergence = "; ".join(str(e) for e in sim_exception)
                else:
                    non_convergence = str(sim_exception)

            print(f"{action_id}")
            if description_unitaire:
                print(f"  {description_unitaire}")
            if rho_before is not None and rho_after is not None:
                print(f"  Rho reduction from {np.round(rho_before, 2)} to {np.round(rho_after, 2)}")
                print(f"  New max rho is {max_rho:.2f} on line {max_rho_line}")

            detailed_actions[action_id] = {
                "action": action,
                "description_unitaire": description_unitaire,
                "rho_before": rho_before,
                "rho_after": rho_after,
                "max_rho": max_rho,
                "max_rho_line": max_rho_line,
                "is_rho_reduction": is_rho_reduction,
                "observation": obs_simu_action,
                "non_convergence": non_convergence,
            }

    pre_existing_info = [
        {"name": str(obs.name_line[i]), "rho_N": pre_existing_rho[i]}
        for i in sorted(pre_existing_rho.keys())
    ]
    return detailed_actions, pre_existing_info


def propagate_non_convergence_to_scores(
    detailed_actions: Dict[str, dict],
    action_scores: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror each action's non-convergence reason into the score table."""
    for action_id, details in detailed_actions.items():
        nc = details.get("non_convergence")
        for category in action_scores:
            if action_id in action_scores[category].get("scores", {}):
                action_scores[category].setdefault("non_convergence", {})
                action_scores[category]["non_convergence"][action_id] = nc
    return action_scores


def compute_combined_pairs(
    detailed_actions: Dict[str, dict],
    context: Dict[str, Any],
) -> Dict[str, dict]:
    """Run the superposition theorem on every detailed-action pair.

    Returns ``{}`` on any failure — superposition is decorative metadata
    and must never break the main flow.
    """
    with Timer("Combined Action Pairs (Superposition)"):
        try:
            from expert_op4grid_recommender.utils.superposition import (
                compute_all_pairs_superposition,
            )
            return compute_all_pairs_superposition(
                obs_start=context["obs_simu_defaut"],
                detailed_actions=detailed_actions,
                classifier=context["classifier"],
                env=context["env"],
                lines_overloaded_ids=context["lines_overloaded_ids"],
                lines_we_care_about=context["lines_we_care_about"],
                pre_existing_rho=context["pre_existing_rho"],
                dict_action=context["dict_action"],
            )
        except Exception as e:
            logger.warning(
                "Failed to compute combined action pairs: %s", e, exc_info=True,
            )
            return {}


def build_recommender_inputs(context: Dict[str, Any]):
    """Project the analysis ``context`` into a :class:`RecommenderInputs`.

    Surfaces three classes of data:

    1. **Observations + networks**: ``(obs, network)`` for the N state
       and ``(obs_defaut, network_defaut)`` for the N-K state.
    2. **Step-1 outcome already computed**: ``lines_overloaded_names``,
       ``lines_overloaded_ids``, ``lines_overloaded_rho`` (pre-extracted
       from ``obs_defaut.rho``), ``lines_overloaded_ids_kept`` (post-
       island-guard subset), and ``pre_existing_rho`` (N-state rho of
       lines that were already overloaded before the contingency).
       Models read these instead of recomputing.
    3. **Overflow-graph artefacts** when step-2 produced them, including
       ``filtered_candidate_actions`` — the action IDs retained by the
       expert :class:`ActionRuleValidator`. Forwarded so non-expert
       models that declare ``requires_overflow_graph=True`` (e.g.
       :class:`RandomOverflowRecommender`) can sample inside the same
       reduced action space the expert sees.
    """
    from expert_op4grid_recommender.models.base import RecommenderInputs

    env = context.get("env")
    obs_defaut = context["obs_simu_defaut"]
    lines_overloaded_ids = list(context["lines_overloaded_ids"])

    # --- Network handles (paired with the two observations) -----------
    network = context.get("n_grid")
    if network is None:
        network = _extract_pypowsybl_network(env)

    network_defaut = context.get("n_grid_defaut")
    if network_defaut is None:
        network_defaut = _extract_pypowsybl_network_from_obs(obs_defaut)
    if network_defaut is None:
        network_defaut = _extract_pypowsybl_network(env)

    # --- Pre-computed step-1 outcome ----------------------------------
    lines_overloaded_rho = _extract_overloaded_rho(obs_defaut, lines_overloaded_ids)

    ids_kept_raw = context.get("lines_overloaded_ids_kept")
    lines_overloaded_ids_kept = list(ids_kept_raw) if ids_kept_raw is not None else None

    pre_existing_rho_raw = context.get("pre_existing_rho")
    pre_existing_rho = (
        dict(pre_existing_rho_raw) if pre_existing_rho_raw is not None else None
    )

    # --- Expert-rule-filtered candidate set ---------------------------
    # Populated by ``_run_expert_action_filter`` whenever a recommender
    # that declares ``requires_overflow_graph=True`` enters
    # ``run_analysis_step2_discovery``. None when the filter never ran
    # (e.g. step-2 graph was skipped); empty list when it ran but
    # nothing passed — downstream models distinguish those two cases.
    filtered_raw = context.get("filtered_candidate_actions")
    filtered_candidate_actions = (
        list(filtered_raw) if filtered_raw is not None else None
    )

    return RecommenderInputs(
        obs=context["obs"],
        obs_defaut=obs_defaut,
        lines_defaut=list(context["current_lines_defaut"]),
        lines_overloaded_names=list(context["lines_overloaded_names"]),
        lines_overloaded_ids=lines_overloaded_ids,
        dict_action=context["dict_action"],
        env=env,
        classifier=context["classifier"],
        network=network,
        network_defaut=network_defaut,
        lines_overloaded_rho=lines_overloaded_rho,
        lines_overloaded_ids_kept=lines_overloaded_ids_kept,
        pre_existing_rho=pre_existing_rho,
        timestep=context["current_timestep"],
        overflow_graph=context.get("g_overflow"),
        distribution_graph=context.get("g_distribution_graph"),
        overflow_sim=context.get("overflow_sim"),
        hubs=context.get("hubs"),
        node_name_mapping=context.get("node_name_mapping"),
        non_connected_reconnectable_lines=context.get("non_connected_reconnectable_lines"),
        lines_non_reconnectable=context.get("lines_non_reconnectable"),
        lines_we_care_about=context.get("lines_we_care_about"),
        maintenance_to_reco_at_t=context.get("maintenance_to_reco_at_t"),
        act_reco_maintenance=context.get("act_reco_maintenance"),
        use_dc=context.get("use_dc", False),
        is_pypowsybl=context.get("is_pypowsybl", True),
        fast_mode=context.get("actual_fast_mode", False),
        filtered_candidate_actions=filtered_candidate_actions,
        _context=context,
    )
