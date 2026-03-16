# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender

"""
Unit tests for the improved rho estimation in the superposition module.

Covers:
- _estimate_rho_from_p(): per-extremity power-factor ratio method
- use_p_based_rho parameter in compute_all_pairs_superposition()
- p_ex_combined key in compute_combined_pair_superposition() results
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from expert_op4grid_recommender.utils.superposition import (
    _estimate_rho_from_p,
    compute_combined_pair_superposition,
    compute_all_pairs_superposition,
)
from expert_op4grid_recommender.action_evaluation.classifier import ActionClassifier


# =============================================================================
# Helpers
# =============================================================================

def _obs_no_limits(p_or, p_ex, rho):
    """Observation without per-extremity data — forces fallback path in _estimate_rho_from_p."""
    obs = MagicMock(spec=["p_or", "p_ex", "rho"])
    obs.p_or = np.array(p_or, dtype=float)
    obs.p_ex = np.array(p_ex, dtype=float)
    obs.rho = np.array(rho, dtype=float)
    return obs


def _obs_with_limits(p_or, p_ex, rho, a_or, a_ex, limit_or, limit_ex):
    """Observation with per-extremity current and thermal limit data — best path."""
    obs = MagicMock()
    obs.p_or = np.array(p_or, dtype=float)
    obs.p_ex = np.array(p_ex, dtype=float)
    obs.rho = np.array(rho, dtype=float)
    obs.a_or = np.array(a_or, dtype=float)
    obs.a_ex = np.array(a_ex, dtype=float)
    obs._limit_or = MagicMock()
    obs._limit_or.values = np.array(limit_or, dtype=float)
    obs._limit_ex = MagicMock()
    obs._limit_ex.values = np.array(limit_ex, dtype=float)
    return obs


# =============================================================================
# Tests for _estimate_rho_from_p
# =============================================================================

class TestEstimateRhoFromP:

    def test_scales_proportionally_to_p_or(self):
        """rho_combined ∝ |p_or_combined|/|p_or_start| when no reactive power."""
        obs = _obs_no_limits(p_or=[100.0], p_ex=[-100.0], rho=[1.0])
        # factor_or = rho_start / |p_or_start| = 1.0 / 100 = 0.01
        p_or_combined = np.array([50.0])
        p_ex_combined = np.array([-50.0])
        result = _estimate_rho_from_p(p_or_combined, p_ex_combined, obs)
        # rho_or_est = 50 * 0.01 = 0.5
        np.testing.assert_allclose(result, [0.5], rtol=1e-6)

    def test_multiple_lines_independent_scaling(self):
        """Each line scales independently by its own rho/|P| factor."""
        obs = _obs_no_limits(
            p_or=[100.0, 200.0, 50.0],
            p_ex=[-100.0, -200.0, -50.0],
            rho=[1.0, 0.5, 0.8],
        )
        p_or_combined = np.array([80.0, 100.0, 60.0])
        p_ex_combined = np.array([-80.0, -100.0, -60.0])
        result = _estimate_rho_from_p(p_or_combined, p_ex_combined, obs)
        # Line 0: factor = 1.0/100 = 0.01  → 80 * 0.01 = 0.80
        # Line 1: factor = 0.5/200 = 0.0025 → 100 * 0.0025 = 0.25
        # Line 2: factor = 0.8/50  = 0.016  → 60 * 0.016 = 0.96
        np.testing.assert_allclose(result, [0.80, 0.25, 0.96], rtol=1e-6)

    def test_near_zero_p_start_gives_zero_rho(self):
        """Lines with |P_start| below threshold (0.1 MW) get rho_combined = 0."""
        obs = _obs_no_limits(
            p_or=[0.05, 100.0],   # line 0: near-zero, line 1: normal
            p_ex=[-0.05, -100.0],
            rho=[0.1, 1.0],
        )
        p_or_combined = np.array([10.0, 50.0])
        p_ex_combined = np.array([-10.0, -50.0])
        result = _estimate_rho_from_p(p_or_combined, p_ex_combined, obs)
        # Line 0: both |p_or_start| < 0.1 and |p_ex_start| < 0.1 → rho = 0
        assert result[0] == pytest.approx(0.0)
        # Line 1: factor = 1.0/100 = 0.01 → 50 * 0.01 = 0.5
        assert result[1] == pytest.approx(0.5, rel=1e-6)

    def test_fallback_uses_rho_start_without_limit_data(self):
        """Without _limit_or/_limit_ex, rho_start is used as the per-extremity fallback."""
        # spec= restricts hasattr so _limit_or / a_or are absent
        obs = _obs_no_limits(p_or=[100.0], p_ex=[-100.0], rho=[0.8])
        p_or_combined = np.array([120.0])
        p_ex_combined = np.array([-120.0])
        result = _estimate_rho_from_p(p_or_combined, p_ex_combined, obs)
        # factor_or = rho_start / |p_or_start| = 0.8 / 100 = 0.008
        # rho_or_est = 120 * 0.008 = 0.96
        np.testing.assert_allclose(result, [0.96], rtol=1e-6)

    def test_uses_per_extremity_rho_when_limit_data_available(self):
        """When _limit_or/_limit_ex are present, rho_or/rho_ex are derived from a/limit."""
        # Asymmetric limits: limit_or=100A, limit_ex=200A
        # a_or = 80A → rho_or = 0.80; a_ex = 80A → rho_ex = 0.40
        obs = _obs_with_limits(
            p_or=[100.0], p_ex=[-100.0], rho=[0.80],
            a_or=[80.0], a_ex=[80.0],
            limit_or=[100.0], limit_ex=[200.0],
        )
        p_or_combined = np.array([120.0])
        p_ex_combined = np.array([-120.0])
        result = _estimate_rho_from_p(p_or_combined, p_ex_combined, obs)
        # factor_or = rho_or_start / |p_or_start| = 0.80 / 100 = 0.008
        # rho_or_est = 120 * 0.008 = 0.96
        # MAX_RHO_BOTH_EXTREMITIES=False in test config → rho_combined = rho_or_est
        np.testing.assert_allclose(result, [0.96], rtol=1e-6)

    def test_per_extremity_different_from_rho_start(self):
        """Per-extremity path gives different result from fallback on asymmetric lines."""
        # Asymmetric: limit_or = 100A, limit_ex = 50A; a_or = a_ex = 40A
        # rho_or_start = 40/100 = 0.40; rho_ex_start = 40/50 = 0.80
        # rho_start (as stored) might be max(0.40, 0.80) = 0.80 (or just rho_or = 0.40)
        obs_with = _obs_with_limits(
            p_or=[100.0], p_ex=[-100.0], rho=[0.80],
            a_or=[40.0], a_ex=[40.0],
            limit_or=[100.0], limit_ex=[50.0],
        )
        obs_without = _obs_no_limits(p_or=[100.0], p_ex=[-100.0], rho=[0.80])
        p_or_combined = np.array([80.0])
        p_ex_combined = np.array([-80.0])

        result_with = _estimate_rho_from_p(p_or_combined, p_ex_combined, obs_with)
        result_without = _estimate_rho_from_p(p_or_combined, p_ex_combined, obs_without)

        # With limits: factor_or = rho_or_start/|p_or_start| = 0.40/100 = 0.004
        #              rho_or_est = 80 * 0.004 = 0.32
        np.testing.assert_allclose(result_with, [0.32], rtol=1e-6)
        # Without limits: factor_or = rho_start/|p_or_start| = 0.80/100 = 0.008
        #                 rho_or_est = 80 * 0.008 = 0.64
        np.testing.assert_allclose(result_without, [0.64], rtol=1e-6)

    def test_negative_p_or_combined_uses_absolute_value(self):
        """Flow direction reversal must not produce negative rho."""
        obs = _obs_no_limits(p_or=[100.0], p_ex=[-100.0], rho=[1.0])
        p_or_combined = np.array([-50.0])  # flow reversed
        p_ex_combined = np.array([50.0])
        result = _estimate_rho_from_p(p_or_combined, p_ex_combined, obs)
        assert result[0] >= 0.0
        np.testing.assert_allclose(result, [0.5], rtol=1e-6)

    def test_zero_thermal_limit_clamped(self):
        """Zero thermal limits are clamped to 1e-6 to prevent division by zero."""
        obs = _obs_with_limits(
            p_or=[100.0], p_ex=[-100.0], rho=[1.0],
            a_or=[100.0], a_ex=[100.0],
            limit_or=[0.0], limit_ex=[0.0],  # degenerate: zero limits
        )
        p_or_combined = np.array([50.0])
        p_ex_combined = np.array([-50.0])
        # Should not raise; rho_or_start = 100 / 1e-6 (huge), but factor = (big) / 100 → huge rho
        result = _estimate_rho_from_p(p_or_combined, p_ex_combined, obs)
        assert np.isfinite(result[0])
        assert result[0] >= 0.0


# =============================================================================
# Tests for p_ex_combined in compute_combined_pair_superposition results
# =============================================================================

class TestPExCombinedInResult:

    def test_p_ex_combined_present_in_result(self):
        """compute_combined_pair_superposition returns p_ex_combined alongside p_or_combined."""
        obs_start = MagicMock()
        obs_start.p_or = np.array([100.0, 50.0])
        obs_start.p_ex = np.array([-100.0, -50.0])
        obs_start.line_status = np.array([False, True])

        obs_act1 = MagicMock()
        obs_act1.p_or = np.array([90.0, 50.0])
        obs_act1.p_ex = np.array([-90.0, -50.0])
        obs_act1.line_status = np.array([True, True])

        obs_act2 = MagicMock()
        obs_act2.p_or = np.array([100.0, 40.0])
        obs_act2.p_ex = np.array([-100.0, -40.0])
        obs_act2.line_status = np.array([False, False])

        with patch("expert_op4grid_recommender.utils.superposition.get_delta_theta_line",
                   return_value=0.1), \
             patch("expert_op4grid_recommender.utils.superposition.get_betas_coeff",
                   return_value=np.array([0.5, 0.5])):
            result = compute_combined_pair_superposition(
                obs_start, obs_act1, obs_act2,
                act1_line_idxs=[0], act1_sub_idxs=[],
                act2_line_idxs=[1], act2_sub_idxs=[],
            )

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "p_ex_combined" in result, "p_ex_combined must be in the result dict"
        assert isinstance(result["p_ex_combined"], list)
        assert len(result["p_ex_combined"]) == 2

    def test_p_ex_combined_is_linear_combination(self):
        """p_ex_combined = w*p_ex_start + b1*p_ex_act1 + b2*p_ex_act2."""
        betas = np.array([0.6, 0.4])
        w = 1.0 - betas.sum()  # 0.0

        obs_start = MagicMock()
        obs_start.p_or = np.array([100.0, 50.0])
        obs_start.p_ex = np.array([-90.0, -45.0])
        obs_start.line_status = np.array([False, True])

        obs_act1 = MagicMock()
        obs_act1.p_or = np.array([80.0, 50.0])
        obs_act1.p_ex = np.array([-75.0, -45.0])
        obs_act1.line_status = np.array([True, True])

        obs_act2 = MagicMock()
        obs_act2.p_or = np.array([100.0, 30.0])
        obs_act2.p_ex = np.array([-90.0, -28.0])
        obs_act2.line_status = np.array([False, False])

        with patch("expert_op4grid_recommender.utils.superposition.get_delta_theta_line",
                   return_value=0.1), \
             patch("expert_op4grid_recommender.utils.superposition.get_betas_coeff",
                   return_value=betas):
            result = compute_combined_pair_superposition(
                obs_start, obs_act1, obs_act2,
                act1_line_idxs=[0], act1_sub_idxs=[],
                act2_line_idxs=[1], act2_sub_idxs=[],
            )

        assert "error" not in result
        expected_p_ex = (w * obs_start.p_ex
                         + betas[0] * obs_act1.p_ex
                         + betas[1] * obs_act2.p_ex)
        np.testing.assert_allclose(
            result["p_ex_combined"], expected_p_ex.tolist(), rtol=1e-6
        )


# =============================================================================
# Tests for use_p_based_rho parameter
# =============================================================================

class TestUsePBasedRho:
    """Tests that use_p_based_rho routes to the correct rho computation."""

    def _make_env_and_obs(self):
        env = MagicMock()
        env.name_line = ["L0", "L1", "L2"]

        obs_start = MagicMock()
        obs_start.rho = np.array([1.2, 0.4, 0.6])
        obs_start.p_or = np.array([120.0, 40.0, 60.0])
        obs_start.p_ex = np.array([-120.0, -40.0, -60.0])
        obs_start.a_or = np.array([120.0, 40.0, 60.0])
        obs_start.a_ex = np.array([120.0, 40.0, 60.0])
        obs_start._limit_or = MagicMock()
        obs_start._limit_or.values = np.array([100.0, 100.0, 100.0])
        obs_start._limit_ex = MagicMock()
        obs_start._limit_ex.values = np.array([100.0, 100.0, 100.0])

        obs_act1 = MagicMock()
        obs_act1.rho = np.array([1.0, 0.4, 0.6])
        obs_act1.p_or = np.array([100.0, 40.0, 60.0])
        obs_act1.p_ex = np.array([-100.0, -40.0, -60.0])

        obs_act2 = MagicMock()
        obs_act2.rho = np.array([1.0, 0.4, 0.6])
        obs_act2.p_or = np.array([100.0, 40.0, 60.0])
        obs_act2.p_ex = np.array([-100.0, -40.0, -60.0])

        aid1, aid2 = "act1", "act2"
        detailed_actions = {
            aid1: {"action": MagicMock(), "observation": obs_act1, "description_unitaire": "A1"},
            aid2: {"action": MagicMock(), "observation": obs_act2, "description_unitaire": "A2"},
        }
        return env, obs_start, obs_act1, obs_act2, detailed_actions, aid1, aid2

    def _run_pairs(self, use_p_based_rho, env, obs_start, detailed_actions, aid1, aid2,
                   betas, p_or_combined, p_ex_combined):
        from expert_op4grid_recommender.utils import superposition

        classifier = MagicMock(spec=ActionClassifier)
        classifier._action_space = MagicMock()

        def mock_identify(action, action_id, *args):
            if action_id == aid1: return ([0], [])
            if action_id == aid2: return ([1], [])
            return ([], [])

        orig_identify = superposition._identify_action_elements
        orig_compute = superposition.compute_combined_pair_superposition
        try:
            superposition._identify_action_elements = MagicMock(side_effect=mock_identify)
            superposition.compute_combined_pair_superposition = MagicMock(return_value={
                "betas": betas,
                "p_or_combined": list(p_or_combined),
                "p_ex_combined": list(p_ex_combined),
                "is_islanded": False,
                "disconnected_mw": 0.0,
            })
            return compute_all_pairs_superposition(
                obs_start=obs_start,
                detailed_actions=detailed_actions,
                classifier=classifier,
                env=env,
                lines_overloaded_ids=[0],
                lines_we_care_about=["L0", "L1", "L2"],
                pre_existing_rho={},
                dict_action={},
                use_p_based_rho=use_p_based_rho,
            )
        finally:
            superposition._identify_action_elements = orig_identify
            superposition.compute_combined_pair_superposition = orig_compute

    def test_p_based_rho_true_calls_estimate_rho_from_p(self):
        """use_p_based_rho=True calls _estimate_rho_from_p."""
        env, obs_start, obs_act1, obs_act2, detailed_actions, aid1, aid2 = self._make_env_and_obs()
        betas = [0.5, 0.5]
        p_or_combined = np.array([80.0, 40.0, 60.0])
        p_ex_combined = np.array([-80.0, -40.0, -60.0])

        with patch("expert_op4grid_recommender.utils.superposition._estimate_rho_from_p",
                   wraps=_estimate_rho_from_p) as mock_estimate:
            self._run_pairs(True, env, obs_start, detailed_actions, aid1, aid2,
                            betas, p_or_combined, p_ex_combined)
            mock_estimate.assert_called_once()

    def test_p_based_rho_false_does_not_call_estimate_rho_from_p(self):
        """use_p_based_rho=False skips _estimate_rho_from_p and uses rho arrays directly."""
        env, obs_start, obs_act1, obs_act2, detailed_actions, aid1, aid2 = self._make_env_and_obs()
        betas = [0.5, 0.5]
        p_or_combined = np.array([80.0, 40.0, 60.0])
        p_ex_combined = np.array([-80.0, -40.0, -60.0])

        with patch("expert_op4grid_recommender.utils.superposition._estimate_rho_from_p",
                   wraps=_estimate_rho_from_p) as mock_estimate:
            self._run_pairs(False, env, obs_start, detailed_actions, aid1, aid2,
                            betas, p_or_combined, p_ex_combined)
            mock_estimate.assert_not_called()

    def test_approximate_method_rho_matches_direct_formula(self):
        """use_p_based_rho=False gives rho = |w*rho_start + b1*rho_act1 + b2*rho_act2|."""
        env, obs_start, obs_act1, obs_act2, detailed_actions, aid1, aid2 = self._make_env_and_obs()
        betas = [0.6, 0.5]  # sum = 1.1, w = -0.1
        p_or_combined = np.array([100.0, 40.0, 60.0])
        p_ex_combined = np.array([-100.0, -40.0, -60.0])

        results = self._run_pairs(False, env, obs_start, detailed_actions, aid1, aid2,
                                  betas, p_or_combined, p_ex_combined)

        pair_id = f"{aid1}+{aid2}"
        assert pair_id in results
        res = results[pair_id]

        w = 1.0 - sum(betas)  # -0.1
        expected_rho = np.abs(
            w * obs_start.rho + betas[0] * obs_act1.rho + betas[1] * obs_act2.rho
        )
        # max_rho is from eligible lines (no pre-existing overloads → all eligible)
        assert abs(res["max_rho"] - float(np.max(expected_rho))) < 0.01

    def test_default_is_p_based_rho_true(self):
        """Default value of use_p_based_rho must be True (accurate method)."""
        import inspect
        sig = inspect.signature(compute_all_pairs_superposition)
        default = sig.parameters["use_p_based_rho"].default
        assert default is True, (
            f"use_p_based_rho default should be True, got {default!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
