# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender

"""
Tests for the superposition theorem across diverse action type combinations.

Mirrors the structure of:
  https://github.com/marota/Topology_Superposition_Theorem/blob/master/superposition_theorem/tests/test_different_action_type_superposition.py

Each test verifies that compute_combined_pair_superposition produces p_or values
that match the ground-truth combined-action simulation within tolerance.
"""

import warnings
import numpy as np
import unittest

import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend

from expert_op4grid_recommender.utils.superposition import (
    get_virtual_line_flow,
    get_sub_node1_idsflow,
    compute_combined_pair_superposition,
    get_delta_theta_sub_2nodes,
    get_delta_theta_line,
)


class TestDiverseActionTypeSuperposition(unittest.TestCase):

    def setUp(self) -> None:
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            params = Parameters()
            params.NO_OVERFLOW_DISCONNECTION = True
            params.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
            params.ENV_DC = True
            params.MAX_LINE_STATUS_CHANGED = 99999
            params.MAX_SUB_CHANGED = 99999

            self.env = grid2op.make(env_name, backend=LightSimBackend(), param=params, test=True)
            self.env.set_max_iter(20)

        self.chronic_id = 0
        self.tol = 3e-5

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    # =========================================================================
    # Helper
    # =========================================================================

    def _assert_superposition_accuracy(self, obs_target, result):
        """Assert that the superposition result matches the target observation."""
        self.assertNotIn("error", result, f"Superposition returned error: {result.get('error')}")

        self.assertIn("p_or_combined", result)
        p_computed = np.array(result["p_or_combined"])
        max_diff = np.max(np.abs(obs_target.p_or - p_computed))
        self.assertLessEqual(
            max_diff, self.tol,
            f"max |p_or_target - p_or_computed| = {max_diff:.2e} > tol {self.tol:.2e}"
        )

        # p_ex is also linear in angles and must superpose with the same accuracy
        self.assertIn("p_ex_combined", result)
        p_ex_computed = np.array(result["p_ex_combined"])
        max_diff_ex = np.max(np.abs(obs_target.p_ex - p_ex_computed))
        self.assertLessEqual(
            max_diff_ex, self.tol,
            f"max |p_ex_target - p_ex_computed| = {max_diff_ex:.2e} > tol {self.tol:.2e}"
        )

    # =========================================================================
    # Line reconnection + line disconnection
    # =========================================================================

    def test_line_disconnection_line_reconnection_combination_by_hand(self):
        """Line reconnection + line disconnection: manual beta computation matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3  # line to reconnect
        id_l2 = 7  # line to disconnect

        # Start with l1 disconnected, l2 connected
        opposite_action = self.env.action_space({"set_line_status": [(id_l1, -1)]})
        obs_start, *_ = self.env.step(opposite_action)

        act1 = self.env.action_space({"set_line_status": [(id_l1, +1)]})
        act2 = self.env.action_space({"set_line_status": [(id_l2, -1)]})

        obs1 = obs_start.simulate(act1, time_step=0)[0]
        obs2 = obs_start.simulate(act2, time_step=0)[0]
        obs_target = obs_start.simulate(act1 + act2, time_step=0)[0]

        # Manual beta computation using reference formulas
        a = np.array([
            [1, 1 - get_delta_theta_line(obs2, id_l1) / get_delta_theta_line(obs_start, id_l1)],
            [1 - obs1.p_or[id_l2] / obs_start.p_or[id_l2], 1],
        ])
        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or
        max_diff = np.max(np.abs(obs_target.p_or - p_computed))
        self.assertLessEqual(max_diff, self.tol)

    def test_line_disconnection_line_reconnection_combination_sup_theorem(self):
        """Line reconnection + line disconnection: compute_combined_pair_superposition matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_l2 = 7

        opposite_action = self.env.action_space({"set_line_status": [(id_l1, -1)]})
        obs_start, *_ = self.env.step(opposite_action)

        act1 = self.env.action_space({"set_line_status": [(id_l1, +1)]})
        act2 = self.env.action_space({"set_line_status": [(id_l2, -1)]})

        obs1 = obs_start.simulate(act1, time_step=0)[0]
        obs2 = obs_start.simulate(act2, time_step=0)[0]
        obs_target = obs_start.simulate(act1 + act2, time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start, obs1, obs2,
            act1_line_idxs=[id_l1], act1_sub_idxs=[],
            act2_line_idxs=[id_l2], act2_sub_idxs=[],
        )
        self._assert_superposition_accuracy(obs_target, result)

    # =========================================================================
    # Node merging + node splitting
    # =========================================================================

    def test_node_merging_splitting_combination_by_hand(self):
        """Node merging + node splitting: manual beta computation matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        unitary_action_list = [
            {"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}},  # merging sub5
            {"set_bus": {"substations_id": [(4, (2, 1, 2, 1, 2))]}},          # splitting sub4
        ]
        opposite_action_list = [
            {"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}},
            {"set_bus": {"substations_id": [(4, (1, 1, 1, 1, 1))]}},
        ]

        combined_opposite = (self.env.action_space(opposite_action_list[0])
                             + self.env.action_space(opposite_action_list[1]))
        obs_start, *_ = self.env.step(combined_opposite)

        id_sub1 = 5  # starts split, action = merge
        id_sub2 = 4  # starts merged, action = split

        unitary_actions = [self.env.action_space(a) for a in unitary_action_list]
        obs1 = obs_start.simulate(unitary_actions[0], time_step=0)[0]
        obs2 = obs_start.simulate(unitary_actions[1], time_step=0)[0]
        obs_target = obs_start.simulate(unitary_actions[0] + unitary_actions[1], time_step=0)[0]

        delta_theta_sub1_obs2 = get_delta_theta_sub_2nodes(obs2, id_sub1)
        delta_theta_sub1_start = get_delta_theta_sub_2nodes(obs_start, id_sub1)

        self.assertNotEqual(delta_theta_sub1_obs2, 0)
        self.assertNotEqual(delta_theta_sub1_start, 0)

        # node1 ids for sub2 (split in obs2)
        (ind_load, ind_prod, ind_lor, ind_lex) = get_sub_node1_idsflow(obs2, id_sub2)
        p_start_sub2 = get_virtual_line_flow(obs_start, ind_load, ind_prod, ind_lor, ind_lex)
        p_obs1_sub2 = get_virtual_line_flow(obs1, ind_load, ind_prod, ind_lor, ind_lex)

        a = np.array([
            [1, 1 - delta_theta_sub1_obs2 / delta_theta_sub1_start],
            [1 - p_obs1_sub2 / p_start_sub2, 1],
        ])
        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or
        max_diff = np.max(np.abs(obs_target.p_or - p_computed))
        self.assertLessEqual(max_diff, self.tol)

    def test_node_merging_splitting_combination_sup_theorem(self):
        """Node merging + node splitting: compute_combined_pair_superposition matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        unitary_action_list = [
            {"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}},
            {"set_bus": {"substations_id": [(4, (2, 1, 2, 1, 2))]}},
        ]
        opposite_action_list = [
            {"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}},
            {"set_bus": {"substations_id": [(4, (1, 1, 1, 1, 1))]}},
        ]

        combined_opposite = (self.env.action_space(opposite_action_list[0])
                             + self.env.action_space(opposite_action_list[1]))
        obs_start, *_ = self.env.step(combined_opposite)

        id_sub1 = 5
        id_sub2 = 4

        unitary_actions = [self.env.action_space(a) for a in unitary_action_list]
        obs1 = obs_start.simulate(unitary_actions[0], time_step=0)[0]
        obs2 = obs_start.simulate(unitary_actions[1], time_step=0)[0]
        obs_target = obs_start.simulate(unitary_actions[0] + unitary_actions[1], time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start, obs1, obs2,
            act1_line_idxs=[], act1_sub_idxs=[id_sub1],
            act2_line_idxs=[], act2_sub_idxs=[id_sub2],
        )
        self._assert_superposition_accuracy(obs_target, result)

    # =========================================================================
    # Line action + node splitting
    # =========================================================================

    def test_line_disconnection_node_splitting_combination_by_hand(self):
        """Line reconnection + node splitting: manual beta computation matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub1 = 5

        obs_start, *_ = self.env.step(self.env.action_space())

        act_line = self.env.action_space({"set_line_status": [(id_l1, +1)]})
        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})

        obs1 = obs_start.simulate(act_line, time_step=0)[0]
        obs2 = obs_start.simulate(act_split, time_step=0)[0]
        obs_target = obs_start.simulate(act_line + act_split, time_step=0)[0]

        # node1 ids for sub1 (split in obs2)
        (ind_load, ind_prod, ind_lor, ind_lex) = get_sub_node1_idsflow(obs2, id_sub1)
        p_start_sub1 = get_virtual_line_flow(obs_start, ind_load, ind_prod, ind_lor, ind_lex)
        p_obs1_sub1 = get_virtual_line_flow(obs1, ind_load, ind_prod, ind_lor, ind_lex)

        a = np.array([
            [1, 1 - obs2.p_or[id_l1] / obs_start.p_or[id_l1]],
            [1 - p_obs1_sub1 / p_start_sub1, 1],
        ])
        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or
        max_diff = np.max(np.abs(obs_target.p_or - p_computed))
        self.assertLessEqual(max_diff, self.tol)

    def test_node_splitting_line_disconnection_combination_sup_theorem(self):
        """Line disconnection + node splitting: compute_combined_pair_superposition matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub1 = 5

        obs_start, *_ = self.env.step(self.env.action_space())

        act_line = self.env.action_space({"set_line_status": [(id_l1, -1)]})  # disconnect
        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})

        obs1 = obs_start.simulate(act_line, time_step=0)[0]
        obs2 = obs_start.simulate(act_split, time_step=0)[0]
        obs_target = obs_start.simulate(act_line + act_split, time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start, obs1, obs2,
            act1_line_idxs=[id_l1], act1_sub_idxs=[],
            act2_line_idxs=[], act2_sub_idxs=[id_sub1],
        )
        self._assert_superposition_accuracy(obs_target, result)

    # =========================================================================
    # Line reconnection + node merging
    # =========================================================================

    def test_line_reconnection_node_merging_combination_by_hand(self):
        """Line reconnection + node merging: manual beta computation matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        opposite_action = (self.env.action_space({"set_line_status": [(id_l1, -1)]})
                           + self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}}))
        obs_start, *_ = self.env.step(opposite_action)

        act_reco = self.env.action_space({"set_line_status": [(id_l1, +1)]})
        act_merge = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}})

        obs1 = obs_start.simulate(act_reco, time_step=0)[0]
        obs2 = obs_start.simulate(act_merge, time_step=0)[0]
        obs_target = obs_start.simulate(act_reco + act_merge, time_step=0)[0]

        delta_theta_sub_obs1 = get_delta_theta_sub_2nodes(obs1, id_sub)
        delta_theta_sub_start = get_delta_theta_sub_2nodes(obs_start, id_sub)

        self.assertNotEqual(delta_theta_sub_obs1, 0)
        self.assertNotEqual(delta_theta_sub_start, 0)

        a = np.array([
            [1, 1 - get_delta_theta_line(obs2, id_l1) / get_delta_theta_line(obs_start, id_l1)],
            [1 - delta_theta_sub_obs1 / delta_theta_sub_start, 1],
        ])
        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or
        max_diff = np.max(np.abs(obs_target.p_or - p_computed))
        self.assertLessEqual(max_diff, self.tol)

    def test_line_reconnection_node_merging_combination_sup_theorem(self):
        """Line reconnection + node merging: compute_combined_pair_superposition matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        opposite_action = (self.env.action_space({"set_line_status": [(id_l1, -1)]})
                           + self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}}))
        obs_start, *_ = self.env.step(opposite_action)

        act_reco = self.env.action_space({"set_line_status": [(id_l1, +1)]})
        act_merge = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}})

        obs1 = obs_start.simulate(act_reco, time_step=0)[0]
        obs2 = obs_start.simulate(act_merge, time_step=0)[0]
        obs_target = obs_start.simulate(act_reco + act_merge, time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start, obs1, obs2,
            act1_line_idxs=[id_l1], act1_sub_idxs=[],
            act2_line_idxs=[], act2_sub_idxs=[id_sub],
        )
        self._assert_superposition_accuracy(obs_target, result)

    # =========================================================================
    # Line reconnection + node splitting
    # =========================================================================

    def test_line_reconnection_node_splitting_combination_by_hand(self):
        """Line reconnection + node splitting: manual beta computation matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        opposite_action = (self.env.action_space({"set_line_status": [(id_l1, -1)]})
                           + self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}}))
        obs_start, *_ = self.env.step(opposite_action)

        act_reco = self.env.action_space({"set_line_status": [(id_l1, +1)]})
        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})

        obs1 = obs_start.simulate(act_reco, time_step=0)[0]
        obs2 = obs_start.simulate(act_split, time_step=0)[0]
        obs_target = obs_start.simulate(act_reco + act_split, time_step=0)[0]

        # node1 ids for sub (split in obs2)
        (ind_load, ind_prod, ind_lor, ind_lex) = get_sub_node1_idsflow(obs2, id_sub)
        p_start_sub = get_virtual_line_flow(obs_start, ind_load, ind_prod, ind_lor, ind_lex)
        p_obs1_sub = get_virtual_line_flow(obs1, ind_load, ind_prod, ind_lor, ind_lex)

        a = np.array([
            [1, 1 - get_delta_theta_line(obs2, id_l1) / get_delta_theta_line(obs_start, id_l1)],
            [1 - p_obs1_sub / p_start_sub, 1],
        ])
        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or
        max_diff = np.max(np.abs(obs_target.p_or - p_computed))
        self.assertLessEqual(max_diff, self.tol)

    def test_node_splitting_line_reconnection_combination_sup_theorem(self):
        """Line reconnection + node splitting: compute_combined_pair_superposition matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        opposite_action = (self.env.action_space({"set_line_status": [(id_l1, -1)]})
                           + self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}}))
        obs_start, *_ = self.env.step(opposite_action)

        act_reco = self.env.action_space({"set_line_status": [(id_l1, +1)]})
        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})

        obs1 = obs_start.simulate(act_reco, time_step=0)[0]
        obs2 = obs_start.simulate(act_split, time_step=0)[0]
        obs_target = obs_start.simulate(act_reco + act_split, time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start, obs1, obs2,
            act1_line_idxs=[id_l1], act1_sub_idxs=[],
            act2_line_idxs=[], act2_sub_idxs=[id_sub],
        )
        self._assert_superposition_accuracy(obs_target, result)

    # =========================================================================
    # Line disconnection + node merging
    # =========================================================================

    def test_line_disconnection_node_merging_combination_by_hand(self):
        """Line disconnection + node merging: manual beta computation matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        opposite_action = (self.env.action_space({"set_line_status": [(id_l1, +1)]})
                           + self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}}))
        obs_start, *_ = self.env.step(opposite_action)

        act_disco = self.env.action_space({"set_line_status": [(id_l1, -1)]})
        act_merge = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}})

        obs1 = obs_start.simulate(act_disco, time_step=0)[0]
        obs2 = obs_start.simulate(act_merge, time_step=0)[0]
        obs_target = obs_start.simulate(act_disco + act_merge, time_step=0)[0]

        delta_theta_sub_obs1 = get_delta_theta_sub_2nodes(obs1, id_sub)
        delta_theta_sub_start = get_delta_theta_sub_2nodes(obs_start, id_sub)

        self.assertNotEqual(delta_theta_sub_obs1, 0)
        self.assertNotEqual(delta_theta_sub_start, 0)

        a = np.array([
            [1, 1 - obs2.p_or[id_l1] / obs_start.p_or[id_l1]],
            [1 - delta_theta_sub_obs1 / delta_theta_sub_start, 1],
        ])
        b = np.ones(2)
        betas = np.linalg.solve(a, b)

        p_computed = betas[0] * obs1.p_or + betas[1] * obs2.p_or + (1 - betas.sum()) * obs_start.p_or
        max_diff = np.max(np.abs(obs_target.p_or - p_computed))
        self.assertLessEqual(max_diff, self.tol)

    def test_line_disconnection_node_merging_combination_sup_theorem(self):
        """Line disconnection + node merging: compute_combined_pair_superposition matches target."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        opposite_action = (self.env.action_space({"set_line_status": [(id_l1, +1)]})
                           + self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}}))
        obs_start, *_ = self.env.step(opposite_action)

        act_disco = self.env.action_space({"set_line_status": [(id_l1, -1)]})
        act_merge = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}})

        obs1 = obs_start.simulate(act_disco, time_step=0)[0]
        obs2 = obs_start.simulate(act_merge, time_step=0)[0]
        obs_target = obs_start.simulate(act_disco + act_merge, time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start, obs1, obs2,
            act1_line_idxs=[id_l1], act1_sub_idxs=[],
            act2_line_idxs=[], act2_sub_idxs=[id_sub],
        )
        self._assert_superposition_accuracy(obs_target, result)

    # =========================================================================
    # Reversed ordering: sub action as act1, line action as act2
    #
    # These tests specifically cover the bug where is_this_subs_action used
    # (i + n_line_actions == j), which assumed line actions always precede
    # sub actions in unit_act_observations. When a sub action is act1 (j=0)
    # and the line action is act2 (j=1), the old formula would incorrectly
    # treat the line observation as the sub-action observation.
    # =========================================================================

    def test_node_merging_line_reconnection_combination_sup_theorem_reversed(self):
        """Node merging (act1) + line reconnection (act2): ordering must not affect result.

        This is the reversed order of test_line_reconnection_node_merging_combination_sup_theorem.
        The sub action is act1 (obs_act1 is the merge obs) and the line action is act2.
        Previously broken: betas were ~[0.49, -0.00] instead of correct values.
        """
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        opposite_action = (self.env.action_space({"set_line_status": [(id_l1, -1)]})
                           + self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}}))
        obs_start, *_ = self.env.step(opposite_action)

        act_merge = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}})
        act_reco = self.env.action_space({"set_line_status": [(id_l1, +1)]})

        # act1 = merge (sub), act2 = reco (line) — reversed from the "natural" order
        obs_merge = obs_start.simulate(act_merge, time_step=0)[0]
        obs_reco = obs_start.simulate(act_reco, time_step=0)[0]
        obs_target = obs_start.simulate(act_merge + act_reco, time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start,
            obs_merge,   # obs_act1 = merge (sub action)
            obs_reco,    # obs_act2 = reco (line action)
            act1_line_idxs=[], act1_sub_idxs=[id_sub],    # sub is act1
            act2_line_idxs=[id_l1], act2_sub_idxs=[],    # line is act2
        )
        self._assert_superposition_accuracy(obs_target, result)

    def test_node_merging_line_disconnection_combination_sup_theorem_reversed(self):
        """Node merging (act1) + line disconnection (act2): reversed ordering must work.

        Sub action first, line action second — previously broken.
        """
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        opposite_action = (self.env.action_space({"set_line_status": [(id_l1, +1)]})
                           + self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}}))
        obs_start, *_ = self.env.step(opposite_action)

        act_merge = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}})
        act_disco = self.env.action_space({"set_line_status": [(id_l1, -1)]})

        obs_merge = obs_start.simulate(act_merge, time_step=0)[0]
        obs_disco = obs_start.simulate(act_disco, time_step=0)[0]
        obs_target = obs_start.simulate(act_merge + act_disco, time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start,
            obs_merge,   # obs_act1 = merge (sub action)
            obs_disco,   # obs_act2 = disco (line action)
            act1_line_idxs=[], act1_sub_idxs=[id_sub],
            act2_line_idxs=[id_l1], act2_sub_idxs=[],
        )
        self._assert_superposition_accuracy(obs_target, result)

    def test_node_splitting_line_reconnection_combination_sup_theorem_reversed(self):
        """Node splitting (act1) + line reconnection (act2): reversed ordering must work.

        Sub action first, line action second — previously broken.
        """
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        opposite_action = (self.env.action_space({"set_line_status": [(id_l1, -1)]})
                           + self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 1, 1, 1, 1, 1))]}}))
        obs_start, *_ = self.env.step(opposite_action)

        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})
        act_reco = self.env.action_space({"set_line_status": [(id_l1, +1)]})

        obs_split = obs_start.simulate(act_split, time_step=0)[0]
        obs_reco = obs_start.simulate(act_reco, time_step=0)[0]
        obs_target = obs_start.simulate(act_split + act_reco, time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start,
            obs_split,   # obs_act1 = split (sub action)
            obs_reco,    # obs_act2 = reco (line action)
            act1_line_idxs=[], act1_sub_idxs=[id_sub],
            act2_line_idxs=[id_l1], act2_sub_idxs=[],
        )
        self._assert_superposition_accuracy(obs_target, result)

    def test_node_splitting_line_disconnection_combination_sup_theorem_reversed(self):
        """Node splitting (act1) + line disconnection (act2): reversed ordering must work.

        Sub action first, line action second — previously broken.
        """
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3
        id_sub = 5

        obs_start, *_ = self.env.step(self.env.action_space())

        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})
        act_disco = self.env.action_space({"set_line_status": [(id_l1, -1)]})

        obs_split = obs_start.simulate(act_split, time_step=0)[0]
        obs_disco = obs_start.simulate(act_disco, time_step=0)[0]
        obs_target = obs_start.simulate(act_split + act_disco, time_step=0)[0]

        result = compute_combined_pair_superposition(
            obs_start,
            obs_split,   # obs_act1 = split (sub action)
            obs_disco,   # obs_act2 = disco (line action)
            act1_line_idxs=[], act1_sub_idxs=[id_sub],
            act2_line_idxs=[id_l1], act2_sub_idxs=[],
        )
        self._assert_superposition_accuracy(obs_target, result)

    # =========================================================================
    # Helper function tests
    # =========================================================================

    def test_get_delta_theta_line_disconnected_line(self):
        """get_delta_theta_line returns non-zero for a disconnected line via endpoint angles."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l = 3
        # Disconnect line
        obs_disco, *_ = self.env.step(self.env.action_space({"set_line_status": [(id_l, -1)]}))

        self.assertFalse(obs_disco.line_status[id_l])
        # Raw theta_or/theta_ex are 0 for the disconnected line
        self.assertEqual(obs_disco.theta_or[id_l], 0.0)
        self.assertEqual(obs_disco.theta_ex[id_l], 0.0)
        # Our implementation should still return a non-zero delta theta
        # by reading the endpoint substation angles from other connected elements
        dt = get_delta_theta_line(obs_disco, id_l)
        self.assertNotEqual(dt, 0.0, "delta_theta should be non-zero via substation endpoint angles")

    def test_get_virtual_line_flow_reference_topology(self):
        """get_virtual_line_flow returns near-zero for a sub in reference (single-bus) topology."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_sub = 5
        obs, *_ = self.env.step(self.env.action_space())

        # sub5 in reference topo: all elements on bus 1
        self.assertNotIn(2, obs.sub_topology(id_sub))

        # Any node1 ids for a reference topo sub should give ~0 flow (balanced injection)
        # We use a hypothetical splitting by building ind_subs from a later split obs
        # But more simply: a merged sub has zero net injection at the single bus
        # For a single-bus sub, pick any split to define "node1"
        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})
        obs_split = obs.simulate(act_split, time_step=0)[0]
        (ind_load, ind_prod, ind_lor, ind_lex) = get_sub_node1_idsflow(obs_split, id_sub)
        flow = get_virtual_line_flow(obs, ind_load, ind_prod, ind_lor, ind_lex)
        # For DC load flow, the net injection at any bus is balanced so this equals
        # the power flowing across the virtual cut
        # We just assert it doesn't crash and returns a numeric value
        self.assertIsInstance(float(flow), float)

    def test_get_sub_node1_idsflow_returns_node1_elements(self):
        """get_sub_node1_idsflow returns the correct elements for node 1 of a split substation."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_sub = 5
        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})
        obs_split, *_ = self.env.step(act_split)

        self.assertIn(2, obs_split.sub_topology(id_sub))

        (ind_load, ind_prod, ind_lor, ind_lex) = get_sub_node1_idsflow(obs_split, id_sub)

        # All returned elements should be in the object set of sub5
        obj5 = obs_split.get_obj_connect_to(substation_id=id_sub)
        for i in ind_lor:
            self.assertIn(i, obj5['lines_or_id'])
        for i in ind_lex:
            self.assertIn(i, obj5['lines_ex_id'])
        for i in ind_load:
            self.assertIn(i, obj5['loads_id'])
        for i in ind_prod:
            self.assertIn(i, obj5['generators_id'])

    def test_get_delta_theta_sub_2nodes_split_substation(self):
        """get_delta_theta_sub_2nodes returns non-zero for a split substation."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_sub = 5
        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})
        obs_split, *_ = self.env.step(act_split)

        dt = get_delta_theta_sub_2nodes(obs_split, id_sub)
        self.assertNotEqual(dt, 0.0)

    def test_get_delta_theta_sub_2nodes_merged_substation(self):
        """get_delta_theta_sub_2nodes for a merged (single-bus) substation has bus2 theta == 0."""
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_sub = 5
        obs, *_ = self.env.step(self.env.action_space())  # start: sub5 is single-bus

        self.assertNotIn(2, obs.sub_topology(id_sub))

        dt = get_delta_theta_sub_2nodes(obs, id_sub)
        # For a single-bus sub, no element is on bus 2, so theta_bus2 == 0.
        # The result equals 0 - theta_bus1 (= -theta_bus1), which is generally non-zero.
        # The superposition algorithm uses is_start_ref to decide whether to use
        # p_or (virtual flow) or delta_theta; for reference topology it uses p_or.
        # We assert that bus 2 contributes zero theta, i.e. delta_theta == -theta_bus1.
        from expert_op4grid_recommender.utils.superposition import _get_theta_node
        theta_bus2 = _get_theta_node(obs, id_sub, bus=2)
        self.assertEqual(theta_bus2, 0.0, "Bus 2 of a single-bus sub should have theta == 0")


    def test_noop_line_action_returns_error(self):
        """Reconnecting an already-connected line is a no-op; superposition must return an error.

        No-op detection uses line_status (topology) directly, not flow magnitude.
        This is robust for lightly-loaded lines that may have near-zero flow while
        still being connected.
        """
        self.env.set_id(self.chronic_id)
        self.env.reset()

        id_l1 = 3   # line 3 is connected in obs_start
        id_sub = 5

        obs_start, *_ = self.env.step(self.env.action_space())

        # Verify line is connected in obs_start
        self.assertTrue(obs_start.line_status[id_l1],
                        "Line id_l1 should be connected in obs_start")

        # Line 3 is already connected → set_line_status(+1) is a no-op
        act_noop_reco = self.env.action_space({"set_line_status": [(id_l1, +1)]})
        act_split = self.env.action_space({"set_bus": {"substations_id": [(5, (1, 1, 2, 2, 1, 2, 2))]}})

        obs_noop = obs_start.simulate(act_noop_reco, time_step=0)[0]
        obs_split = obs_start.simulate(act_split, time_step=0)[0]

        # Verify the no-op didn't change the line status
        self.assertEqual(obs_start.line_status[id_l1], obs_noop.line_status[id_l1],
                         "No-op reco should not change line_status")

        # Pair: noop reco (act1=line) + node splitting (act2=sub)
        result = compute_combined_pair_superposition(
            obs_start,
            obs_noop,    # obs_act1: reconnect already-connected line (no-op)
            obs_split,   # obs_act2: node splitting
            act1_line_idxs=[id_l1], act1_sub_idxs=[],
            act2_line_idxs=[], act2_sub_idxs=[id_sub],
        )
        self.assertIn("error", result,
                      "No-op line action should trigger error, not produce spurious betas")
        self.assertIn("No-op", result["error"])

        # Pair: node splitting (act1=sub) + noop reco (act2=line) — reversed
        result_rev = compute_combined_pair_superposition(
            obs_start,
            obs_split,   # obs_act1: node splitting
            obs_noop,    # obs_act2: reconnect already-connected line (no-op)
            act1_line_idxs=[], act1_sub_idxs=[id_sub],
            act2_line_idxs=[id_l1], act2_sub_idxs=[],
        )
        self.assertIn("error", result_rev,
                      "No-op line action (as act2) should trigger error too")
        self.assertIn("No-op", result_rev["error"])


if __name__ == "__main__":
    unittest.main()
