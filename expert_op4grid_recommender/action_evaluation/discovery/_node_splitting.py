# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Node splitting (bus split) discovery and scoring mixin."""
import numpy as np
import networkx as nx
from typing import Any, Dict, List, Tuple

from alphaDeesp.core.alphadeesp import AlphaDeesp_warmStart
from expert_op4grid_recommender.utils.helpers import sort_actions_by_score

class NodeSplittingMixin:
    """Node splitting (bus split) discovery and scoring mixin."""

    def identify_bus_of_interest_in_node_splitting_(
        self, node_type, buses, buses_negative_inflow, buses_negative_out_flow
    ):
        """
        Identifies the specific bus within a split node that is most critical for relieving the constraint.

        The logic depends on whether the node is upstream (amont) or downstream (aval) of the constraint:
        - **Amont (Upstream):** We look for the bus with the most negative dispatch outflow (pushing flow away).
        - **Aval (Downstream):** We look for the bus with the most negative dispatch inflow.

        Args:
            node_type (str): The classification of the node ('amont', 'aval', or other).
            buses (list): List of bus identifiers involved in the split (e.g., [1, 2]).
            buses_negative_inflow (list or np.array): Magnitude of negative inflow for each bus.
            buses_negative_out_flow (list or np.array): Magnitude of negative outflow for each bus.

        Returns:
            int: The identifier of the bus of interest (e.g., 1 or 2).
        """
        bus_of_interest = 1
        buses_negative_inflow = np.array(buses_negative_inflow)
        buses_negative_out_flow = np.array(buses_negative_out_flow)

        # Check if there is any negative flow to analyze
        if np.sum(buses_negative_inflow) != 0 or np.sum(buses_negative_out_flow) != 0:
            if node_type == "amont":
                # a) is_Amont: At least one out_edge belongs to the constrained path.
                # Strategy: Identify the bus with the most negative dispatch outflow.
                bus_of_interest = buses[
                    np.argmax(buses_negative_out_flow - buses_negative_inflow)
                ]
            elif node_type == "aval":
                # b) is_Aval: At least one in_edge belongs to the constrained path.
                # Strategy: Identify the bus with the most negative dispatch inflow.
                bus_of_interest = buses[
                    np.argmax(buses_negative_inflow - buses_negative_out_flow)
                ]
            else:
                # Fallback for loop nodes or unclassified nodes: find max absolute difference
                id_of_interest = np.argmax(
                    abs(buses_negative_inflow - buses_negative_out_flow)
                )
                bus_of_interest = buses[id_of_interest]
        else:
            print("Warning: no negative flow detected")

        return bus_of_interest

    def computing_buses_values_of_interest(self, graph, node, dict_edge_names_buses):
        """
        Aggregates flow attributes for all buses within a specific node (substation).

        Calculates total positive/negative inflows and outflows for every bus configuration
        in the provided graph.

        Args:
            graph (nx.Graph): The network graph containing edge attributes (flow values).
            node (int): The ID of the node (substation) being analyzed.
            dict_edge_names_buses (dict): Mapping of line names to their assigned bus (1 or 2).

        Returns:
            tuple:
                - buses (list): Unique bus IDs (excluding 0 or -1 which represent disconnection).
                - buses_negative_inflow (list): Sum of negative inflows per bus.
                - buses_negative_out_flow (list): Sum of negative outflows per bus.
                - buses_positive_inflow (list): Sum of positive inflows per bus.
                - buses_positive_out_flow (list): Sum of positive outflows per bus.
        """
        all_edges_value_attributes = nx.get_edge_attributes(
            graph, "label"
        )  # dict[edge_tuple] -> flow value
        all_edges_names = nx.get_edge_attributes(graph, "name")

        # Filter for valid buses (Grid2Op buses usually start at 1; 0/-1 are disconnected)
        buses = list(set(dict_edge_names_buses.values()) - set([0, -1]))

        # Helper function to sum flows based on direction and sign
        # Note: 'keys=True' is used because this is likely a MultiGraph
        buses_negative_inflow = [
            np.sum(
                [
                    abs(float(all_edges_value_attributes[edge]))
                    for edge in graph.in_edges(node, keys=True)
                    if dict_edge_names_buses[all_edges_names[edge]] == bus
                    and float(all_edges_value_attributes[edge]) < 0
                ]
            )
            for bus in buses
        ]

        buses_negative_out_flow = [
            np.sum(
                [
                    abs(float(all_edges_value_attributes[edge]))
                    for edge in graph.out_edges(node, keys=True)
                    if dict_edge_names_buses[all_edges_names[edge]] == bus
                    and float(all_edges_value_attributes[edge]) < 0
                ]
            )
            for bus in buses
        ]

        buses_positive_inflow = [
            np.sum(
                [
                    abs(float(all_edges_value_attributes[edge]))
                    for edge in graph.in_edges(node, keys=True)
                    if dict_edge_names_buses[all_edges_names[edge]] == bus
                    and float(all_edges_value_attributes[edge]) >= 0
                ]
            )
            for bus in buses
        ]

        buses_positive_out_flow = [
            np.sum(
                [
                    abs(float(all_edges_value_attributes[edge]))
                    for edge in graph.out_edges(node, keys=True)
                    if dict_edge_names_buses[all_edges_names[edge]] == bus
                    and float(all_edges_value_attributes[edge]) >= 0
                ]
            )
            for bus in buses
        ]

        return (
            buses,
            buses_negative_inflow,
            buses_negative_out_flow,
            buses_positive_inflow,
            buses_positive_out_flow,
        )

    def identify_node_splitting_type(self, node, g_distribution_graph):
        """
        Classifies a node based on its position relative to the constrained path.

        Args:
            node (int): The node ID.
            g_distribution_graph (object): An object containing the constrained path and loop data.

        Returns:
            str: Node type ('amont', 'aval', 'loop', or None).
        """
        constrained_path = g_distribution_graph.get_constrained_path()
        red_loops = g_distribution_graph.get_loops()
        red_loops_nodes = set(
            [x for loop in range(len(red_loops.Path)) for x in red_loops.Path[loop]]
        )

        node_type = None
        if node in constrained_path.n_amont():
            node_type = "amont"  # Upstream
        elif node in constrained_path.n_aval():
            node_type = "aval"  # Downstream
        elif node in red_loops_nodes:
            node_type = "loop"  # Part of a topological loop

        return node_type

    def compute_node_splitting_action_bus_score(
        self,
        node_type,
        bus_of_interest,
        buses,
        buses_negative_inflow,
        buses_negative_out_flow,
        buses_positive_inflow,
        buses_positive_out_flow,
    ):
        """
        Calculates a heuristic score for a splitting action on a specific bus.

        The score represents how effectively the split separates "helpful" flows (negative dispatch)
        from "harmful" flows. It uses a combination of a weight factor and a repulsion factor.

        Score Formula:
            $$Score = WeightFactor \times Repulsion$$

        Where for **Amont** (Upstream):
            - $WeightFactor = \frac{NegOut - (NegIn + PosIn + PosOut)}{TotalFlow}$
            - $Repulsion = NegOut - PosOut$

        Args:
            node_type (str): 'amont', 'aval', or implied by flow direction.
            bus_of_interest (int): The bus ID being scored.
            buses (list): List of all bus IDs.
            buses_negative_inflow (list): Negative inflow values per bus.
            buses_negative_out_flow (list): Negative outflow values per bus.
            buses_positive_inflow (list): Positive inflow values per bus.
            buses_positive_out_flow (list): Positive outflow values per bus.

        Returns:
            float: The calculated score. Higher is better.
        """
        idx_bus_of_interest = [
            i for i, bus in enumerate(buses) if bus == bus_of_interest
        ][0]

        # Extract flows for the specific bus
        bus_negative_inflow = buses_negative_inflow[idx_bus_of_interest]
        bus_negative_out_flow = buses_negative_out_flow[idx_bus_of_interest]
        bus_positive_inflow = buses_positive_inflow[idx_bus_of_interest]
        bus_positive_out_flow = buses_positive_out_flow[idx_bus_of_interest]

        TotalInOutDispatchFlow = (
            bus_negative_inflow
            + bus_negative_out_flow
            + bus_positive_inflow
            + bus_positive_out_flow
        )

        harmonized_node_type = node_type

        # If not explicitly Amont or Aval, infer type based on dominant negative flow direction
        if node_type not in ["amont", "aval"]:
            # TODO: interesting cases to test for
            # 1) GEN.PP6 on Full France for SAOL31RONCI contingency on case 20240828T0100Z => should not be effective
            # 2) FRON5L31LOUHA_chronic_20240828_0000_timestep_36 and subs CREYSP7, MAGNYP3, GEN.PP6, FLEYRP6
            # 3) P.SAOL31RONCI_chronic_20240828_0000_timestep_1 and sub MAGNYP6
            # 4) a case where VOUGLP6 is in the middle of loop paths but with negative dispatch flows. But which case was this ? There is P.SAOL31RONCI so should probably be BEON L31CPVAN contingency
            if bus_negative_out_flow >= bus_negative_inflow:
                harmonized_node_type = (
                    "amont"  # Treat as upstream (negative out edge is dominant)
                )
            else:
                harmonized_node_type = "aval"

        weight_factor = 0
        repulsion = 0

        if TotalInOutDispatchFlow == 0:
            score = -9999
        else:
            if harmonized_node_type == "amont":
                # Strategy: Separate negative outflow from all other flows.
                # Repulsion: Contrast negative outflow against positive outflow (separation of paths).
                repulsion = bus_negative_out_flow - bus_positive_out_flow
                # Weight: Ratio of the desired flow vs everything else.
                weight_factor = (
                    bus_negative_out_flow
                    - (
                        bus_negative_inflow
                        + bus_positive_inflow
                        + bus_positive_out_flow
                    )
                ) / TotalInOutDispatchFlow

            elif harmonized_node_type == "aval":
                # Strategy: Separate negative inflow from all other flows.
                repulsion = bus_negative_inflow - bus_positive_inflow
                if TotalInOutDispatchFlow == 0:
                    weight_factor = -9999
                else:
                    weight_factor = (
                        bus_negative_inflow
                        - (
                            bus_negative_out_flow
                            + bus_positive_inflow
                            + bus_positive_out_flow
                        )
                    ) / TotalInOutDispatchFlow

            score = weight_factor * repulsion

            # Penalize configurations where the split results in opposing indicators (both negative)
            if weight_factor < 0 and repulsion < 0:
                score = -score

        return score

    def compute_node_splitting_action_score_value(
        self,
        overflow_graph,
        g_distribution_graph,
        node: int,
        dict_edge_names_buses=None,
    ):
        """
        Main orchestration function to score a specific node splitting topology.

        Steps:
        1. Identify the node type (Upstream, Downstream, Loop).
        2. Compute flow values for all buses in the proposed split.
        3. Identify the 'bus of interest' (the one carrying the relieving flow).
        4. Compute the score based on flow separation logic.

        Args:
            overflow_graph (nx.Graph): Graph representing flows and overflows.
            g_distribution_graph (object): Object containing path constraints and loops.
            node (int): The ID of the substation being split.
            topo_vect_buses (list): The topology vector representing the split (e.g., [1, 1, 2, 2]).
            dict_edge_names_buses (dict): Mapping of edge names to bus IDs.

        Returns:
            Tuple[float, Dict]: A tuple of (score, details) where details contains:
                - node_type: str ("amont", "aval", or other)
                - targeted_node_bus: int (bus number, replaced by targeted_node_assets downstream)
                - in_negative_flows, out_negative_flows, in_positive_flows, out_positive_flows: floats
        """
        # 1) Identify node type
        node_type = self.identify_node_splitting_type(node, g_distribution_graph)

        # 2) Compute buses values of interest (aggregate flows)
        (
            buses,
            buses_negative_inflow,
            buses_negative_out_flow,
            buses_positive_inflow,
            buses_positive_out_flow,
        ) = self.computing_buses_values_of_interest(
            overflow_graph, node, dict_edge_names_buses
        )

        # Handle edge case: no valid buses found (all disconnected or empty)
        if not buses:
            print(f"Warning: No valid buses found for node {node}, returning score 0")
            return 0.0, {}

        # 3) Detect bus of interest
        bus_of_interest = self.identify_bus_of_interest_in_node_splitting_(
            node_type, buses, buses_negative_inflow, buses_negative_out_flow
        )

        # Handle edge case: bus_of_interest not in buses list
        # This can happen when identify_bus_of_interest returns default value 1
        # but the actual buses are different (e.g., [2] or [2, 3])
        if bus_of_interest not in buses:
            # Fall back to the first available bus
            bus_of_interest = buses[0]
            print(
                f"Warning: Default bus_of_interest not in buses list, using {bus_of_interest}"
            )

        # 4) Compute score
        bus_of_interest_score = self.compute_node_splitting_action_bus_score(
            node_type,
            bus_of_interest,
            buses,
            buses_negative_inflow,
            buses_negative_out_flow,
            buses_positive_inflow,
            buses_positive_out_flow,
        )

        # 5) Build per-action details for the bus of interest
        bus_idx = buses.index(bus_of_interest)
        details = {
            "node_type": node_type,
            "targeted_node_bus": bus_of_interest,
            "in_negative_flows": float(buses_negative_inflow[bus_idx]),
            "out_negative_flows": float(buses_negative_out_flow[bus_idx]),
            "in_positive_flows": float(buses_positive_inflow[bus_idx]),
            "out_positive_flows": float(buses_positive_out_flow[bus_idx]),
        }

        return bus_of_interest_score, details

    def compute_node_splitting_action_score(
        self, action_dict: Any, sub_impacted_id: int, alphaDeesp_ranker: Any
    ) -> Tuple[float, Dict]:
        """
        Computes the heuristic score for a single node splitting action.

        Args:
            action_dict: The Grid2Op node splitting action dictionary.
            sub_impacted_id: The integer index of the affected substation.
            alphaDeesp_ranker: The initialized AlphaDeesp ranker.

        Returns:
            Tuple[float, Dict]: The heuristic score and per-action details dict.
        """
        # Extract bus assignments directly from action dictionary (backend-agnostic)
        # This avoids relying on topology vector operations which may not be available
        # in all backends (e.g., pypowsybl)

        action = self.action_space(action_dict)
        action_topo_vect, is_single_node = self._get_action_topo_vect(
            sub_impacted_id, action
        )
        action_topo_vect_alphadeesp = action_topo_vect - 1

        #########"
        dict_edge_names_buses = self._edge_names_buses_dict(
            self.obs_defaut, action_topo_vect, sub_impacted_id
        )
        # dict_edge_names_buses=self._edge_names_buses_dict_new(action_dict)#self._edge_names_buses_dict(self.obs_defaut,action_topo_vect,sub_impacted_id)

        result = self.compute_node_splitting_action_score_value(
            self.g_overflow.g,
            self.g_distribution_graph,
            node=sub_impacted_id,
            dict_edge_names_buses=dict_edge_names_buses,
        )

        # Handle both old (float) and new (tuple) return formats for backward compatibility
        if isinstance(result, tuple):
            score, details = result
        else:
            score, details = result, {}

        # Replace the raw bus number with the named assets on the targeted node
        if details and "targeted_node_bus" in details:
            details["targeted_node_assets"] = self._get_assets_on_bus_for_sub(
                sub_impacted_id,
                details["targeted_node_bus"],
                bus_assignments=action_topo_vect,
            )
            del details["targeted_node_bus"]

        return score, details

    def identify_and_score_node_splitting_actions(
        self,
        hubs_names: List[str],
        nodes_blue_path_names: List[str],
        alphaDeesp_ranker: Any,
    ) -> Tuple[Dict, List]:
        """
        Identifies relevant node splitting actions and calculates their scores.

        Args:
            hubs_names: List of hub substation names.
            nodes_blue_path_names: List of substation names on the blue path.
            alphaDeesp_ranker: The initialized AlphaDeesp ranker.

        Returns:
            A tuple: (map_action_score, ignored_actions)
        """
        map_action_score, ignored_actions = {}, []
        for action_id, action_desc in self.dict_action.items():
            if action_id not in self.actions_unfiltered:
                ignored_actions.append(action_desc)
                continue

            action_type = self.classifier.identify_action_type(
                action_desc, by_description=True
            )

            if "open_coupling" in action_type:
                action = self.action_space(action_desc["content"])

                # Get impacted substations from action description (backend-agnostic)
                # topology_impact = action.impact_on_objects()["topology"] if grid2op
                # subs_impacted = list(set([assignment['substation'] for assignment in topology_impact["assigned_bus"]]))
                # sub_impacted_id = subs_impacted[0]
                subs_impacted = self._get_subs_impacted_from_action_desc(action_desc)
                if not subs_impacted:
                    ignored_actions.append(action_desc)
                    continue

                sub_impacted_id = subs_impacted[0]
                sub_impacted_name = self.obs_defaut.name_sub[sub_impacted_id]

                if (
                    sub_impacted_name in hubs_names
                    or sub_impacted_name in nodes_blue_path_names
                ):
                    score, details = self.compute_node_splitting_action_score(
                        action_desc["content"], sub_impacted_id, alphaDeesp_ranker
                    )
                    map_action_score[action_id] = {
                        "action": action,
                        "score": score,
                        "sub_impacted": sub_impacted_name,
                        "details": details,
                    }
                    # print(action_desc["content"]["set_bus"])
                    # print(action_id+": "+str(score))
                else:
                    ignored_actions.append(action_desc)
            else:
                ignored_actions.append(action_desc)
        return map_action_score, ignored_actions

    def find_relevant_node_splitting(
        self, hubs_names: List[str], nodes_blue_path_names: List[str]
    ):
        """
        Finds, scores, sorts, and evaluates node splitting actions.

        Populates `self.identified_splits`, `self.effective_splits`, `self.ineffective_splits`,
        `self.ignored_splits`, and `self.scores_splits`.

        Args:
            hubs_names: List of hub substation names.
            nodes_blue_path_names: List of substation names on the blue path.
        """
        alphaDeesp_ranker = AlphaDeesp_warmStart(
            self.g_overflow.g, self.g_distribution_graph, self.simulator_data
        )

        map_action_score, ignored = self.identify_and_score_node_splitting_actions(
            hubs_names, nodes_blue_path_names, alphaDeesp_ranker
        )
        actions, subs_impacted, scores = sort_actions_by_score(
            map_action_score
        )  # Higher score first

        effective, ineffective = [], []
        if self.check_action_simulation and actions:
            print("  Simulating effectiveness...")
            act_defaut = self._create_default_action(
                self.action_space, self.lines_defaut
            )
            for action_id, sub_impacted in zip(actions.keys(), subs_impacted):
                action = actions[action_id]
                is_rho_reduction, _ = self._check_rho_reduction(
                    self.obs,
                    self.timestep,
                    act_defaut,
                    action,
                    self.lines_overloaded_ids,
                    self.act_reco_maintenance,
                    self.lines_we_care_about,
                )
                is_hub = sub_impacted in hubs_names
                if is_rho_reduction:
                    effective.append(action)
                    print(
                        f"    Effective node split: {action_id} at {sub_impacted} (hub: {is_hub})"
                    )
                else:
                    ineffective.append(action)
                    print(
                        f"    Ineffective node split: {action_id} at {sub_impacted} (hub: {is_hub})"
                    )

        self.identified_splits = actions
        self.effective_splits = effective
        self.ineffective_splits = ineffective
        self.ignored_splits = ignored
        self.scores_splits = scores
        self.scores_splits_dict = {
            action_id: map_action_score[action_id]["score"]
            for action_id in map_action_score
        }
        self.params_splits_dict = {
            action_id: map_action_score[action_id].get("details", {})
            for action_id in map_action_score
        }
