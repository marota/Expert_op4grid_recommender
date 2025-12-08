# expert_op4grid_recommender/graph_analysis/builder.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

import networkx as nx
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph, Structured_Overload_Distribution_Graph
#from ..config import PARAM_OPTIONS_EXPERT_OP
from expert_op4grid_recommender.config import PARAM_OPTIONS_EXPERT_OP

def inhibit_swapped_flows(df_of_g):
    df_of_g.loc[df_of_g.new_flows_swapped, "delta_flows"] = -df_of_g[df_of_g.new_flows_swapped]["delta_flows"]
    idx_or = df_of_g[df_of_g.new_flows_swapped]["idx_or"]
    df_of_g.loc[df_of_g.new_flows_swapped, "idx_or"] = df_of_g[df_of_g.new_flows_swapped]["idx_ex"]
    df_of_g.loc[df_of_g.new_flows_swapped, "idx_ex"] = idx_or
    return df_of_g


def build_overflow_graph(env, obs_overloaded, overloaded_line_ids, non_connected_reconnectable_lines,
                         lines_non_reconnectable, timestep, do_consolidate_graph=True,
                         inhibit_swapped_flow_reversion=True, node_renaming=True,param_options=PARAM_OPTIONS_EXPERT_OP):
    """
    Constructs and refines an overflow graph based on a Grid2Op simulation state.

    This function simulates the disconnection of specified overloaded lines and uses the
    resulting changes in power flow (calculated by `Grid2opSimulation` from `alphaDeesp`)
    to build an initial overflow graph (`OverFlowGraph`). It then performs several optional
    refinement steps:
    1.  Inhibits legacy flow swapping behavior if requested.
    2.  Renames graph nodes from integer IDs to substation names.
    3.  Consolidates the graph using heuristics from `alphaDeesp` if requested.
    4.  Adds potentially relevant disconnected but reconnectable lines to the graph.

    Args:
        env (grid2op.Environment): The Grid2Op environment instance.
        obs_overloaded (grid2op.Observation): The observation object representing the grid
            state *before* disconnecting the overloaded lines (typically after an N-1 contingency).
        overloaded_line_ids (list[int]): List of integer indices of the lines considered
            overloaded and whose disconnection will be simulated.
        non_connected_reconnectable_lines (list[str]): List of line names that are currently
            disconnected but could potentially be reconnected.
        lines_non_reconnectable (list[str]): List of line names that are permanently
            disconnected or should not be considered for reconnection.
        timestep (int): The simulation timestep index.
        do_consolidate_graph (bool, optional): If True, applies graph consolidation heuristics.
            Defaults to True.
        inhibit_swapped_flow_reversion (bool, optional): If True, corrects potential flow
            direction swapping from older `alphaDeesp` logic. Defaults to True.
        node_renaming (bool, optional): If True, renames graph nodes from indices to
            substation names. Defaults to True.
        param_options (dict, optional): Configuration dictionary for the `Grid2opSimulation`
            (e.g., thresholds). Defaults to `PARAM_OPTIONS_EXPERT_OP` from config.

    Returns:
        tuple: A tuple containing:
            - df_of_g (pd.DataFrame): DataFrame detailing flow changes on lines after simulating
              the disconnection of `overloaded_line_ids`.
            - overflow_sim (Grid2opSimulation): The `alphaDeesp` simulation object used for
              calculating flow changes.
            - g_overflow (OverFlowGraph): The constructed and potentially refined overflow graph object.
            - real_hubs (list): List of nodes identified as hubs by `Structured_Overload_Distribution_Graph`
              before adding reconnectable lines.
            - g_distribution_graph (Structured_Overload_Distribution_Graph): The final `alphaDeesp`
              distribution graph object derived from the refined `g_overflow`.
            - node_name_mapping (dict): A dictionary mapping original node indices to substation names.
    """
    overflow_sim = Grid2opSimulation(
        obs_overloaded, env.action_space, env.observation_space,
        param_options=param_options, debug=False,
        ltc=overloaded_line_ids, plot=True, simu_step=timestep
    )
    df_of_g = overflow_sim.get_dataframe()
    df_of_g["line_name"] = obs_overloaded.name_line

    if inhibit_swapped_flow_reversion:
        df_of_g = inhibit_swapped_flows(df_of_g)

    g_overflow = OverFlowGraph(overflow_sim.topo, overloaded_line_ids, df_of_g, float_precision="%.0f")
    node_name_mapping = {i: name for i, name in enumerate(obs_overloaded.name_sub)}

    if node_renaming:
        g_overflow.g = nx.relabel_nodes(g_overflow.g, node_name_mapping, copy=True)

    g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)
    c_path_init = g_distribution_graph.constrained_path.full_n_constrained_path()

    if len(g_distribution_graph.g_only_red_components.nodes) != 0 and do_consolidate_graph:
        g_overflow.consolidate_graph(g_distribution_graph,
                                     non_connected_lines_to_ignore=non_connected_reconnectable_lines + lines_non_reconnectable,
                                     no_desambiguation=True)
        g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)

    real_hubs = g_distribution_graph.get_hubs()

    #for _ in range(2):  # Iterate to find complex paths
    g_overflow.add_relevant_null_flow_lines_all_paths(g_distribution_graph,
                                                      non_connected_lines=non_connected_reconnectable_lines,
                                                      non_reconnectable_lines=lines_non_reconnectable)
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g,possible_hubs=c_path_init)

    return df_of_g, overflow_sim, g_overflow, real_hubs, g_distribution_graph, node_name_mapping