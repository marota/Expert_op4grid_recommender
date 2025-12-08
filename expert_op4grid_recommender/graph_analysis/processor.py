# expert_op4grid_recommender/graph_analysis/processor.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

import numpy as np
import networkx as nx
import re
from alphaDeesp.core.graphsAndPaths import Structured_Overload_Distribution_Graph


def get_n_connected_components_graph_with_overloads(obs_simu, lines_overloaded_ids):
    """
    Analyzes the change in connected components of the grid graph as specified overloaded lines are disconnected.

    This function takes the current grid state and a list of overloaded line IDs.
    It simulates the disconnection of these lines in two stages: first, only the line
    with the maximum load (`rho`) among the specified overloads, and second, all
    other specified overloaded lines. It returns the sets of connected components
    at each stage.

    Args:
        obs_simu (grid2op.Observation): The Grid2Op observation object representing
            the current state of the power grid, including topology and line loadings.
        lines_overloaded_ids (list[int]): A list of integer indices corresponding to the
            lines considered overloaded. The analysis focuses only on these lines.

    Returns:
        tuple[list[set], list[set], list[set]]: A tuple containing three lists of sets:
            - comps_init: A list where each set contains the nodes (bus IDs) belonging
              to a connected component in the initial state (after removing lines
              that are already disconnected in the observation).
            - comps_wo_max_overload: A list of sets representing the connected components
              after removing the single edge corresponding to the line with the highest
              `rho` value among `lines_overloaded_ids`.
            - comps_wo_all_overloads: A list of sets representing the connected components
              after removing all edges corresponding to the lines listed in
              `lines_overloaded_ids`.
    """
    #TODO: adapt by simplifying, just get topo graph and add it a_or and rho attributes per edges
    # 2. List of edges: (Node1, Node2, {attributes})
    edge_list = [("subid_"+str(or_subid)+"_bus_"+str(or_bus),"subid_"+str(ex_subid)+"_bus_"+str(ex_bus),{'name': edge_name,"a_or": a_or,"rho":rho})
                 for edge_name, or_subid, or_bus, ex_subid, ex_bus, line_status, a_or, rho in
                 zip(obs_simu.name_line, obs_simu.line_or_to_subid, obs_simu.line_or_bus, obs_simu.line_ex_to_subid,
                     obs_simu.line_ex_bus,obs_simu.line_status,obs_simu.a_or,obs_simu.rho)]# if line_status]

    obs_graph=nx.from_edgelist(edge_list)

    #previous approach with energy graph, wass buggy in French grid
    #obs_graph = obs_simu.get_energy_graph()
    max_rho = max([obs_simu.rho[i] for i in lines_overloaded_ids])
    edges_disconnected = [edge for edge, a_or in nx.get_edge_attributes(obs_graph, 'a_or').items() if
                          np.round(a_or, 3) == 0]
    recover_max_overload_edge = \
    [edge for edge, rho in nx.get_edge_attributes(obs_graph, 'rho').items() if rho == max_rho][0]
    recover_other_overload_edge = [
        edge for edge, rho in nx.get_edge_attributes(obs_graph, 'rho').items()
        for line_id in lines_overloaded_ids if rho == obs_simu.rho[line_id] and edge != recover_max_overload_edge
    ]

    graph = nx.Graph(obs_graph)
    graph.remove_edges_from(edges_disconnected)
    comps_init = list(nx.connected_components(graph))

    graph.remove_edges_from([recover_max_overload_edge])
    comps_wo_max_overload = list(nx.connected_components(graph))

    graph.remove_edges_from(recover_other_overload_edge)
    comps_wo_all_overloads = list(nx.connected_components(graph))

    return sorted(comps_init, key=len,reverse=True), sorted(comps_wo_max_overload, key=len,reverse=True), sorted(comps_wo_all_overloads, key=len,reverse=True)


def get_subs_islanded_by_overload_disconnections(obs_simu, comps_init, comp_overloads, max_overload_name):
    """
    Identifies substations that become islanded after the disconnection of overloaded lines.

    This function compares the set of nodes in the main connected component of the grid
    before and after simulating the disconnection of overloaded lines. Nodes present
    initially but missing afterwards are considered part of islanded sections. It maps
    these node IDs to substation names where possible and prints the results.

    Args:
        obs_simu (grid2op.Observation): The Grid2Op observation object, used to map
            node indices to substation names (`obs_simu.name_sub`).
        comps_init (list[set]): A list of sets, where `comps_init[0]` is expected to be
            the set of node indices in the main connected component *before*
            disconnection.
        comp_overloads (list[set]): A list of sets, where `comp_overloads[0]` is expected
            to be the set of node indices in the main connected component *after*
            disconnection.
        max_overload_name (str): The name of the primary overloaded line whose disconnection
            event is being analyzed (used for informative printing).

    Returns:
        list[str]: A list containing the names of the substations that were identified
                   as becoming islanded (disconnected from the main component) due to
                   the simulated disconnections.
    """
    n_subs = len(obs_simu.name_sub)
    if type(list(comps_init[0])[0])==int:
        subs_broken_apart_ids = comps_init[0] - comp_overloads[0]
    else:
        identified_subs_broken_text = list(comps_init[0] - comp_overloads[0])
        subs_broken_apart_ids=[int(subs_text.split('_')[1]) for subs_text in identified_subs_broken_text]

    identified_subs_broken_apart = [obs_simu.name_sub[i] for i in subs_broken_apart_ids if i < n_subs]
    print(
        f"These identified substations are broken apart by only disconnecting the max overload {max_overload_name}: {identified_subs_broken_apart}")

    if len(identified_subs_broken_apart) < len(subs_broken_apart_ids):
        print(
            f"These non-identified multi-node IDs are broken apart: {[i for i in subs_broken_apart_ids if i >= n_subs]}")

    return identified_subs_broken_apart


def identify_overload_lines_to_keep_overflow_graph_connected(obs_simu, lines_overloaded_ids,
                                                             force_keep_max_overload_id=False):
    """
    Determines which overloaded lines to consider for overflow graph analysis based on grid connectivity.

    This function analyzes whether disconnecting the specified overloaded lines would
    fragment the power grid into multiple connected components. The goal is to select
    a subset of overloads for the `Grid2opSimulation` such that the simulated
    disconnection doesn't immediately lead to an islanded grid, which would make
    topological solutions difficult or impossible.

    The logic is as follows:
    1. Check if disconnecting *all* specified `lines_overloaded_ids` increases the
       number of connected components compared to the initial state.
       - If NO: Keep *all* `lines_overloaded_ids` for the overflow graph analysis.
    2. If YES (disconnecting all breaks the grid): Check if disconnecting *only* the
       line with the *maximum rho* among `lines_overloaded_ids` increases the
       number of components.
       - If NO (or if `force_keep_max_overload_id` is True): Keep *only* the line(s)
         with the maximum rho. This focuses the analysis on the most severe overload
         while avoiding immediate grid fragmentation. Also identifies substations islanded
         by disconnecting *all* overloads.
       - If YES (even disconnecting the single max overload breaks the grid): Return `None`
         for the lines to keep, indicating that the situation might require load shedding
         or other non-topological actions. Also identifies substations islanded by
         disconnecting just the max overload.

    Args:
        obs_simu (grid2op.Observation): The Grid2Op observation representing the current
            grid state (usually after an N-1 contingency).
        lines_overloaded_ids (list[int]): A list of integer indices for the lines initially
            identified as overloaded.
        force_keep_max_overload_id (bool, optional): If True, forces the function to keep
            at least the maximum overloaded line, even if disconnecting it alone would
            normally suggest the problem is topologically unsolvable. This might be useful
            if reconnectable lines could potentially bridge the gap later. Defaults to False.

    Returns:
        tuple:
            - lines_overloaded_ids_to_keep (list[int] or None): A list containing the indices
              of the overloaded lines recommended for inclusion in the `Grid2opSimulation`
              for building the overflow graph. Returns `None` if disconnecting even the
              single max overload fragments the grid (and force_keep is False).
            - prevent_islanded_subs (list[str]): A list of substation names that would
              become islanded if the disconnections causing grid fragmentation were performed.
              This list is populated only when the function decides to keep a subset
              of lines (or none).
    """
    if not lines_overloaded_ids:
        return [], []
    max_rho = max([obs_simu.rho[i] for i in lines_overloaded_ids])
    max_overload_name = [obs_simu.name_line[i] for i in lines_overloaded_ids if obs_simu.rho[i] == max_rho][0]

    comps_init, comps_wo_max_overload, comps_wo_all_overloads = get_n_connected_components_graph_with_overloads(
        obs_simu, lines_overloaded_ids)
    n_connected_comp_init = len(comps_init)
    n_connected_comp_max_overload = len(comps_wo_max_overload)
    n_connected_comp_all_overload = len(comps_wo_all_overloads)

    prevent_islanded_subs = []
    if n_connected_comp_init == n_connected_comp_all_overload:
        lines_overloaded_ids_to_keep = lines_overloaded_ids
    elif n_connected_comp_max_overload == n_connected_comp_init or force_keep_max_overload_id:
        lines_overloaded_ids_to_keep = [
            [line_id for line_id in lines_overloaded_ids if obs_simu.rho[line_id] == max_rho][0]]
        prevent_islanded_subs = get_subs_islanded_by_overload_disconnections(obs_simu, comps_init,
                                                                             comps_wo_all_overloads, max_overload_name)
        print(
            f"we reduce the problem by focusing on the deepest overload {obs_simu.name_line[lines_overloaded_ids_to_keep[0]]}")
    else:
        prevent_islanded_subs = get_subs_islanded_by_overload_disconnections(obs_simu, comps_init,
                                                                             comps_wo_max_overload, max_overload_name)
        lines_overloaded_ids_to_keep = None

    return lines_overloaded_ids_to_keep, prevent_islanded_subs


def get_constrained_and_dispatch_paths(g_distribution_graph, obs, lines_overloaded_ids, lines_overloaded_ids_kept):
    """
    Extracts constrained (blue) and dispatch (red) paths from a structured overload distribution graph.

    This function utilizes methods from the `Structured_Overload_Distribution_Graph` object
    to identify key paths relevant for analyzing power flow and applying expert rules.
    It retrieves the lines and nodes belonging to the "constrained path" (where flow must pass)
    and the "dispatch paths" (alternative routes where flow can be potentially redirected).

    Additionally, it performs a sanity check: if the analysis focuses on a subset of the
    originally identified overloads (`lines_overloaded_ids_kept` vs `lines_overloaded_ids`),
    it verifies that all original overloads are present within the identified constrained paths.
    A warning is printed if any original overloads are missing, suggesting a potential
    inconsistency in the graph structure or analysis assumptions.

    Args:
        g_distribution_graph (Structured_Overload_Distribution_Graph): An instance of the
            `alphaDeesp` graph object containing the structured analysis of power flow
            distribution based on the overflow graph.
        obs (grid2op.Observation): The Grid2Op observation object, used here to map line
            indices (`lines_overloaded_ids`) to line names for the warning message.
        lines_overloaded_ids (list[int]): List of indices for all lines initially identified
            as overloaded in the scenario.
        lines_overloaded_ids_kept (list[int]): List of indices for the overloaded lines that
            were actually used to build the overflow graph (potentially a subset of
            `lines_overloaded_ids` if some were excluded to maintain connectivity).

    Returns:
        tuple[list, list, list, list]: A tuple containing four lists:
            - lines_blue_paths (list): List of line identifiers (usually names or indices,
              depending on `g_distribution_graph`) belonging to the constrained paths (blue edges).
            - nodes_blue_path (list): List of node identifiers (usually names or indices)
              belonging to the constrained paths (blue nodes).
            - lines_dispatch (list): List of line identifiers belonging to the dispatch paths
              (red edges, including loops and non-loops).
            - nodes_dispatch_path (list): List of node identifiers belonging to the dispatch paths
              (red nodes, including loops and non-loops).
    """
    lines_constrained_path, nodes_constrained_path, other_blue_edges, other_blue_nodes = g_distribution_graph.get_constrained_edges_nodes()
    lines_blue_paths = lines_constrained_path + other_blue_edges
    nodes_blue_path = nodes_constrained_path + other_blue_nodes

    if len(lines_overloaded_ids_kept) != len(lines_overloaded_ids):
        missing_overload_constrained_path = set(obs.name_line[lines_overloaded_ids]) - set(lines_constrained_path)
        if missing_overload_constrained_path:
            print(
                f"⚠️ WARNING: The overload graph might be inconsistent — lines {missing_overload_constrained_path} are not present in the constrained path.")

    lines_dispatch, nodes_dispatch_path = g_distribution_graph.get_dispatch_edges_nodes(only_loop_paths=False)
    return lines_blue_paths, nodes_blue_path, lines_dispatch, nodes_dispatch_path


def pre_process_graph_alphadeesp(g_overflow, overflow_sim, node_name_mapping):
    """
    Prepares the overflow graph and extracts simulation data for use with AlphaDeesp functions.

    AlphaDeesp's core functions often expect specific data formats, such as integer node IDs
    and purely numeric edge labels. This function performs the necessary transformations:
    1. Extracts relevant simulation metadata (substation elements, mappings) from the
       `Grid2opSimulation` object.
    2. Reverts the graph node labels from substation names back to their original integer indices
       using the provided `node_name_mapping`.
    3. Cleans up edge 'label' attributes, attempting to extract the primary numeric value
       (representing flow change) if the label contains additional text (e.g., units or line names).
    4. Re-initializes the `Structured_Overload_Distribution_Graph` using the processed graph.

    Args:
        g_overflow (OverFlowGraph): The overflow graph object, potentially with nodes already
            relabeled to substation names. Its internal graph (`g_overflow.g`) will be modified.
        overflow_sim (Grid2opSimulation): The `alphaDeesp` simulation object used to generate
            the overflow graph, containing necessary metadata.
        node_name_mapping (dict): A dictionary mapping original integer node indices (keys) to
            substation names (values). Used to reverse the node labeling.

    Returns:
        tuple: A tuple containing:
            - g_overflow (OverFlowGraph): The modified overflow graph object, now with integer node IDs
              and cleaned edge labels.
            - g_distribution_graph (Structured_Overload_Distribution_Graph): A new distribution graph
              object created from the processed `g_overflow.g`.
            - simulator_data (dict): A dictionary containing metadata extracted from `overflow_sim`,
              required by certain AlphaDeesp functions (e.g., `AlphaDeesp_warmStart`). Includes keys
              like 'substations_elements', 'substation_to_node_mapping', etc.
    """
    simulator_data = {
        "substations_elements": overflow_sim.get_substation_elements(),
        "substation_to_node_mapping": overflow_sim.get_substation_to_node_mapping(),
        "internal_to_external_mapping": overflow_sim.get_internal_to_external_mapping()
    }
    reverse_node_name_mapping = {name: i for i, name in node_name_mapping.items()}
    g_overflow.g = nx.relabel_nodes(g_overflow.g, reverse_node_name_mapping, copy=True)

    all_edges_xlabel_attributes = nx.get_edge_attributes(g_overflow.g, "label")
    for edge, label in all_edges_xlabel_attributes.items():
        try:
            float(label)
        except (ValueError, TypeError):
            match = re.search(r'[-+]?\d+', str(label))
            if match:
                all_edges_xlabel_attributes[edge] = match.group()

    nx.set_edge_attributes(g_overflow.g, all_edges_xlabel_attributes, "label")
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)
    return g_overflow, g_distribution_graph, simulator_data