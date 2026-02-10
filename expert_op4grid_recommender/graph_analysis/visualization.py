# expert_op4grid_recommender/graph_analysis/visualization.py
#!/usr/bin/python3
# Copyright (c) 2025-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of expert_op4grid_recommender, Expert system analyzer based on ExpertOp4Grid principles. ⚡️ This tool builds overflow graphs,
# applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

import os
import glob
import shutil
import numpy as np
import pypowsybl as pp

# STEP 1: Import your config module
from expert_op4grid_recommender import config
from expert_op4grid_recommender.config import DRAW_ONLY_SIGNIFICANT_EDGES, USE_GRID_LAYOUT, DO_CONSOLIDATE_GRAPH


def get_zone_voltage_levels(env_path):
    """
    Loads voltage level information for substations from a PowSyBl network file.

    This function reads a network definition file (expected to be 'grid.xiidm')
    located within the specified environment path using the `pypowsybl` library.
    It extracts the nominal voltage for each voltage level defined in the file
    and returns a mapping from the substation/voltage level identifier to its
    nominal voltage value.

    Args:
        env_path (str): The file path to the directory containing the Grid2Op
            environment definition, which must include the 'grid.xiidm' file.

    Returns:
        dict: A dictionary where keys are the substation or voltage level
              identifiers (typically strings) and values are their corresponding
              nominal voltage levels (usually floats or integers, e.g., 400.0, 225.0).

    Raises:
        FileNotFoundError: If the 'grid.xiidm' file cannot be found at the
                           specified `env_path`.
        Exception: Potential exceptions from `pypowsybl.network.load` if the
                   network file is invalid or cannot be parsed.
    """
    file_iidm = "grid.xiidm"
    network_file_path = os.path.join(env_path, file_iidm)
    n_zone = pp.network.load(network_file_path)
    df_volt = n_zone.get_voltage_levels()
    return {sub: volt for sub, volt in zip(df_volt.index, df_volt.nominal_v)}


def get_graph_file_name(lines_defaut, chronic_name, timestep, use_dc_load_flow):
    """
    Loads voltage level information for substations from a PowSyBl network file.

    This function reads a network definition file (expected to be 'grid.xiidm')
    located within the specified environment path using the `pypowsybl` library.
    It extracts the nominal voltage for each voltage level defined in the file
    and returns a mapping from the substation/voltage level identifier to its
    nominal voltage value.

    Args:
        env_path (str): The file path to the directory containing the Grid2Op
            environment definition, which must include the 'grid.xiidm' file.

    Returns:
        dict: A dictionary where keys are the substation or voltage level
              identifiers (typically strings) and values are their corresponding
              nominal voltage levels (usually floats or integers, e.g., 400.0, 225.0).

    Raises:
        FileNotFoundError: If the 'grid.xiidm' file cannot be found at the
                           specified `env_path`.
        Exception: Potential exceptions from `pypowsybl.network.load` if the
                   network file is invalid or cannot be parsed.
    """
    graph_file_name = f"Overflow_Graph_{'_'.join(map(str, lines_defaut))}_chronic_{chronic_name}_timestep_{timestep}"
    graph_file_name += "_geo" if USE_GRID_LAYOUT else "_hierarchi"
    graph_file_name += "_only_signif_edges" if DRAW_ONLY_SIGNIFICANT_EDGES else "_all_edges"
    graph_file_name += "_consoli" if DO_CONSOLIDATE_GRAPH else "_no_consoli"
    if use_dc_load_flow:
        graph_file_name += "_in_DC"
    return graph_file_name


def make_overflow_graph_visualization(env, overflow_sim, g_overflow,hubs, obs_simu, save_folder, graph_file_name,
                                      lines_swapped, custom_layout=None):
    """
    Generates and saves a visualization of the overflow graph with various annotations.

    This function takes a constructed overflow graph and enhances it with visual information
    before plotting and saving it as a PDF file. The enhancements include:
    - Coloring nodes based on substation voltage levels.
    - Annotating nodes with the number of connected electrical buses.
    - Highlighting hub nodes with a distinct shape.
    - Highlighting edges where flow direction might have been swapped.
    - Annotating edges with significant line loading changes (before vs. after overload disconnection),
      unless the simulation uses a DC power flow model.

    Args:
        env (grid2op.Environment): The Grid2Op environment object (used to get line names).
        overflow_sim (alphaDeesp.Grid2opSimulation): The alphaDeesp simulation object used to
            generate the graph, containing simulation results like `obs_linecut`.
        g_overflow (alphaDeesp.OverFlowGraph): The overflow graph object to be visualized.
            This object will be modified with annotations.
        hubs (list[str]):  A list of hubs to highlight with a specific shape (diamond)
        obs_simu (grid2op.Observation): The Grid2Op observation object representing the grid state
            *before* the simulated overload disconnection (used for 'before' line loading).
        save_folder (str): The path to the directory where the output PDF should be saved.
        graph_file_name (str): The base name for the output PDF file (without the extension).
        lines_swapped (list[str]): A list of line names where the flow direction might have
            been swapped during the overflow graph calculation.
        custom_layout (dict | list | None, optional): A predefined layout for positioning the graph nodes.
            Can be a dictionary mapping node names to (x, y) coordinates or a list of coordinates
            matching the node order. If None, the plotting function determines the layout.
            Defaults to None.

    Returns:
        svg: The SVG data generated by the `g_overflow.plot` method, if any. Note that the
             primary output is the saved PDF file.

    Side Effects:
        - Creates a temporary folder within `save_folder`.
        - Saves the graph visualization as a PDF file named `<graph_file_name>.pdf` in `save_folder`.
        - Deletes the temporary folder.
        - Prints the path to the saved PDF file to the console.
    """
    rescale_factor = 3
    fontsize = 10
    node_thickness = 2
    shape_hub = "diamond"

    # STEP 2: Use the path from the config file, not env.path
    df_volt_dict = get_zone_voltage_levels(config.ENV_PATH)

    voltage_levels=set(df_volt_dict.values())
    voltage_colors = {400: "red",350: "red", 225: "darkgreen", 150: "turquoise",110: "orange",90: "gold", 63: "purple", 20: "pink", 24: "pink", 10: "pink",
                      15: "pink", 33: "pink"}
    #add default colors if missing in dictionnary
    for voltage in voltage_levels:
        if voltage not in voltage_colors:
           if voltage <63:
               voltage_colors[voltage]="pink"
           else:
               voltage_colors[voltage] = "grey"
    #set colors
    g_overflow.set_voltage_level_color(df_volt_dict, voltage_colors)

    # Add node numbers
    number_nodal_dict = {sub_name: len(set(obs_simu.sub_topology(i)) - {-1}) for i, sub_name in
                         enumerate(obs_simu.name_sub)}
    g_overflow.set_electrical_node_number(number_nodal_dict)

    # Add hubs
    g_overflow.set_hubs_shape(hubs, shape_hub=shape_hub)

    # Highlight swapped flows
    g_overflow.highlight_swapped_flows(lines_swapped)

    # Highlight significant line loading
    # Detect if DC mode - handle both grid2op and pypowsybl observations
    is_DC = False
    try:
        # Grid2op observation
        is_DC = obs_simu._obs_env._parameters.ENV_DC
    except AttributeError:
        # Pypowsybl observation - check if network_manager has DC flag
        try:
            is_DC = obs_simu._network_manager._default_dc
        except AttributeError:
            # Default to False if we can't determine
            is_DC = False
    
    if not is_DC:
        ind_assets_to_monitor = np.where(overflow_sim.obs_linecut.rho >= 0.9)[0]
        ind_assets_to_monitor = np.append(ind_assets_to_monitor, overflow_sim.ltc)
        dict_significant_change = {
            env.name_line[ind]: {"before": int(obs_simu.rho[ind] * 100),
                                 "after": int(overflow_sim.obs_linecut.rho[ind] * 100)}
            for ind in ind_assets_to_monitor
        }
        g_overflow.highlight_significant_line_loading(dict_significant_change)

    g_overflow.collapse_red_loops()
    # Generate and save visualization
    tmp_save_folder = os.path.join(save_folder, graph_file_name)
    svg = g_overflow.plot(
        custom_layout,
        save_folder=tmp_save_folder,
        fontsize=fontsize,
        without_gray_edges=DRAW_ONLY_SIGNIFICANT_EDGES,
        node_thickness=node_thickness,
        rescale_factor=rescale_factor
    )

    pdf_files = glob.glob(f"{tmp_save_folder}/Base graph/*.pdf")
    if pdf_files:
        file_path = os.path.join(save_folder, graph_file_name + ".pdf")
        os.makedirs(save_folder, exist_ok=True)
        shutil.move(pdf_files[0], file_path)
        shutil.rmtree(tmp_save_folder)
        print("Overflow graph visualization has been saved in: " + file_path)

    return svg