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
import networkx as nx
import pypowsybl as pp

# STEP 1: Import your config module
from expert_op4grid_recommender import config
from expert_op4grid_recommender.config import DRAW_ONLY_SIGNIFICANT_EDGES, USE_GRID_LAYOUT, DO_CONSOLIDATE_GRAPH


# Memoized result of get_zone_voltage_levels keyed by (resolved network file
# path, mtime). The un-cached implementation re-parsed the FULL .xiidm network
# from disk on every visualization call — several seconds on a national grid —
# even though the voltage-level table never changes within a study. The mtime
# key auto-invalidates when the file is replaced on disk.
_zone_voltage_levels_cache = {}


def _extract_network_zip(zip_path):
    """Extract the first ``.xiidm`` / ``.xml`` member of ``zip_path`` and
    return the path to the decompressed file.

    The member is written next to the archive (so it is cached and reused
    across calls); if that directory is read-only we fall back to a temp
    dir. This mirrors Co-Study4Grid's ``network_service`` behaviour so the
    recommender can load a zipped network (e.g. the game-mode
    ``network.xiidm.zip``) on its own.
    """
    import os
    import tempfile
    import zipfile
    from pathlib import Path

    with zipfile.ZipFile(str(zip_path)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith((".xiidm", ".xml", ".iidm"))]
        if not members:
            raise FileNotFoundError(f"No .xiidm/.xml member found inside {zip_path}")
        member = members[0]
        out_name = os.path.basename(member)
        out_dir = os.path.dirname(os.path.abspath(str(zip_path)))
        out_path = os.path.join(out_dir, out_name)
        if os.path.isfile(out_path):
            return Path(out_path)  # already decompressed — reuse
        data = zf.read(member)
        try:
            with open(out_path, "wb") as f:
                f.write(data)
        except OSError:
            out_path = os.path.join(tempfile.mkdtemp(prefix="eo4g_net_"), out_name)
            with open(out_path, "wb") as f:
                f.write(data)
        return Path(out_path)


def _resolve_network_file(env_path):
    """Resolve ``env_path`` to a pypowsybl-loadable network file.

    ``env_path`` may be a direct network file (``.xiidm`` / ``.iidm`` /
    ``.xml``), a **zip archive** of one (``pypowsybl.network.load`` cannot
    read an arbitrary ``.zip``, so we decompress it), or a directory
    containing ``grid.xiidm`` (the Grid2Op-style layout).

    The previous logic only recognised ``.xiidm`` / ``.iidm`` / ``.xml`` as
    files, so a zipped network (e.g. the game-mode ``network.xiidm.zip``)
    fell through to the directory branch and produced a bogus
    ``network.xiidm.zip/grid.xiidm`` path that failed to load — aborting
    the whole overflow-graph render even though action discovery (which
    reuses the already-extracted network) succeeded.
    """
    from pathlib import Path
    env_path = Path(env_path)

    if env_path.is_file():
        if env_path.suffix.lower() == ".zip":
            return _extract_network_zip(env_path)
        return env_path

    if env_path.is_dir():
        grid = env_path / "grid.xiidm"
        if grid.is_file():
            return grid
        direct = [f for f in env_path.iterdir() if f.suffix.lower() in (".xiidm", ".iidm", ".xml")]
        if direct:
            return direct[0]
        zips = [f for f in env_path.iterdir() if f.suffix.lower() == ".zip"]
        if zips:
            return _extract_network_zip(zips[0])
        return grid

    # Missing path: try a sibling/companion .zip (``foo.xiidm`` -> ``foo.xiidm.zip``).
    for cand in (Path(str(env_path) + ".zip"), env_path.with_suffix(".zip")):
        if cand.is_file():
            return _extract_network_zip(cand)
    return env_path / "grid.xiidm"


def get_zone_voltage_levels(env_path):
    """
    Loads voltage level information for substations from a PowSyBl network file.

    The result is cached per (file path, mtime): the overflow-graph
    visualization runs once per Step-2 and only needs the static
    voltage-level -> nominal-voltage mapping, so re-parsing the whole
    network each time is pure waste.
    """
    network_file_path = _resolve_network_file(env_path)

    try:
        cache_key = (str(network_file_path), os.path.getmtime(network_file_path))
    except OSError:
        cache_key = None
    if cache_key is not None and cache_key in _zone_voltage_levels_cache:
        return _zone_voltage_levels_cache[cache_key]

    n_zone = pp.network.load(str(network_file_path))
    df_volt = n_zone.get_voltage_levels()
    result = {sub: volt for sub, volt in zip(df_volt.index, df_volt.nominal_v)}
    if cache_key is not None:
        _zone_voltage_levels_cache.clear()  # keep at most the current network
        _zone_voltage_levels_cache[cache_key] = result
    return result


# Memoized result of get_zone_voltage_level_names keyed by (resolved network
# file path, mtime). Same rationale as `_zone_voltage_levels_cache`: the
# voltage-level name table is static within a study, so re-parsing the whole
# .xiidm on every visualization call is wasted work.
_zone_voltage_level_names_cache = {}


def get_zone_voltage_level_names(env_path):
    """Return ``{voltage_level_id: human_readable_name}`` from the network.

    PyPSA-derived networks expose machine voltage-level IDs (``VL_way_...``)
    as the node identity (``obs.name_sub``) while carrying a readable name
    (e.g. ``"Saucats 400kV"``) in the voltage-level ``name`` column. This
    helper extracts that mapping so the overflow-graph visualization can
    relabel node *display text* without touching node identity.

    Only entries whose ``name`` is non-empty AND differs from the ID are
    returned: when a network has no separate readable names (the usual RTE
    case, where ``name == id``) the result is empty and the graph is left
    unchanged. The result is cached per (file path, mtime).
    """
    network_file_path = _resolve_network_file(env_path)

    try:
        cache_key = (str(network_file_path), os.path.getmtime(network_file_path))
    except OSError:
        cache_key = None
    if cache_key is not None and cache_key in _zone_voltage_level_names_cache:
        return _zone_voltage_level_names_cache[cache_key]

    n_zone = pp.network.load(str(network_file_path))
    df_volt = n_zone.get_voltage_levels()
    result = {}
    if "name" in df_volt.columns:
        for vl_id, name in zip(df_volt.index, df_volt["name"]):
            name = "" if name is None else str(name)
            if name and name != "nan" and name != str(vl_id):
                result[str(vl_id)] = name

    if cache_key is not None:
        _zone_voltage_level_names_cache.clear()  # keep at most the current network
        _zone_voltage_level_names_cache[cache_key] = result
    return result


def _sanitize_graph_label(text):
    """Make a readable name safe to use as a Graphviz node label.

    Older ``pydot`` (< 2.0) does **not** escape embedded double quotes in
    attribute values, so a voltage-level name such as
    ``LAAT "SET Gurrea - SET Sabiñánigo"`` serializes to malformed DOT
    (``label="LAAT "SET Gurrea ..."``) that makes ``dot`` / ``neato``
    crash — which aborts the *entire* overflow-graph render. We therefore
    neutralise the characters that can break DOT across pydot versions:
    double quotes and backslashes become harmless equivalents, control
    characters / newlines collapse to single spaces, and a leading ``<`` is
    pushed in so Graphviz can't mistake the value for an HTML-like label.
    """
    s = str(text).replace('"', "'").replace("\\", "/")
    s = " ".join(s.split())  # collapse newlines / control chars / whitespace runs
    if s.startswith("<"):
        s = " " + s
    return s


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


def make_overflow_graph_visualization(env, overflow_sim, g_overflow, hubs, obs_simu, save_folder, graph_file_name,
                                      lines_swapped, custom_layout=None, lines_we_care_about=None,
                                      monitoring_factor_thermal_limits=1.0,
                                      lines_constrained_path=None, nodes_constrained_path=None,
                                      lines_red_loops=None, nodes_red_loops=None,
                                      extra_lines_to_cut_ids=None):
    """
    Generates and saves a visualization of the overflow graph with various annotations.

    This function takes a constructed overflow graph and enhances it with visual information
    before plotting and saving it as either a PDF or an interactive HTML file, depending on
    :attr:`config.VISUALIZATION_FORMAT` (``"pdf"`` by default; ``"html"`` uses the interactive
    viewer introduced by ExpertOp4Grid PR #74). The enhancements include:
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
        save_folder (str): The path to the directory where the output file should be saved.
        graph_file_name (str): The base name for the output file (without the extension).
        lines_swapped (list[str]): A list of line names where the flow direction might have
            been swapped during the overflow graph calculation.
        custom_layout (dict | list | None, optional): A predefined layout for positioning the graph nodes.
            Can be a dictionary mapping node names to (x, y) coordinates or a list of coordinates
            matching the node order. If None, the plotting function determines the layout.
            Defaults to None.
        lines_we_care_about (array-like | None, optional): Array/list of monitored line names.
            When provided, only these lines are considered when highlighting significant loading
            (rho >= 0.9). Lines not in this set are excluded from the loading annotation even
            if their rho is high. The overloaded lines used to build the graph (overflow_sim.ltc)
            are always included regardless of this filter. Defaults to None (all lines annotated).
        monitoring_factor_thermal_limits (float, optional): Factor by which thermal limits were
            scaled before computing rho (e.g. 0.95 means limits were set at 95% of the real
            value). Multiplying rho by this factor converts displayed percentages back to the
            real thermal limit. Defaults to 1.0 (no rescaling).

    Returns:
        svg: The SVG data generated by the `g_overflow.plot` method, if any. Note that the
             primary output is the saved PDF or HTML file (selected via
             ``config.VISUALIZATION_FORMAT``).

    Side Effects:
        - Creates a temporary folder within `save_folder`.
        - Saves the graph visualization as ``<graph_file_name>.<ext>`` in `save_folder`,
          where ``<ext>`` is ``pdf`` or ``html`` according to ``config.VISUALIZATION_FORMAT``.
        - Deletes the temporary folder.
        - Prints the path to the saved file to the console.
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

    # Use human-readable voltage-level names as the displayed node labels.
    # The graph nodes are identified by their voltage-level ID (from
    # ``obs.name_sub``); for PyPSA-derived networks those IDs are opaque
    # (``VL_way_...``) while a readable name is available in the network.
    # We set the Graphviz ``label`` node attribute (which controls the
    # *rendered* text) and leave the node identity untouched, so the SVG
    # ``<title>`` / ``data-name`` — and therefore pin overlays, SLD lookups
    # and search — keep using the stable ID.
    applied_label_nodes = []
    if getattr(config, "USE_VOLTAGE_LEVEL_NAMES_IN_GRAPH", True):
        try:
            vl_name_map = get_zone_voltage_level_names(config.ENV_PATH)
            if vl_name_map:
                label_map = {
                    node: _sanitize_graph_label(vl_name_map[str(node)])
                    for node in g_overflow.g.nodes
                    if str(node) in vl_name_map
                }
                if label_map:
                    nx.set_node_attributes(g_overflow.g, label_map, "label")
                    applied_label_nodes = list(label_map.keys())
        except Exception as exc:
            # Readable labels are a presentational nicety — never let a
            # name-lookup failure abort the (otherwise valid) graph render.
            print(
                "Could not apply voltage-level name labels to the overflow "
                f"graph (keeping IDs): {type(exc).__name__}: {exc}"
            )

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
        # Filter to keep only non-grey edges for highlighting
        non_grey_line_names = {
            data.get("name")
            for _, _, _, data in g_overflow.g.edges(data=True, keys=True)
            if data.get("color") != "gray"
        }

        ind_assets_to_monitor = np.where(overflow_sim.obs_linecut.rho >= 0.9)[0]
        if lines_we_care_about is not None and len(lines_we_care_about) > 0:
            monitored_mask = np.isin(env.name_line, lines_we_care_about)
            ind_assets_to_monitor = ind_assets_to_monitor[monitored_mask[ind_assets_to_monitor]]

        # Keep only assets that are non-grey in the graph
        ind_assets_to_monitor = np.array([ind for ind in ind_assets_to_monitor if env.name_line[ind] in non_grey_line_names])

        ind_assets_to_monitor = np.append(ind_assets_to_monitor, overflow_sim.ltc).astype(int)
        # Rescale rho percentages back to real thermal limits: the monitoring factor
        # reduces thermal limits (e.g. 0.95 → limits set at 95% of real), so rho is
        # inflated by 1/factor.  Multiplying by the factor converts back to the
        # percentage of the real thermal limit that the user expects to see.
        dict_significant_change = {
            env.name_line[ind]: {"before": int(obs_simu.rho[ind] * 100 * monitoring_factor_thermal_limits),
                                 "after": int(overflow_sim.obs_linecut.rho[ind] * 100 * monitoring_factor_thermal_limits)}
            for ind in ind_assets_to_monitor
        }
        g_overflow.highlight_significant_line_loading(dict_significant_change)

    # Tag the constrained-path lines/nodes (source-of-truth flags consumed
    # by the interactive HTML viewer's "Constrained path" layer toggle).
    # Caller passes these from the recommender pipeline to avoid having
    # the visualization layer reinterpret edge colour/style.
    if lines_constrained_path or nodes_constrained_path:
        g_overflow.tag_constrained_path(
            lines_constrained_path=lines_constrained_path,
            nodes_constrained_path=nodes_constrained_path,
        )

    # Tag the dispatch loop paths (the "red loops") from the structured
    # analysis. This must run AFTER `collapse_red_loops` so the visual
    # collapse heuristic does not stomp on or pre-empt the
    # source-of-truth tags. When the caller does not supply the lists
    # the layer simply stays empty (no false positives).
    g_overflow.collapse_red_loops()
    if lines_red_loops or nodes_red_loops:
        g_overflow.tag_red_loops(
            lines_red_loops=lines_red_loops,
            nodes_red_loops=nodes_red_loops,
        )
    # Generate and save visualization
    tmp_save_folder = os.path.join(save_folder, graph_file_name)

    def _plot():
        return g_overflow.plot(
            custom_layout,
            save_folder=tmp_save_folder,
            fontsize=fontsize,
            without_gray_edges=DRAW_ONLY_SIGNIFICANT_EDGES,
            node_thickness=node_thickness,
            rescale_factor=rescale_factor
        )

    try:
        svg = _plot()
    except Exception as exc:
        # The readable-name labels must never be able to break the graph
        # render: some Graphviz/pydot combos choke on label text that the
        # local stack here tolerates. If the labelled render fails, drop the
        # display labels (node identity is unchanged) and retry once so the
        # operator still gets the graph — just with VL IDs as text.
        if applied_label_nodes:
            print(
                "Overflow-graph render failed with readable node labels "
                f"({type(exc).__name__}: {exc}); retrying with VL IDs."
            )
            for node in applied_label_nodes:
                g_overflow.g.nodes[node].pop("label", None)
            svg = _plot()
        else:
            raise

    # Pick output format from config ("pdf" default, or "html" from PR #74).
    output_format = getattr(config, "VISUALIZATION_FORMAT", "pdf").lower()
    if output_format not in ("pdf", "html"):
        raise ValueError(
            f"Unsupported VISUALIZATION_FORMAT={output_format!r}; expected 'pdf' or 'html'."
        )

    generated_files = glob.glob(f"{tmp_save_folder}/Base graph/*.{output_format}")
    if generated_files:
        file_path = os.path.join(save_folder, graph_file_name + f".{output_format}")
        os.makedirs(save_folder, exist_ok=True)
        shutil.move(generated_files[0], file_path)
        shutil.rmtree(tmp_save_folder)
        print("Overflow graph visualization has been saved in: " + file_path)
    elif output_format == "html":
        print(
            "HTML export not found — make sure expertop4grid includes the "
            "interactive HTML viewer from PR #74."
        )

    return svg