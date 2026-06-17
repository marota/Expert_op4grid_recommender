import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import os

# Import the function to test
from expert_op4grid_recommender.graph_analysis.visualization import make_overflow_graph_visualization

def test_make_overflow_graph_visualization_filtering():
    """
    Test that make_overflow_graph_visualization correctly filters out grey edges
    from being highlighted, even if they have high rho.
    """
    # 1. Setup Mock Environment
    env = MagicMock()
    # Indices: 0: contingency, 1: significant, 2: insignificant, 3: high rho but grey
    env.name_line = np.array(["line1", "line2", "line3", "line4"])
    
    # 2. Setup Mock Simulation Results
    overflow_sim = MagicMock()
    # Significant lines (> 0.9) are indices 0, 1, 3
    overflow_sim.obs_linecut.rho = np.array([1.1, 0.95, 0.2, 0.92]) 
    overflow_sim.ltc = np.array([0]) # line1 is the contingency
    
    # 3. Setup Mock Overflow Graph
    g_overflow = MagicMock()
    # Mock g_overflow.g.edges to return colors
    # Format: (u, v, key, data)
    edges = [
        (0, 1, 0, {"name": "line1", "color": "black"}), # Contingency
        (1, 2, 0, {"name": "line2", "color": "blue"}),  # Significant (non-grey)
        (2, 3, 0, {"name": "line3", "color": "gray"}),  # Insignificant
        (3, 0, 0, {"name": "line4", "color": "gray"}),  # High rho BUT grey
    ]
    g_overflow.g.edges.return_value = edges
    
    # 4. Setup Mock Observation
    obs_simu = MagicMock()
    obs_simu.rho = np.array([1.0, 0.8, 0.2, 0.85]) # Before loading
    obs_simu.name_sub = ["sub1", "sub2", "sub3", "sub4"]
    obs_simu.sub_topology.return_value = [1]
    
    # Bypass is_DC check
    # The code tries: obs_simu._obs_env._parameters.ENV_DC
    obs_simu._obs_env._parameters.ENV_DC = False
    
    # 5. Patch external dependencies
    with patch("expert_op4grid_recommender.graph_analysis.visualization.get_zone_voltage_levels") as mock_get_volt, \
         patch("expert_op4grid_recommender.graph_analysis.visualization.get_zone_voltage_level_names", return_value={}), \
         patch("expert_op4grid_recommender.graph_analysis.visualization.config") as mock_config, \
         patch("os.path.join", side_effect=os.path.join), \
         patch("os.makedirs"), \
         patch("shutil.move"), \
         patch("shutil.rmtree"), \
         patch("glob.glob") as mock_glob, \
         patch("builtins.print"):
        
        mock_get_volt.return_value = {}
        mock_config.ENV_PATH = "/dummy/path"
        mock_config.SAVE_FOLDER_VISUALIZATION = "/dummy/save"
        mock_config.DRAW_ONLY_SIGNIFICANT_EDGES = False
        mock_config.USE_GRID_LAYOUT = False
        mock_config.DO_CONSOLIDATE_GRAPH = False
        mock_config.VISUALIZATION_FORMAT = "pdf"
        mock_glob.return_value = ["/dummy/save/test_graph/Base graph/plot.pdf"]

        # 6. Call the function
        make_overflow_graph_visualization(
            env, overflow_sim, g_overflow, hubs=[], obs_simu=obs_simu,
            save_folder="/dummy/save", graph_file_name="test_graph",
            lines_swapped=[], monitoring_factor_thermal_limits=1.0
        )
        
        # 7. Verification
        # Get the dictionary passed to highlight_significant_line_loading
        assert g_overflow.highlight_significant_line_loading.called
        assert g_overflow.collapse_red_loops.called, "collapse_red_loops should be called before plotting"
        dict_highlight = g_overflow.highlight_significant_line_loading.call_args[0][0]
        
        # Check that line1 and line2 are included
        assert "line1" in dict_highlight, "Contingency line (LTC) should be highlighted"
        assert "line2" in dict_highlight, "Significant non-grey line should be highlighted"
        
        # Check that line4 is NOT included (high rho but grey)
        assert "line4" not in dict_highlight, "Grey line should NOT be highlighted even if rho >= 0.9"
        
        # Check that line3 is NOT included (low rho and grey)
        assert "line3" not in dict_highlight

def test_get_zone_voltage_level_names_filters_ids_and_empty():
    """Only VLs with a readable name that differs from the ID are returned."""
    import pandas as pd
    from expert_op4grid_recommender.graph_analysis import visualization

    df = pd.DataFrame(
        {"name": ["Saucats 400kV", "VL_way_2", "", None]},
        index=["VL_way_1", "VL_way_2", "VL_way_3", "VL_way_4"],
    )
    fake_network = MagicMock()
    fake_network.get_voltage_levels.return_value = df

    with patch.object(visualization.pp.network, "load", return_value=fake_network), \
         patch("os.path.getmtime", side_effect=OSError):  # disable caching
        result = visualization.get_zone_voltage_level_names("/some/grid.xiidm")

    # Readable, distinct name kept; name==id, empty and None dropped.
    assert result == {"VL_way_1": "Saucats 400kV"}


def test_resolve_network_file_handles_zip_and_dir(tmp_path):
    """A zipped network is decompressed; a directory falls back to its
    xiidm; a plain file is used as-is."""
    import zipfile
    from expert_op4grid_recommender.graph_analysis.visualization import (
        _resolve_network_file,
    )

    # Plain file used as-is.
    xiidm = tmp_path / "net.xiidm"
    xiidm.write_text("<network/>")
    assert _resolve_network_file(xiidm) == xiidm

    # Zip archive -> extracted sibling .xiidm (pypowsybl can't read the .zip).
    zip_path = tmp_path / "network.xiidm.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("network.xiidm", "<network/>")
    resolved = _resolve_network_file(zip_path)
    assert resolved.suffix == ".xiidm"
    assert resolved.is_file()
    assert resolved.read_text() == "<network/>"

    # Directory with grid.xiidm.
    d = tmp_path / "envdir"
    d.mkdir()
    (d / "grid.xiidm").write_text("<network/>")
    assert _resolve_network_file(d) == d / "grid.xiidm"


def test_extract_network_zip_extracts_and_reuses(tmp_path):
    """The zip member is extracted next to the archive and reused on a
    second call (no re-extraction)."""
    import zipfile
    from expert_op4grid_recommender.graph_analysis.visualization import (
        _extract_network_zip,
    )

    zip_path = tmp_path / "network.xiidm.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("network.xiidm", "<network>v1</network>")

    out1 = _extract_network_zip(zip_path)
    assert out1.is_file()
    assert out1.parent == tmp_path
    assert out1.read_text() == "<network>v1</network>"

    # Reuse path: the extracted sibling already exists, so a second call
    # returns it without rewriting (even if the zip member differs).
    out2 = _extract_network_zip(zip_path)
    assert out2 == out1


def test_extract_network_zip_raises_without_xiidm_member(tmp_path):
    import zipfile
    from expert_op4grid_recommender.graph_analysis.visualization import (
        _extract_network_zip,
    )

    zip_path = tmp_path / "bogus.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "no network here")
    with pytest.raises(FileNotFoundError):
        _extract_network_zip(zip_path)


def test_resolve_network_file_companion_zip(tmp_path):
    """A missing ``foo.xiidm`` resolves to its companion ``foo.xiidm.zip``."""
    import zipfile
    from expert_op4grid_recommender.graph_analysis.visualization import (
        _resolve_network_file,
    )

    zip_path = tmp_path / "network.xiidm.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("network.xiidm", "<network/>")

    resolved = _resolve_network_file(tmp_path / "network.xiidm")  # does not exist
    assert resolved.suffix == ".xiidm"
    assert resolved.is_file()


def test_get_zone_voltage_level_names_loads_through_zip(tmp_path):
    """get_zone_voltage_level_names decompresses a .zip before loading
    (pypowsybl can't read the archive directly)."""
    import zipfile
    import pandas as pd
    from expert_op4grid_recommender.graph_analysis import visualization

    zip_path = tmp_path / "network.xiidm.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("network.xiidm", "<network/>")

    df = pd.DataFrame({"name": ["Readable A"]}, index=["VL_way_1"])
    fake_network = MagicMock()
    fake_network.get_voltage_levels.return_value = df

    with patch.object(visualization.pp.network, "load", return_value=fake_network) as mock_load:
        result = visualization.get_zone_voltage_level_names(zip_path)

    # Loaded from the extracted .xiidm, NOT the raw .zip.
    loaded_arg = str(mock_load.call_args[0][0])
    assert loaded_arg.endswith(".xiidm")
    assert not loaded_arg.endswith(".zip")
    assert result == {"VL_way_1": "Readable A"}


def test_sanitize_graph_label_neutralizes_dot_breaking_chars():
    """Embedded quotes / backslashes / newlines must not survive into the
    label (older pydot mis-escapes them and crashes Graphviz)."""
    from expert_op4grid_recommender.graph_analysis.visualization import (
        _sanitize_graph_label,
    )

    assert _sanitize_graph_label('LAAT "SET Gurrea - SET Sabiñánigo"') == \
        "LAAT 'SET Gurrea - SET Sabiñánigo'"
    assert '"' not in _sanitize_graph_label('a "b" c')
    assert "\\" not in _sanitize_graph_label("a\\b")
    assert "\n" not in _sanitize_graph_label("line1\nline2")
    # A value that would otherwise be read as an HTML-like label is defused.
    assert not _sanitize_graph_label("<html>").startswith("<")
    # Plain names with accents / dashes pass through unchanged.
    assert _sanitize_graph_label("Subestación 220kV") == "Subestación 220kV"


def test_visualization_sets_readable_node_labels():
    """Graph node `label` attribute is set to the readable VL name while the
    node identity (ID) is preserved."""
    import networkx as nx
    from expert_op4grid_recommender.graph_analysis.visualization import (
        make_overflow_graph_visualization,
    )

    env = MagicMock()
    env.name_line = np.array(["line1"])

    overflow_sim = MagicMock()
    overflow_sim.obs_linecut.rho = np.array([0.5])
    overflow_sim.ltc = np.array([0])

    # Real graph so set_node_attributes actually mutates it.
    real_g = nx.MultiDiGraph()
    real_g.add_node("VL_way_1")
    real_g.add_node("VL_way_2")
    real_g.add_edge("VL_way_1", "VL_way_2", key=0, name="line1", color="blue")

    g_overflow = MagicMock()
    g_overflow.g = real_g

    obs_simu = MagicMock()
    obs_simu.rho = np.array([0.5])
    obs_simu.name_sub = ["VL_way_1", "VL_way_2"]
    obs_simu.sub_topology.return_value = [1]
    obs_simu._obs_env._parameters.ENV_DC = False

    with patch("expert_op4grid_recommender.graph_analysis.visualization.get_zone_voltage_levels", return_value={}), \
         patch("expert_op4grid_recommender.graph_analysis.visualization.get_zone_voltage_level_names",
               return_value={"VL_way_1": "Saucats 400kV"}), \
         patch("expert_op4grid_recommender.graph_analysis.visualization.config") as mock_config, \
         patch("os.makedirs"), patch("shutil.move"), patch("shutil.rmtree"), \
         patch("glob.glob", return_value=["/dummy/save/test_graph/Base graph/plot.pdf"]), \
         patch("builtins.print"):

        mock_config.ENV_PATH = "/dummy/path"
        mock_config.USE_VOLTAGE_LEVEL_NAMES_IN_GRAPH = True
        mock_config.DRAW_ONLY_SIGNIFICANT_EDGES = False
        mock_config.USE_GRID_LAYOUT = False
        mock_config.DO_CONSOLIDATE_GRAPH = False
        mock_config.VISUALIZATION_FORMAT = "pdf"

        make_overflow_graph_visualization(
            env, overflow_sim, g_overflow, hubs=[], obs_simu=obs_simu,
            save_folder="/dummy/save", graph_file_name="test_graph",
            lines_swapped=[], monitoring_factor_thermal_limits=1.0,
        )

    # Readable label applied to the matching node, identity untouched.
    assert real_g.nodes["VL_way_1"]["label"] == "Saucats 400kV"
    # Node without a readable name keeps no label override (renders its ID).
    assert "label" not in real_g.nodes["VL_way_2"]
    # Node identities are unchanged (still the VL IDs).
    assert set(real_g.nodes) == {"VL_way_1", "VL_way_2"}


def test_visualization_retries_without_labels_when_plot_fails():
    """If the labelled render crashes (e.g. a Graphviz/pydot quirk), the
    labels are dropped and the plot retried so the operator still gets a
    graph."""
    import networkx as nx
    from expert_op4grid_recommender.graph_analysis.visualization import (
        make_overflow_graph_visualization,
    )

    env = MagicMock()
    env.name_line = np.array(["line1"])
    overflow_sim = MagicMock()
    overflow_sim.obs_linecut.rho = np.array([0.5])
    overflow_sim.ltc = np.array([0])

    real_g = nx.MultiDiGraph()
    real_g.add_node("VL_way_1")
    real_g.add_edge("VL_way_1", "VL_way_1", key=0, name="line1", color="blue")

    g_overflow = MagicMock()
    g_overflow.g = real_g
    # First plot call raises (labels present), second succeeds.
    g_overflow.plot.side_effect = [RuntimeError("graphviz boom"), "svg-data"]

    obs_simu = MagicMock()
    obs_simu.rho = np.array([0.5])
    obs_simu.name_sub = ["VL_way_1"]
    obs_simu.sub_topology.return_value = [1]
    obs_simu._obs_env._parameters.ENV_DC = False

    with patch("expert_op4grid_recommender.graph_analysis.visualization.get_zone_voltage_levels", return_value={}), \
         patch("expert_op4grid_recommender.graph_analysis.visualization.get_zone_voltage_level_names",
               return_value={"VL_way_1": "Saucats 400kV"}), \
         patch("expert_op4grid_recommender.graph_analysis.visualization.config") as mock_config, \
         patch("os.makedirs"), patch("shutil.move"), patch("shutil.rmtree"), \
         patch("glob.glob", return_value=["/dummy/save/test_graph/Base graph/plot.pdf"]), \
         patch("builtins.print"):

        mock_config.ENV_PATH = "/dummy/path"
        mock_config.USE_VOLTAGE_LEVEL_NAMES_IN_GRAPH = True
        mock_config.DRAW_ONLY_SIGNIFICANT_EDGES = False
        mock_config.USE_GRID_LAYOUT = False
        mock_config.DO_CONSOLIDATE_GRAPH = False
        mock_config.VISUALIZATION_FORMAT = "pdf"

        svg = make_overflow_graph_visualization(
            env, overflow_sim, g_overflow, hubs=[], obs_simu=obs_simu,
            save_folder="/dummy/save", graph_file_name="test_graph",
            lines_swapped=[], monitoring_factor_thermal_limits=1.0,
        )

    # Retried (two plot calls) and the label was stripped before the retry.
    assert g_overflow.plot.call_count == 2
    assert "label" not in real_g.nodes["VL_way_1"]
    assert svg == "svg-data"


if __name__ == "__main__":
    pytest.main([__file__])
