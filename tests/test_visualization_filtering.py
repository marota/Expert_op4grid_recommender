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
        dict_highlight = g_overflow.highlight_significant_line_loading.call_args[0][0]
        
        # Check that line1 and line2 are included
        assert "line1" in dict_highlight, "Contingency line (LTC) should be highlighted"
        assert "line2" in dict_highlight, "Significant non-grey line should be highlighted"
        
        # Check that line4 is NOT included (high rho but grey)
        assert "line4" not in dict_highlight, "Grey line should NOT be highlighted even if rho >= 0.9"
        
        # Check that line3 is NOT included (low rho and grey)
        assert "line3" not in dict_highlight

if __name__ == "__main__":
    pytest.main([__file__])
