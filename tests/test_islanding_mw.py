import pytest
import numpy as np
import pypowsybl
from pathlib import Path
from expert_op4grid_recommender.pypowsybl_backend import NetworkManager, ActionSpace, PypowsyblObservation

# Skip all tests if pypowsybl is not available
pypowsybl = pytest.importorskip("pypowsybl")

def test_islanding_small_grid_case():
    """Test islanding case suggested by user on the small grid."""
    grid_path = Path("/home/marotant/dev/Expert_op4grid_recommender/data/bare_env_small_grid_test/grid.xiidm")
    if not grid_path.exists():
        pytest.skip(f"Small grid xiidm not found at {grid_path}")
        
    net = pypowsybl.network.load(str(grid_path))
    nm = NetworkManager(network=net)
    action_space = ActionSpace(nm)
    obs = PypowsyblObservation(nm, action_space)
    
    initial_load = obs.main_component_load_mw
    initial_components = obs.n_components
    print(f"Initial loads: {initial_load}, components: {initial_components}")
    
    # Corrected names based on network inspection
    to_disconnect = ["P.SAOL31RONCI", "BEON L31CPVAN", "LOUHAL31SSUSU"]
    
    all_lines_trafos = nm.name_line.tolist()
    valid_disconnects = [name for name in to_disconnect if name in all_lines_trafos]
    print(f"Valid elements to disconnect: {valid_disconnects}")
    
    assert len(valid_disconnects) == 3, f"Not all elements found: {valid_disconnects}"

    action = action_space({"set_line_status": [(lid, -1) for lid in valid_disconnects]})
    
    obs_simu, _, _, _ = obs.simulate(action)
    
    print(f"Simulated components: {obs_simu.n_components}")
    print(f"Simulated mainland load: {obs_simu.main_component_load_mw}")
    
    # verify snapshotting worked
    assert obs_simu.n_components > initial_components
    assert obs_simu.main_component_load_mw < initial_load
    
    disconnected_mw = initial_load - obs_simu.main_component_load_mw
    print(f"Disconnected MW: {disconnected_mw}")
    assert disconnected_mw > 0
