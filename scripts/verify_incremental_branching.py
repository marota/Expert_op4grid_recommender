import time
import numpy as np
import pypowsybl
from expert_op4grid_recommender.pypowsybl_backend import NetworkManager, PypowsyblObservation, ActionSpace

def verify():
    # Create a built-in IEEE14 network
    net = pypowsybl.network.create_ieee14()
    nm = NetworkManager(network=net)
    action_space = ActionSpace(nm)
    
    # Initial observation
    obs = PypowsyblObservation(nm, action_space)
    print(f"Network has lines: {obs.name_line[:10]}...")
    
    # 1. Simulate N-1 (contingency) and KEEP variant
    line_to_cut = obs.name_line[0]  # Use the first line name
    act_n_1 = action_space({"set_line_status": [(line_to_cut, -1)]})
    
    print(f"Simulating N-1 (Line {line_to_cut})...")
    obs_n_1, _, _, info = obs.simulate(act_n_1, keep_variant=True)
    n_1_variant = obs_n_1._variant_id
    print(f"N-1 Variant ID: {n_1_variant}")
    
    if info["exception"]:
        print(f"N-1 simulation failed: {info['exception']}")
        return

    # 2. Simulate remedial action branching from N-1
    remedial_line = obs.name_line[1] # L1-5-1
    act_remedial = action_space({"set_line_status": [(remedial_line, -1)]})
    
    print(f"\nSimulating remedial action (Line {remedial_line}) branching from N-1...")
    start = time.time()
    # Branching from obs_n_1 automatically clones from n_1_variant
    obs_action, _, _, info_action = obs_n_1.simulate(act_remedial)
    end = time.time()
    
    print(f"Direct Branching took: {end - start:.4f}s")
    if info_action["exception"]:
        print(f"Remedial simulation failed: {info_action['exception']}")
        return

    # Verify state: both lines should be disconnected
    line_idx_n_1 = nm.get_line_idx(line_to_cut)
    line_idx_remedial = nm.get_line_idx(remedial_line)
    
    print(f"Line {line_to_cut} status (should be False): {obs_action.line_status[line_idx_n_1]}")
    print(f"Line {remedial_line} status (should be False): {obs_action.line_status[line_idx_remedial]}")
    
    if not obs_action.line_status[line_idx_n_1] and not obs_action.line_status[line_idx_remedial]:
        print("\n✅ SUCCESS: Incremental branching correctly maintained both contingency and action.")
    else:
        print("\n❌ FAILURE: Incremental branching failed to maintain state.")

    # 3. Compare with "Full" simulation (N + Action + Contingency)
    print("\nComparing with full simulation from base N...")
    act_combined = act_n_1 + act_remedial
    start = time.time()
    obs_combined, _, _, _ = obs.simulate(act_combined)
    end = time.time()
    print(f"Full Simulation took: {end - start:.4f}s")
    
    # Check equality of rho (within tolerance)
    if np.allclose(obs_action.rho, obs_combined.rho, atol=1e-5):
        print("✅ SUCCESS: Incremental results match full simulation results.")
    else:
        print("❌ FAILURE: Incremental results differ from full simulation.")

if __name__ == "__main__":
    verify()
