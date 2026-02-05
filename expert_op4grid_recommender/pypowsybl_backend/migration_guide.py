# expert_op4grid_recommender/pypowsybl_backend/migration_guide.py
"""
Migration Guide: Grid2Op to Pure pypowsybl

This file documents the migration strategy and provides examples of how to
convert existing grid2op-dependent code to use the pure pypowsybl backend.

=============================================================================
OVERVIEW
=============================================================================

The migration involves replacing:
1. grid2op.make() -> SimulationEnvironment()
2. env.reset() / env.get_obs() -> env.get_obs()  
3. obs.simulate() -> obs.simulate() (same interface, different implementation)
4. env.action_space({...}) -> env.action_space({...}) (same interface)

=============================================================================
STEP-BY-STEP MIGRATION
=============================================================================

PHASE 1: Environment Creation
-----------------------------
BEFORE (grid2op):
    import grid2op
    from pypowsybl2grid import PyPowSyBlBackend
    
    backend = PyPowSyBlBackend(n_busbar_per_sub=12, ...)
    env = grid2op.make(path, backend=backend, ...)
    obs = env.reset()

AFTER (pure pypowsybl):
    from expert_op4grid_recommender.pypowsybl_backend import SimulationEnvironment
    
    env = SimulationEnvironment(
        network_path="/path/to/network.xiidm",
        thermal_limits_path="/path/to/thermal_limits.json"
    )
    obs = env.get_obs()


PHASE 2: Actions
----------------
BEFORE (grid2op):
    # Disconnect a line
    action = env.action_space({"set_line_status": [("LINE_NAME", -1)]})
    
    # Topology change
    action = env.action_space({
        "set_bus": {
            "lines_or_id": {"LINE_NAME": 1},
            "lines_ex_id": {"LINE_NAME": 2}
        }
    })

AFTER (pure pypowsybl):
    # SAME INTERFACE! No changes needed for action creation
    action = env.action_space({"set_line_status": [("LINE_NAME", -1)]})
    action = env.action_space({
        "set_bus": {
            "lines_or_id": {"LINE_NAME": 1},
            "lines_ex_id": {"LINE_NAME": 2}
        }
    })


PHASE 3: Simulation
-------------------
BEFORE (grid2op):
    obs_simu, reward, done, info = obs.simulate(action, time_step=timestep)
    if info["exception"]:
        print("Simulation failed")
    print(f"Line loading: {obs_simu.rho}")

AFTER (pure pypowsybl):
    # SAME INTERFACE! 
    obs_simu, reward, done, info = obs.simulate(action, time_step=timestep)
    if info["exception"]:
        print("Simulation failed")
    print(f"Line loading: {obs_simu.rho}")


PHASE 4: Observation Properties
-------------------------------
All common observation properties are maintained:
- obs.rho              -> Line loading ratios
- obs.line_status      -> Line connection status (boolean array)
- obs.theta_or         -> Voltage angles at line origins
- obs.theta_ex         -> Voltage angles at line extremities
- obs.load_p, load_q   -> Load active/reactive power
- obs.gen_p, gen_q     -> Generator active/reactive power
- obs.name_line        -> Line names
- obs.name_sub         -> Substation names
- obs.line_or_to_subid -> Origin substation index per line
- obs.line_ex_to_subid -> Extremity substation index per line
- obs.sub_topology(id) -> Topology vector for substation
- obs.topo_vect        -> Full topology vector

=============================================================================
FILE-BY-FILE MIGRATION CHECKLIST
=============================================================================

1. environment.py
   [ ] Replace make_grid2op_assistant_env with SimulationEnvironment
   [ ] Update get_env_first_obs() to work without chronics
   [ ] Remove grid2op.Parameters usage (use lf_parameters instead)

2. utils/simulation.py
   [ ] No changes needed - functions work with the compatible interface

3. utils/make_assistant_env.py
   [ ] Can be deprecated or converted to thin wrapper

4. action_evaluation/classifier.py
   [ ] No changes needed - uses action.* attributes that are preserved

5. action_evaluation/discovery.py
   [ ] No changes needed - uses env.action_space and obs.simulate

6. graph_analysis/builder.py
   [ ] IMPORTANT: alphaDeesp.Grid2opSimulation needs adaptation
   [ ] Create pypowsybl-native overflow calculation

=============================================================================
ALPHADESP INTEGRATION
=============================================================================

The alphaDeesp library currently expects grid2op objects. Options:

OPTION A: Create adapter classes
    - Wrap pypowsybl observation to look like grid2op observation
    - This is what we've done with PypowsyblObservation

OPTION B: Fork/modify alphaDeesp
    - Create version that works directly with pypowsybl
    - More work but cleaner long-term

OPTION C: Re-implement overflow graph logic
    - The core logic (PTDF-based flow changes) can be computed directly
    - Use pypowsybl's sensitivity analysis: pp.sensitivity.create_dc_analysis()

=============================================================================
EXAMPLE: Converting main.py run_analysis()
=============================================================================
"""

# Example of converted run_analysis function

def run_analysis_pypowsybl(network_path: str,
                           thermal_limits_path: str = None,
                           lines_defaut: list = None,
                           lines_we_care_about: list = None):
    """
    Example of how run_analysis would look with pure pypowsybl.
    
    This is a template showing the migration pattern.
    """
    from expert_op4grid_recommender.pypowsybl_backend import SimulationEnvironment
    import numpy as np
    
    # ===== PHASE 1: Environment Setup =====
    # BEFORE: env = make_grid2op_assistant_env(env_folder, env_name)
    # AFTER:
    env = SimulationEnvironment(
        network_path=network_path,
        thermal_limits_path=thermal_limits_path,
        threshold_thermal_limit=0.95
    )
    
    # Get observation
    # BEFORE: obs = env.reset() or get_first_obs_on_chronic(...)
    # AFTER:
    obs = env.get_obs()
    
    # ===== PHASE 2: Simulate Contingency =====
    # Create contingency action - SAME INTERFACE
    act_deco_defaut = env.action_space({
        "set_line_status": [(line, -1) for line in lines_defaut]
    })
    
    # Simulate - SAME INTERFACE
    obs_simu_defaut, _, _, info = obs.simulate(act_deco_defaut, time_step=0)
    
    if info["exception"]:
        raise RuntimeError(f"Contingency simulation failed: {info['exception']}")
    
    # ===== PHASE 3: Find Overloads =====
    # SAME CODE - obs.rho and obs.name_line work identically
    lines_overloaded_ids = [
        i for i, l in enumerate(obs_simu_defaut.name_line)
        if l in lines_we_care_about and obs_simu_defaut.rho[i] >= 1
    ]
    
    print(f"Found {len(lines_overloaded_ids)} overloaded lines")
    
    # ===== PHASE 4: Test Remedial Actions =====
    # Create and simulate remedial actions - SAME INTERFACE
    test_action = env.action_space({
        "set_bus": {
            "lines_or_id": {"SOME_LINE": 1},
            "lines_ex_id": {"SOME_LINE": 1}
        }
    })
    
    # Combine actions with + operator - SAME INTERFACE
    combined_action = act_deco_defaut + test_action
    obs_simu_remedial, _, _, info = obs.simulate(combined_action, time_step=0)
    
    # Check if overloads reduced
    new_rho = obs_simu_remedial.rho[lines_overloaded_ids]
    old_rho = obs_simu_defaut.rho[lines_overloaded_ids]
    
    if np.all(new_rho < old_rho):
        print("Remedial action effective!")
    
    return obs_simu_defaut, lines_overloaded_ids


"""
=============================================================================
REMOVED DEPENDENCIES
=============================================================================

After migration, these can be removed from pyproject.toml:
- grid2op
- lightsim2grid  
- pypowsybl2grid (though you might keep it for compatibility)

Keep:
- pypowsybl
- numpy
- pandas
- networkx
- expertop4grid (may need adaptation)

=============================================================================
TESTING STRATEGY
=============================================================================

1. Create test cases that run same scenarios with both backends
2. Compare obs.rho values (should be very close)
3. Compare simulation results for same actions
4. Verify topology changes produce same effects

Example test:
    def test_same_results():
        # Run with grid2op
        obs_g2op = run_with_grid2op(scenario)
        
        # Run with pypowsybl
        obs_pp = run_with_pypowsybl(scenario)
        
        # Compare
        np.testing.assert_allclose(obs_g2op.rho, obs_pp.rho, rtol=0.01)
"""
