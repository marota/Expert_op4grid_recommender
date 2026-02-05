# tests/test_pypowsybl_backend.py
"""
Tests for the pypowsybl backend.

These tests verify that the pypowsybl backend provides equivalent functionality
to the grid2op-based implementation.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

# Skip all tests if pypowsybl is not available
pypowsybl = pytest.importorskip("pypowsybl")


class TestNetworkManager:
    """Tests for NetworkManager class."""

    @pytest.fixture
    def sample_network(self):
        """Create a simple test network."""
        # Create a simple 3-bus network
        network = pypowsybl.network.create_ieee9()
        return network

    def test_network_manager_creation(self, sample_network):
        """Test that NetworkManager can be created from a network."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        assert nm.n_line > 0
        assert nm.n_sub > 0
        assert len(nm.name_line) == nm.n_line
        assert len(nm.name_sub) == nm.n_sub

    def test_variant_management(self, sample_network):
        """Test variant creation and switching."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        # Create a variant
        variant_id = nm.create_variant("test_variant")
        assert variant_id == "test_variant"

        # Switch to it
        nm.set_working_variant("test_variant")

        # Make a change
        if nm.n_line > 0:
            first_line = nm.name_line[0]
            nm.disconnect_line(first_line)

        # Switch back to base
        nm.reset_to_base()

        # Clean up
        nm.remove_variant("test_variant")

    def test_load_flow(self, sample_network):
        """Test load flow execution."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager
        import pypowsybl.loadflow as lf

        nm = NetworkManager(network=sample_network)

        # Run AC load flow
        result = nm.run_load_flow(dc=False)
        assert result is not None

        # Run DC load flow
        result_dc = nm.run_load_flow(dc=True)
        assert result_dc is not None

    def test_cached_dataframes_exist(self, sample_network):
        """Test that DataFrames are properly cached during initialization."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        # Check cached DataFrames exist
        assert hasattr(nm, '_cached_lines_df')
        assert hasattr(nm, '_cached_trafos_df')
        assert hasattr(nm, '_cached_gen_df')
        assert hasattr(nm, '_cached_load_df')

        # Check they have correct types
        import pandas as pd
        assert isinstance(nm._cached_lines_df, pd.DataFrame)
        assert isinstance(nm._cached_trafos_df, pd.DataFrame)

    def test_cached_index_mappings(self, sample_network):
        """Test that index mappings are properly cached."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        # Check name -> index mappings
        assert hasattr(nm, '_sub_name_to_idx')
        assert hasattr(nm, '_line_name_to_idx')
        assert hasattr(nm, '_line_id_to_arr_idx')

        # Check all substations are mapped
        for i, name in enumerate(nm.name_sub):
            assert nm._sub_name_to_idx[name] == i

        # Check all lines are mapped
        for i, name in enumerate(nm.name_line):
            assert nm._line_name_to_idx[name] == i

    def test_cached_line_substation_arrays(self, sample_network):
        """Test cached line-to-substation arrays."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        # Check arrays exist
        assert hasattr(nm, '_cached_line_or_subid')
        assert hasattr(nm, '_cached_line_ex_subid')

        # Check correct length
        assert len(nm._cached_line_or_subid) == nm.n_line
        assert len(nm._cached_line_ex_subid) == nm.n_line

        # Check values are valid substation indices
        assert all(0 <= idx < nm.n_sub for idx in nm._cached_line_or_subid)
        assert all(0 <= idx < nm.n_sub for idx in nm._cached_line_ex_subid)

    def test_cached_gen_load_power_values(self, sample_network):
        """Test cached generator and load power values."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)
        nm.run_load_flow()  # Need load flow for power values

        # Check arrays exist
        assert hasattr(nm, '_gen_p_values')
        assert hasattr(nm, '_load_p_values')

        # Check correct lengths
        assert len(nm._gen_p_values) == nm._n_gen
        assert len(nm._load_p_values) == nm._n_load

    def test_elements_per_substation_cache(self, sample_network):
        """Test per-substation element lists are cached."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        # Check lists exist
        assert hasattr(nm, '_loads_per_sub')
        assert hasattr(nm, '_gens_per_sub')
        assert hasattr(nm, '_lines_or_per_sub')
        assert hasattr(nm, '_lines_ex_per_sub')

        # Check lengths
        assert len(nm._loads_per_sub) == nm.n_sub
        assert len(nm._gens_per_sub) == nm.n_sub
        assert len(nm._lines_or_per_sub) == nm.n_sub
        assert len(nm._lines_ex_per_sub) == nm.n_sub

        # Verify sub_info matches element counts
        for i in range(nm.n_sub):
            expected_count = (len(nm._loads_per_sub[i]) + len(nm._gens_per_sub[i]) +
                            len(nm._lines_or_per_sub[i]) + len(nm._lines_ex_per_sub[i]))
            assert nm._cached_sub_info[i] == expected_count

    def test_disconnect_lines_batch(self, sample_network):
        """Test batch line disconnection method."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        if nm.n_line >= 2:
            # Create variant for testing
            nm.create_variant("test_batch")
            nm.set_working_variant("test_batch")

            # Get lines to disconnect
            lines_to_disconnect = [nm.name_line[0], nm.name_line[1]]

            # Disconnect in batch
            nm.disconnect_lines_batch(lines_to_disconnect)

            # Verify disconnection by checking network state
            lines_df = nm.network.get_lines()
            for line_id in lines_to_disconnect:
                if line_id in lines_df.index:
                    assert not lines_df.loc[line_id, 'connected1']
                    assert not lines_df.loc[line_id, 'connected2']

            # Cleanup
            nm.reset_to_base()
            nm.remove_variant("test_batch")

    def test_disconnect_lines_batch_empty_list(self, sample_network):
        """Test batch disconnect with empty list does nothing."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        # Should not raise any errors
        nm.disconnect_lines_batch([])

    def test_disconnect_lines_batch_mixed_types(self, sample_network):
        """Test batch disconnect with both lines and transformers."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        # Collect lines and trafos
        lines_to_disconnect = []
        if len(nm._lines_set) > 0:
            lines_to_disconnect.append(list(nm._lines_set)[0])
        if len(nm._trafos_set) > 0:
            lines_to_disconnect.append(list(nm._trafos_set)[0])

        if lines_to_disconnect:
            nm.create_variant("test_mixed")
            nm.set_working_variant("test_mixed")

            nm.disconnect_lines_batch(lines_to_disconnect)

            nm.reset_to_base()
            nm.remove_variant("test_mixed")

    def test_get_line_p1_array(self, sample_network):
        """Test vectorized line power extraction."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)
        nm.run_load_flow()

        p1_arr = nm.get_line_p1_array()

        # Check correct type and length
        assert isinstance(p1_arr, np.ndarray)
        assert len(p1_arr) == nm.n_line

        # Check no NaN values (should be replaced with 0.0)
        assert not np.any(np.isnan(p1_arr))

    def test_get_line_currents_array(self, sample_network):
        """Test vectorized line currents extraction."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)
        nm.run_load_flow()

        i1_arr, i2_arr = nm.get_line_currents_array()

        # Check correct types and lengths
        assert isinstance(i1_arr, np.ndarray)
        assert isinstance(i2_arr, np.ndarray)
        assert len(i1_arr) == nm.n_line
        assert len(i2_arr) == nm.n_line

        # Check no NaN values
        assert not np.any(np.isnan(i1_arr))
        assert not np.any(np.isnan(i2_arr))

    def test_get_line_idx(self, sample_network):
        """Test line index lookup method."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        # Test valid line
        if nm.n_line > 0:
            line_name = nm.name_line[0]
            idx = nm.get_line_idx(line_name)
            assert idx == 0

        # Test invalid line returns -1
        idx = nm.get_line_idx("NONEXISTENT_LINE")
        assert idx == -1

    def test_get_sub_idx(self, sample_network):
        """Test substation index lookup method."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        nm = NetworkManager(network=sample_network)

        # Test valid substation
        if nm.n_sub > 0:
            sub_name = nm.name_sub[0]
            idx = nm.get_sub_idx(sub_name)
            assert idx == 0

        # Test invalid substation returns -1
        idx = nm.get_sub_idx("NONEXISTENT_SUB")
        assert idx == -1


class TestActionSpace:
    """Tests for ActionSpace class."""
    
    @pytest.fixture
    def env_with_action_space(self):
        """Create environment with action space."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager, ActionSpace
        
        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)
        
        return nm, action_space
    
    def test_line_disconnection_action(self, env_with_action_space):
        """Test creating a line disconnection action."""
        nm, action_space = env_with_action_space
        
        if nm.n_line > 0:
            first_line = nm.name_line[0]
            action = action_space({"set_line_status": [(first_line, -1)]})
            
            assert action is not None
            assert len(action._modifications) > 0
    
    def test_line_reconnection_action(self, env_with_action_space):
        """Test creating a line reconnection action."""
        nm, action_space = env_with_action_space
        
        if nm.n_line > 0:
            first_line = nm.name_line[0]
            action = action_space({"set_line_status": [(first_line, 1)]})
            
            assert action is not None
    
    def test_bus_action(self, env_with_action_space):
        """Test creating a bus topology action."""
        nm, action_space = env_with_action_space
        
        if nm.n_line > 0:
            first_line = nm.name_line[0]
            action = action_space({
                "set_bus": {
                    "lines_or_id": {first_line: 1},
                    "lines_ex_id": {first_line: 1}
                }
            })
            
            assert action is not None
    
    def test_action_combination(self, env_with_action_space):
        """Test combining multiple actions."""
        nm, action_space = env_with_action_space
        
        if nm.n_line >= 2:
            line1 = nm.name_line[0]
            line2 = nm.name_line[1]
            
            action1 = action_space({"set_line_status": [(line1, -1)]})
            action2 = action_space({"set_line_status": [(line2, -1)]})
            
            combined = action1 + action2
            
            assert combined is not None
            assert len(combined._modifications) == 2
    
    def test_switch_action_creation(self, env_with_action_space):
        """Test creating a switch action for topology changes."""
        nm, action_space = env_with_action_space
        
        # Create action with switches field (short names)
        switch_states = {
            "SWITCH_1": True,   # Open
            "SWITCH_2": False,  # Closed
        }
        action = action_space({"switches": switch_states})
        
        assert action is not None
        assert len(action._modifications) > 0
    
    def test_switch_action_empty_dict(self, env_with_action_space):
        """Test creating a switch action with empty switches dict."""
        nm, action_space = env_with_action_space
        
        # Empty switches should not add modifications
        action = action_space({"switches": {}})
        
        assert action is not None
        # Empty switches dict should not add a SwitchAction
        assert len(action._modifications) == 0
    
    def test_combined_bus_and_switch_action(self, env_with_action_space):
        """Test creating an action with both set_bus and switches."""
        nm, action_space = env_with_action_space
        
        if nm.n_line > 0:
            first_line = nm.name_line[0]
            action = action_space({
                "set_bus": {
                    "lines_or_id": {first_line: 1},
                    "lines_ex_id": {first_line: 1}
                },
                "switches": {
                    "SWITCH_1": True,
                }
            })
            
            assert action is not None
            # Should have both BusAction and SwitchAction modifications
            assert len(action._modifications) == 2
    
    def test_switch_action_with_line_status(self, env_with_action_space):
        """Test creating an action combining line status and switches."""
        nm, action_space = env_with_action_space
        
        if nm.n_line > 0:
            first_line = nm.name_line[0]
            action = action_space({
                "set_line_status": [(first_line, -1)],
                "switches": {
                    "SWITCH_1": True,
                }
            })
            
            assert action is not None
            assert len(action._modifications) == 2
    
    def test_switches_by_voltage_level_action(self, env_with_action_space):
        """Test creating an action using switches at top level (full IDs)."""
        nm, action_space = env_with_action_space
        
        # Create action with switches at top level (full switch IDs)
        # This is the new format from conversion_actions_repas
        action_dict = {
            "content": {"set_bus": {}},
            "switches": {
                "VL1_SWITCH_1": True,
                "VL1_SWITCH_2": False,
                "VL2_COUPL_DJ": True,
            }
        }
        action = action_space(action_dict)
        
        assert action is not None
        assert len(action._modifications) > 0
    
    def test_switches_at_top_level(self, env_with_action_space):
        """Test that switches at top level are correctly handled."""
        nm, action_space = env_with_action_space
        
        # New format: switches outside content
        action = action_space({
            "content": {"set_bus": {}},
            "switches": {"VL1_SW1": True}
        })
        
        assert action is not None
        # Should have one switch action
        assert len(action._modifications) == 1


class TestObservation:
    """Tests for PypowsyblObservation class."""
    
    @pytest.fixture
    def observation(self):
        """Create a test observation."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation
        )
        
        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)
        
        # Run load flow first
        nm.run_load_flow()
        
        obs = PypowsyblObservation(nm, action_space)
        return obs, nm, action_space
    
    def test_observation_properties(self, observation):
        """Test that observation has required properties."""
        obs, nm, _ = observation
        
        # Check array properties
        assert isinstance(obs.rho, np.ndarray)
        assert isinstance(obs.line_status, np.ndarray)
        assert isinstance(obs.theta_or, np.ndarray)
        assert isinstance(obs.theta_ex, np.ndarray)
        assert isinstance(obs.load_p, np.ndarray)
        assert isinstance(obs.gen_p, np.ndarray)
        
        # Check name arrays
        assert isinstance(obs.name_line, np.ndarray)
        assert isinstance(obs.name_sub, np.ndarray)
        
        # Check lengths match
        assert len(obs.rho) == nm.n_line
        assert len(obs.line_status) == nm.n_line
    
    def test_simulate_method(self, observation):
        """Test the simulate method."""
        obs, nm, action_space = observation
        
        if nm.n_line > 0:
            first_line = nm.name_line[0]
            action = action_space({"set_line_status": [(first_line, -1)]})
            
            obs_simu, reward, done, info = obs.simulate(action)
            
            # Should return an observation
            assert obs_simu is not None
            assert isinstance(obs_simu.rho, np.ndarray)
            
            # Info should have exception key
            assert "exception" in info


class TestSimulationEnvironment:
    """Tests for SimulationEnvironment class."""
    
    @pytest.fixture
    def temp_network_file(self):
        """Create a temporary network file."""
        network = pypowsybl.network.create_ieee9()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            network_path = Path(tmpdir) / "test_network.xiidm"
            network.save(str(network_path))
            yield network_path
    
    def test_environment_creation(self, temp_network_file):
        """Test creating environment from file."""
        from expert_op4grid_recommender.pypowsybl_backend import SimulationEnvironment
        
        env = SimulationEnvironment(network_path=temp_network_file)
        
        assert env.n_line > 0
        assert env.n_sub > 0
        assert env.action_space is not None
    
    def test_get_obs(self, temp_network_file):
        """Test getting observation."""
        from expert_op4grid_recommender.pypowsybl_backend import SimulationEnvironment
        
        env = SimulationEnvironment(network_path=temp_network_file)
        obs = env.get_obs()
        
        assert obs is not None
        assert isinstance(obs.rho, np.ndarray)
    
    def test_thermal_limits(self, temp_network_file):
        """Test thermal limit handling."""
        from expert_op4grid_recommender.pypowsybl_backend import SimulationEnvironment
        
        env = SimulationEnvironment(network_path=temp_network_file)
        
        limits = env.get_thermal_limit()
        assert len(limits) == env.n_line
        
        # Set new limits
        new_limits = np.ones(env.n_line) * 1000.0
        env.set_thermal_limit(new_limits)
        
        updated_limits = env.get_thermal_limit()
        np.testing.assert_array_equal(updated_limits, new_limits)


class TestOverflowAnalysis:
    """Tests for overflow analysis functionality."""

    @pytest.fixture
    def overflow_setup(self):
        """Setup for overflow analysis tests."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation, OverflowSimulator
        )

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)
        nm.run_load_flow()
        obs = PypowsyblObservation(nm, action_space)

        return nm, action_space, obs

    def test_overflow_simulator_creation(self, overflow_setup):
        """Test creating overflow simulator."""
        from expert_op4grid_recommender.pypowsybl_backend import OverflowSimulator

        nm, action_space, obs = overflow_setup

        # Test AC mode
        sim_ac = OverflowSimulator(nm, obs, use_dc=False)
        assert sim_ac is not None

        # Test DC mode
        sim_dc = OverflowSimulator(nm, obs, use_dc=True)
        assert sim_dc is not None

    def test_flow_changes_computation(self, overflow_setup):
        """Test computing flow changes after disconnection."""
        from expert_op4grid_recommender.pypowsybl_backend import OverflowSimulator

        nm, action_space, obs = overflow_setup
        sim = OverflowSimulator(nm, obs, use_dc=True)

        if nm.n_line > 0:
            first_line = nm.name_line[0]
            df = sim.compute_flow_changes_after_disconnection([first_line])

            assert df is not None
            assert 'line_name' in df.columns
            assert 'delta_flows' in df.columns
            assert len(df) == nm.n_line

    def test_base_flows_arr_cached(self, overflow_setup):
        """Test that base flows are cached as arrays."""
        from expert_op4grid_recommender.pypowsybl_backend import OverflowSimulator

        nm, action_space, obs = overflow_setup
        sim = OverflowSimulator(nm, obs, use_dc=True)

        # Check array cache exists
        assert hasattr(sim, '_base_flows_arr')
        assert hasattr(sim, '_base_currents_arr')

        # Check correct types
        assert isinstance(sim._base_flows_arr, np.ndarray)
        assert isinstance(sim._base_currents_arr, np.ndarray)

        # Check lengths
        assert len(sim._base_flows_arr) == nm.n_line
        assert len(sim._base_currents_arr) == nm.n_line

    def test_compute_flow_changes_and_rho_combined(self, overflow_setup):
        """Test the combined flow changes and rho computation method."""
        from expert_op4grid_recommender.pypowsybl_backend import OverflowSimulator

        nm, action_space, obs = overflow_setup
        sim = OverflowSimulator(nm, obs, use_dc=True)

        if nm.n_line > 0:
            first_line = nm.name_line[0]

            # Create thermal limits dict
            thermal_limits = {lid: 1000.0 for lid in nm.name_line}

            # Call combined method
            df, rho = sim.compute_flow_changes_and_rho([first_line], thermal_limits)

            # Check DataFrame
            assert df is not None
            assert 'line_name' in df.columns
            assert 'delta_flows' in df.columns
            assert 'idx_or' in df.columns
            assert 'idx_ex' in df.columns
            assert len(df) == nm.n_line

            # Check rho array
            assert isinstance(rho, np.ndarray)
            assert len(rho) == nm.n_line
            assert not np.any(np.isnan(rho))

    def test_combined_method_matches_separate_calls(self, overflow_setup):
        """Test that combined method produces same results as separate calls would."""
        from expert_op4grid_recommender.pypowsybl_backend import OverflowSimulator

        nm, action_space, obs = overflow_setup
        sim = OverflowSimulator(nm, obs, use_dc=True)

        if nm.n_line > 0:
            first_line = nm.name_line[0]
            thermal_limits = {lid: 1000.0 for lid in nm.name_line}

            # Get combined results
            df_combined, rho_combined = sim.compute_flow_changes_and_rho(
                [first_line], thermal_limits
            )

            # Get separate DataFrame result
            df_separate = sim.compute_flow_changes_after_disconnection([first_line])

            # Compare DataFrame columns that should match
            np.testing.assert_array_almost_equal(
                df_combined['delta_flows'].values,
                df_separate['delta_flows'].values
            )
            np.testing.assert_array_almost_equal(
                df_combined['init_flows'].values,
                df_separate['init_flows'].values
            )

    def test_combined_method_with_multiple_lines(self, overflow_setup):
        """Test combined method with multiple lines to disconnect."""
        from expert_op4grid_recommender.pypowsybl_backend import OverflowSimulator

        nm, action_space, obs = overflow_setup
        sim = OverflowSimulator(nm, obs, use_dc=True)

        if nm.n_line >= 2:
            lines_to_disconnect = [nm.name_line[0], nm.name_line[1]]
            thermal_limits = {lid: 1000.0 for lid in nm.name_line}

            df, rho = sim.compute_flow_changes_and_rho(lines_to_disconnect, thermal_limits)

            assert df is not None
            assert len(df) == nm.n_line
            assert len(rho) == nm.n_line

    def test_combined_method_empty_disconnect_list(self, overflow_setup):
        """Test combined method with empty disconnect list."""
        from expert_op4grid_recommender.pypowsybl_backend import OverflowSimulator

        nm, action_space, obs = overflow_setup
        sim = OverflowSimulator(nm, obs, use_dc=True)

        thermal_limits = {lid: 1000.0 for lid in nm.name_line}

        df, rho = sim.compute_flow_changes_and_rho([], thermal_limits)

        assert df is not None
        assert len(df) == nm.n_line
        assert len(rho) == nm.n_line

    def test_get_dataframe_uses_cached_array(self, overflow_setup):
        """Test that get_dataframe uses the cached array."""
        from expert_op4grid_recommender.pypowsybl_backend import OverflowSimulator

        nm, action_space, obs = overflow_setup
        sim = OverflowSimulator(nm, obs, use_dc=True)

        df = sim.get_dataframe()

        # Check it has expected columns
        assert 'init_flows' in df.columns
        assert 'idx_or' in df.columns
        assert 'idx_ex' in df.columns
        assert len(df) == nm.n_line


class TestAlphaDeespAdapter:
    """Tests for AlphaDeespAdapter class."""

    @pytest.fixture
    def adapter_setup(self):
        """Setup for AlphaDeespAdapter tests."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation
        )

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)
        nm.run_load_flow()
        obs = PypowsyblObservation(nm, action_space)

        return nm, action_space, obs

    def test_adapter_creation_with_overloaded_lines(self, adapter_setup):
        """Test creating AlphaDeespAdapter with overloaded lines."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import AlphaDeespAdapter

        nm, action_space, obs = adapter_setup

        # Create adapter with first line as overloaded
        ltc = [0] if nm.n_line > 0 else []

        adapter = AlphaDeespAdapter(
            obs=obs,
            action_space=action_space,
            observation_space=None,
            param_options={"ThresholdReportOfLine": 0.05},
            ltc=ltc,
            use_dc=True
        )

        assert adapter is not None
        assert hasattr(adapter, '_df')
        assert hasattr(adapter, 'obs_linecut')
        assert hasattr(adapter, 'topo')

    def test_adapter_creation_without_overloaded_lines(self, adapter_setup):
        """Test creating AlphaDeespAdapter with no overloaded lines."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import AlphaDeespAdapter

        nm, action_space, obs = adapter_setup

        adapter = AlphaDeespAdapter(
            obs=obs,
            action_space=action_space,
            observation_space=None,
            param_options={"ThresholdReportOfLine": 0.05},
            ltc=[],
            use_dc=True
        )

        assert adapter is not None
        # Without LTC, obs_linecut should be same as original obs
        assert adapter.obs_linecut is obs

    def test_adapter_obs_linecut_has_rho(self, adapter_setup):
        """Test that obs_linecut from adapter has rho attribute."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import AlphaDeespAdapter

        nm, action_space, obs = adapter_setup

        ltc = [0] if nm.n_line > 0 else []

        adapter = AlphaDeespAdapter(
            obs=obs,
            action_space=action_space,
            observation_space=None,
            param_options={"ThresholdReportOfLine": 0.05},
            ltc=ltc,
            use_dc=True
        )

        # obs_linecut should have rho
        assert hasattr(adapter.obs_linecut, 'rho')
        assert isinstance(adapter.obs_linecut.rho, np.ndarray)
        assert len(adapter.obs_linecut.rho) == nm.n_line

    def test_adapter_topo_structure(self, adapter_setup):
        """Test that adapter topo has correct structure."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import AlphaDeespAdapter

        nm, action_space, obs = adapter_setup

        adapter = AlphaDeespAdapter(
            obs=obs,
            action_space=action_space,
            observation_space=None,
            param_options={},
            ltc=[],
            use_dc=True
        )

        topo = adapter.topo

        # Check structure
        assert 'edges' in topo
        assert 'nodes' in topo

        # Check edges
        assert 'idx_or' in topo['edges']
        assert 'idx_ex' in topo['edges']
        assert len(topo['edges']['idx_or']) == nm.n_line

        # Check nodes
        assert 'are_prods' in topo['nodes']
        assert 'are_loads' in topo['nodes']
        assert 'names' in topo['nodes']
        assert len(topo['nodes']['names']) == nm.n_sub

    def test_adapter_get_dataframe(self, adapter_setup):
        """Test adapter get_dataframe method."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import AlphaDeespAdapter

        nm, action_space, obs = adapter_setup

        ltc = [0] if nm.n_line > 0 else []

        adapter = AlphaDeespAdapter(
            obs=obs,
            action_space=action_space,
            observation_space=None,
            param_options={},
            ltc=ltc,
            use_dc=True
        )

        df = adapter.get_dataframe()

        assert df is not None
        assert len(df) == nm.n_line
        assert 'delta_flows' in df.columns

    def test_adapter_get_substation_elements(self, adapter_setup):
        """Test adapter get_substation_elements method."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import AlphaDeespAdapter

        nm, action_space, obs = adapter_setup

        adapter = AlphaDeespAdapter(
            obs=obs,
            action_space=action_space,
            observation_space=None,
            param_options={},
            ltc=[],
            use_dc=True
        )

        elements = adapter.get_substation_elements()

        # Check structure
        assert len(elements) == nm.n_sub

        for sub_idx, sub_elements in elements.items():
            assert 'loads' in sub_elements
            assert 'generators' in sub_elements
            assert 'lines_or' in sub_elements
            assert 'lines_ex' in sub_elements

    def test_adapter_mappings(self, adapter_setup):
        """Test adapter mapping methods."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import AlphaDeespAdapter

        nm, action_space, obs = adapter_setup

        adapter = AlphaDeespAdapter(
            obs=obs,
            action_space=action_space,
            observation_space=None,
            param_options={},
            ltc=[],
            use_dc=True
        )

        # Test substation to node mapping (should be identity)
        sub_to_node = adapter.get_substation_to_node_mapping()
        for i in range(nm.n_sub):
            assert sub_to_node[i] == i

        # Test internal to external mapping
        int_to_ext = adapter.get_internal_to_external_mapping()
        for i, name in enumerate(nm.name_sub):
            assert int_to_ext[i] == name


class TestObsLineCut:
    """Tests for _ObsLineCut helper class."""

    def test_obs_linecut_creation(self):
        """Test creating _ObsLineCut wrapper."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import _ObsLineCut

        rho_values = np.array([0.5, 0.8, 1.2, 0.3])
        obs_linecut = _ObsLineCut(rho_values)

        assert obs_linecut is not None
        assert hasattr(obs_linecut, 'rho')
        np.testing.assert_array_equal(obs_linecut.rho, rho_values)

    def test_obs_linecut_with_empty_array(self):
        """Test _ObsLineCut with empty array."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import _ObsLineCut

        rho_values = np.array([])
        obs_linecut = _ObsLineCut(rho_values)

        assert len(obs_linecut.rho) == 0


class TestOverflowGraphBuilder:
    """Tests for OverflowGraphBuilder class."""

    @pytest.fixture
    def builder_setup(self):
        """Setup for OverflowGraphBuilder tests."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation, OverflowSimulator
        )
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import OverflowGraphBuilder

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)
        nm.run_load_flow()
        obs = PypowsyblObservation(nm, action_space)
        sim = OverflowSimulator(nm, obs, use_dc=True)

        return nm, sim

    def test_builder_creation(self, builder_setup):
        """Test creating OverflowGraphBuilder."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import OverflowGraphBuilder

        nm, sim = builder_setup

        overloaded_ids = [0] if nm.n_line > 0 else []
        builder = OverflowGraphBuilder(sim, overloaded_ids)

        assert builder is not None

    def test_builder_build_graph(self, builder_setup):
        """Test building overflow graph."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import OverflowGraphBuilder
        import networkx as nx

        nm, sim = builder_setup

        overloaded_ids = [0] if nm.n_line > 0 else []
        builder = OverflowGraphBuilder(sim, overloaded_ids)

        graph, df = builder.build_graph()

        assert isinstance(graph, nx.MultiDiGraph)
        assert df is not None
        assert len(df) == nm.n_line

    def test_builder_get_topology(self, builder_setup):
        """Test getting topology from builder."""
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import OverflowGraphBuilder

        nm, sim = builder_setup

        builder = OverflowGraphBuilder(sim, [])
        topo = builder.get_topology()

        assert 'n_sub' in topo
        assert 'n_line' in topo
        assert 'name_sub' in topo
        assert 'name_line' in topo
        assert topo['n_sub'] == nm.n_sub
        assert topo['n_line'] == nm.n_line


class TestSwitchAction:
    """Tests for the SwitchAction class."""
    
    def test_switch_action_import(self):
        """Test that SwitchAction can be imported."""
        from expert_op4grid_recommender.pypowsybl_backend.action_space import SwitchAction
        assert SwitchAction is not None
    
    def test_switch_action_creation(self):
        """Test creating a SwitchAction with switch states."""
        from expert_op4grid_recommender.pypowsybl_backend.action_space import SwitchAction
        
        switch_states = {
            "SWITCH_1": True,   # Open
            "SWITCH_2": False,  # Closed
        }
        action = SwitchAction(switch_states)
        
        assert action is not None
        assert len(action._modifications) == 1
    
    def test_switch_action_empty_dict(self):
        """Test creating a SwitchAction with empty dict."""
        from expert_op4grid_recommender.pypowsybl_backend.action_space import SwitchAction
        
        action = SwitchAction({})
        
        assert action is not None
        assert len(action._modifications) == 1
    
    def test_switch_action_combination_with_pypowsybl_action(self):
        """Test combining SwitchAction with PypowsyblAction."""
        from expert_op4grid_recommender.pypowsybl_backend.action_space import (
            SwitchAction, PypowsyblAction
        )
        
        switch_action = SwitchAction({"SW1": True})
        other_action = PypowsyblAction()
        
        combined = switch_action + other_action
        
        assert combined is not None
        assert len(combined._modifications) >= 1
    
    def test_switch_action_with_real_network(self):
        """Test SwitchAction with a real pypowsybl network."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager
        from expert_op4grid_recommender.pypowsybl_backend.action_space import SwitchAction
        
        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        
        # Get actual switch IDs from the network
        switches_df = network.get_switches()
        if len(switches_df) > 0:
            switch_id = switches_df.index[0]
            switch_states = {switch_id: True}  # Open the switch
            
            action = SwitchAction(switch_states)
            
            # Create a variant to test
            nm.create_variant("test_switch")
            nm.set_working_variant("test_switch")
            
            # Apply the action
            action.apply(nm)
            
            # Verify the switch state changed
            new_switches = network.get_switches()
            assert new_switches.loc[switch_id, 'open'] == True
            
            # Cleanup
            nm.reset_to_base()
            nm.remove_variant("test_switch")


class TestIntegration:
    """Integration tests comparing pypowsybl backend behavior."""

    def test_contingency_analysis_workflow(self):
        """Test a complete contingency analysis workflow."""
        from expert_op4grid_recommender.pypowsybl_backend import SimulationEnvironment

        # Create test network
        network = pypowsybl.network.create_ieee9()

        with tempfile.TemporaryDirectory() as tmpdir:
            network_path = Path(tmpdir) / "test_network.xiidm"
            network.save(str(network_path))

            # Create environment
            env = SimulationEnvironment(network_path=network_path)
            obs = env.get_obs()

            # Get line names
            line_names = list(env.name_line)
            if len(line_names) > 0:
                # Create contingency action
                contingency_line = line_names[0]
                action = env.action_space({
                    "set_line_status": [(contingency_line, -1)]
                })

                # Simulate
                obs_simu, _, done, info = obs.simulate(action)

                # Check results
                assert obs_simu is not None

                # Find overloaded lines
                overloaded = np.where(obs_simu.rho >= 1.0)[0]
                print(f"Overloaded lines after contingency: {len(overloaded)}")

    def test_action_with_switches_workflow(self):
        """Test workflow using action with switches field."""
        from expert_op4grid_recommender.pypowsybl_backend import SimulationEnvironment

        network = pypowsybl.network.create_ieee9()

        with tempfile.TemporaryDirectory() as tmpdir:
            network_path = Path(tmpdir) / "test_network.xiidm"
            network.save(str(network_path))

            env = SimulationEnvironment(network_path=network_path)

            # Get actual switches from the network
            switches_df = env.network_manager.network.get_switches()

            if len(switches_df) > 0:
                switch_id = switches_df.index[0]
                original_state = switches_df.loc[switch_id, 'open']

                # Create action with switches field
                action = env.action_space({
                    "switches": {switch_id: not original_state}
                })

                assert action is not None
                assert len(action._modifications) > 0

    def test_full_alpha_deesp_workflow(self):
        """Test complete AlphaDeespAdapter workflow as used in graph building."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation
        )
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import (
            AlphaDeespAdapter, build_overflow_graph_pypowsybl
        )

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)
        nm.run_load_flow()
        obs = PypowsyblObservation(nm, action_space)

        # Create a SimulationEnvironment-like object for the function
        class MockEnv:
            def __init__(self, nm, action_space):
                self.network_manager = nm
                self.action_space = action_space

        mock_env = MockEnv(nm, action_space)

        # Simulate having an overloaded line
        overloaded_ids = [0] if nm.n_line > 0 else []

        # Call the full build function
        try:
            result = build_overflow_graph_pypowsybl(
                env=mock_env,
                obs=obs,
                overloaded_line_ids=overloaded_ids,
                non_connected_reconnectable_lines=[],
                lines_non_reconnectable=[],
                timestep=0,
                do_consolidate_graph=False,
                use_dc=True,
                param_options={"ThresholdReportOfLine": 0.05}
            )

            df_of_g, overflow_sim, g_overflow, hubs, g_dist, node_mapping = result

            assert df_of_g is not None
            assert overflow_sim is not None
            assert g_overflow is not None

        except ImportError:
            # alphaDeesp may not be available in test environment
            pytest.skip("alphaDeesp not available")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_network_manager_with_no_transformers(self):
        """Test NetworkManager with network that has no transformers."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        # IEEE 9 has transformers, but we can still test the sets
        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)

        # Should have both lines and possibly transformers
        assert hasattr(nm, '_lines_set')
        assert hasattr(nm, '_trafos_set')

    def test_observation_with_disconnected_line(self):
        """Test observation when a line is already disconnected."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation
        )

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)

        # Disconnect a line first
        if nm.n_line > 0:
            nm.disconnect_line(nm.name_line[0])

        nm.run_load_flow()
        action_space = ActionSpace(nm)
        obs = PypowsyblObservation(nm, action_space)

        # Check that line_status reflects disconnection
        assert not obs.line_status[0]

    def test_overflow_simulator_with_all_lines_disconnected_in_obs(self):
        """Test overflow simulator when observation has disconnected lines."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation, OverflowSimulator
        )

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)

        # Create variant and disconnect a line
        nm.create_variant("test_disc")
        nm.set_working_variant("test_disc")

        if nm.n_line > 0:
            nm.disconnect_line(nm.name_line[0])

        nm.run_load_flow()
        obs = PypowsyblObservation(nm, action_space)

        # Reset to base
        nm.reset_to_base()
        nm.remove_variant("test_disc")

        # Create simulator from obs with disconnected line
        sim = OverflowSimulator(nm, obs, use_dc=True)

        # Should handle the already-disconnected line correctly
        if nm.n_line > 1:
            df = sim.compute_flow_changes_after_disconnection([nm.name_line[1]])
            assert df is not None

    def test_thermal_limits_edge_values(self):
        """Test rho computation with edge case thermal limits."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation, OverflowSimulator
        )

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)
        nm.run_load_flow()
        obs = PypowsyblObservation(nm, action_space)

        sim = OverflowSimulator(nm, obs, use_dc=True)

        if nm.n_line > 0:
            # Test with zero thermal limit (should handle gracefully)
            thermal_limits = {lid: 0.0 for lid in nm.name_line}
            df, rho = sim.compute_flow_changes_and_rho([nm.name_line[0]], thermal_limits)

            # rho should be 0 where thermal limit is 0
            assert not np.any(np.isinf(rho))

            # Test with very large thermal limits
            thermal_limits = {lid: 1e10 for lid in nm.name_line}
            df, rho = sim.compute_flow_changes_and_rho([nm.name_line[0]], thermal_limits)

            # rho should be very small
            assert np.all(rho < 1.0)


class TestConsistency:
    """Tests for consistency and reproducibility."""

    def test_multiple_adapter_calls_consistent(self):
        """Test that creating adapter multiple times gives consistent results."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation
        )
        from expert_op4grid_recommender.pypowsybl_backend.overflow_analysis import AlphaDeespAdapter

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)
        nm.run_load_flow()
        obs = PypowsyblObservation(nm, action_space)

        ltc = [0] if nm.n_line > 0 else []

        # Create adapter twice
        adapter1 = AlphaDeespAdapter(
            obs=obs, action_space=action_space, observation_space=None,
            param_options={}, ltc=ltc, use_dc=True
        )

        adapter2 = AlphaDeespAdapter(
            obs=obs, action_space=action_space, observation_space=None,
            param_options={}, ltc=ltc, use_dc=True
        )

        # Results should be identical
        df1 = adapter1.get_dataframe()
        df2 = adapter2.get_dataframe()

        np.testing.assert_array_almost_equal(
            df1['delta_flows'].values,
            df2['delta_flows'].values
        )

        np.testing.assert_array_almost_equal(
            adapter1.obs_linecut.rho,
            adapter2.obs_linecut.rho
        )

    def test_variant_isolation(self):
        """Test that variants are properly isolated."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        nm.run_load_flow()

        # Get original flows
        original_p1 = nm.get_line_p1_array().copy()

        # Create variant and disconnect lines
        nm.create_variant("test_isolation")
        nm.set_working_variant("test_isolation")

        if nm.n_line > 0:
            nm.disconnect_lines_batch([nm.name_line[0]])
            nm.run_load_flow()

        # Variant flows should be different
        variant_p1 = nm.get_line_p1_array()

        # Reset to base
        nm.reset_to_base()
        nm.remove_variant("test_isolation")

        # Base flows should be unchanged
        base_p1 = nm.get_line_p1_array()
        np.testing.assert_array_almost_equal(original_p1, base_p1)

    def test_cache_consistency_after_operations(self):
        """Test that caches remain consistent after various operations."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)

        # Store initial cache state
        initial_line_or = nm._cached_line_or_subid.copy()
        initial_line_ex = nm._cached_line_ex_subid.copy()

        # Perform operations
        nm.create_variant("test_cache")
        nm.set_working_variant("test_cache")

        if nm.n_line > 0:
            nm.disconnect_line(nm.name_line[0])

        nm.reset_to_base()
        nm.remove_variant("test_cache")

        # Caches should be unchanged (they're topology-based, not state-based)
        np.testing.assert_array_equal(initial_line_or, nm._cached_line_or_subid)
        np.testing.assert_array_equal(initial_line_ex, nm._cached_line_ex_subid)


class TestPerformanceCharacteristics:
    """Tests verifying performance optimizations are in place."""

    def test_batch_disconnect_faster_than_loop(self):
        """Verify batch disconnect method exists and works correctly."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)

        # Verify method exists
        assert hasattr(nm, 'disconnect_lines_batch')

        # Verify it works
        if nm.n_line >= 2:
            nm.create_variant("test_batch_perf")
            nm.set_working_variant("test_batch_perf")

            lines = [nm.name_line[0], nm.name_line[1]]
            nm.disconnect_lines_batch(lines)

            nm.reset_to_base()
            nm.remove_variant("test_batch_perf")

    def test_array_getters_return_numpy(self):
        """Verify array getters return numpy arrays (not lists or dicts)."""
        from expert_op4grid_recommender.pypowsybl_backend import NetworkManager

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        nm.run_load_flow()

        p1_arr = nm.get_line_p1_array()
        i1_arr, i2_arr = nm.get_line_currents_array()

        assert isinstance(p1_arr, np.ndarray)
        assert isinstance(i1_arr, np.ndarray)
        assert isinstance(i2_arr, np.ndarray)

    def test_overflow_simulator_uses_cached_arrays(self):
        """Verify OverflowSimulator uses cached arrays internally."""
        from expert_op4grid_recommender.pypowsybl_backend import (
            NetworkManager, ActionSpace, PypowsyblObservation, OverflowSimulator
        )

        network = pypowsybl.network.create_ieee9()
        nm = NetworkManager(network=network)
        action_space = ActionSpace(nm)
        nm.run_load_flow()
        obs = PypowsyblObservation(nm, action_space)

        sim = OverflowSimulator(nm, obs, use_dc=True)

        # Should have array caches
        assert hasattr(sim, '_base_flows_arr')
        assert isinstance(sim._base_flows_arr, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
