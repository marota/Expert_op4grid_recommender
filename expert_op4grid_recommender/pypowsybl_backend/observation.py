# expert_op4grid_recommender/pypowsybl_backend/observation.py
"""
Observation class providing grid2op-compatible interface for pypowsybl networks.

This module provides a read-only view of the network state with properties
matching the grid2op observation interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import pypowsybl.loadflow as lf

if TYPE_CHECKING:
    from .network_manager import NetworkManager
    from .action_space import ActionSpace


class PypowsyblObservation:
    """
    Grid2op-compatible observation interface for pypowsybl networks.
    
    Provides read-only access to network state including:
    - Line flows and loading ratios (rho)
    - Line status (connected/disconnected)
    - Bus voltage magnitudes and angles
    - Load and generation values
    - Topology information
    
    The simulate() method creates a temporary variant, applies an action,
    runs load flow, extracts results, and cleans up the variant.
    
    Attributes:
        _network_manager: Reference to the NetworkManager
        _thermal_limits: Cached thermal limits for rho calculation
    """
    
    def __init__(self, 
                 network_manager: 'NetworkManager',
                 action_space: 'ActionSpace',
                 thermal_limits: Optional[Dict[str, float]] = None):
        """
        Initialize observation from network state.
        
        Args:
            network_manager: The NetworkManager instance
            action_space: The ActionSpace instance for simulate()
            thermal_limits: Optional override for thermal limits
        """
        self._network_manager = network_manager
        self._action_space = action_space
        self._thermal_limits = thermal_limits or network_manager.get_thermal_limits()
        
        # Cache current state
        self._refresh_state()
    
    def _refresh_state(self):
        """Refresh cached state from network."""
        nm = self._network_manager
        net = nm.network
        
        # Line information
        self._line_flows = nm.get_line_flows()
        
        # Compute rho (loading ratio)
        self._compute_rho()
        
        # Line status
        self._line_status = self._line_flows['connected'].values
        
        # Bus voltages and angles
        bus_df = nm.get_bus_voltages()
        self._v_mag = bus_df['v_mag'].values
        self._v_angle = bus_df['v_angle'].values
        
        # Get angles per line terminal
        self._compute_line_angles()
        
        # Compute bus assignments for line terminals
        self._compute_line_buses()
        
        # Load and generation
        self._load_p = net.get_loads()['p'].values
        self._load_q = net.get_loads()['q'].values
        self._gen_p = net.get_generators()['p'].values
        self._gen_q = net.get_generators()['q'].values
    
    def _compute_rho(self):
        """Compute line loading ratios.
        
        Uses i1 (side 1 current) for rho calculation since thermal limits 
        are defined for side 1 in pypowsybl convention.
        """
        nm = self._network_manager
        
        rho = np.zeros(nm.n_line)
        
        for i, line_id in enumerate(nm.name_line):
            if line_id in self._line_flows.index:
                i1 = abs(self._line_flows.loc[line_id, 'i1'])
                i1 = i1 if not np.isnan(i1) else 0.0
                
                thermal_limit = self._thermal_limits.get(line_id, 9999.0)
                if thermal_limit > 0:
                    rho[i] = i1 / thermal_limit
                else:
                    rho[i] = 0.0
            else:
                rho[i] = 0.0
        
        self._rho = rho
    
    def _compute_line_angles(self):
        """Compute voltage angles at line terminals."""
        nm = self._network_manager
        net = nm.network
        
        n_line = nm.n_line
        self._theta_or = np.zeros(n_line)
        self._theta_ex = np.zeros(n_line)
        
        # Get bus information
        lines_df = net.get_lines()
        trafos_df = net.get_2_windings_transformers()
        buses_df = net.get_buses()
        
        for i, line_id in enumerate(nm.name_line):
            try:
                if line_id in lines_df.index:
                    bus1_id = lines_df.loc[line_id, 'bus1_id']
                    bus2_id = lines_df.loc[line_id, 'bus2_id']
                elif line_id in trafos_df.index:
                    bus1_id = trafos_df.loc[line_id, 'bus1_id']
                    bus2_id = trafos_df.loc[line_id, 'bus2_id']
                else:
                    continue
                
                if pd.notna(bus1_id) and bus1_id in buses_df.index:
                    self._theta_or[i] = buses_df.loc[bus1_id, 'v_angle']
                if pd.notna(bus2_id) and bus2_id in buses_df.index:
                    self._theta_ex[i] = buses_df.loc[bus2_id, 'v_angle']
            except (KeyError, TypeError):
                pass
    
    def _compute_line_buses(self):
        """
        Compute bus assignments for line terminals.
        
        In grid2op, line_or_bus and line_ex_bus indicate which local bus (1 or 2)
        each line terminal is connected to within its substation.
        -1 means disconnected.
        
        In pypowsybl with bus/breaker topology, we map the bus_id to a local
        bus number within the voltage level (substation).
        """
        nm = self._network_manager
        net = nm.network
        
        n_line = nm.n_line
        self._line_or_bus = np.ones(n_line, dtype=int)  # Default to bus 1
        self._line_ex_bus = np.ones(n_line, dtype=int)  # Default to bus 1
        
        # Get line and transformer data
        lines_df = net.get_lines()
        trafos_df = net.get_2_windings_transformers()
        
        # Build a mapping of voltage_level -> list of bus_ids
        # This helps us determine local bus number within each substation
        buses_df = net.get_buses()
        vl_to_buses = {}
        for bus_id, row in buses_df.iterrows():
            vl_id = row.get('voltage_level_id', '')
            if vl_id not in vl_to_buses:
                vl_to_buses[vl_id] = []
            vl_to_buses[vl_id].append(bus_id)
        
        # Sort buses within each voltage level for consistent numbering
        for vl_id in vl_to_buses:
            vl_to_buses[vl_id] = sorted(vl_to_buses[vl_id])
        
        for i, line_id in enumerate(nm.name_line):
            try:
                # Get bus IDs for this line
                if line_id in lines_df.index:
                    bus1_id = lines_df.loc[line_id, 'bus1_id']
                    bus2_id = lines_df.loc[line_id, 'bus2_id']
                    connected1 = lines_df.loc[line_id, 'connected1'] if 'connected1' in lines_df.columns else True
                    connected2 = lines_df.loc[line_id, 'connected2'] if 'connected2' in lines_df.columns else True
                elif line_id in trafos_df.index:
                    bus1_id = trafos_df.loc[line_id, 'bus1_id']
                    bus2_id = trafos_df.loc[line_id, 'bus2_id']
                    connected1 = trafos_df.loc[line_id, 'connected1'] if 'connected1' in trafos_df.columns else True
                    connected2 = trafos_df.loc[line_id, 'connected2'] if 'connected2' in trafos_df.columns else True
                else:
                    continue
                
                # Determine local bus number for origin (bus1)
                if pd.isna(bus1_id) or (isinstance(connected1, bool) and not connected1):
                    self._line_or_bus[i] = -1  # Disconnected
                else:
                    # Find which voltage level this bus belongs to
                    if bus1_id in buses_df.index:
                        vl_id = buses_df.loc[bus1_id, 'voltage_level_id']
                        if vl_id in vl_to_buses:
                            # Local bus number is position in sorted list + 1
                            try:
                                local_bus = vl_to_buses[vl_id].index(bus1_id) + 1
                                self._line_or_bus[i] = local_bus
                            except ValueError:
                                self._line_or_bus[i] = 1
                
                # Determine local bus number for extremity (bus2)
                if pd.isna(bus2_id) or (isinstance(connected2, bool) and not connected2):
                    self._line_ex_bus[i] = -1  # Disconnected
                else:
                    if bus2_id in buses_df.index:
                        vl_id = buses_df.loc[bus2_id, 'voltage_level_id']
                        if vl_id in vl_to_buses:
                            try:
                                local_bus = vl_to_buses[vl_id].index(bus2_id) + 1
                                self._line_ex_bus[i] = local_bus
                            except ValueError:
                                self._line_ex_bus[i] = 1
                                
            except (KeyError, TypeError) as e:
                # Default to bus 1 if we can't determine
                pass
    
    # ========== Grid2op-compatible properties ==========
    
    @property
    def rho(self) -> np.ndarray:
        """Line loading ratios (I / I_max)."""
        return self._rho.copy()
    
    @property
    def line_status(self) -> np.ndarray:
        """Boolean array of line connection status."""
        return self._line_status.copy()
    
    @property
    def line_or_bus(self) -> np.ndarray:
        """
        Bus assignment at line origin terminals.
        
        Values:
        - 1 or 2: Local bus number within the substation
        - -1: Disconnected
        """
        return self._line_or_bus.copy()
    
    @property
    def line_ex_bus(self) -> np.ndarray:
        """
        Bus assignment at line extremity terminals.
        
        Values:
        - 1 or 2: Local bus number within the substation
        - -1: Disconnected
        """
        return self._line_ex_bus.copy()
    
    @property
    def a_or(self) -> np.ndarray:
        """
        Current intensity at line origin terminals (A).
        """
        return self._get_line_i_or()
    
    @property
    def a_ex(self) -> np.ndarray:
        """
        Current intensity at line extremity terminals (A).
        """
        return self._get_line_i_ex()
    
    @property
    def p_or(self) -> np.ndarray:
        """Active power at line origin terminals (MW)."""
        return self._get_line_p_or()
    
    @property
    def p_ex(self) -> np.ndarray:
        """Active power at line extremity terminals (MW)."""
        return self._get_line_p_ex()
    
    def _get_line_i_or(self) -> np.ndarray:
        """Get current intensity at line origins (A)."""
        i_or = np.zeros(self._network_manager.n_line)
        for i, line_id in enumerate(self._network_manager.name_line):
            if line_id in self._line_flows.index:
                i1 = self._line_flows.loc[line_id, 'i1']
                i_or[i] = i1 if not np.isnan(i1) else 0.0
        return i_or
    
    def _get_line_i_ex(self) -> np.ndarray:
        """Get current intensity at line extremities (A)."""
        i_ex = np.zeros(self._network_manager.n_line)
        for i, line_id in enumerate(self._network_manager.name_line):
            if line_id in self._line_flows.index:
                i2 = self._line_flows.loc[line_id, 'i2']
                i_ex[i] = i2 if not np.isnan(i2) else 0.0
        return i_ex
    
    def _get_line_p_or(self) -> np.ndarray:
        """Get active power at line origins (MW)."""
        p_or = np.zeros(self._network_manager.n_line)
        for i, line_id in enumerate(self._network_manager.name_line):
            if line_id in self._line_flows.index:
                p1 = self._line_flows.loc[line_id, 'p1']
                p_or[i] = p1 if not np.isnan(p1) else 0.0
        return p_or
    
    def _get_line_p_ex(self) -> np.ndarray:
        """Get active power at line extremities (MW)."""
        p_ex = np.zeros(self._network_manager.n_line)
        for i, line_id in enumerate(self._network_manager.name_line):
            if line_id in self._line_flows.index:
                p2 = self._line_flows.loc[line_id, 'p2']
                p_ex[i] = p2 if not np.isnan(p2) else 0.0
        return p_ex
    
    @property
    def theta_or(self) -> np.ndarray:
        """Voltage angles at line origin terminals (degrees)."""
        return self._theta_or.copy()
    
    @property
    def theta_ex(self) -> np.ndarray:
        """Voltage angles at line extremity terminals (degrees)."""
        return self._theta_ex.copy()
    
    @property
    def load_p(self) -> np.ndarray:
        """Active power consumption at loads (MW)."""
        return self._load_p.copy()
    
    @property
    def load_q(self) -> np.ndarray:
        """Reactive power consumption at loads (MVAr)."""
        return self._load_q.copy()
    
    @property
    def gen_p(self) -> np.ndarray:
        """Active power generation (MW)."""
        return self._gen_p.copy()
    
    @property
    def gen_q(self) -> np.ndarray:
        """Reactive power generation (MVAr)."""
        return self._gen_q.copy()
    
    @property
    def name_line(self) -> np.ndarray:
        """Array of line names."""
        return self._network_manager.name_line
    
    @property
    def name_sub(self) -> np.ndarray:
        """Array of substation names."""
        return self._network_manager.name_sub
    
    @property
    def name_gen(self) -> np.ndarray:
        """Array of generator names."""
        return self._network_manager.name_gen
    
    @property
    def name_load(self) -> np.ndarray:
        """Array of load names."""
        return self._network_manager.name_load
    
    @property
    def line_or_to_subid(self) -> np.ndarray:
        """Substation index at line origin."""
        return self._network_manager.get_line_or_subid()
    
    @property
    def line_ex_to_subid(self) -> np.ndarray:
        """Substation index at line extremity."""
        return self._network_manager.get_line_ex_subid()
    
    @property
    def n_line(self) -> int:
        """Number of lines."""
        return self._network_manager.n_line
    
    @property
    def n_sub(self) -> int:
        """Number of substations."""
        return self._network_manager.n_sub
    
    @property
    def sub_info(self) -> np.ndarray:
        """
        Number of elements per substation.
        
        Note: This is a simplified version - in grid2op this includes
        all elements (gens, loads, lines_or, lines_ex) per substation.
        """
        # For now, return a placeholder - needs proper implementation
        # based on how elements map to substations
        return np.ones(self.n_sub, dtype=int) * 10  # Placeholder
    
    @property
    def topo_vect(self) -> np.ndarray:
        """
        Topology vector (bus assignment for all elements).
        
        Note: This requires tracking bus assignments in pypowsybl.
        Simplified implementation - full version needs bus/breaker model.
        """
        # Placeholder - needs proper implementation
        return np.ones(sum(self.sub_info), dtype=int)
    
    def sub_topology(self, sub_id: int) -> np.ndarray:
        """
        Get topology vector for a specific substation.
        
        Args:
            sub_id: Substation index
            
        Returns:
            Array of bus assignments for elements in this substation
        """
        # Placeholder implementation
        # In full implementation, need to track which elements
        # are connected to which bus within each voltage level
        n_elements = int(self.sub_info[sub_id])
        return np.ones(n_elements, dtype=int)
    
    def get_obj_connect_to(self, substation_id: int) -> Dict[str, List[int]]:
        """
        Get objects connected to a substation.
        
        Args:
            substation_id: Substation index
            
        Returns:
            Dictionary with keys 'loads_id', 'generators_id', 'lines_or_id', 
            'lines_ex_id' containing lists of element indices.
        """
        sub_name = self.name_sub[substation_id]
        net = self._network_manager.network
        
        result = {
            'loads_id': [],
            'generators_id': [],
            'lines_or_id': [],
            'lines_ex_id': []
        }
        
        # Find loads in this substation
        loads_df = net.get_loads()
        for i, (load_id, row) in enumerate(loads_df.iterrows()):
            if row.get('voltage_level_id') == sub_name:
                result['loads_id'].append(i)
        
        # Find generators
        gens_df = net.get_generators()
        for i, (gen_id, row) in enumerate(gens_df.iterrows()):
            if row.get('voltage_level_id') == sub_name:
                result['generators_id'].append(i)
        
        # Find lines (origin and extremity)
        nm = self._network_manager
        for i, line_id in enumerate(nm.name_line):
            if nm._line_or_sub.get(line_id) == sub_name:
                result['lines_or_id'].append(i)
            if nm._line_ex_sub.get(line_id) == sub_name:
                result['lines_ex_id'].append(i)
        
        return result
    
    def topo_vect_element(self, topo_vect_pos: int) -> Dict[str, Any]:
        """
        Get element information for a topology vector position.
        
        Args:
            topo_vect_pos: Position in the topology vector
            
        Returns:
            Dictionary with element type and ID
        """
        # Placeholder - needs proper implementation based on
        # how elements are ordered in the topology vector
        return {'type': 'unknown', 'id': topo_vect_pos}
    
    def get_time_stamp(self):
        """Get timestamp of the observation (placeholder)."""
        from datetime import datetime
        return datetime.now()
    
    def simulate(self, 
                 action: 'PypowsyblAction',
                 time_step: int = 0) -> Tuple['PypowsyblObservation', float, bool, Dict]:
        """
        Simulate the effect of an action without modifying the base network.
        
        This is the key method that replaces grid2op's obs.simulate().
        It uses pypowsybl variants to create a temporary copy of the network,
        apply the action, run load flow, and return results.
        
        Args:
            action: Action to simulate (PypowsyblAction or combined action)
            time_step: Simulation timestep (for compatibility, not used in static analysis)
            
        Returns:
            Tuple of (new_observation, reward, done, info)
            - new_observation: PypowsyblObservation after action
            - reward: Always 0.0 (not used in this application)
            - done: True if simulation failed
            - info: Dictionary with 'exception' key if errors occurred
        """
        nm = self._network_manager
        variant_id = f"simulate_{id(action)}_{time_step}"
        info = {"exception": []}
        
        try:
            # Create temporary variant
            nm.create_variant(variant_id)
            nm.set_working_variant(variant_id)
            
            # Apply action
            action.apply(nm)
            
            # Run load flow
            result = nm.run_load_flow()
            
            if result is None or result.status != lf.ComponentStatus.CONVERGED:
                info["exception"].append(
                    Exception(f"Load flow did not converge: {result.status if result else 'No result'}")
                )
                # Return observation with NaN values
                obs_simu = PypowsyblObservation(nm, self._action_space, self._thermal_limits)
                return obs_simu, 0.0, True, info
            
            # Create observation from simulated state
            obs_simu = PypowsyblObservation(nm, self._action_space, self._thermal_limits)
            
            return obs_simu, 0.0, False, info
            
        except Exception as e:
            info["exception"].append(e)
            # Try to create an observation anyway
            try:
                obs_simu = PypowsyblObservation(nm, self._action_space, self._thermal_limits)
            except:
                obs_simu = self  # Return self if we can't create new obs
            return obs_simu, 0.0, True, info
            
        finally:
            # Always clean up: return to base variant and remove temp
            nm.set_working_variant(nm.base_variant_id)
            nm.remove_variant(variant_id)
    
    def __add__(self, action: 'PypowsyblAction') -> 'PypowsyblObservation':
        """
        Support obs + action syntax (returns observation with action applied).
        
        Note: This creates a temporary copy. For simulation, use simulate().
        """
        # This is used in grid2op for impact_on_objects() type operations
        # For now, return self - actual implementation would need to
        # create a modified observation
        return self


class PypowsyblAction:
    """
    Represents an action that can be applied to a pypowsybl network.
    
    This is the base class for actions. Actions are created by ActionSpace
    and can be combined using the + operator.
    """
    
    def __init__(self):
        self._modifications = []
    
    def apply(self, network_manager: 'NetworkManager'):
        """Apply this action to the network."""
        for modification in self._modifications:
            modification(network_manager)
    
    def __add__(self, other: 'PypowsyblAction') -> 'PypowsyblAction':
        """Combine two actions."""
        combined = PypowsyblAction()
        combined._modifications = self._modifications + other._modifications
        return combined
    
    def __radd__(self, other):
        """Support other + action syntax."""
        if other == 0 or other is None:
            return self
        return other.__add__(self)
