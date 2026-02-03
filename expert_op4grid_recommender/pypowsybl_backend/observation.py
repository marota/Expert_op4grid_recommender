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
                    self._theta_or[i] = buses_df.loc[bus1_id, 'v_angle']/360*2*np.pi#to have values in radians
                if pd.notna(bus2_id) and bus2_id in buses_df.index:
                    self._theta_ex[i] = buses_df.loc[bus2_id, 'v_angle']/360*2*np.pi
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
    
    def _count_sub_elements(self, sub_id: int) -> int:
        """
        Count the number of elements in a substation.

        Elements include loads, generators, line origins, and line extremities.
        This must match the ordering used in sub_topology().

        Args:
            sub_id: Substation index

        Returns:
            Number of elements in this substation
        """
        sub_name = self.name_sub[sub_id]
        nm = self._network_manager
        net = nm.network
        count = 0

        # Count loads in this substation
        loads_df = net.get_loads()
        for load_id, row in loads_df.iterrows():
            if row.get('voltage_level_id') == sub_name:
                count += 1

        # Count generators in this substation
        gens_df = net.get_generators()
        for gen_id, row in gens_df.iterrows():
            if row.get('voltage_level_id') == sub_name:
                count += 1

        # Count line origins in this substation
        for line_id in nm.name_line:
            if nm._line_or_sub.get(line_id) == sub_name:
                count += 1

        # Count line extremities in this substation
        for line_id in nm.name_line:
            if nm._line_ex_sub.get(line_id) == sub_name:
                count += 1

        return count

    @property
    def sub_info(self) -> np.ndarray:
        """
        Number of elements per substation.

        Counts all elements (gens, loads, lines_or, lines_ex) per substation,
        matching the ordering in sub_topology().
        """
        counts = np.array([
            self._count_sub_elements(sub_id)
            for sub_id in range(self.n_sub)
        ], dtype=int)
        return counts

    @property
    def topo_vect(self) -> np.ndarray:
        """
        Topology vector (bus assignment for all elements).

        Concatenates the topology vectors for all substations in order.
        Each element's value indicates which bus (1 or 2) it is connected to,
        or -1 if disconnected.
        """
        topo_arrays = [self.sub_topology(sub_id) for sub_id in range(self.n_sub)]
        return np.concatenate(topo_arrays) if topo_arrays else np.array([], dtype=int)
    
    def sub_topology(self, sub_id: int) -> np.ndarray:
        """
        Get topology vector for a specific substation.

        Returns bus assignments for all elements connected to this substation.
        Elements are ordered as: loads, generators, lines_or, lines_ex

        Args:
            sub_id: Substation index

        Returns:
            Array of bus assignments for elements in this substation.
            Values: 1 or 2 for connected buses, -1 for disconnected.
        """
        sub_name = self.name_sub[sub_id]
        bus_assignments = []

        nm = self._network_manager
        net = nm.network

        # Find loads in this substation
        loads_df = net.get_loads()
        for i, (load_id, row) in enumerate(loads_df.iterrows()):
            if row.get('voltage_level_id') == sub_name:
                # Get bus assignment from bus_id
                bus_id = row.get('bus_id')
                if pd.isna(bus_id) or not row.get('connected', True):
                    bus_assignments.append(-1)
                else:
                    bus_num = self._get_local_bus_number(bus_id, sub_name)
                    bus_assignments.append(bus_num)

        # Find generators in this substation
        gens_df = net.get_generators()
        for i, (gen_id, row) in enumerate(gens_df.iterrows()):
            if row.get('voltage_level_id') == sub_name:
                bus_id = row.get('bus_id')
                if pd.isna(bus_id) or not row.get('connected', True):
                    bus_assignments.append(-1)
                else:
                    bus_num = self._get_local_bus_number(bus_id, sub_name)
                    bus_assignments.append(bus_num)

        # Find line origins in this substation
        for i, line_id in enumerate(nm.name_line):
            if nm._line_or_sub.get(line_id) == sub_name:
                bus_assignments.append(int(self._line_or_bus[i]))

        # Find line extremities in this substation
        for i, line_id in enumerate(nm.name_line):
            if nm._line_ex_sub.get(line_id) == sub_name:
                bus_assignments.append(int(self._line_ex_bus[i]))

        return np.array(bus_assignments, dtype=int)

    def _get_local_bus_number(self, bus_id: str, vl_id: str) -> int:
        """
        Convert a pypowsybl bus_id to a local bus number (1 or 2) within a voltage level.

        Args:
            bus_id: The pypowsybl bus ID
            vl_id: The voltage level (substation) ID

        Returns:
            Local bus number (1 or 2)
        """
        net = self._network_manager.network
        buses_df = net.get_buses()

        # Get all buses in this voltage level
        vl_buses = []
        for bid, row in buses_df.iterrows():
            if row.get('voltage_level_id') == vl_id:
                vl_buses.append(bid)

        vl_buses = sorted(vl_buses)  # Consistent ordering

        if bus_id in vl_buses:
            return vl_buses.index(bus_id) + 1  # 1-indexed
        return 1  # Default to bus 1
    
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

        The topology vector ordering matches sub_topology():
        loads, generators, lines_or, lines_ex for each substation.

        Args:
            topo_vect_pos: Position in the topology vector

        Returns:
            Dictionary with element type and ID. For lines:
            - {'line_id': True, 'line_or_id': line_idx} for line origins
            - {'line_id': True, 'line_ex_id': line_idx} for line extremities
        """
        nm = self._network_manager
        net = nm.network

        # Find which substation this position belongs to
        sub_info = self.sub_info
        cumsum = 0
        sub_id = 0
        for i, count in enumerate(sub_info):
            if cumsum + count > topo_vect_pos:
                sub_id = i
                break
            cumsum += count

        # Local position within the substation
        local_pos = topo_vect_pos - cumsum
        sub_name = self.name_sub[sub_id]

        # Count elements in this substation to find what's at local_pos
        # Order: loads, generators, lines_or, lines_ex
        loads_df = net.get_loads()
        gens_df = net.get_generators()

        # Count loads in this substation
        load_count = 0
        load_ids_in_sub = []
        for i, (load_id, row) in enumerate(loads_df.iterrows()):
            if row.get('voltage_level_id') == sub_name:
                load_ids_in_sub.append(i)
                load_count += 1

        if local_pos < load_count:
            return {'type': 'load', 'load_id': load_ids_in_sub[local_pos]}

        local_pos -= load_count

        # Count generators in this substation
        gen_count = 0
        gen_ids_in_sub = []
        for i, (gen_id, row) in enumerate(gens_df.iterrows()):
            if row.get('voltage_level_id') == sub_name:
                gen_ids_in_sub.append(i)
                gen_count += 1

        if local_pos < gen_count:
            return {'type': 'gen', 'gen_id': gen_ids_in_sub[local_pos]}

        local_pos -= gen_count

        # Count line origins in this substation
        line_or_count = 0
        line_or_ids_in_sub = []
        for i, line_id in enumerate(nm.name_line):
            if nm._line_or_sub.get(line_id) == sub_name:
                line_or_ids_in_sub.append(i)
                line_or_count += 1

        if local_pos < line_or_count:
            return {'line_id': True, 'line_or_id': line_or_ids_in_sub[local_pos]}

        local_pos -= line_or_count

        # Count line extremities in this substation
        line_ex_ids_in_sub = []
        for i, line_id in enumerate(nm.name_line):
            if nm._line_ex_sub.get(line_id) == sub_name:
                line_ex_ids_in_sub.append(i)

        if local_pos < len(line_ex_ids_in_sub):
            return {'line_id': True, 'line_ex_id': line_ex_ids_in_sub[local_pos]}

        # Fallback
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

        Creates a wrapper observation that reflects the action's topology changes
        without actually modifying the network.
        """
        return ObservationWithTopologyOverride(self, action)


class ObservationWithTopologyOverride(PypowsyblObservation):
    """
    Observation wrapper that applies topology overrides from an action.

    This class doesn't actually modify the network - it just presents a view
    of what the topology would look like after the action is applied.
    Used by grid2op-style code that does `obs + action` to see the impact.
    """

    def __init__(self, base_obs: 'PypowsyblObservation', action: 'PypowsyblAction'):
        """
        Initialize with base observation and action.

        Args:
            base_obs: The original observation
            action: The action whose topology changes should be reflected
        """
        # Don't call super().__init__ - we're wrapping base_obs instead
        self._base_obs = base_obs
        self._action = action
        self._network_manager = base_obs._network_manager
        self._action_space = base_obs._action_space
        self._thermal_limits = base_obs._thermal_limits

        # Extract topology overrides from the action
        self._lines_or_bus_override = {}
        self._lines_ex_bus_override = {}
        self._loads_bus_override = {}
        self._gens_bus_override = {}
        self._substations_override = {}

        self._extract_topology_overrides(action)

    def _extract_topology_overrides(self, action: 'PypowsyblAction'):
        """Extract topology changes from the action.

        All PypowsyblAction objects now store topology info directly as attributes,
        and the __add__ method merges these when actions are combined.
        """
        if hasattr(action, 'lines_or_bus'):
            self._lines_or_bus_override = dict(action.lines_or_bus) if action.lines_or_bus else {}
        if hasattr(action, 'lines_ex_bus'):
            self._lines_ex_bus_override = dict(action.lines_ex_bus) if action.lines_ex_bus else {}
        if hasattr(action, 'loads_bus'):
            self._loads_bus_override = dict(action.loads_bus) if action.loads_bus else {}
        if hasattr(action, 'gens_bus'):
            self._gens_bus_override = dict(action.gens_bus) if action.gens_bus else {}
        if hasattr(action, 'substations'):
            self._substations_override = dict(action.substations) if action.substations else {}

    @property
    def topo_vect(self) -> np.ndarray:
        """
        Topology vector with action overrides applied.
        """
        topo_arrays = [self.sub_topology(sub_id) for sub_id in range(self.n_sub)]
        return np.concatenate(topo_arrays) if topo_arrays else np.array([], dtype=int)

    def sub_topology(self, sub_id: int) -> np.ndarray:
        """
        Get topology vector for a substation with action overrides applied.
        """
        # Check if there's a full substation override
        if sub_id in self._substations_override:
            return np.array(self._substations_override[sub_id], dtype=int)

        # Get base topology from the underlying observation
        base_topo = self._base_obs.sub_topology(sub_id)

        # Apply element-specific overrides
        sub_name = self.name_sub[sub_id]
        nm = self._network_manager
        net = nm.network

        # Track current position in the topology vector
        pos = 0

        # Process loads in this substation
        loads_df = net.get_loads()
        for load_id, row in loads_df.iterrows():
            if row.get('voltage_level_id') == sub_name:
                if load_id in self._loads_bus_override:
                    base_topo[pos] = self._loads_bus_override[load_id]
                pos += 1

        # Process generators in this substation
        gens_df = net.get_generators()
        for gen_id, row in gens_df.iterrows():
            if row.get('voltage_level_id') == sub_name:
                if gen_id in self._gens_bus_override:
                    base_topo[pos] = self._gens_bus_override[gen_id]
                pos += 1

        # Process line origins in this substation
        for line_id in nm.name_line:
            if nm._line_or_sub.get(line_id) == sub_name:
                if line_id in self._lines_or_bus_override:
                    base_topo[pos] = self._lines_or_bus_override[line_id]
                pos += 1

        # Process line extremities in this substation
        for line_id in nm.name_line:
            if nm._line_ex_sub.get(line_id) == sub_name:
                if line_id in self._lines_ex_bus_override:
                    base_topo[pos] = self._lines_ex_bus_override[line_id]
                pos += 1

        return base_topo

    # Delegate all other properties to base observation
    @property
    def rho(self) -> np.ndarray:
        return self._base_obs.rho

    @property
    def line_status(self) -> np.ndarray:
        return self._base_obs.line_status

    @property
    def name_line(self) -> List[str]:
        return self._base_obs.name_line

    @property
    def name_sub(self) -> List[str]:
        return self._base_obs.name_sub

    @property
    def name_gen(self) -> List[str]:
        return self._base_obs.name_gen

    @property
    def name_load(self) -> List[str]:
        return self._base_obs.name_load

    @property
    def n_line(self) -> int:
        return self._base_obs.n_line

    @property
    def n_sub(self) -> int:
        return self._base_obs.n_sub

    @property
    def sub_info(self) -> np.ndarray:
        return self._base_obs.sub_info

    @property
    def a_or(self) -> np.ndarray:
        return self._base_obs.a_or

    @property
    def a_ex(self) -> np.ndarray:
        return self._base_obs.a_ex

    @property
    def p_or(self) -> np.ndarray:
        return self._base_obs.p_or

    @property
    def p_ex(self) -> np.ndarray:
        return self._base_obs.p_ex

    @property
    def q_or(self) -> np.ndarray:
        return self._base_obs.q_or

    @property
    def q_ex(self) -> np.ndarray:
        return self._base_obs.q_ex

    @property
    def v_or(self) -> np.ndarray:
        return self._base_obs.v_or

    @property
    def v_ex(self) -> np.ndarray:
        return self._base_obs.v_ex

    @property
    def theta_or(self) -> np.ndarray:
        return self._base_obs.theta_or

    @property
    def theta_ex(self) -> np.ndarray:
        return self._base_obs.theta_ex

    @property
    def line_or_to_subid(self) -> np.ndarray:
        return self._base_obs.line_or_to_subid

    @property
    def line_ex_to_subid(self) -> np.ndarray:
        return self._base_obs.line_ex_to_subid


class PypowsyblAction:
    """
    Represents an action that can be applied to a pypowsybl network.

    This is the base class for actions. Actions are created by ActionSpace
    and can be combined using the + operator.

    Topology info (lines_or_bus, lines_ex_bus, etc.) is stored as attributes
    so that ObservationWithTopologyOverride can extract it for computing
    the impact of the action on topology.
    """

    def __init__(self):
        self._modifications = []
        # Topology info for observation + action operations
        self.lines_or_bus = {}
        self.lines_ex_bus = {}
        self.loads_bus = {}
        self.gens_bus = {}
        self.substations = {}

    def apply(self, network_manager: 'NetworkManager'):
        """Apply this action to the network."""
        for modification in self._modifications:
            modification(network_manager)

    def __add__(self, other: 'PypowsyblAction') -> 'PypowsyblAction':
        """Combine two actions, merging their topology info."""
        combined = PypowsyblAction()
        combined._modifications = self._modifications + other._modifications
        # Merge topology info from both actions
        combined.lines_or_bus = {**self.lines_or_bus, **other.lines_or_bus}
        combined.lines_ex_bus = {**self.lines_ex_bus, **other.lines_ex_bus}
        combined.loads_bus = {**self.loads_bus, **other.loads_bus}
        combined.gens_bus = {**self.gens_bus, **other.gens_bus}
        combined.substations = {**self.substations, **other.substations}
        return combined

    def __radd__(self, other):
        """Support other + action syntax."""
        if other == 0 or other is None:
            return self
        return other.__add__(self)
