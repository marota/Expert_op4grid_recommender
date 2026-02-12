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
        """Refresh cached state from network. OPTIMIZED with vectorized operations."""
        nm = self._network_manager
        net = nm.network

        # Line information - get once and cache columns as dicts for fast lookup
        self._line_flows = nm.get_line_flows()
        self._line_flows_i1 = self._line_flows['i1'].to_dict()
        self._line_flows_i2 = self._line_flows['i2'].to_dict()
        self._line_flows_p1 = self._line_flows['p1'].to_dict()
        self._line_flows_p2 = self._line_flows['p2'].to_dict()
        self._line_flows_index_set = set(self._line_flows.index)

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

        # Load and generation - get once
        loads_df = net.get_loads()
        gens_df = net.get_generators()
        self._load_p = loads_df['p'].values
        self._load_q = loads_df['q'].values
        self._gen_p = gens_df['p'].values
        self._gen_q = gens_df['q'].values

        # Cache line currents and powers as arrays for fast access
        self._cache_line_arrays()
    
    def _cache_line_arrays(self):
        """Cache line current and power arrays for fast property access."""
        nm = self._network_manager
        n_line = nm.n_line
        line_names = nm.name_line

        # Pre-allocate arrays
        self._cached_i_or = np.zeros(n_line)
        self._cached_i_ex = np.zeros(n_line)
        self._cached_p_or = np.zeros(n_line)
        self._cached_p_ex = np.zeros(n_line)

        # Fill arrays using cached dicts (much faster than df.loc in loop)
        for i, line_id in enumerate(line_names):
            if line_id in self._line_flows_index_set:
                i1 = self._line_flows_i1.get(line_id, 0.0)
                i2 = self._line_flows_i2.get(line_id, 0.0)
                p1 = self._line_flows_p1.get(line_id, 0.0)
                p2 = self._line_flows_p2.get(line_id, 0.0)

                self._cached_i_or[i] = i1 if not np.isnan(i1) else 0.0
                self._cached_i_ex[i] = i2 if not np.isnan(i2) else 0.0
                self._cached_p_or[i] = p1 if not np.isnan(p1) else 0.0
                self._cached_p_ex[i] = p2 if not np.isnan(p2) else 0.0

    def _compute_rho(self):
        """Compute line loading ratios. OPTIMIZED with cached dicts.

        Uses i1 (side 1 current) for rho calculation since thermal limits
        are defined for side 1 in pypowsybl convention.
        """
        nm = self._network_manager
        n_line = nm.n_line
        line_names = nm.name_line

        rho = np.zeros(n_line)

        # Use cached dict for fast lookup instead of df.loc
        for i, line_id in enumerate(line_names):
            if line_id in self._line_flows_index_set:
                i1 = abs(self._line_flows_i1.get(line_id, 0.0))
                i1 = i1 if not np.isnan(i1) else 0.0

                thermal_limit = self._thermal_limits.get(line_id, 9999.0)
                if thermal_limit > 0:
                    rho[i] = i1 / thermal_limit

        self._rho = rho
    
    def _compute_line_angles(self):
        """Compute voltage angles at line terminals. OPTIMIZED with dict lookups."""
        nm = self._network_manager
        net = nm.network

        n_line = nm.n_line
        self._theta_or = np.zeros(n_line)
        self._theta_ex = np.zeros(n_line)

        # Get bus information - extract as dicts for fast lookup
        lines_df = net.get_lines()
        trafos_df = net.get_2_windings_transformers()
        buses_df = net.get_buses()

        # Pre-extract columns as dicts
        lines_bus1 = lines_df['bus1_id'].to_dict() if len(lines_df) > 0 else {}
        lines_bus2 = lines_df['bus2_id'].to_dict() if len(lines_df) > 0 else {}
        trafos_bus1 = trafos_df['bus1_id'].to_dict() if len(trafos_df) > 0 else {}
        trafos_bus2 = trafos_df['bus2_id'].to_dict() if len(trafos_df) > 0 else {}
        bus_angles = buses_df['v_angle'].to_dict() if len(buses_df) > 0 else {}

        # Use cached sets from NetworkManager
        lines_set = nm._lines_set
        trafos_set = nm._trafos_set

        for i, line_id in enumerate(nm.name_line):
            try:
                if line_id in lines_set:
                    bus1_id = lines_bus1.get(line_id)
                    bus2_id = lines_bus2.get(line_id)
                elif line_id in trafos_set:
                    bus1_id = trafos_bus1.get(line_id)
                    bus2_id = trafos_bus2.get(line_id)
                else:
                    continue

                if pd.notna(bus1_id) and bus1_id in bus_angles:
                    self._theta_or[i] = bus_angles[bus1_id] / 360 * 2 * np.pi
                if pd.notna(bus2_id) and bus2_id in bus_angles:
                    self._theta_ex[i] = bus_angles[bus2_id] / 360 * 2 * np.pi
            except (KeyError, TypeError):
                pass
    
    def _compute_line_buses(self):
        """
        Compute bus assignments for line terminals. OPTIMIZED with dict lookups.

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

        # Get line and transformer data - extract columns as dicts
        lines_df = net.get_lines()
        trafos_df = net.get_2_windings_transformers()

        lines_bus1 = lines_df['bus1_id'].to_dict() if len(lines_df) > 0 else {}
        lines_bus2 = lines_df['bus2_id'].to_dict() if len(lines_df) > 0 else {}
        lines_conn1 = lines_df['connected1'].to_dict() if 'connected1' in lines_df.columns else {}
        lines_conn2 = lines_df['connected2'].to_dict() if 'connected2' in lines_df.columns else {}

        trafos_bus1 = trafos_df['bus1_id'].to_dict() if len(trafos_df) > 0 else {}
        trafos_bus2 = trafos_df['bus2_id'].to_dict() if len(trafos_df) > 0 else {}
        trafos_conn1 = trafos_df['connected1'].to_dict() if 'connected1' in trafos_df.columns else {}
        trafos_conn2 = trafos_df['connected2'].to_dict() if 'connected2' in trafos_df.columns else {}

        # Build a mapping of voltage_level -> list of bus_ids
        buses_df = net.get_buses()
        bus_to_vl = buses_df['voltage_level_id'].to_dict() if len(buses_df) > 0 else {}
        buses_set = set(buses_df.index)

        vl_to_buses = {}
        for bus_id, vl_id in bus_to_vl.items():
            if vl_id not in vl_to_buses:
                vl_to_buses[vl_id] = []
            vl_to_buses[vl_id].append(bus_id)

        # Sort buses within each voltage level for consistent numbering
        # and build index lookup for O(1) access
        vl_bus_to_local = {}
        for vl_id, bus_list in vl_to_buses.items():
            sorted_buses = sorted(bus_list)
            vl_to_buses[vl_id] = sorted_buses
            for idx, bus_id in enumerate(sorted_buses):
                vl_bus_to_local[(vl_id, bus_id)] = idx + 1

        # Use cached sets from NetworkManager
        lines_set = nm._lines_set
        trafos_set = nm._trafos_set

        for i, line_id in enumerate(nm.name_line):
            try:
                # Get bus IDs for this line
                if line_id in lines_set:
                    bus1_id = lines_bus1.get(line_id)
                    bus2_id = lines_bus2.get(line_id)
                    connected1 = lines_conn1.get(line_id, True)
                    connected2 = lines_conn2.get(line_id, True)
                elif line_id in trafos_set:
                    bus1_id = trafos_bus1.get(line_id)
                    bus2_id = trafos_bus2.get(line_id)
                    connected1 = trafos_conn1.get(line_id, True)
                    connected2 = trafos_conn2.get(line_id, True)
                else:
                    continue

                # Determine local bus number for origin (bus1)
                if pd.isna(bus1_id) or (isinstance(connected1, bool) and not connected1):
                    self._line_or_bus[i] = -1  # Disconnected
                elif bus1_id in buses_set:
                    vl_id = bus_to_vl.get(bus1_id, '')
                    local_bus = vl_bus_to_local.get((vl_id, bus1_id), 1)
                    self._line_or_bus[i] = local_bus

                # Determine local bus number for extremity (bus2)
                if pd.isna(bus2_id) or (isinstance(connected2, bool) and not connected2):
                    self._line_ex_bus[i] = -1  # Disconnected
                elif bus2_id in buses_set:
                    vl_id = bus_to_vl.get(bus2_id, '')
                    local_bus = vl_bus_to_local.get((vl_id, bus2_id), 1)
                    self._line_ex_bus[i] = local_bus

            except (KeyError, TypeError):
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
        """Get current intensity at line origins (A). Returns cached array."""
        return self._cached_i_or.copy()

    def _get_line_i_ex(self) -> np.ndarray:
        """Get current intensity at line extremities (A). Returns cached array."""
        return self._cached_i_ex.copy()

    def _get_line_p_or(self) -> np.ndarray:
        """Get active power at line origins (MW). Returns cached array."""
        return self._cached_p_or.copy()

    def _get_line_p_ex(self) -> np.ndarray:
        """Get active power at line extremities (MW). Returns cached array."""
        return self._cached_p_ex.copy()
    
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
        Count the number of elements in a substation. OPTIMIZED using cached data.

        Elements include loads, generators, line origins, and line extremities.
        This must match the ordering used in sub_topology().

        Args:
            sub_id: Substation index

        Returns:
            Number of elements in this substation
        """
        nm = self._network_manager
        return (len(nm._loads_per_sub[sub_id]) + len(nm._gens_per_sub[sub_id]) +
                len(nm._lines_or_per_sub[sub_id]) + len(nm._lines_ex_per_sub[sub_id]))

    @property
    def sub_info(self) -> np.ndarray:
        """
        Number of elements per substation. OPTIMIZED using cached data.

        Counts all elements (gens, loads, lines_or, lines_ex) per substation,
        matching the ordering in sub_topology().
        """
        return self._network_manager._cached_sub_info.copy()

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
        Get topology vector for a specific substation. OPTIMIZED using cached data.

        Returns bus assignments for all elements connected to this substation.
        Elements are ordered as: loads, generators, lines_or, lines_ex

        Args:
            sub_id: Substation index

        Returns:
            Array of bus assignments for elements in this substation.
            Values: 1 or 2 for connected buses, -1 for disconnected.
        """
        nm = self._network_manager
        sub_name = nm._substation_ids[sub_id]

        # Use cached element lists
        load_indices = nm._loads_per_sub[sub_id]
        gen_indices = nm._gens_per_sub[sub_id]
        line_or_indices = nm._lines_or_per_sub[sub_id]
        line_ex_indices = nm._lines_ex_per_sub[sub_id]

        bus_assignments = []

        # Ensure we have cached bus info for loads and gens
        if not hasattr(self, '_load_bus_cache'):
            self._cache_element_buses()

        # Get bus assignments for loads
        for load_idx in load_indices:
            bus_assignments.append(self._load_bus_cache[load_idx])

        # Get bus assignments for generators
        for gen_idx in gen_indices:
            bus_assignments.append(self._gen_bus_cache[gen_idx])

        # Get bus assignments for line origins
        for line_idx in line_or_indices:
            bus_assignments.append(int(self._line_or_bus[line_idx]))

        # Get bus assignments for line extremities
        for line_idx in line_ex_indices:
            bus_assignments.append(int(self._line_ex_bus[line_idx]))

        return np.array(bus_assignments, dtype=int)

    def _cache_element_buses(self):
        """Cache bus assignments for loads and generators."""
        nm = self._network_manager
        net = nm.network

        # Get DataFrames once
        loads_df = net.get_loads()
        gens_df = net.get_generators()
        buses_df = net.get_buses()

        # Build voltage level -> sorted bus list mapping
        bus_to_vl = buses_df['voltage_level_id'].to_dict() if len(buses_df) > 0 else {}
        vl_to_buses = {}
        for bus_id, vl_id in bus_to_vl.items():
            if vl_id not in vl_to_buses:
                vl_to_buses[vl_id] = []
            vl_to_buses[vl_id].append(bus_id)

        # Build local bus number lookup
        vl_bus_to_local = {}
        for vl_id, bus_list in vl_to_buses.items():
            sorted_buses = sorted(bus_list)
            for idx, bus_id in enumerate(sorted_buses):
                vl_bus_to_local[(vl_id, bus_id)] = idx + 1

        # Cache load bus assignments
        self._load_bus_cache = np.ones(nm._n_load, dtype=int)
        if len(loads_df) > 0:
            load_bus_ids = loads_df['bus_id'].to_dict() if 'bus_id' in loads_df.columns else {}
            load_connected = loads_df['connected'].to_dict() if 'connected' in loads_df.columns else {}
            load_vl_ids = loads_df['voltage_level_id'].to_dict() if 'voltage_level_id' in loads_df.columns else {}

            for i, load_id in enumerate(nm._load_ids):
                bus_id = load_bus_ids.get(load_id)
                connected = load_connected.get(load_id, True)
                vl_id = load_vl_ids.get(load_id, '')

                if pd.isna(bus_id) or not connected:
                    self._load_bus_cache[i] = -1
                else:
                    self._load_bus_cache[i] = vl_bus_to_local.get((vl_id, bus_id), 1)

        # Cache generator bus assignments
        self._gen_bus_cache = np.ones(nm._n_gen, dtype=int)
        if len(gens_df) > 0:
            gen_bus_ids = gens_df['bus_id'].to_dict() if 'bus_id' in gens_df.columns else {}
            gen_connected = gens_df['connected'].to_dict() if 'connected' in gens_df.columns else {}
            gen_vl_ids = gens_df['voltage_level_id'].to_dict() if 'voltage_level_id' in gens_df.columns else {}

            for i, gen_id in enumerate(nm._gen_ids):
                bus_id = gen_bus_ids.get(gen_id)
                connected = gen_connected.get(gen_id, True)
                vl_id = gen_vl_ids.get(gen_id, '')

                if pd.isna(bus_id) or not connected:
                    self._gen_bus_cache[i] = -1
                else:
                    self._gen_bus_cache[i] = vl_bus_to_local.get((vl_id, bus_id), 1)

    def _get_local_bus_number(self, bus_id: str, vl_id: str) -> int:
        """
        Convert a pypowsybl bus_id to a local bus number (1 or 2) within a voltage level.
        OPTIMIZED - uses cached data when possible.

        Args:
            bus_id: The pypowsybl bus ID
            vl_id: The voltage level (substation) ID

        Returns:
            Local bus number (1 or 2)
        """
        # Try to use cached lookup if available
        if hasattr(self, '_vl_bus_to_local_cache'):
            return self._vl_bus_to_local_cache.get((vl_id, bus_id), 1)

        # Fallback to original implementation
        net = self._network_manager.network
        buses_df = net.get_buses()

        # Build cache for future use
        bus_to_vl = buses_df['voltage_level_id'].to_dict() if len(buses_df) > 0 else {}
        vl_to_buses = {}
        for bid, v in bus_to_vl.items():
            if v not in vl_to_buses:
                vl_to_buses[v] = []
            vl_to_buses[v].append(bid)

        self._vl_bus_to_local_cache = {}
        for v, bus_list in vl_to_buses.items():
            sorted_buses = sorted(bus_list)
            for idx, bid in enumerate(sorted_buses):
                self._vl_bus_to_local_cache[(v, bid)] = idx + 1

        return self._vl_bus_to_local_cache.get((vl_id, bus_id), 1)
    
    def get_obj_connect_to(self, substation_id: int) -> Dict[str, List[int]]:
        """
        Get objects connected to a substation. OPTIMIZED using cached data.

        Args:
            substation_id: Substation index

        Returns:
            Dictionary with keys 'loads_id', 'generators_id', 'lines_or_id',
            'lines_ex_id' containing lists of element indices.
        """
        nm = self._network_manager

        return {
            'loads_id': list(nm._loads_per_sub[substation_id]),
            'generators_id': list(nm._gens_per_sub[substation_id]),
            'lines_or_id': list(nm._lines_or_per_sub[substation_id]),
            'lines_ex_id': list(nm._lines_ex_per_sub[substation_id])
        }
    
    def topo_vect_element(self, topo_vect_pos: int) -> Dict[str, Any]:
        """
        Get element information for a topology vector position. OPTIMIZED using cached data.

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

        # Find which substation this position belongs to using cached sub_info
        sub_info = nm._cached_sub_info
        cumsum = 0
        sub_id = 0
        for i, count in enumerate(sub_info):
            if cumsum + count > topo_vect_pos:
                sub_id = i
                break
            cumsum += count

        # Local position within the substation
        local_pos = topo_vect_pos - cumsum

        # Use cached element lists for this substation
        load_ids_in_sub = nm._loads_per_sub[sub_id]
        gen_ids_in_sub = nm._gens_per_sub[sub_id]
        line_or_ids_in_sub = nm._lines_or_per_sub[sub_id]
        line_ex_ids_in_sub = nm._lines_ex_per_sub[sub_id]

        # Check loads
        load_count = len(load_ids_in_sub)
        if local_pos < load_count:
            return {'type': 'load', 'load_id': load_ids_in_sub[local_pos]}
        local_pos -= load_count

        # Check generators
        gen_count = len(gen_ids_in_sub)
        if local_pos < gen_count:
            return {'type': 'gen', 'gen_id': gen_ids_in_sub[local_pos]}
        local_pos -= gen_count

        # Check line origins
        line_or_count = len(line_or_ids_in_sub)
        if local_pos < line_or_count:
            return {'line_id': True, 'line_or_id': line_or_ids_in_sub[local_pos]}
        local_pos -= line_or_count

        # Check line extremities
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
                 time_step: int = 0,
                 keep_variant: bool = False) -> Tuple['PypowsyblObservation', float, bool, Dict]:
        """
        Simulate the effect of an action without modifying the base network.

        This is the key method that replaces grid2op's obs.simulate().
        It uses pypowsybl variants to create a temporary copy of the network,
        apply the action, run load flow, and return results.

        Args:
            action: Action to simulate (PypowsyblAction or combined action)
            time_step: Simulation timestep (for compatibility, not used in static analysis)
            keep_variant: If True, the network variant is kept alive after simulation
                         and stored as ``_variant_id`` on the returned observation.
                         The caller is responsible for cleanup via
                         ``nm.remove_variant(obs._variant_id)`` when done.

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
                if keep_variant:
                    obs_simu._variant_id = variant_id
                return obs_simu, 0.0, True, info

            # Create observation from simulated state
            obs_simu = PypowsyblObservation(nm, self._action_space, self._thermal_limits)
            if keep_variant:
                obs_simu._variant_id = variant_id

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
            # Always return to base variant
            nm.set_working_variant(nm.base_variant_id)
            # Only remove variant if not keeping it
            if not keep_variant:
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
        OPTIMIZED using cached data.
        """
        # Check if there's a full substation override
        if sub_id in self._substations_override:
            return np.array(self._substations_override[sub_id], dtype=int)

        # Get base topology from the underlying observation
        base_topo = self._base_obs.sub_topology(sub_id)

        nm = self._network_manager

        # Use cached element lists for this substation
        load_indices = nm._loads_per_sub[sub_id]
        gen_indices = nm._gens_per_sub[sub_id]
        line_or_indices = nm._lines_or_per_sub[sub_id]
        line_ex_indices = nm._lines_ex_per_sub[sub_id]

        # Track current position in the topology vector
        pos = 0

        # Process loads in this substation
        for load_idx in load_indices:
            load_id = nm._load_ids[load_idx]
            if load_id in self._loads_bus_override:
                base_topo[pos] = self._loads_bus_override[load_id]
            pos += 1

        # Process generators in this substation
        for gen_idx in gen_indices:
            gen_id = nm._gen_ids[gen_idx]
            if gen_id in self._gens_bus_override:
                base_topo[pos] = self._gens_bus_override[gen_id]
            pos += 1

        # Process line origins in this substation
        for line_idx in line_or_indices:
            line_id = nm._line_ids[line_idx]
            if line_id in self._lines_or_bus_override:
                base_topo[pos] = self._lines_or_bus_override[line_id]
            pos += 1

        # Process line extremities in this substation
        for line_idx in line_ex_indices:
            line_id = nm._line_ids[line_idx]
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
