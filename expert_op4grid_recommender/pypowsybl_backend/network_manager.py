# expert_op4grid_recommender/pypowsybl_backend/network_manager.py
"""
Network Manager for pypowsybl-based simulation.

Handles loading networks, managing variants, and running load flows.
"""

import pypowsybl as pp
import pypowsybl.loadflow as lf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union


class NetworkManager:
    """
    Manages a pypowsybl network with variant support for efficient what-if analysis.
    
    This class wraps pypowsybl.network.Network and provides:
    - Network loading from various formats (XIIDM, CGMES, etc.)
    - Variant management for hypothetical scenarios
    - Load flow execution with configurable parameters
    - Element naming and topology information extraction
    
    Attributes:
        network (pp.network.Network): The underlying pypowsybl network.
        lf_parameters (pp.loadflow.Parameters): Load flow parameters.
        base_variant_id (str): The ID of the base/reference variant.
    """
    
    BASE_VARIANT = "base"
    
    def __init__(self, 
                 network_path: Optional[Union[str, Path]] = None,
                 network: Optional[pp.network.Network] = None,
                 lf_parameters: Optional[lf.Parameters] = None):
        """
        Initialize the NetworkManager.
        
        Args:
            network_path: Path to network file (XIIDM, CGMES, etc.)
            network: Pre-loaded pypowsybl network (alternative to path)
            lf_parameters: Load flow parameters. If None, uses RTE defaults.
        """
        if network is not None:
            self.network = network
        elif network_path is not None:
            self.network = pp.network.load(str(network_path))
        else:
            raise ValueError("Either network_path or network must be provided")
        
        self.lf_parameters = lf_parameters or self._create_default_lf_parameters()
        self.base_variant_id = self.BASE_VARIANT
        
        # DC mode flag (can be set externally)
        self._default_dc = False
        
        # Ensure we have a working variant
        self._setup_base_variant()
        
        # Cache element information
        self._cache_element_info()
    
    def _create_default_lf_parameters(self) -> lf.Parameters:
        """Create default load flow parameters (RTE-style configuration)."""
        return lf.Parameters(
            read_slack_bus=False,
            write_slack_bus=False,
            voltage_init_mode=lf.VoltageInitMode.DC_VALUES,
            transformer_voltage_control_on=True,
            use_reactive_limits=True,
            shunt_compensator_voltage_control_on=True,
            phase_shifter_regulation_on=True,
            distributed_slack=True,
            dc_use_transformer_ratio=False,
            twt_split_shunt_admittance=True,
            provider_parameters={
                "useActiveLimits": "true",
                "svcVoltageMonitoring": "false",
                "voltageRemoteControl": "false",
                "writeReferenceTerminals": "false",
                "slackBusSelectionMode": "MOST_MESHED"
            }
        )
    
    def _setup_base_variant(self):
        """Setup the base variant for the network."""
        existing_variants = self.network.get_variant_ids()
        if self.BASE_VARIANT not in existing_variants:
            # Clone from initial variant
            self.network.clone_variant(existing_variants[0], self.BASE_VARIANT)
        self.network.set_working_variant(self.BASE_VARIANT)
    
    def _cache_element_info(self):
        """Cache element information for fast access."""
        # Lines (AC lines + transformers)
        lines_df = self.network.get_lines()
        trafos_df = self.network.get_2_windings_transformers()

        self._line_ids = list(lines_df.index) + list(trafos_df.index)
        self._n_line = len(self._line_ids)

        # Store line/trafo sets for O(1) membership tests
        self._lines_set = set(lines_df.index)
        self._trafos_set = set(trafos_df.index)

        # Substations (voltage levels in pypowsybl terminology)
        vl_df = self.network.get_voltage_levels()
        self._substation_ids = list(vl_df.index)
        self._n_sub = len(self._substation_ids)

        # OPTIMIZATION: Pre-compute name -> index mappings for O(1) lookups
        self._sub_name_to_idx = {name: idx for idx, name in enumerate(self._substation_ids)}
        self._line_name_to_idx = {name: idx for idx, name in enumerate(self._line_ids)}

        # Map lines to substations
        self._line_or_sub = {}
        self._line_ex_sub = {}

        for line_id in lines_df.index:
            self._line_or_sub[line_id] = lines_df.loc[line_id, 'voltage_level1_id']
            self._line_ex_sub[line_id] = lines_df.loc[line_id, 'voltage_level2_id']

        for trafo_id in trafos_df.index:
            self._line_or_sub[trafo_id] = trafos_df.loc[trafo_id, 'voltage_level1_id']
            self._line_ex_sub[trafo_id] = trafos_df.loc[trafo_id, 'voltage_level2_id']

        # OPTIMIZATION: Pre-compute line_or_subid and line_ex_subid arrays
        self._cached_line_or_subid = np.array([
            self._sub_name_to_idx.get(self._line_or_sub[lid], -1)
            for lid in self._line_ids
        ])
        self._cached_line_ex_subid = np.array([
            self._sub_name_to_idx.get(self._line_ex_sub[lid], -1)
            for lid in self._line_ids
        ])

        # Generators
        gen_df = self.network.get_generators()
        self._gen_ids = list(gen_df.index)
        self._n_gen = len(self._gen_ids)
        self._gen_name_to_idx = {name: idx for idx, name in enumerate(self._gen_ids)}

        # Cache generator -> substation mapping
        self._gen_to_sub = {}
        if len(gen_df) > 0 and 'voltage_level_id' in gen_df.columns:
            self._gen_to_sub = gen_df['voltage_level_id'].to_dict()

        # Loads
        load_df = self.network.get_loads()
        self._load_ids = list(load_df.index)
        self._n_load = len(self._load_ids)
        self._load_name_to_idx = {name: idx for idx, name in enumerate(self._load_ids)}

        # Cache load -> substation mapping
        self._load_to_sub = {}
        if len(load_df) > 0 and 'voltage_level_id' in load_df.columns:
            self._load_to_sub = load_df['voltage_level_id'].to_dict()

        # OPTIMIZATION: Pre-compute elements per substation
        self._cache_elements_per_substation()

    def _cache_elements_per_substation(self):
        """Cache which elements belong to each substation."""
        n_sub = self._n_sub

        # Initialize per-substation element lists
        self._loads_per_sub = [[] for _ in range(n_sub)]
        self._gens_per_sub = [[] for _ in range(n_sub)]
        self._lines_or_per_sub = [[] for _ in range(n_sub)]
        self._lines_ex_per_sub = [[] for _ in range(n_sub)]

        # Map loads to substations
        for i, load_id in enumerate(self._load_ids):
            sub_name = self._load_to_sub.get(load_id, '')
            sub_idx = self._sub_name_to_idx.get(sub_name, -1)
            if sub_idx >= 0:
                self._loads_per_sub[sub_idx].append(i)

        # Map generators to substations
        for i, gen_id in enumerate(self._gen_ids):
            sub_name = self._gen_to_sub.get(gen_id, '')
            sub_idx = self._sub_name_to_idx.get(sub_name, -1)
            if sub_idx >= 0:
                self._gens_per_sub[sub_idx].append(i)

        # Map line origins to substations
        for i, line_id in enumerate(self._line_ids):
            sub_idx = self._cached_line_or_subid[i]
            if sub_idx >= 0:
                self._lines_or_per_sub[sub_idx].append(i)

        # Map line extremities to substations
        for i, line_id in enumerate(self._line_ids):
            sub_idx = self._cached_line_ex_subid[i]
            if sub_idx >= 0:
                self._lines_ex_per_sub[sub_idx].append(i)

        # Pre-compute sub_info (element count per substation)
        self._cached_sub_info = np.array([
            len(self._loads_per_sub[i]) + len(self._gens_per_sub[i]) +
            len(self._lines_or_per_sub[i]) + len(self._lines_ex_per_sub[i])
            for i in range(n_sub)
        ], dtype=int)
    
    @property
    def name_line(self) -> np.ndarray:
        """Array of line names (compatible with grid2op interface)."""
        return np.array(self._line_ids)
    
    @property
    def name_sub(self) -> np.ndarray:
        """Array of substation names (voltage level IDs)."""
        return np.array(self._substation_ids)
    
    @property
    def name_gen(self) -> np.ndarray:
        """Array of generator names."""
        return np.array(self._gen_ids)
    
    @property
    def name_load(self) -> np.ndarray:
        """Array of load names."""
        return np.array(self._load_ids)
    
    @property
    def n_line(self) -> int:
        """Number of lines (AC lines + transformers)."""
        return self._n_line
    
    @property
    def n_sub(self) -> int:
        """Number of substations."""
        return self._n_sub
    
    def get_line_or_subid(self) -> np.ndarray:
        """Get origin substation index for each line (cached)."""
        return self._cached_line_or_subid.copy()

    def get_line_ex_subid(self) -> np.ndarray:
        """Get extremity substation index for each line (cached)."""
        return self._cached_line_ex_subid.copy()

    def get_sub_idx(self, sub_name: str) -> int:
        """Get substation index by name (O(1) lookup)."""
        return self._sub_name_to_idx.get(sub_name, -1)

    def get_line_idx(self, line_name: str) -> int:
        """Get line index by name (O(1) lookup)."""
        return self._line_name_to_idx.get(line_name, -1)
    
    def create_variant(self, variant_id: str, from_variant: Optional[str] = None) -> str:
        """
        Create a new network variant for what-if analysis.
        
        Args:
            variant_id: ID for the new variant
            from_variant: Source variant to clone from (default: base)
            
        Returns:
            The variant_id
        """
        source = from_variant or self.base_variant_id
        
        # Remove existing variant if it exists
        if variant_id in self.network.get_variant_ids():
            self.network.remove_variant(variant_id)
        
        self.network.clone_variant(source, variant_id)
        return variant_id
    
    def set_working_variant(self, variant_id: str):
        """Set the active working variant."""
        self.network.set_working_variant(variant_id)
    
    def remove_variant(self, variant_id: str):
        """Remove a variant (cannot remove base variant)."""
        if variant_id != self.base_variant_id:
            if variant_id in self.network.get_variant_ids():
                self.network.remove_variant(variant_id)
    
    def run_load_flow(self, dc: Optional[bool] = None) -> lf.ComponentResult:
        """
        Run load flow on the current working variant.
        
        Args:
            dc: If True, run DC load flow instead of AC. 
                If None, uses the _default_dc attribute.
            
        Returns:
            Load flow result object
        """
        # Use default if not specified
        use_dc = dc if dc is not None else self._default_dc
        
        try:
            if use_dc:
                # DC load flow - run_dc uses its own default parameters
                results = lf.run_dc(self.network)
            else:
                results = lf.run_ac(self.network, parameters=self.lf_parameters)
            
            return results[0] if results else None
        except Exception as e:
            print(f"Load flow failed: {e}")
            return None
    
    def disconnect_line(self, line_id: str):
        """Disconnect a line (open both terminals)."""
        if line_id in self.network.get_lines().index:
            self.network.update_lines(id=line_id, connected1=False, connected2=False)
        elif line_id in self.network.get_2_windings_transformers().index:
            self.network.update_2_windings_transformers(
                id=line_id, connected1=False, connected2=False
            )
    
    def reconnect_line(self, line_id: str, bus_or: int = 1, bus_ex: int = 1):
        """
        Reconnect a line to specified buses.
        
        Args:
            line_id: Line identifier
            bus_or: Bus number at origin (1 or 2)
            bus_ex: Bus number at extremity (1 or 2)
        """
        if line_id in self.network.get_lines().index:
            self.network.update_lines(id=line_id, connected1=True, connected2=True)
        elif line_id in self.network.get_2_windings_transformers().index:
            self.network.update_2_windings_transformers(
                id=line_id, connected1=True, connected2=True
            )
    
    def get_line_flows(self) -> pd.DataFrame:
        """
        Get current line flows (P, Q, I) for all lines.
        
        Returns:
            DataFrame with columns: p1, q1, i1, p2, q2, i2, connected
        """
        lines_df = self.network.get_lines()[['p1', 'q1', 'i1', 'p2', 'q2', 'i2', 'connected1', 'connected2']]
        trafos_df = self.network.get_2_windings_transformers()[['p1', 'q1', 'i1', 'p2', 'q2', 'i2', 'connected1', 'connected2']]
        
        all_branches = pd.concat([lines_df, trafos_df])
        all_branches['connected'] = all_branches['connected1'] & all_branches['connected2']
        
        return all_branches
    
    def get_thermal_limits(self) -> Dict[str, float]:
        """
        Get thermal limits for all lines.
        
        Returns:
            Dictionary mapping line_id to thermal limit (A)
        """
        limits_df = self.network.get_operational_limits()
        limits_df = limits_df.reset_index()
        
        # Get permanent limits
        perm_limits = limits_df[limits_df['name'] == 'permanent_limit']
        
        thermal_limits = {}
        for _, row in perm_limits.iterrows():
            element_id = row['element_id']
            thermal_limits[element_id] = row['value']
        
        # Fill missing with high value
        default_limit = 9999.0
        for line_id in self._line_ids:
            if line_id not in thermal_limits:
                thermal_limits[line_id] = default_limit
        
        return thermal_limits
    
    def get_bus_voltages(self) -> pd.DataFrame:
        """Get bus voltage magnitudes and angles."""
        return self.network.get_buses()[['v_mag', 'v_angle']]
    
    def reset_to_base(self):
        """Reset working variant to base state."""
        self.set_working_variant(self.base_variant_id)
