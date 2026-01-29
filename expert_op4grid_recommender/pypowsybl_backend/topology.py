# expert_op4grid_recommender/pypowsybl_backend/topology.py
"""
Topology management for pypowsybl networks.

Handles the mapping between grid2op-style topology vectors (topo_vect)
and pypowsybl's bus/breaker or node/breaker models.

In pypowsybl, there are different topology representations:
- Bus/Breaker view: Each voltage level has numbered buses (like grid2op's bus 1, 2, etc.)
- Node/Breaker view: Fine-grained view with individual switches

This module provides translation between these representations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import pypowsybl as pp

if TYPE_CHECKING:
    from .network_manager import NetworkManager


class TopologyManager:
    """
    Manages topology information and bus assignments for a pypowsybl network.
    
    Provides grid2op-compatible topology vector (topo_vect) interface:
    - Each substation has N elements (lines_or, lines_ex, gens, loads)
    - Each element is assigned to a bus (1, 2, ...) or disconnected (-1, 0)
    
    The topology vector is ordered by substation, with elements within each
    substation in a specific order: loads, generators, lines_or, lines_ex.
    """
    
    def __init__(self, network_manager: 'NetworkManager'):
        """
        Initialize topology manager.
        
        Args:
            network_manager: The NetworkManager instance
        """
        self._nm = network_manager
        self._network = network_manager.network
        
        # Build element-to-substation mapping
        self._build_element_mapping()
        
        # Build topology vector structure
        self._build_topo_vect_structure()
    
    def _build_element_mapping(self):
        """Build mapping of elements to their substations."""
        net = self._network
        
        # Get all element DataFrames
        self._loads_df = net.get_loads()
        self._gens_df = net.get_generators()
        self._lines_df = net.get_lines()
        self._trafos_df = net.get_2_windings_transformers()
        
        # Map elements to voltage levels (substations)
        self._load_to_vl = {}
        for load_id in self._loads_df.index:
            vl = self._loads_df.loc[load_id, 'voltage_level_id']
            self._load_to_vl[load_id] = vl
        
        self._gen_to_vl = {}
        for gen_id in self._gens_df.index:
            vl = self._gens_df.loc[gen_id, 'voltage_level_id']
            self._gen_to_vl[gen_id] = vl
        
        # Lines connect two voltage levels
        self._line_or_to_vl = {}
        self._line_ex_to_vl = {}
        
        for line_id in self._lines_df.index:
            self._line_or_to_vl[line_id] = self._lines_df.loc[line_id, 'voltage_level1_id']
            self._line_ex_to_vl[line_id] = self._lines_df.loc[line_id, 'voltage_level2_id']
        
        for trafo_id in self._trafos_df.index:
            self._line_or_to_vl[trafo_id] = self._trafos_df.loc[trafo_id, 'voltage_level1_id']
            self._line_ex_to_vl[trafo_id] = self._trafos_df.loc[trafo_id, 'voltage_level2_id']
    
    def _build_topo_vect_structure(self):
        """
        Build the structure of the topology vector.
        
        The topology vector concatenates elements from all substations.
        Within each substation, elements are ordered:
        1. Loads
        2. Generators  
        3. Lines (origin terminal)
        4. Lines (extremity terminal)
        """
        self._sub_info = []  # Number of elements per substation
        self._topo_vect_elements = []  # List of (element_type, element_id, sub_idx)
        
        sub_names = self._nm.name_sub
        line_ids = self._nm.name_line
        
        for sub_idx, sub_name in enumerate(sub_names):
            elements_in_sub = []
            
            # Add loads in this substation
            for load_id, vl in self._load_to_vl.items():
                if vl == sub_name:
                    elements_in_sub.append(('load', load_id))
                    self._topo_vect_elements.append(('load', load_id, sub_idx))
            
            # Add generators in this substation
            for gen_id, vl in self._gen_to_vl.items():
                if vl == sub_name:
                    elements_in_sub.append(('gen', gen_id))
                    self._topo_vect_elements.append(('gen', gen_id, sub_idx))
            
            # Add line origins in this substation
            for line_id in line_ids:
                if self._line_or_to_vl.get(line_id) == sub_name:
                    elements_in_sub.append(('line_or', line_id))
                    self._topo_vect_elements.append(('line_or', line_id, sub_idx))
            
            # Add line extremities in this substation
            for line_id in line_ids:
                if self._line_ex_to_vl.get(line_id) == sub_name:
                    elements_in_sub.append(('line_ex', line_id))
                    self._topo_vect_elements.append(('line_ex', line_id, sub_idx))
            
            self._sub_info.append(len(elements_in_sub))
        
        self._sub_info = np.array(self._sub_info, dtype=int)
        self._topo_vect_to_sub = np.array([e[2] for e in self._topo_vect_elements], dtype=int)
    
    @property
    def sub_info(self) -> np.ndarray:
        """Number of elements per substation."""
        return self._sub_info.copy()
    
    @property
    def topo_vect_to_sub(self) -> np.ndarray:
        """Substation index for each position in topology vector."""
        return self._topo_vect_to_sub.copy()
    
    def get_topo_vect(self) -> np.ndarray:
        """
        Get current topology vector.
        
        Each element is assigned:
        - 1 or 2: Connected to bus 1 or 2
        - -1 or 0: Disconnected
        
        Returns:
            Numpy array with bus assignments
        """
        topo_vect = np.ones(len(self._topo_vect_elements), dtype=int)
        
        for i, (elem_type, elem_id, sub_idx) in enumerate(self._topo_vect_elements):
            if elem_type == 'load':
                connected = self._loads_df.loc[elem_id, 'connected']
                topo_vect[i] = 1 if connected else -1
            
            elif elem_type == 'gen':
                connected = self._gens_df.loc[elem_id, 'connected']
                topo_vect[i] = 1 if connected else -1
            
            elif elem_type == 'line_or':
                if elem_id in self._lines_df.index:
                    connected = self._lines_df.loc[elem_id, 'connected1']
                else:
                    connected = self._trafos_df.loc[elem_id, 'connected1']
                topo_vect[i] = 1 if connected else -1
            
            elif elem_type == 'line_ex':
                if elem_id in self._lines_df.index:
                    connected = self._lines_df.loc[elem_id, 'connected2']
                else:
                    connected = self._trafos_df.loc[elem_id, 'connected2']
                topo_vect[i] = 1 if connected else -1
        
        return topo_vect
    
    def get_sub_topology(self, sub_id: int) -> np.ndarray:
        """
        Get topology vector for a specific substation.
        
        Args:
            sub_id: Substation index
            
        Returns:
            Array of bus assignments for elements in this substation
        """
        if sub_id < 0 or sub_id >= len(self._sub_info):
            return np.array([])
        
        # Find start and end positions
        start = int(np.sum(self._sub_info[:sub_id]))
        length = int(self._sub_info[sub_id])
        
        full_topo = self.get_topo_vect()
        return full_topo[start:start + length]
    
    def get_element_at_topo_pos(self, pos: int) -> Dict[str, Any]:
        """
        Get element information for a topology vector position.
        
        Args:
            pos: Position in topology vector
            
        Returns:
            Dictionary with element type and ID
        """
        if pos < 0 or pos >= len(self._topo_vect_elements):
            return {'type': 'unknown', 'id': None}
        
        elem_type, elem_id, sub_idx = self._topo_vect_elements[pos]
        
        result = {
            'type': elem_type,
            'id': elem_id,
            'sub_id': sub_idx
        }
        
        # Add specific keys for grid2op compatibility
        if elem_type == 'load':
            load_idx = list(self._loads_df.index).index(elem_id)
            result['load_id'] = load_idx
        elif elem_type == 'gen':
            gen_idx = list(self._gens_df.index).index(elem_id)
            result['gen_id'] = gen_idx
        elif elem_type == 'line_or':
            line_idx = list(self._nm.name_line).index(elem_id)
            result['line_or_id'] = line_idx
            result['line_id'] = line_idx
        elif elem_type == 'line_ex':
            line_idx = list(self._nm.name_line).index(elem_id)
            result['line_ex_id'] = line_idx
            result['line_id'] = line_idx
        
        return result
    
    def get_objects_in_substation(self, sub_id: int) -> Dict[str, List[int]]:
        """
        Get all objects connected to a substation.
        
        Args:
            sub_id: Substation index
            
        Returns:
            Dictionary with lists of element indices by type
        """
        result = {
            'loads_id': [],
            'generators_id': [],
            'lines_or_id': [],
            'lines_ex_id': []
        }
        
        sub_name = self._nm.name_sub[sub_id]
        
        # Find loads
        for i, (load_id, vl) in enumerate(self._load_to_vl.items()):
            if vl == sub_name:
                result['loads_id'].append(i)
        
        # Find generators
        for i, (gen_id, vl) in enumerate(self._gen_to_vl.items()):
            if vl == sub_name:
                result['generators_id'].append(i)
        
        # Find lines
        line_ids = list(self._nm.name_line)
        for i, line_id in enumerate(line_ids):
            if self._line_or_to_vl.get(line_id) == sub_name:
                result['lines_or_id'].append(i)
            if self._line_ex_to_vl.get(line_id) == sub_name:
                result['lines_ex_id'].append(i)
        
        return result
    
    def apply_topology_change(self, 
                               sub_id: int, 
                               new_topo: List[int]) -> bool:
        """
        Apply a topology change to a substation.
        
        Args:
            sub_id: Substation index
            new_topo: New topology vector for this substation
            
        Returns:
            True if successful, False otherwise
        """
        if sub_id < 0 or sub_id >= len(self._sub_info):
            return False
        
        expected_len = int(self._sub_info[sub_id])
        if len(new_topo) != expected_len:
            return False
        
        # Find start position
        start = int(np.sum(self._sub_info[:sub_id]))
        
        # Apply changes
        for i, bus in enumerate(new_topo):
            pos = start + i
            elem_type, elem_id, _ = self._topo_vect_elements[pos]
            
            # Convert bus number to connected status
            # In this simplified version, bus >= 1 means connected
            connected = bus >= 1
            
            if elem_type == 'load':
                self._network.update_loads(id=elem_id, connected=connected)
            elif elem_type == 'gen':
                self._network.update_generators(id=elem_id, connected=connected)
            elif elem_type == 'line_or':
                if elem_id in self._lines_df.index:
                    self._network.update_lines(id=elem_id, connected1=connected)
                else:
                    self._network.update_2_windings_transformers(id=elem_id, connected1=connected)
            elif elem_type == 'line_ex':
                if elem_id in self._lines_df.index:
                    self._network.update_lines(id=elem_id, connected2=connected)
                else:
                    self._network.update_2_windings_transformers(id=elem_id, connected2=connected)
        
        # Refresh DataFrames
        self._loads_df = self._network.get_loads()
        self._gens_df = self._network.get_generators()
        self._lines_df = self._network.get_lines()
        self._trafos_df = self._network.get_2_windings_transformers()
        
        return True
