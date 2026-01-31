# expert_op4grid_recommender/pypowsybl_backend/action_space.py
"""
Action Space for pypowsybl-based simulation.

Provides a grid2op-compatible interface for creating actions that modify
the network topology (line switching, bus reconfigurations).
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
from .observation import PypowsyblAction

if TYPE_CHECKING:
    from .network_manager import NetworkManager


class LineStatusAction(PypowsyblAction):
    """Action to change line connection status."""
    
    def __init__(self, line_changes: List[tuple]):
        """
        Args:
            line_changes: List of (line_name, status) tuples
                         status: -1 to disconnect, 1 to reconnect
        """
        super().__init__()
        self.line_changes = line_changes
        
        def apply_line_changes(nm: 'NetworkManager'):
            for line_name, status in self.line_changes:
                if status == -1:
                    nm.disconnect_line(line_name)
                elif status == 1:
                    nm.reconnect_line(line_name)
        
        self._modifications.append(apply_line_changes)


class SwitchAction(PypowsyblAction):
    """Action to change switch states (open/close) for topology changes."""
    
    def __init__(self, switch_states: Dict[str, bool]):
        """
        Args:
            switch_states: Dict mapping switch_id -> open (True=open, False=closed)
        """
        super().__init__()
        self.switch_states = switch_states
        
        def apply_switch_changes(nm: 'NetworkManager'):
            net = nm.network
            for switch_id, is_open in self.switch_states.items():
                try:
                    net.update_switches(id=switch_id, open=is_open)
                except Exception as e:
                    print(f"Warning: Could not update switch {switch_id}: {e}")
        
        self._modifications.append(apply_switch_changes)


class BusAction(PypowsyblAction):
    """Action to change bus assignments (topology changes)."""
    
    def __init__(self, 
                 lines_or_bus: Optional[Dict[str, int]] = None,
                 lines_ex_bus: Optional[Dict[str, int]] = None,
                 loads_bus: Optional[Dict[str, int]] = None,
                 gens_bus: Optional[Dict[str, int]] = None,
                 substations: Optional[Dict[int, List[int]]] = None):
        """
        Args:
            lines_or_bus: Dict mapping line_name -> bus_number at origin
            lines_ex_bus: Dict mapping line_name -> bus_number at extremity  
            loads_bus: Dict mapping load_name -> bus_number
            gens_bus: Dict mapping gen_name -> bus_number
            substations: Dict mapping sub_id -> full topology vector
        """
        super().__init__()
        self.lines_or_bus = lines_or_bus or {}
        self.lines_ex_bus = lines_ex_bus or {}
        self.loads_bus = loads_bus or {}
        self.gens_bus = gens_bus or {}
        self.substations = substations or {}
        
        def apply_bus_changes(nm: 'NetworkManager'):
            net = nm.network
            
            # Handle line origin bus changes
            for line_name, bus in self.lines_or_bus.items():
                if bus == -1:
                    # Disconnect at origin
                    if line_name in net.get_lines().index:
                        net.update_lines(id=line_name, connected1=False)
                    elif line_name in net.get_2_windings_transformers().index:
                        net.update_2_windings_transformers(id=line_name, connected1=False)
                elif bus >= 1:
                    # Connect to specified bus
                    # Note: pypowsybl bus handling depends on the topology model
                    # This is a simplified version
                    if line_name in net.get_lines().index:
                        net.update_lines(id=line_name, connected1=True)
                    elif line_name in net.get_2_windings_transformers().index:
                        net.update_2_windings_transformers(id=line_name, connected1=True)
            
            # Handle line extremity bus changes
            for line_name, bus in self.lines_ex_bus.items():
                if bus == -1:
                    if line_name in net.get_lines().index:
                        net.update_lines(id=line_name, connected2=False)
                    elif line_name in net.get_2_windings_transformers().index:
                        net.update_2_windings_transformers(id=line_name, connected2=False)
                elif bus >= 1:
                    if line_name in net.get_lines().index:
                        net.update_lines(id=line_name, connected2=True)
                    elif line_name in net.get_2_windings_transformers().index:
                        net.update_2_windings_transformers(id=line_name, connected2=True)
            
            # Handle load bus changes
            for load_name, bus in self.loads_bus.items():
                if bus == -1:
                    net.update_loads(id=load_name, connected=False)
                elif bus >= 1:
                    net.update_loads(id=load_name, connected=True)
            
            # Handle generator bus changes
            for gen_name, bus in self.gens_bus.items():
                if bus == -1:
                    net.update_generators(id=gen_name, connected=False)
                elif bus >= 1:
                    net.update_generators(id=gen_name, connected=True)
        
        self._modifications.append(apply_bus_changes)


class ActionSpace:
    """
    Grid2op-compatible action space for pypowsybl networks.
    
    Provides the familiar interface for creating actions:
    - action_space({"set_line_status": [...]})
    - action_space({"set_bus": {...}})
    
    Actions can be combined using the + operator.
    """
    
    def __init__(self, network_manager: 'NetworkManager'):
        """
        Initialize action space.
        
        Args:
            network_manager: The NetworkManager instance
        """
        self._network_manager = network_manager
        
        # Cache line and element information for validation
        self._line_names = set(network_manager.name_line)
        self._sub_names = set(network_manager.name_sub)
        self._gen_names = set(network_manager.name_gen)
        self._load_names = set(network_manager.name_load)
    
    def __call__(self, action_dict: Dict[str, Any]) -> PypowsyblAction:
        """
        Create an action from a dictionary specification.
        
        Supports grid2op-style action dictionaries:
        - {"set_line_status": [(line_name, status), ...]}
        - {"set_bus": {"lines_or_id": {name: bus}, "lines_ex_id": {name: bus}, ...}}
        - {"set_bus": {"substations_id": [(sub_id, [topo_vector]), ...]}}
        
        And pypowsybl-specific switch actions:
        - {"switches": {switch_id: open_state, ...}}  # Full switch IDs
        
        Also handles action dicts from conversion_actions_repas:
        - {"content": {"set_bus": {...}}, "switches": {...}}
        
        Args:
            action_dict: Dictionary specifying the action
            
        Returns:
            PypowsyblAction that can be applied or simulated
        """
        combined_action = PypowsyblAction()
        
        # Handle nested content structure from conversion_actions_repas
        # set_bus is inside content, switches is at top level
        working_dict = action_dict
        if "content" in action_dict and isinstance(action_dict["content"], dict):
            # Merge content with top-level for set_bus access
            working_dict = {**action_dict, **action_dict["content"]}
        
        # Handle set_line_status
        if "set_line_status" in working_dict:
            line_changes = working_dict["set_line_status"]
            if line_changes:
                line_action = LineStatusAction(line_changes)
                combined_action = combined_action + line_action
        
        # Handle switches (full switch IDs)
        # Check top-level first (new format from conversion_actions_repas)
        # Then check inside working_dict (legacy format)
        switch_states = None
        if "switches" in action_dict:
            switch_states = action_dict["switches"]
        elif "switches" in working_dict:
            switch_states = working_dict["switches"]
        
        if switch_states:
            switch_action = SwitchAction(switch_states)
            combined_action = combined_action + switch_action
        
        # Handle set_bus
        if "set_bus" in working_dict:
            bus_config = working_dict["set_bus"]
            
            lines_or_bus = {}
            lines_ex_bus = {}
            loads_bus = {}
            gens_bus = {}
            substations = {}
            
            # Lines origin bus
            if "lines_or_id" in bus_config:
                for name, bus in bus_config["lines_or_id"].items():
                    lines_or_bus[name] = bus
            
            # Lines extremity bus
            if "lines_ex_id" in bus_config:
                for name, bus in bus_config["lines_ex_id"].items():
                    lines_ex_bus[name] = bus
            
            # Loads bus
            if "loads_id" in bus_config:
                for name, bus in bus_config["loads_id"].items():
                    loads_bus[name] = bus
            
            # Generators bus
            if "generators_id" in bus_config:
                for name, bus in bus_config["generators_id"].items():
                    gens_bus[name] = bus
            
            # Full substation topology
            if "substations_id" in bus_config:
                for sub_id, topo_vect in bus_config["substations_id"]:
                    substations[sub_id] = topo_vect
            
            if any([lines_or_bus, lines_ex_bus, loads_bus, gens_bus, substations]):
                bus_action = BusAction(
                    lines_or_bus=lines_or_bus,
                    lines_ex_bus=lines_ex_bus,
                    loads_bus=loads_bus,
                    gens_bus=gens_bus,
                    substations=substations
                )
                combined_action = combined_action + bus_action
        
        return combined_action
    
    def get_do_nothing_action(self) -> PypowsyblAction:
        """Return an action that does nothing."""
        return PypowsyblAction()
    
    # ========== Additional methods for compatibility ==========
    
    @property
    def n_line(self) -> int:
        """Number of lines."""
        return self._network_manager.n_line
    
    @property
    def n_sub(self) -> int:
        """Number of substations."""
        return self._network_manager.n_sub
