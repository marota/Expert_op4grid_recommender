# expert_op4grid_recommender/pypowsybl_backend/__init__.py
"""
Pure pypowsybl backend for expert_op4grid_recommender.

This module provides a grid2op-free interface to power system simulation
using pypowsybl's network variants for efficient what-if analysis.

Main Components:
----------------
- NetworkManager: Low-level network access and variant management
- SimulationEnvironment: High-level grid2op-compatible environment
- PypowsyblObservation: Read-only network state with simulate() method
- ActionSpace: Create actions for topology changes and line switching
- OverflowSimulator: Compute flow changes after line disconnections
- TopologyManager: Handle bus assignments and topology vectors

Usage Example:
--------------
    from expert_op4grid_recommender.pypowsybl_backend import SimulationEnvironment
    
    # Create environment from network file
    env = SimulationEnvironment(
        network_path="/path/to/network.xiidm",
        thermal_limits_path="/path/to/thermal_limits.json"
    )
    
    # Get observation
    obs = env.get_obs()
    
    # Create and simulate an action
    action = env.action_space({"set_line_status": [("LINE_NAME", -1)]})
    obs_simu, reward, done, info = obs.simulate(action)
    
    # Check results
    print(f"Overloaded lines: {obs_simu.rho[obs_simu.rho >= 1.0]}")
"""

from .network_manager import NetworkManager
from .observation import PypowsyblObservation, PypowsyblAction
from .action_space import ActionSpace, LineStatusAction, BusAction
from .simulation_env import SimulationEnvironment, make_simulation_env
from .topology import TopologyManager
from .overflow_analysis import (
    OverflowSimulator,
    OverflowGraphBuilder,
    AlphaDeespAdapter,
    build_overflow_graph_pypowsybl
)

__all__ = [
    # Core classes
    'NetworkManager',
    'SimulationEnvironment',
    'PypowsyblObservation',
    'PypowsyblAction',
    'ActionSpace',
    'TopologyManager',
    
    # Action types
    'LineStatusAction',
    'BusAction',
    
    # Overflow analysis
    'OverflowSimulator',
    'OverflowGraphBuilder',
    'AlphaDeespAdapter',
    
    # Factory functions
    'make_simulation_env',
    'build_overflow_graph_pypowsybl',
]

# Version
__version__ = '0.1.0.post1'
