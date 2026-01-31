# expert_op4grid_recommender/pypowsybl_backend/overflow_analysis.py
"""
Overflow graph analysis using pure pypowsybl.

This module provides the overflow graph building functionality that was
previously dependent on alphaDeesp's Grid2opSimulation. It uses pypowsybl's
sensitivity analysis for PTDF-based flow change calculations.

The core idea:
- When a line disconnects, power redistributes according to Power Transfer 
  Distribution Factors (PTDFs)
- We can compute these factors using DC sensitivity analysis
- The overflow graph shows how flows change after disconnecting overloaded lines

Supports both AC and DC load flow for flexibility.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from math import fabs
import pypowsybl as pp
import pypowsybl.loadflow as lf

if TYPE_CHECKING:
    from .network_manager import NetworkManager
    from .observation import PypowsyblObservation
    from .simulation_env import SimulationEnvironment


class OverflowSimulator:
    """
    Computes flow changes when lines are disconnected.
    
    Uses pypowsybl's load flow (AC or DC) to compute flow redistribution 
    after line outages.
    
    This replaces alphaDeesp's Grid2opSimulation for overflow graph building.
    
    Attributes:
        use_dc: If True, uses DC load flow. If False, uses AC load flow.
    """
    
    def __init__(self, 
                 network_manager: 'NetworkManager',
                 obs: 'PypowsyblObservation',
                 use_dc: bool = False,
                 param_options: Optional[Dict] = None):
        """
        Initialize the overflow simulator.
        
        Args:
            network_manager: NetworkManager instance
            obs: Current observation (after initial contingency)
            use_dc: If True, use DC load flow. If False, use AC (default).
            param_options: Configuration parameters (thresholds, etc.)
        """
        self._nm = network_manager
        self._obs = obs
        self._net = network_manager.network
        self._use_dc = use_dc
        self._param_options = param_options or {}
        
        # Get current line flows FROM THE OBSERVATION (not network)
        # This ensures we have the correct base flows even if the observation
        # was captured at a different network state
        self._base_flows = self._get_flows_from_obs(obs)
        self._base_currents = self._get_currents_from_obs(obs)
    
    def _get_flows_from_obs(self, obs: 'PypowsyblObservation') -> Dict[str, float]:
        """Get active power flows from observation's p_or values."""
        flows = {}
        for i, line_id in enumerate(self._nm.name_line):
            flows[line_id] = obs.p_or[i]
        return flows
    
    def _get_currents_from_obs(self, obs: 'PypowsyblObservation') -> Dict[str, float]:
        """Get current magnitudes from observation's a_or values."""
        currents = {}
        for i, line_id in enumerate(self._nm.name_line):
            # Use max of a_or and a_ex like we do elsewhere
            i_or = obs.a_or[i] if hasattr(obs, 'a_or') else 0.0
            i_ex = obs.a_ex[i] if hasattr(obs, 'a_ex') else 0.0
            currents[line_id] = max(abs(i_or), abs(i_ex))
        return currents
    
    def _run_load_flow(self) -> lf.ComponentResult:
        """Run load flow with configured mode (AC or DC)."""
        return self._nm.run_load_flow(dc=self._use_dc)
    
    def _get_line_flows(self) -> Dict[str, float]:
        """Get active power flows for all lines (MW at terminal 1)."""
        lines_df = self._net.get_lines()[['p1']]
        trafos_df = self._net.get_2_windings_transformers()[['p1']]
        
        flows = {}
        for line_id in lines_df.index:
            p1 = lines_df.loc[line_id, 'p1']
            flows[line_id] = p1 if not np.isnan(p1) else 0.0
        for trafo_id in trafos_df.index:
            p1 = trafos_df.loc[trafo_id, 'p1']
            flows[trafo_id] = p1 if not np.isnan(p1) else 0.0
        
        return flows
    
    def _get_line_currents(self) -> Dict[str, float]:
        """Get current magnitudes for all lines (A, max of both terminals)."""
        lines_df = self._net.get_lines()[['i1', 'i2']]
        trafos_df = self._net.get_2_windings_transformers()[['i1', 'i2']]
        
        currents = {}
        for line_id in lines_df.index:
            i1 = lines_df.loc[line_id, 'i1']
            i2 = lines_df.loc[line_id, 'i2']
            i1 = i1 if not np.isnan(i1) else 0.0
            i2 = i2 if not np.isnan(i2) else 0.0
            currents[line_id] = max(abs(i1), abs(i2))
        for trafo_id in trafos_df.index:
            i1 = trafos_df.loc[trafo_id, 'i1']
            i2 = trafos_df.loc[trafo_id, 'i2']
            i1 = i1 if not np.isnan(i1) else 0.0
            i2 = i2 if not np.isnan(i2) else 0.0
            currents[trafo_id] = max(abs(i1), abs(i2))
        
        return currents
    
    def _get_line_reactive_flows(self) -> Dict[str, float]:
        """Get reactive power flows for all lines (MVAr at terminal 1)."""
        lines_df = self._net.get_lines()[['q1']]
        trafos_df = self._net.get_2_windings_transformers()[['q1']]
        
        flows = {}
        for line_id in lines_df.index:
            q1 = lines_df.loc[line_id, 'q1']
            flows[line_id] = q1 if not np.isnan(q1) else 0.0
        for trafo_id in trafos_df.index:
            q1 = trafos_df.loc[trafo_id, 'q1']
            flows[trafo_id] = q1 if not np.isnan(q1) else 0.0
        
        return flows
    
    def compute_ptdf_for_line(self, line_id: str) -> Dict[str, float]:
        """
        Compute PTDF row for disconnecting a specific line.
        
        This shows how much each other line's flow changes when
        we disconnect the given line (per unit of the disconnected flow).
        
        Args:
            line_id: ID of line to disconnect
            
        Returns:
            Dictionary mapping other_line_id -> PTDF factor
        """
        original_flow = self._base_flows.get(line_id, 0.0)
        if abs(original_flow) < 1e-6:
            return {}
        
        # Create a variant for this calculation
        variant_id = f"ptdf_{line_id}"
        self._nm.create_variant(variant_id)
        self._nm.set_working_variant(variant_id)
        
        try:
            # Disconnect the line
            self._nm.disconnect_line(line_id)
            
            # Run load flow (AC or DC based on configuration)
            result = self._run_load_flow()
            
            if result is None or result.status != lf.ComponentStatus.CONVERGED:
                return {}
            
            # Get new flows
            new_flows = self._get_line_flows()
            
            # Compute PTDFs
            ptdf = {}
            for other_line, new_flow in new_flows.items():
                if other_line != line_id:
                    old_flow = self._base_flows.get(other_line, 0.0)
                    delta = new_flow - old_flow
                    ptdf[other_line] = delta / abs(original_flow) if abs(original_flow) > 1e-6 else 0.0
            
            return ptdf
            
        finally:
            self._nm.set_working_variant(self._nm.base_variant_id)
            self._nm.remove_variant(variant_id)
    
    def compute_flow_changes_after_disconnection(self, 
                                                   lines_to_disconnect: List[str]
                                                   ) -> pd.DataFrame:
        """
        Compute flow changes when specified lines are disconnected.
        
        This method replicates alphaDeesp's Simulation.create_df() logic exactly.
        
        Args:
            lines_to_disconnect: List of line IDs to disconnect
            
        Returns:
            DataFrame with columns matching alphaDeesp's Grid2opSimulation format
        """
        # Create variant
        variant_id = "flow_change_analysis"
        self._nm.create_variant(variant_id)
        self._nm.set_working_variant(variant_id)
        
        try:
            # IMPORTANT: First, re-apply any lines that are already disconnected in the observation
            # The observation was created from a simulated state (with contingency applied),
            # but the network manager's base variant doesn't have those disconnections.
            for i, line_id in enumerate(self._nm.name_line):
                if not self._obs.line_status[i]:  # Line is disconnected in observation
                    self._nm.disconnect_line(line_id)
            
            # Run load flow to verify we match the observation state
            result = self._run_load_flow()
            
            if result is None or result.status != lf.ComponentStatus.CONVERGED:
                print(f"Warning: Load flow did not converge when re-applying contingency state")
            
            # Now disconnect the additional lines (overloaded lines)
            for line_id in lines_to_disconnect:
                self._nm.disconnect_line(line_id)
            
            # Run load flow again after disconnecting overloaded lines
            result = self._run_load_flow()
            
            converged = (result is not None and result.status == lf.ComponentStatus.CONVERGED)
            
            # Get new flows (signed values, like alphaDeesp's cut_lines_and_recomputes_flows)
            if converged:
                new_flows_dict = {}
                lines_df = self._net.get_lines()
                trafos_df = self._net.get_2_windings_transformers()
                
                for line_id in self._nm.name_line:
                    if line_id in lines_df.index:
                        p1 = lines_df.loc[line_id, 'p1']
                        new_flows_dict[line_id] = p1 if not np.isnan(p1) else 0.0
                    elif line_id in trafos_df.index:
                        p1 = trafos_df.loc[line_id, 'p1']
                        new_flows_dict[line_id] = p1 if not np.isnan(p1) else 0.0
                    else:
                        new_flows_dict[line_id] = 0.0
            else:
                new_flows_dict = {k: 0.0 for k in self._base_flows}
            
            # ===== STEP 1: Build initial DataFrame (like alphaDeesp's d["edges"]) =====
            data = []
            for line_id in self._nm.name_line:
                flow_before = self._base_flows.get(line_id, 0.0)
                if np.isnan(flow_before):
                    flow_before = 0.0
                
                or_sub = self._nm._line_or_sub.get(line_id, '')
                ex_sub = self._nm._line_ex_sub.get(line_id, '')
                
                try:
                    idx_or = list(self._nm.name_sub).index(or_sub)
                except ValueError:
                    idx_or = -1
                try:
                    idx_ex = list(self._nm.name_sub).index(ex_sub)
                except ValueError:
                    idx_ex = -1
                
                data.append({
                    'idx_or': idx_or,
                    'idx_ex': idx_ex,
                    'init_flows': flow_before,  # Signed value initially
                    'line_name': line_id,
                })
            
            df = pd.DataFrame(data)
            
            # ===== STEP 2: branch_direction_swaps - swap when init_flows < 0 =====
            swapped = []
            for i, row in df.iterrows():
                a = row["init_flows"]
                if a < 0 and a != 0.:
                    # Swap origin and extremity
                    idx_or = row["idx_or"]
                    df.at[i, "idx_or"] = row["idx_ex"]
                    df.at[i, "idx_ex"] = idx_or
                    df.at[i, "init_flows"] = fabs(row["init_flows"])
                    swapped.append(True)
                else:
                    swapped.append(False)
            df["swapped"] = swapped
            
            # ===== STEP 3: Get new_flows and apply swapped correction =====
            n_flows = []
            for line_id, is_swapped in zip(self._nm.name_line, df["swapped"]):
                f = new_flows_dict.get(line_id, 0.0)
                if np.isnan(f):
                    f = 0.0
                if is_swapped:
                    n_flows.append(f * -1)
                else:
                    n_flows.append(f)
            df["new_flows"] = n_flows
            
            # ===== STEP 4: Compute new_flows_swapped =====
            # True if new_flows < 0 AND |new_flows| > |init_flows|
            new_flows_swapped = []
            for i, row in df.iterrows():
                if row["new_flows"] < 0 and fabs(row["new_flows"]) > fabs(row["init_flows"]):
                    new_flows_swapped.append(True)
                else:
                    new_flows_swapped.append(False)
            df["new_flows_swapped"] = new_flows_swapped
            
            # ===== STEP 5: Compute delta_flows (exactly like alphaDeesp) =====
            delta_flo = []
            for i, row in df.iterrows():
                if row["new_flows_swapped"]:
                    # Flow reversed and is stronger - positive delta, swap indices
                    delta_flo.append(fabs(row["new_flows"]) + fabs(row["init_flows"]))
                    # Swap origin and extremity
                    idx_or = row["idx_or"]
                    df.at[i, "idx_or"] = row["idx_ex"]
                    df.at[i, "idx_ex"] = idx_or
                    df.at[i, "init_flows"] = fabs(row["init_flows"])
                elif (np.sign(row["new_flows"]) != np.sign(row["init_flows"])) and \
                     (row["new_flows"] != 0) and (row["init_flows"] != 0):
                    # Signs differ but new flow is weaker - negative delta (flow relieved)
                    delta_flo.append(-(fabs(row["new_flows"]) + fabs(row["init_flows"])))
                else:
                    # Same direction - simple difference
                    delta_flo.append(fabs(row["new_flows"]) - fabs(row["init_flows"]))
            df["delta_flows"] = delta_flo
            
            # ===== STEP 6: Compute gray_edges based on significance (like alphaDeesp) =====
            # gray_edges = True when |delta_flows| < ltc_report * ThresholdReportOfLine
            # ltc_report is the delta_flows of the first line to cut
            if lines_to_disconnect:
                # Get the index of the first line to cut
                first_ltc_idx = None
                for i, line_id in enumerate(self._nm.name_line):
                    if line_id == lines_to_disconnect[0]:
                        first_ltc_idx = i
                        break
                
                if first_ltc_idx is not None:
                    ltc_report = fabs(df["delta_flows"].iloc[first_ltc_idx])
                else:
                    ltc_report = df["delta_flows"].abs().max()
            else:
                ltc_report = df["delta_flows"].abs().max()
            
            # Get threshold from param_options
            threshold = self._param_options.get("ThresholdReportOfLine", 0.05)
            max_overload = ltc_report * float(threshold)
            
            gray_edges = []
            for delta in df["delta_flows"]:
                if fabs(delta) < max_overload:
                    gray_edges.append(True)
                else:
                    gray_edges.append(False)
            df["gray_edges"] = gray_edges
            
            # NOTE: new_flows stays as signed value (not converted to absolute)
            # This matches alphaDeesp behavior where new_flows can be negative
            
            return df
            
        finally:
            self._nm.set_working_variant(self._nm.base_variant_id)
            self._nm.remove_variant(variant_id)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get base flow information as DataFrame.
        
        Provides interface compatible with alphaDeesp's Grid2opSimulation.
        
        Returns:
            DataFrame with flow information for all lines in alphaDeesp format
        """
        data = []
        for line_id in self._nm.name_line:
            flow = self._base_flows.get(line_id, 0.0)
            
            or_sub = self._nm._line_or_sub.get(line_id, '')
            ex_sub = self._nm._line_ex_sub.get(line_id, '')
            
            try:
                idx_or = list(self._nm.name_sub).index(or_sub)
            except ValueError:
                idx_or = -1
            try:
                idx_ex = list(self._nm.name_sub).index(ex_sub)
            except ValueError:
                idx_ex = -1
            
            # Determine if flow is "swapped" (negative = flow from ex to or)
            swapped = flow < 0
            
            data.append({
                'idx_or': idx_or,
                'idx_ex': idx_ex,
                'init_flows': abs(flow),
                'swapped': swapped,
                'new_flows': abs(flow),  # Same as init when no disconnection
                'new_flows_swapped': False,
                'delta_flows': 0.0,  # No disconnection yet
                'gray_edges': False,
                'line_name': line_id,
            })
        
        return pd.DataFrame(data)


class OverflowGraphBuilder:
    """
    Builds overflow graphs compatible with alphaDeesp's OverFlowGraph format.
    
    The overflow graph is a directed graph where:
    - Nodes are substations
    - Edges are power lines with flow values
    - Edge weights represent flow changes after disconnecting overloaded lines
    """
    
    def __init__(self,
                 overflow_simulator: OverflowSimulator,
                 overloaded_line_ids: List[int],
                 param_options: Optional[Dict] = None):
        """
        Initialize the graph builder.
        
        Args:
            overflow_simulator: OverflowSimulator instance
            overloaded_line_ids: List of indices of overloaded lines
            param_options: Configuration parameters (thresholds, etc.)
        """
        self._sim = overflow_simulator
        self._overloaded_ids = overloaded_line_ids
        self._nm = overflow_simulator._nm
        
        self._params = param_options or {
            "ThresholdReportOfLine": 0.05,
            "ThersholdMinPowerOfLoop": 0.1,
            "ratioToKeepLoop": 0.25,
            "ratioToReconsiderFlowDirection": 0.75,
            "maxUnusedLines": 3,
        }
    
    def build_graph(self) -> Tuple[nx.MultiDiGraph, pd.DataFrame]:
        """
        Build the overflow graph.
        
        Returns:
            Tuple of (graph, flow_changes_df)
        """
        # Get overloaded line names
        line_names = list(self._nm.name_line)
        overloaded_line_names = [line_names[i] for i in self._overloaded_ids]
        
        # Compute flow changes
        df = self._sim.compute_flow_changes_after_disconnection(overloaded_line_names)
        
        # Build graph
        G = nx.MultiDiGraph()
        
        # Add nodes (substations)
        for i, sub_name in enumerate(self._nm.name_sub):
            G.add_node(i, name=sub_name)
        
        # Add edges (lines with significant flow changes)
        threshold = self._params.get("ThresholdReportOfLine", 0.05)
        max_delta = df['delta_flows'].abs().max() if not df.empty else 1.0
        
        for _, row in df.iterrows():
            if row['gray_edges']:  # Skip disconnected lines (marked as gray)
                continue  # Skip disconnected lines
            
            delta = row['delta_flows']
            if np.isnan(delta):
                continue
            
            # Filter by significance
            if abs(delta) < threshold * max_delta:
                continue
            
            idx_or = int(row['idx_or'])
            idx_ex = int(row['idx_ex'])
            
            if idx_or < 0 or idx_ex < 0:
                continue
            
            # Determine edge direction based on flow direction
            if delta >= 0:
                u, v = idx_or, idx_ex
            else:
                u, v = idx_ex, idx_or
                delta = -delta
            
            # Add edge with attributes
            G.add_edge(u, v, 
                       name=row['line_name'],
                       capacity=delta,
                       label=f"{delta:.0f}",
                       flow_before=row['init_flows'],
                       flow_after=row['new_flows'])
        
        return G, df
    
    def get_topology(self) -> Dict[str, Any]:
        """
        Get topology information for alphaDeesp compatibility.
        
        Returns:
            Dictionary with topology data
        """
        return {
            'n_sub': self._nm.n_sub,
            'n_line': self._nm.n_line,
            'line_or_to_subid': self._nm.get_line_or_subid(),
            'line_ex_to_subid': self._nm.get_line_ex_subid(),
            'name_sub': self._nm.name_sub,
            'name_line': self._nm.name_line,
        }


def build_overflow_graph_pypowsybl(env: 'SimulationEnvironment',
                                    obs: 'PypowsyblObservation', 
                                    overloaded_line_ids: List[int],
                                    non_connected_reconnectable_lines: List[str],
                                    lines_non_reconnectable: List[str],
                                    timestep: int,
                                    do_consolidate_graph: bool = True,
                                    use_dc: bool = False,
                                    param_options: Optional[Dict] = None):
    """
    Build overflow graph using pure pypowsybl.
    
    This is a drop-in replacement for the grid2op-based build_overflow_graph function.
    Uses alphaDeesp's OverFlowGraph and Structured_Overload_Distribution_Graph for 
    proper graph construction with color attributes.
    
    Args:
        env: SimulationEnvironment instance
        obs: Current observation (after contingency)
        overloaded_line_ids: List of indices of overloaded lines
        non_connected_reconnectable_lines: Lines that could be reconnected
        lines_non_reconnectable: Lines that cannot be reconnected
        timestep: Current timestep (for compatibility)
        do_consolidate_graph: Whether to consolidate the graph
        use_dc: If True, use DC load flow. If False, use AC load flow.
        param_options: Configuration parameters
        
    Returns:
        Tuple containing:
        - df_of_g: DataFrame with flow changes
        - overflow_sim: AlphaDeespAdapter instance (compatible with Grid2opSimulation)
        - g_overflow: OverFlowGraph instance
        - hubs: List of hub nodes
        - g_distribution_graph: Structured_Overload_Distribution_Graph
        - node_name_mapping: Dict mapping node indices to names
    """
    from expert_op4grid_recommender.config import PARAM_OPTIONS_EXPERT_OP
    from alphaDeesp.core.graphsAndPaths import OverFlowGraph, Structured_Overload_Distribution_Graph
    
    params = param_options or PARAM_OPTIONS_EXPERT_OP
    
    # Create alphaDeesp-compatible adapter (replaces Grid2opSimulation)
    overflow_sim = AlphaDeespAdapter(
        obs=obs,
        action_space=env.action_space,
        observation_space=None,  # Not needed for our use case
        param_options=params,
        debug=False,
        ltc=overloaded_line_ids,
        plot=False,
        simu_step=timestep,
        use_dc=use_dc
    )
    
    # Get flow changes DataFrame
    df_of_g = overflow_sim.get_dataframe()
    df_of_g["line_name"] = obs.name_line
    
    # Apply flow swap correction - this negates delta_flows for lines where
    # new_flows_swapped=True, making them negative (blue) instead of positive (red)
    df_of_g = _inhibit_swapped_flows(df_of_g)
    
    # Build topology info for alphaDeesp
    topo = overflow_sim.topo
    
    # Use alphaDeesp's OverFlowGraph to construct graph with proper color attributes
    g_overflow = OverFlowGraph(topo, overloaded_line_ids, df_of_g, float_precision="%.0f")
    
    # Node name mapping
    node_name_mapping = {i: name for i, name in enumerate(obs.name_sub)}
    
    # Rename nodes from indices to substation names
    g_overflow.g = nx.relabel_nodes(g_overflow.g, node_name_mapping, copy=True)
    
    # Create distribution graph for path analysis
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)
    c_path_init = g_distribution_graph.constrained_path.full_n_constrained_path()
    
    # Consolidate graph if requested
    if len(g_distribution_graph.g_only_red_components.nodes) != 0 and do_consolidate_graph:
        g_overflow.consolidate_graph(
            g_distribution_graph,
            non_connected_lines_to_ignore=non_connected_reconnectable_lines + lines_non_reconnectable,
            no_desambiguation=True
        )
        g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)
    
    # Get hubs before adding reconnectable lines
    real_hubs = g_distribution_graph.get_hubs()
    
    # Add relevant null flow lines for reconnection analysis
    if len(non_connected_reconnectable_lines) >= 0:
        g_overflow.add_relevant_null_flow_lines_all_paths(
            g_distribution_graph,
            non_connected_lines=non_connected_reconnectable_lines,
            non_reconnectable_lines=lines_non_reconnectable
        )
        # Recreate distribution graph after adding lines
        # Note: possible_hubs parameter may not be supported in all alphaDeesp versions
        try:
            g_distribution_graph = Structured_Overload_Distribution_Graph(
                g_overflow.g, 
                possible_hubs=c_path_init
            )
        except TypeError:
            # Fallback for older alphaDeesp versions without possible_hubs support
            g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)
    
    return df_of_g, overflow_sim, g_overflow, real_hubs, g_distribution_graph, node_name_mapping


def _inhibit_swapped_flows(df_of_g: pd.DataFrame) -> pd.DataFrame:
    """
    Correct flow direction swapping in the DataFrame.
    
    When flows are swapped, the delta_flows sign and idx_or/idx_ex need to be corrected.
    
    NOTE: This function is kept for compatibility but may not be needed when using
    the new compute_flow_changes_after_disconnection which handles swapping internally.
    
    Args:
        df_of_g: DataFrame with flow changes
        
    Returns:
        Corrected DataFrame
    """
    if 'new_flows_swapped' not in df_of_g.columns:
        return df_of_g
    
    swapped_mask = df_of_g.new_flows_swapped
    if not swapped_mask.any():
        return df_of_g
    
    # Negate delta_flows for swapped lines
    df_of_g.loc[swapped_mask, "delta_flows"] = -df_of_g.loc[swapped_mask, "delta_flows"]
    
    # Swap idx_or and idx_ex
    idx_or = df_of_g.loc[swapped_mask, "idx_or"].copy()
    df_of_g.loc[swapped_mask, "idx_or"] = df_of_g.loc[swapped_mask, "idx_ex"]
    df_of_g.loc[swapped_mask, "idx_ex"] = idx_or
    
    return df_of_g


def _find_hubs_simple(graph: nx.MultiDiGraph) -> List[int]:
    """
    Simple hub detection based on node degree.
    
    Hubs are nodes with high connectivity (degree).
    
    Args:
        graph: NetworkX graph
        
    Returns:
        List of hub node indices
    """
    if len(graph.nodes()) == 0:
        return []
    
    degrees = dict(graph.degree())
    if not degrees:
        return []
    
    avg_degree = np.mean(list(degrees.values()))
    std_degree = np.std(list(degrees.values()))
    
    threshold = avg_degree + std_degree
    hubs = [node for node, degree in degrees.items() if degree > threshold]
    
    return hubs


class AlphaDeespAdapter:
    """
    Adapter to make pypowsybl simulation compatible with alphaDeesp.
    
    This class wraps the pypowsybl-based simulation to provide the interface
    expected by alphaDeesp's Grid2opSimulation.
    """
    
    def __init__(self,
                 obs: 'PypowsyblObservation',
                 action_space: Any,
                 observation_space: Any,
                 param_options: Dict,
                 debug: bool = False,
                 ltc: List[int] = None,
                 plot: bool = False,
                 simu_step: int = 0,
                 use_dc: bool = False):
        """
        Initialize adapter with alphaDeesp-compatible interface.
        
        Args:
            obs: PypowsyblObservation instance
            action_space: ActionSpace instance  
            observation_space: Observation space (for compatibility)
            param_options: alphaDeesp parameters
            debug: Enable debug output
            ltc: List of overloaded line indices (Lines To Consider)
            plot: Enable plotting
            simu_step: Simulation timestep
            use_dc: Use DC load flow
        """
        self._obs = obs
        self._action_space = action_space
        self._params = param_options
        self._ltc = ltc or []
        self._use_dc = use_dc
        
        # Store ltc for visualization
        self.ltc = ltc or []
        
        # Store param_options for create_df compatibility
        self.param_options = param_options
        
        # Create overflow simulator
        self._overflow_sim = OverflowSimulator(
            network_manager=obs._network_manager,
            obs=obs,
            use_dc=use_dc,
            param_options=param_options
        )
        
        # Build topology info
        self.topo = self._build_topo()
        
        # Compute flow changes for overloaded lines
        if ltc:
            line_names = [obs.name_line[i] for i in ltc]
            self._df = self._overflow_sim.compute_flow_changes_after_disconnection(line_names)
            # Create obs_linecut - observation after disconnecting overloaded lines
            # This is used for visualization to show rho changes
            self.obs_linecut = self._create_obs_linecut(obs, line_names)
        else:
            self._df = self._overflow_sim.get_dataframe()
            self.obs_linecut = obs  # No lines cut, same as original
    
    def _build_topo(self) -> Dict:
        """
        Build topology structure for alphaDeesp.
        
        This must match the structure expected by OverFlowGraph:
        - 'edges': dict with 'idx_or' and 'idx_ex' arrays
        - 'nodes': dict with node info including 'are_prods', 'are_loads', 'names',
                   'prods_values', 'loads_values'
        """
        nm = self._obs._network_manager
        net = nm.network
        
        # Get substation indices for each line
        idx_or = nm.get_line_or_subid()
        idx_ex = nm.get_line_ex_subid()
        
        # Determine which substations have generators and loads, and their values
        n_sub = nm.n_sub
        are_prods = [False] * n_sub
        are_loads = [False] * n_sub
        prods_values = [0.0] * n_sub
        loads_values = [0.0] * n_sub
        
        # Check for generators at each substation and sum their production
        gens_df = net.get_generators()
        for gen_id, row in gens_df.iterrows():
            vl_id = row.get('voltage_level_id', '')
            try:
                sub_idx = list(nm.name_sub).index(vl_id)
                are_prods[sub_idx] = True
                p = row.get('p', 0.0)
                if not np.isnan(p):
                    prods_values[sub_idx] += abs(p)  # Production is typically negative in pypowsybl convention
            except ValueError:
                pass
        
        # Check for loads at each substation and sum their consumption
        loads_df = net.get_loads()
        for load_id, row in loads_df.iterrows():
            vl_id = row.get('voltage_level_id', '')
            try:
                sub_idx = list(nm.name_sub).index(vl_id)
                are_loads[sub_idx] = True
                p = row.get('p', 0.0)
                if not np.isnan(p):
                    loads_values[sub_idx] += abs(p)
            except ValueError:
                pass
        
        return {
            'edges': {
                'idx_or': list(idx_or),
                'idx_ex': list(idx_ex),
            },
            'nodes': {
                'are_prods': are_prods,
                'are_loads': are_loads,
                'prods_values': prods_values,
                'loads_values': loads_values,
                'names': list(nm.name_sub),
            }
        }
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get flow change DataFrame (alphaDeesp interface)."""
        return self._df.copy()
    
    def _create_obs_linecut(self, obs: 'PypowsyblObservation', lines_to_cut: List[str]):
        """
        Create an observation-like object representing the state after cutting lines.
        
        This is used by the visualization to show rho changes before/after.
        
        Args:
            obs: Original observation
            lines_to_cut: List of line names to disconnect
            
        Returns:
            Object with rho attribute representing state after line cuts
        """
        nm = obs._network_manager
        
        # Create a variant to simulate line cuts
        variant_id = "obs_linecut_temp"
        nm.create_variant(variant_id)
        nm.set_working_variant(variant_id)
        
        try:
            # IMPORTANT: First, re-apply any lines that are already disconnected in the observation
            # The observation was created from a simulated state (with contingency applied),
            # but the network manager's base variant doesn't have those disconnections.
            for i, line_id in enumerate(nm.name_line):
                if not obs.line_status[i]:  # Line is disconnected in observation
                    nm.disconnect_line(line_id)
            
            # Run load flow to get to the observation state
            nm.run_load_flow(dc=self._use_dc)
            
            # Now disconnect the additional lines (overloaded lines)
            for line_id in lines_to_cut:
                nm.disconnect_line(line_id)
            
            # Run load flow after cutting the overloaded lines
            result = nm.run_load_flow(dc=self._use_dc)
            
            # Create a simple object with rho attribute
            class ObsLineCut:
                def __init__(self, rho_values):
                    self.rho = rho_values
            
            if result is not None and result.status == lf.ComponentStatus.CONVERGED:
                # Use the same thermal limits as the original observation
                # This ensures consistent rho calculations
                thermal_limits = obs._thermal_limits
                
                # Get transformer IDs to handle them differently
                net = nm.network
                trafos_df = net.get_2_windings_transformers()
                trafo_ids = set(trafos_df.index)
                
                # Get line flows and compute rho
                line_flows = nm.get_line_flows()
                rho = np.zeros(nm.n_line)
                
                for i, line_id in enumerate(nm.name_line):
                    if line_id in line_flows.index:
                        i1 = abs(line_flows.loc[line_id, 'i1'])
                        i2 = abs(line_flows.loc[line_id, 'i2'])
                        i1 = i1 if not np.isnan(i1) else 0.0
                        i2 = i2 if not np.isnan(i2) else 0.0
                        
                        if line_id in trafo_ids:
                            # For transformers: use i1 (side 1 = high voltage side)
                            # Thermal limits are defined for side 1 in pypowsybl
                            i_for_rho = i1
                        else:
                            # For AC lines: use max of both terminals
                            i_for_rho = max(i1, i2)
                        
                        thermal_limit = thermal_limits.get(line_id, 9999.0)
                        if thermal_limit > 0:
                            rho[i] = i_for_rho / thermal_limit
                
                return ObsLineCut(rho)
            else:
                # Return zeros if didn't converge
                return ObsLineCut(np.zeros(nm.n_line))
                
        finally:
            nm.set_working_variant(nm.base_variant_id)
            nm.remove_variant(variant_id)
    
    def get_substation_elements(self) -> Dict:
        """
        Get elements per substation (alphaDeesp interface).
        
        Returns dict mapping substation index to dict of element types.
        """
        nm = self._obs._network_manager
        net = nm.network
        
        result = {i: {'loads': [], 'generators': [], 'lines_or': [], 'lines_ex': []} 
                  for i in range(nm.n_sub)}
        
        # Map loads to substations
        loads_df = net.get_loads()
        for load_id, row in loads_df.iterrows():
            vl_id = row.get('voltage_level_id', '')
            try:
                sub_idx = list(nm.name_sub).index(vl_id)
                result[sub_idx]['loads'].append(load_id)
            except ValueError:
                pass
        
        # Map generators to substations
        gens_df = net.get_generators()
        for gen_id, row in gens_df.iterrows():
            vl_id = row.get('voltage_level_id', '')
            try:
                sub_idx = list(nm.name_sub).index(vl_id)
                result[sub_idx]['generators'].append(gen_id)
            except ValueError:
                pass
        
        # Map lines to substations
        for i, line_id in enumerate(nm.name_line):
            or_sub = nm._line_or_sub.get(line_id, '')
            ex_sub = nm._line_ex_sub.get(line_id, '')
            try:
                or_idx = list(nm.name_sub).index(or_sub)
                result[or_idx]['lines_or'].append(line_id)
            except ValueError:
                pass
            try:
                ex_idx = list(nm.name_sub).index(ex_sub)
                result[ex_idx]['lines_ex'].append(line_id)
            except ValueError:
                pass
        
        return result
    
    def get_substation_to_node_mapping(self) -> Dict[int, int]:
        """Get mapping from substation index to node index (alphaDeesp interface)."""
        # In our case, substations are nodes, so it's identity mapping
        nm = self._obs._network_manager
        return {i: i for i in range(nm.n_sub)}
    
    def get_internal_to_external_mapping(self) -> Dict[int, str]:
        """Get mapping from internal node index to external name (alphaDeesp interface)."""
        nm = self._obs._network_manager
        return {i: name for i, name in enumerate(nm.name_sub)}
