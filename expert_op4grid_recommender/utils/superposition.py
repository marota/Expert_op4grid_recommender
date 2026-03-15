# expert_op4grid_recommender/utils/superposition.py
"""
Superposition theorem for inferring combined action pair effects.

This module adapts the superposition theorem from
https://github.com/marota/Topology_Superposition_Theorem
to work with the pypowsybl-based observations used in expert_op4grid_recommender.

The key formula:
  p_or_combined = (1 - sum(betas)) * obs_start.p_or + sum(betas[i] * obs_unit[i].p_or)

where betas are solved from a linear system using p_or and delta_theta values
at the lines/substations involved in each action.
"""

import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional


# =============================================================================
# Low-level helper functions (adapted from superposition_theorem/core)
# =============================================================================

def _get_theta_node(obs, sub_id: int, bus: int) -> float:
    """Get the median voltage angle at a specific bus of a substation.

    Uses all connected elements at the bus (lines_or, lines_ex) to estimate
    the bus voltage angle, even for disconnected lines.

    Args:
        obs: Observation with theta_or, theta_ex, line_or_bus, line_ex_bus
        sub_id: Substation index
        bus: Bus number (1 or 2)

    Returns:
        Median theta at the bus (radians), or 0.0 if no connected elements
    """
    obj = obs.get_obj_connect_to(substation_id=sub_id)

    lines_or = [i for i in obj['lines_or_id'] if obs.line_or_bus[i] == bus]
    lines_ex = [i for i in obj['lines_ex_id'] if obs.line_ex_bus[i] == bus]

    thetas = np.concatenate([obs.theta_or[lines_or], obs.theta_ex[lines_ex]])
    thetas = thetas[thetas != 0]

    if len(thetas) == 0:
        return 0.0
    return float(np.median(thetas))


def get_delta_theta_line(obs, line_idx: int) -> float:
    """Compute delta theta (theta_or - theta_ex) for a line.

    Uses endpoint substation angles via connected elements, so this works
    correctly even when the line itself is disconnected (theta_or/theta_ex
    would be 0, but the substation bus still has a non-zero angle).

    Args:
        obs: Observation with theta_or, theta_ex, line_or_bus, line_ex_bus
        line_idx: Index of the line

    Returns:
        Delta theta value in radians
    """
    sub_or = obs.line_or_to_subid[line_idx]
    sub_ex = obs.line_ex_to_subid[line_idx]
    bus_or = int(obs.line_or_bus[line_idx])
    bus_ex = int(obs.line_ex_bus[line_idx])

    # For a disconnected line bus is -1, fall back to bus 1 (where it will reconnect)
    if bus_or == -1:
        bus_or = 1
    if bus_ex == -1:
        bus_ex = 1

    theta_or = _get_theta_node(obs, sub_or, bus_or)
    theta_ex = _get_theta_node(obs, sub_ex, bus_ex)
    return theta_or - theta_ex


def get_delta_theta_sub_2nodes(obs, sub_id: int) -> float:
    """Compute delta theta between bus 2 and bus 1 at a substation.

    This is the "virtual line" delta theta for substation topology actions.

    Args:
        obs: Observation
        sub_id: Substation index

    Returns:
        Delta theta (bus2 - bus1) in radians
    """
    theta_bus1 = _get_theta_node(obs, sub_id, bus=1)
    theta_bus2 = _get_theta_node(obs, sub_id, bus=2)
    return theta_bus2 - theta_bus1


def get_sub_node1_idsflow(obs, sub_id: int):
    """Identify elements belonging to node 1 (local bus 1) at a substation.

    Adapted from the reference superposition_theorem implementation.
    "Node 1" is the first bus (local bus number 1) within the substation.
    This must be called with an observation where the substation is already
    split (the post-split observation), so the bus assignments reflect
    the future topology.

    Works with both Grid2Op observations (via _get_bus_id) and
    PypowsyblObservation (via line_or_bus / get_obj_connect_to directly).

    Args:
        obs: Post-split observation where the substation has two buses
        sub_id: Substation index

    Returns:
        Tuple (ind_load_node1, ind_prod_node1, ind_lor_node1, ind_lex_node1)
    """
    if hasattr(obs, '_get_bus_id'):
        # Grid2Op path: _get_bus_id returns global bus IDs.
        # Global bus == sub_id means local bus 1.
        ind_prod, _ = obs._get_bus_id(obs.gen_pos_topo_vect, obs.gen_to_subid)
        ind_load, _ = obs._get_bus_id(obs.load_pos_topo_vect, obs.load_to_subid)
        ind_lor, _ = obs._get_bus_id(obs.line_or_pos_topo_vect, obs.line_or_to_subid)
        ind_lex, _ = obs._get_bus_id(obs.line_ex_pos_topo_vect, obs.line_ex_to_subid)

        ind_lor_node1 = [i for i in range(obs.n_line) if ind_lor[i] == sub_id]
        ind_lex_node1 = [i for i in range(obs.n_line) if ind_lex[i] == sub_id]
        ind_load_node1 = [i for i in range(obs.n_load) if ind_load[i] == sub_id]
        ind_prod_node1 = [i for i in range(obs.n_gen) if ind_prod[i] == sub_id]
    else:
        # PypowsyblObservation path: use get_obj_connect_to to get element lists
        # at this substation, then filter by local bus == 1.
        obj = obs.get_obj_connect_to(substation_id=sub_id)

        # line_or_bus / line_ex_bus give the local bus number (1 or 2) per line
        line_or_bus = obs.line_or_bus
        line_ex_bus = obs.line_ex_bus

        ind_lor_node1 = [i for i in obj['lines_or_id'] if line_or_bus[i] == 1]
        ind_lex_node1 = [i for i in obj['lines_ex_id'] if line_ex_bus[i] == 1]

        # For loads and gens, use sub_topology which returns bus assignments
        # in order: [loads..., gens..., lines_or..., lines_ex...]
        topo = obs.sub_topology(sub_id)
        n_loads_sub = len(obj['loads_id'])
        n_gens_sub = len(obj['generators_id'])

        ind_load_node1 = [
            obj['loads_id'][k] for k in range(n_loads_sub)
            if topo[k] == 1
        ]
        ind_prod_node1 = [
            obj['generators_id'][k] for k in range(n_gens_sub)
            if topo[n_loads_sub + k] == 1
        ]

    return (ind_load_node1, ind_prod_node1, ind_lor_node1, ind_lex_node1)


def get_virtual_line_flow(obs, ind_load, ind_prod, ind_lor, ind_lex) -> float:
    """Compute the virtual line flow at a substation using KCL at node 1.

    The virtual line flow represents the power flowing between the two buses
    of a split substation, computed from Kirchhoff's Current Law applied
    to node 1 elements.

    Args:
        obs: Observation with p_or, load_p, gen_p
        ind_load: Indices of loads on node 1
        ind_prod: Indices of generators on node 1
        ind_lor: Indices of lines with origin at node 1
        ind_lex: Indices of lines with extremity at node 1

    Returns:
        Virtual line active power flow (MW)
    """
    flow = 0.0
    flow += sum(-obs.p_or[i] for i in ind_lor)
    flow += sum(obs.p_or[i] for i in ind_lex)  # p_ex flows in opposite direction
    flow += sum(-obs.load_p[i] for i in ind_load)
    flow += sum(obs.gen_p[i] for i in ind_prod)
    return flow


def _is_sub_reference_topology(obs, sub_id: int) -> bool:
    """Check if a substation is in reference (single bus) topology.

    Returns True if the substation has no element on bus 2.
    """
    topo = obs.sub_topology(sub_id)
    return 2 not in topo


# =============================================================================
# Beta coefficients computation
# =============================================================================

def get_betas_coeff(delta_theta_unit_acts, delta_theta_start,
                    p_or_unit_acts, p_or_start, idls):
    """Compute the superposition theorem beta coefficients.

    Adapted from get_betas_coeff_N_unit_acts_ultimate in the reference repo.

    The betas satisfy the linear system:
      A * betas = [1, 1, ..., 1]
    where A[j][i] = 1 - (feature_unit_act[i][j] / feature_start[j])
    with diagonal = 1.

    Args:
        delta_theta_unit_acts: array (n_actions, n_actions) of delta_theta
            for each unit action observation at each involved element
        delta_theta_start: array (n_actions,) delta_theta of start obs
            at each involved element
        p_or_unit_acts: array (n_actions, n_actions) of p_or
            for each unit action observation at each involved element
        p_or_start: array (n_actions,) p_or of start obs at involved elements
        idls: list of element indices (for reference, len = n_actions)

    Returns:
        betas: array of beta coefficients, one per action
    """
    n = len(idls)

    # Build the A matrix
    a = np.zeros((n, n))
    threshold = 1e-6
    for j in range(n):
        for i in range(n):
            if i == j:
                a[j][i] = 1.0
            else:
                # Use p_or if both are non-zero, otherwise use delta_theta
                if abs(p_or_start[j]) > threshold:
                    a[j][i] = 1.0 - p_or_unit_acts[i][j] / p_or_start[j]
                elif abs(delta_theta_start[j]) > threshold:
                    a[j][i] = 1.0 - delta_theta_unit_acts[i][j] / delta_theta_start[j]
                else:
                    a[j][i] = 1.0  # No coupling info available

    b = np.ones(n)
    try:
        betas = np.linalg.solve(a, b)
        
    except (np.linalg.LinAlgError, ValueError):
        # Singular matrix or other math error — can't solve
        print("[Superposition] Warning: could not solve linear system for betas. Falling back to [1.0, 1.0].")
        betas = np.ones(n)

    return betas


# =============================================================================
# Action element identification
# =============================================================================

def _identify_action_elements(action, action_id: str, dict_action: dict,
                              classifier, env) -> Tuple[List[int], List[int]]:
    """Identify the line indices and substation indices affected by an action.

    Args:
        action: The PypowsyblAction object
        action_id: Action ID string
        dict_action: Full action dictionary
        classifier: ActionClassifier instance
        env: Environment with name_line, action_space

    Returns:
        (line_indices, sub_indices) — lists of integer indices
    """
    line_indices = []
    sub_indices = []

    # Determine action type from the dict
    action_desc = dict_action.get(action_id, {})
    # Initialize classifier with env action space if possible for by_description=False
    if classifier._action_space is None:
        classifier._action_space = env.action_space

    # Try classification
    action_type = classifier.identify_action_type(action_desc, by_description=True)
    if action_type == "unknown":
        # Guess from ID prefixes
        if "reco_" in action_id or "close_line" in action_id:
            action_type = "close_line"
        elif "node_merging" in action_id or "close_coupling" in action_id:
            action_type = "close_coupling"
        elif "disco_" in action_id or "open_line" in action_id:
            action_type = "open_line"
        elif "open_coupling" in action_id:
            action_type = "open_coupling"
        elif "pst_" in action_id or "pst_tap_" in action_id:
            action_type = "pst"

    name_line = list(env.name_line)
    name_sub = list(env.name_sub)

    if "line" in action_type:
        # Line action: find which line(s) are affected
        affected_lines = set()
        
        # Check explicit attributes in action object
        for field in ('lines_or_bus', 'lines_ex_bus'):
            val = getattr(action, field, None)
            if val:
                affected_lines.update(val.keys())

        # Check for topo_vect if it's a grid2op-like action
        if hasattr(action, 'line_set_status'):
             for i, set_val in enumerate(action.line_set_status):
                 if set_val != 0:
                     affected_lines.add(name_line[i])
        
        # Also check for prefixes if still empty
        if not affected_lines:
            if "reco_" in action_id:
                line_name = action_id[len("reco_"):]
                affected_lines.add(line_name)
            elif "disco_" in action_id:
                line_name = action_id[len("disco_"):]
                affected_lines.add(line_name)
            elif "open_line_" in action_id:
                line_name = action_id[len("open_line_"):]
                affected_lines.add(line_name)
            elif "close_line_" in action_id:
                line_name = action_id[len("close_line_"):]
                affected_lines.add(line_name)

        for line_name in affected_lines:
            if line_name in name_line:
                line_indices.append(name_line.index(line_name))

    if "pst" in action_type:
        # PST action: find which branch(es) are affected
        affected_lines = set()
        pst_tap = action_desc.get("pst_tap", {})
        if pst_tap:
            affected_lines.update(pst_tap.keys())
        
        # Fallback to ElementId if it's there
        resid = action_desc.get("ElementId")
        if resid:
            affected_lines.add(resid)

        # Fallback to ID-based extraction if still empty
        if not affected_lines:
            if "pst_tap_" in action_id:
                affected_lines.add(action_id[len("pst_tap_"):])
            elif "pst_" in action_id:
                affected_lines.add(action_id[len("pst_"):])
             
        for line_name in affected_lines:
            if line_name in name_line:
                line_indices.append(name_line.index(line_name))

    if "coupling" in action_type or "node_merging" in action_id or not line_indices:
        # Substation topology action: find which substation(s) are affected
        affected_subs = set()
        
        vl_id = action_desc.get("VoltageLevelId")
        if vl_id:
            affected_subs.add(vl_id)
        
        # Check action's substations field
        if hasattr(action, 'substations') and action.substations:
            affected_subs.update(action.substations.keys())

        # Check from ID suffix if node_merging
        if "node_merging_" in action_id:
            sub_name = action_id[len("node_merging_"):]
            affected_subs.add(sub_name)
        elif "open_coupling_" in action_id:
            sub_name = action_id[len("open_coupling_"):]
            affected_subs.add(sub_name)
        elif "close_coupling_" in action_id:
            sub_name = action_id[len("close_coupling_"):]
            affected_subs.add(sub_name)

        for sub_name in affected_subs:
            if sub_name in name_sub:
                sub_indices.append(name_sub.index(sub_name))

    return line_indices, sub_indices


# =============================================================================
# Pair superposition computation
# =============================================================================

def compute_combined_pair_superposition(
        obs_start,
        obs_act1,
        obs_act2,
        act1_line_idxs: List[int],
        act1_sub_idxs: List[int],
        act2_line_idxs: List[int],
        act2_sub_idxs: List[int],
        obs_combined: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute the superposition theorem for a pair of unitary actions.

    Args:
        obs_start: Base observation (N-1 state)
        obs_act1: Observation after applying action 1 alone
        obs_act2: Observation after applying action 2 alone
        act1_line_idxs: Line indices involved in action 1
        act1_sub_idxs: Sub indices involved in action 1
        act2_line_idxs: Line indices involved in action 2
        act2_sub_idxs: Sub indices involved in action 2

    Returns:
        Dict with betas, p_or_combined, rho_combined
    """
    # Total number of elements must equal 2 (one per action)
    n_elements_act1 = len(act1_line_idxs) + len(act1_sub_idxs)
    n_elements_act2 = len(act2_line_idxs) + len(act2_sub_idxs)

    if n_elements_act1 == 0 or n_elements_act2 == 0:
        return {"error": "Cannot identify elements for one or both actions"}

    # For multi-element actions, use only the first element for simplicity
    # (the superposition theorem works element-by-element)
    if n_elements_act1 > 1:
        act1_line_idxs = act1_line_idxs[:1] if act1_line_idxs else []
        act1_sub_idxs = act1_sub_idxs[:1] if act1_sub_idxs and not act1_line_idxs else []

    if n_elements_act2 > 1:
        act2_line_idxs = act2_line_idxs[:1] if act2_line_idxs else []
        act2_sub_idxs = act2_sub_idxs[:1] if act2_sub_idxs and not act2_line_idxs else []

    n_actions = len(act1_line_idxs) + len(act1_sub_idxs) + len(act2_line_idxs) + len(act2_sub_idxs)
    if n_actions != 2:
        return {"error": f"Expected 2 elements for pair, got {n_actions}"}

    unit_act_observations = [obs_act1, obs_act2]

    # ---- Determine per-action element type and index ----
    # act1_is_line: True if act1's characteristic element is a line, False if sub.
    # Element ordering in all feature arrays must match action ordering in
    # unit_act_observations (index 0 = act1, index 1 = act2) so that the
    # diagonal A[i][i] = 1 in get_betas_coeff correctly marks element i as
    # "owned by" action i.  We therefore place act1's element at position 0
    # and act2's element at position 1, regardless of their types.
    act1_is_line = len(act1_line_idxs) > 0
    act2_is_line = len(act2_line_idxs) > 0

    # ---- Build p_or and delta_theta arrays, one value per action ----
    # For each action i (0 or 1) and each observation j (0 or 1), compute the
    # feature value for action i's characteristic element observed in obs j.

    def _line_features(obs, line_idx):
        """Return (p_or, delta_theta) for a line element in a given obs."""
        return obs.p_or[line_idx], get_delta_theta_line(obs, line_idx)

    def _sub_features(obs, sub_idx, ind_sub, is_ref, action_applied):
        """Return (p_or_virtual, delta_theta_virtual) for a sub element.

        action_applied: True when the sub's own topology action is active in obs.
        is_ref: True when the sub is in single-bus reference topology at start.
        """
        if action_applied:
            if is_ref:
                # Node splitting applied → virtual line was cut → p_or = 0
                return 0.0, get_delta_theta_sub_2nodes(obs, sub_idx)
            else:
                # Node merging applied → buses merged → delta_theta = 0
                (il, ip, ilor, ilex) = ind_sub
                return get_virtual_line_flow(obs, il, ip, ilor, ilex), 0.0
        else:
            if is_ref:
                # Sub still in reference (single-bus) topology → delta_theta = 0
                (il, ip, ilor, ilex) = ind_sub
                return get_virtual_line_flow(obs, il, ip, ilor, ilex), 0.0
            else:
                # Sub still split → virtual line still cut → p_or = 0
                return 0.0, get_delta_theta_sub_2nodes(obs, sub_idx)

    # Pre-compute node-1 element lists for sub actions (needed by _sub_features)
    ind_sub_act1 = None
    is_start_ref_act1 = None
    if not act1_is_line:
        sid1 = act1_sub_idxs[0]
        is_start_ref_act1 = _is_sub_reference_topology(obs_start, sid1)
        if is_start_ref_act1:
            ind_sub_act1 = get_sub_node1_idsflow(obs_act1, sid1)
        else:
            ind_sub_act1 = get_sub_node1_idsflow(obs_start, sid1)

    ind_sub_act2 = None
    is_start_ref_act2 = None
    if not act2_is_line:
        sid2 = act2_sub_idxs[0]
        is_start_ref_act2 = _is_sub_reference_topology(obs_start, sid2)
        if is_start_ref_act2:
            ind_sub_act2 = get_sub_node1_idsflow(obs_act2, sid2)
        else:
            ind_sub_act2 = get_sub_node1_idsflow(obs_start, sid2)

    # p_or_unit_act[obs_j][element_i]:
    #   outer index j iterates over observations (unit_act_observations),
    #   inner index i iterates over elements in action order [act1_elem, act2_elem].
    p_or_unit_act_list = []
    delta_theta_unit_act_list = []
    for j, obs in enumerate(unit_act_observations):
        row_p = []
        row_dt = []
        # Element 0: act1's characteristic element
        if act1_is_line:
            p, dt = _line_features(obs, act1_line_idxs[0])
        else:
            p, dt = _sub_features(obs, act1_sub_idxs[0], ind_sub_act1,
                                   is_start_ref_act1, action_applied=(j == 0))
        row_p.append(p)
        row_dt.append(dt)
        # Element 1: act2's characteristic element
        if act2_is_line:
            p, dt = _line_features(obs, act2_line_idxs[0])
        else:
            p, dt = _sub_features(obs, act2_sub_idxs[0], ind_sub_act2,
                                   is_start_ref_act2, action_applied=(j == 1))
        row_p.append(p)
        row_dt.append(dt)
        p_or_unit_act_list.append(row_p)
        delta_theta_unit_act_list.append(row_dt)

    p_or_unit_act = np.array(p_or_unit_act_list)
    delta_theta_unit_act = np.array(delta_theta_unit_act_list)

    # Start-observation feature values for each element
    p_or_start_list = []
    delta_theta_start_list = []
    for act_is_line, line_idxs, sub_idxs, ind_sub, is_ref in [
        (act1_is_line, act1_line_idxs, act1_sub_idxs, ind_sub_act1, is_start_ref_act1),
        (act2_is_line, act2_line_idxs, act2_sub_idxs, ind_sub_act2, is_start_ref_act2),
    ]:
        if act_is_line:
            p, dt = _line_features(obs_start, line_idxs[0])
        else:
            if is_ref:
                (il, ip, ilor, ilex) = ind_sub
                p = get_virtual_line_flow(obs_start, il, ip, ilor, ilex)
                dt = 0.0
            else:
                p = 0.0
                dt = get_delta_theta_sub_2nodes(obs_start, sub_idxs[0])
        p_or_start_list.append(p)
        delta_theta_start_list.append(dt)

    p_or_start = np.array(p_or_start_list)
    delta_theta_start = np.array(delta_theta_start_list)

    # idls is only used by get_betas_coeff to determine n; order matches
    # action order: [act1_element, act2_element].
    if act1_is_line:
        idls = [act1_line_idxs[0]]
    else:
        idls = [act1_sub_idxs[0]]
    if act2_is_line:
        idls.append(act2_line_idxs[0])
    else:
        idls.append(act2_sub_idxs[0])

    # ---- Compute betas ----
    betas = get_betas_coeff(
        delta_theta_unit_act, delta_theta_start,
        p_or_unit_act, p_or_start,
        idls
    )

    if np.any(np.isnan(betas)):
        return {"error": "Singular system — cannot compute betas", "betas": betas.tolist()}

    # ---- Compute combined p_or ----
    p_or_combined = (1.0 - np.sum(betas)) * obs_start.p_or
    for i in range(2):
        p_or_combined = p_or_combined + betas[i] * unit_act_observations[i].p_or

    # ---- Islanding detection (if obs_combined is provided) ----
    is_islanded = False
    disconnected_mw = 0.0
    if obs_combined is not None and hasattr(obs_combined, 'n_components'):
         if obs_combined.n_components > obs_start.n_components:
             is_islanded = True
             if hasattr(obs_combined, 'main_component_load_mw') and hasattr(obs_start, 'main_component_load_mw'):
                 disconnected_mw = float(max(0.0, obs_start.main_component_load_mw - obs_combined.main_component_load_mw))

    return {
        "betas": betas.tolist(),
        "p_or_combined": p_or_combined.tolist(),
        "is_islanded": is_islanded,
        "disconnected_mw": disconnected_mw,
    }


# =============================================================================
# Main entry point: compute all pairs
# =============================================================================

def compute_all_pairs_superposition(
        obs_start,
        detailed_actions: Dict[str, Dict],
        classifier,
        env,
        lines_overloaded_ids: List[int],
        lines_we_care_about,
        pre_existing_rho: Dict[int, float],
        dict_action: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Compute superposition theorem for all pairs of converged prioritized actions.

    Args:
        obs_start: Base observation (N-1 state, obs_simu_defaut)
        detailed_actions: Dict of action_id -> action details (from Reassessment)
        classifier: ActionClassifier instance
        env: Environment with name_line, action_space
        lines_overloaded_ids: Indices of overloaded lines
        lines_we_care_about: Set/list of line names we monitor
        pre_existing_rho: Dict of line_idx -> pre-existing rho values
        dict_action: Full action dictionary (for action type classification)

    Returns:
        Dict of "action1_id+action2_id" -> result dict with betas, p_or_combined,
        max_rho, max_rho_line, etc.
    """
    if dict_action is None:
        dict_action = {}

    # Filter to converged actions only
    converged_ids = [
        aid for aid, details in detailed_actions.items()
        if details.get("non_convergence") is None
    ]

    if len(converged_ids) < 2:
        print(f"[Superposition] Only {len(converged_ids)} converged action(s), "
              f"need at least 2 for pair combination.")
        return {}

    print(f"[Superposition] Computing {len(converged_ids)} * "
          f"{len(converged_ids) - 1} / 2 = "
          f"{len(converged_ids) * (len(converged_ids) - 1) // 2} pairs")

    # Pre-compute element indices for each action
    action_elements = {}
    for aid in converged_ids:
        action_obj = detailed_actions[aid]["action"]
        line_idxs, sub_idxs = _identify_action_elements(
            action_obj, aid, dict_action, classifier, env
        )
        action_elements[aid] = (line_idxs, sub_idxs)

    # Monitoring setup for max_rho computation
    from expert_op4grid_recommender import config
    name_line = list(env.name_line)
    num_lines = len(name_line)
    worsening_threshold = getattr(config, 'PRE_EXISTING_OVERLOAD_WORSENING_THRESHOLD', 0.02)

    pre_existing_baseline = np.zeros(num_lines)
    is_pre_existing = np.zeros(num_lines, dtype=bool)
    for idx, rho_val in pre_existing_rho.items():
        pre_existing_baseline[idx] = rho_val
        is_pre_existing[idx] = True

    if lines_we_care_about is not None and len(lines_we_care_about) > 0:
        care_mask = np.isin(name_line, list(lines_we_care_about))
    else:
        care_mask = np.ones(num_lines, dtype=bool)

    results = {}
    baseline_rho = obs_start.rho[lines_overloaded_ids]

    for aid1, aid2 in combinations(converged_ids, 2):
        line_idxs1, sub_idxs1 = action_elements[aid1]
        line_idxs2, sub_idxs2 = action_elements[aid2]

        # Skip if we can't identify elements for either action
        if len(line_idxs1) + len(sub_idxs1) == 0:
            continue
        if len(line_idxs2) + len(sub_idxs2) == 0:
            continue

        obs_act1 = detailed_actions[aid1]["observation"]
        obs_act2 = detailed_actions[aid2]["observation"]

        try:
            result = compute_combined_pair_superposition(
                obs_start=obs_start,
                obs_act1=obs_act1,
                obs_act2=obs_act2,
                act1_line_idxs=line_idxs1,
                act1_sub_idxs=sub_idxs1,
                act2_line_idxs=line_idxs2,
                act2_sub_idxs=sub_idxs2,
            )
        except Exception as e:
            print(f"[Superposition] Error computing pair {aid1}+{aid2}: {e}")
            result = {"error": str(e)}

        if "error" in result:
            print(f"[Superposition] Skipping pair {aid1}+{aid2}: {result['error']}")
            results[f"{aid1}+{aid2}"] = result
            continue

        # Compute combined rho and max_rho
        # p_or_combined = np.array(result["p_or_combined"])

        # Approximate rho from p_or: rho ≈ |p_or| / (sqrt(3) * V * I_max)
        # Since p_or = sqrt(3) * V * I * cos(φ) and I_max is what we stored,
        # and rho = I / I_max, a more direct approach is:
        # rho_combined ≈ |p_or_combined / p_or_start| * rho_start for each line
        # But safer: we compute rho ratio from the linear combination of rho arrays
        rho_combined = np.abs(
            (1.0 - sum(result["betas"])) * obs_start.rho +
            result["betas"][0] * obs_act1.rho +
            result["betas"][1] * obs_act2.rho
        )

        # Rho on overloaded lines
        rho_after = rho_combined[lines_overloaded_ids]
        is_rho_reduction = bool(np.all(rho_after + 0.01 < baseline_rho))

        # Max rho among monitored lines
        worsened_mask = rho_combined > pre_existing_baseline * (1 + worsening_threshold)
        eligible_mask = care_mask & (~is_pre_existing | worsened_mask)

        max_rho = 0.0
        max_rho_line = "N/A"
        if np.any(eligible_mask):
            masked_rho = rho_combined[eligible_mask]
            max_idx = np.argmax(masked_rho)
            max_rho = float(masked_rho[max_idx])
            max_rho_line = name_line[np.where(eligible_mask)[0][max_idx]]
        
        # Islanding and disconnected MW
        is_islanded = result.get("is_islanded", False)
        disconnected_mw = result.get("disconnected_mw", 0.0)

        # Build descriptions
        desc1 = detailed_actions[aid1].get("description_unitaire", aid1)
        desc2 = detailed_actions[aid2].get("description_unitaire", aid2)

        # Scaling factor for monitoring limits
        monitoring_factor = getattr(config, 'MONITORING_FACTOR_THERMAL_LIMITS', 1.0)

        result.update({
            "max_rho": max_rho,
            "max_rho_line": max_rho_line,
            "is_rho_reduction": is_rho_reduction,
            "p_or_combined": result["p_or_combined"],
            "description": f"{desc1} + {desc2}",
            "action1_id": aid1,
            "action2_id": aid2,
            "is_islanded": is_islanded,
            "disconnected_mw": disconnected_mw,
            "rho_after": (rho_combined[lines_overloaded_ids] * monitoring_factor).tolist(),
            "rho_before": (obs_start.rho[lines_overloaded_ids] * monitoring_factor).tolist(),
        })

        # Note: p_or_combined is kept in result for frontend diagram/test consistency

        pair_key = f"{aid1}+{aid2}"
        results[pair_key] = result

        print(f"  {pair_key}: betas={np.round(result['betas'], 4)}, "
              f"max_rho={max_rho:.3f} on {max_rho_line}, "
              f"rho_reduction={is_rho_reduction}")

    print(f"[Superposition] Computed {len(results)} pair combinations")
    return results
