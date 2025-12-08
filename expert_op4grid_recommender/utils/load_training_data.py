import json
import os
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from expert_op4grid_recommender.utils.data_utils import StateInfo
from grid2op.Environment import Environment
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from grid2op.Exceptions import Grid2OpException

from expert_op4grid_recommender.utils.make_training_env import make_grid2op_training_env

#: name of the powerline that are now removed (non existant)
#: but are in the environment because we need them when we will use
#: historical dataset
DELETED_LINE_NAME = set(['BXNE L32BXNE5', 'BXNE5L32MTAGN', 'BXNE5L32CORGO', 'BXNE5L31MTAGN'])


def list_all_obs_files(root: str,
                       ext : str =".gz",
                       sort_results: bool =False,
                       exclude : Optional[str]="20241205") -> List[str]:
    res = []
    tmp = os.path.splitext(ext)
    tmp = [el for el in tmp if el != ""]
    if len(tmp) != 1:
        real_ext = tmp[-1]
    else:
        real_ext = ext
    for path, subdirs, files in os.walk(root):
        for name in files:
            excluded = (exclude is not None) and (exclude in name)
            if (ext in name) and (os.path.splitext(name)[1] == real_ext) and not excluded:
                res.append(os.path.join(path, name))
    if sort_results:
        res = sorted(res)
    return res


def set_state(env: Environment,
              action_path : Union[str, StateInfo],
              with_timings=False) -> Tuple[BaseObservation, StateInfo]:
    """NOT thread safe ! It modifies the environment (first argument)"""
    timings = {}
    if isinstance(action_path, StateInfo):
        state = action_path[0]
    else:
        pth, ext_ = os.path.splitext(action_path)
        beg_read = time.perf_counter()
        if ext_ == ".json":
            with open(action_path, "r", encoding="utf-8") as f:
                init_state = json.load(f)
        elif ext_ == ".gz":
            import gzip
            with gzip.open(action_path, "rt", encoding="utf-8") as f:
                init_state = json.load(f)
        else:
            raise RuntimeError(f"Unsuported time series file extension '{ext_}'")
        timings["reading"] = time.perf_counter() - beg_read
        
        beg_set = time.perf_counter()
        state = StateInfo.from_init_state(env, init_state)
        timings["extract"] = time.perf_counter() - beg_set
    
    beg_reset = time.perf_counter()
    res =  env.reset(options=state.init_state)
    timings["glop_reset"] = time.perf_counter() - beg_reset
    
    if with_timings:
        return res, state, timings
    else:
        return res, state

def aux_prevent_prod_conso_reconnection(obs : BaseObservation,
                                        state :StateInfo,
                                        act: BaseAction) -> BaseAction:
    """
    Given a grid2op state / observation (obs), the information about
    the original "point figes" (full French grid state) and a
    proposed action, this function will ensure that, if a generator
    or a load is not connected on the observation, it will
    not be reconnected by the action.
    
    The reconnection of such element is assumed to be a "non desired"
    side effect of the modeling of opening / closing 
    busbar coupler in the "point figes". But as the actions are not
    modeled the same way in grid2op and in RePAS we had to "translate"
    them which lead to this kind of "errors" that are post processed
    now.
    
    .. danger::
        act arg is updated in place !
    """
    # state is not used currently, but could be used for refinements later if we want to let the reconnection happen under some conditions
    #also added to have same signature as aux_prevent_line_reconnection
    if act.gen_change_bus.any():
        raise RuntimeError("Change bus action for generators are not expected for the action space we defined")

    if act.load_change_bus.any():
        raise RuntimeError("Change bus action for generators are not expected for the action space we defined")

    gen_disco_obs_co_act = (obs.gen_bus == -1) & (act.gen_set_bus >= 1)
    if gen_disco_obs_co_act.any():
        disco_gens = [(gen_id, 0) for gen_id in gen_disco_obs_co_act.nonzero()[0]]
        act.update({"set_bus": {"generators_id": disco_gens}})
        
    load_disco_obs_co_act = (obs.load_bus == -1) & (act.load_set_bus >= 1)
    if load_disco_obs_co_act.any():
        disco_loads = [(load_id, 0) for load_id in load_disco_obs_co_act.nonzero()[0]]
        act.update({"set_bus": {"loads_id": disco_loads}})

    return act


def aux_prevent_line_reconnection_cond1(obs: BaseObservation,
                                        should_not_reco: set,
                                        act: BaseAction) -> Tuple[BaseAction, np.ndarray]:
    line_or_set = act.line_or_set_bus
    line_ex_set = act.line_ex_set_bus
    line_or_change = act.line_or_change_bus
    line_ex_change = act.line_ex_change_bus
    line_change_status = act.line_change_status
    line_set_status = act.line_set_status
    lines_treated = np.full(type(act).n_line, dtype=bool, fill_value=False)
    
    for line_name in should_not_reco:
        if line_name is None:
            continue
        line_id, *_ = type(act).get_line_info(line_name=line_name)
        do_deco = False
        do_deco |= line_or_change[line_id]
        do_deco |= line_ex_change[line_id]
        do_deco |= (line_or_set[line_id] > 0)
        do_deco |= (line_ex_set[line_id] > 0)
        do_deco |= line_change_status[line_id]
        do_deco |= (line_set_status[line_id] > 0)
        if do_deco:
            act.update({"set_line_status": [(line_name, -1)]})
            act.remove_line_status_from_topo(check_cooldown=False)
            lines_treated[line_id] = True
    return act, lines_treated


def aux_prevent_line_reconnection_cond2(obs: BaseObservation,
                                        lines_treated: np.ndarray,
                                        act: BaseAction) -> BaseAction:
    # act is an "action on a busbar coupler"
    # if it affects through set_bus at least 2 disctinct elements
    subs_impacted = type(act).grid_objects_types[act.set_bus >= 1, type(act).SUB_COL]
    subs_impacted, count = np.unique(subs_impacted, return_counts=True)
    if not (count >= 2).any():
       return act
   
    # prevent reconnect of powerlines on the substations
    # that are affected by the coupler
    line_obs_disco = ~obs.line_status
    line_act_reco = ((act.set_line_status == 1) | 
                     (act.line_or_set_bus >= 1) | 
                     (act.line_ex_set_bus >= 1))
    line_disco_but_act_reco = line_obs_disco & line_act_reco & (~lines_treated)
    if not line_disco_but_act_reco.any():
        return act
    
    # I find the substation where there are coupler
    sub_coupler = subs_impacted[count >= 2]
    # discard action on a line that is reconnected
    # with the action on the "coupler"
    line_to_remove = (line_disco_but_act_reco & 
                      (np.isin(type(act).line_or_to_subid, sub_coupler) |
                       np.isin(type(act).line_ex_to_subid, sub_coupler))
                      )
    line_id_to_remove = line_to_remove.nonzero()[0]
    act.update({"set_bus": [(type(act).line_or_pos_topo_vect[el_id], 0) for el_id in line_id_to_remove]})
    act.update({"set_bus": [(type(act).line_ex_pos_topo_vect[el_id], 0) for el_id in line_id_to_remove]})
    act.update({"set_line_status": [(el_id, 0) for el_id in line_id_to_remove]})
    
    return act
    
    
def aux_prevent_line_reconnection(obs: BaseObservation,
                                  state: StateInfo,
                                  act: BaseAction) -> BaseAction:
    """
    Given a grid state (obs), some extra information about the "points figes"
    (French grid) and an action, this function will:
    1) if it was not possible to reconnect the line in the "point figes", by a
       simple action on a switch, , prevent its reconnection (in this case,
       this information is in StateInfo)
    2) if the action is tagged as a "busbar coupler action" (right now an action
       that modifies 2 elements with "set_bus" at the same substation), then for 
       all lines that are disconnected in the obs, but reconnected by this "busbar
       coupler action" then these lines should not be reconnected by the action.
       

    In case 2, the reconnection of such element is assumed to be a "non desired"
    side effect of the modeling of opening / closing 
    busbar coupler in the "point figes". But as the actions are not
    modeled the same way in grid2op and in RePAS we had to "translate"
    them which lead to this kind of "errors" that are post processed
    now.
    
    In case 1 it is because the information has been "lost" during the
    "projection" process.
    
    .. danger::
        act arg is updated in place !    
    """
    
    # obs is here for the date time
    if state.should_not_reco is None:
        should_not_reco = [line for line in DELETED_LINE_NAME if line in obs.name_line]
    else:
        should_not_reco = state.should_not_reco


    # check if the action affect any lines that should not
    # be reconnected
    # case 1)
    act, lines_treated = aux_prevent_line_reconnection_cond1(obs, should_not_reco, act)
    
    # implement condition 2:
    act = aux_prevent_line_reconnection_cond2(obs, lines_treated, act)
    
    return act

def aux_prevent_asset_reconnection(obs : BaseObservation,
                                   state :StateInfo,
                                   act: BaseAction) -> BaseAction:
    act = aux_prevent_prod_conso_reconnection(obs,state, act)
    act = aux_prevent_line_reconnection(obs, state, act)
    return act


def load_interesting_lines(path : Optional[str]=None, file_name : str ="lignes_a_monitorer.csv") -> np.ndarray:
    if path is None:
        path = os.path.abspath(".")
    fn_ = os.path.join(path, file_name)
    if not os.path.exists(fn_):
        #raise RuntimeError(f"Impossible to locate the file describing the interesting lines, looked at {fn_}")
        print(f"Impossible to locate the file describing the interesting lines, looked at {fn_}")
        print("these will not be considered")
        return np.array([])
    return np.array([el.rstrip().lstrip() for el in pd.read_csv(fn_)["branches"].values])
        

def filter_out_non_reproductible_observation(env : Environment, all_obs_files : List[str], lines_we_care_about: List[str]) -> List[StateInfo]:
    nb_errors = 0
    usable_obs_files = []
    not_usable_obs_file = []
    for el in tqdm(all_obs_files):
        try:
            obs, this_state = set_state(env, el)
        except Grid2OpException as exc_:
            print(f"{exc_}")
            not_usable_obs_file.append(el)
            nb_errors += 1
            continue
        
        # some check data are consistent
        try:
            if this_state.n1_id is not None:
                # n-1 should be disconnected
                assert not obs.line_status[this_state.n1_id]
                # n-1 should be in the n-1 list
                assert this_state.n1_name in line_we_disconnect
        except AssertionError as exc_:
            print(f"{exc_}")
            not_usable_obs_file.append(el)
            nb_errors += 1
            continue
        
        # line in overflow are the correct ones (n-1 case)
        if this_state.n1_id is not None:
            li_overflow = type(obs).name_line[obs.rho > 1].tolist()
            li_overflow = [el for el in li_overflow if el in lines_we_care_about]

            # li_overflow_filtered = [item for item in li_overflow if item not in ["C.FOUL31NAVIL", "C.SAUL31ZCRIM"]]
            li_overflow_filtered = li_overflow
            if sorted(li_overflow_filtered) != sorted(this_state.overflow_names):
                print(f"{sorted(li_overflow_filtered)} vs {sorted(this_state.overflow_names)}")
                not_usable_obs_file.append(el)
                nb_errors += 1
            elif len(li_overflow) == 0:
                print(f"No overflow detected for the file {el}")
                not_usable_obs_file.append(el)
                nb_errors += 1
            else:
                usable_obs_files.append(el)
        else:
            # case "N" => save all
            usable_obs_files.append(el)
            
        # check that theta are correctly defined
        assert (obs.theta_or != 0.).any()
    print(f"Found {nb_errors} mismatch between RTE-server and local, finally using only {len(usable_obs_files)} / {len(all_obs_files)}")
    return usable_obs_files, not_usable_obs_file

def save_observation_files(usable_obs_files, not_usable_obs_file):
    # Saving usable_obs_files
    with open('usable_obs_files.json', 'w') as f:
        json.dump(usable_obs_files, f, indent=4)

    # Saving not_usable_obs_file
    with open('not_usable_obs_file.json', 'w') as f:
        json.dump(not_usable_obs_file, f, indent=4)
    
    
if __name__ == "__main__":
    path_env = '.'
    nm_env = "env_dijon_v2_training"
    
    # find all usable data
    # all_obs_files = list_all_obs_files("time_series", sort_results=True,ext=".json")
    all_obs_files = list_all_obs_files("/home/donnotben/Documents/assistflux/read_history/20250228_livraison_LJN/time_series/20250527",
                                       sort_results=True,
                                       ext=".gz")
    print(f"Found {len(all_obs_files)} observations.")

    # interesting lines
    lines_we_care_about = load_interesting_lines()
    line_we_disconnect = load_interesting_lines(file_name="lignes_a_deconnecter.csv")
    
    # make the environment
    env = make_grid2op_training_env(path_env, nm_env)
    
    # id des lignes dont on doit se preoccuper des "overflows"
    id_interesting_lines = np.array([type(env).get_line_info(line_name=el)[0] for el in lines_we_care_about], dtype=int)
    
    # read all available actions
    with open("all_actions.json", "r", encoding="utf-8") as f:
        all_possible_act = json.load(f)
    print(f"Found {len(all_possible_act)} possible actions.")
    
    # perform some consistency check
    usable_obs_files, not_usable_obs_file  = filter_out_non_reproductible_observation(env, all_obs_files, lines_we_care_about)
    save_observation_files(usable_obs_files, not_usable_obs_file)
    # find some possible actions
    this_file = usable_obs_files[0]
    obs, this_state = set_state(env, this_file)
    # set it to the proper state
    obs = this_state.set_env_state(env)
    grid2op_acts_this_obs = [aux_prevent_asset_reconnection(obs, this_state, env.action_space(el["content"])) for k, el in all_possible_act.items()]
    
    # test I can perfom a step and some obs.simulate
    une_action = grid2op_acts_this_obs[1]
    sim_o, sim_r, sim_d, sim_i = obs.simulate(une_action)
    sim_o2, sim_r2, sim_d2, sim_i2 = obs.simulate(une_action, time_step=0)
    next_o, next_r, next_d, next_i = env.step(une_action)
    assert (np.abs(sim_o.rho - next_o.rho) <= 1e-5).all()
    assert (np.abs(sim_o2.rho - next_o.rho) <= 1e-5).all()
    
    # try to find some good actions among the possible ones
    obs = this_state.set_env_state(env)
    max_rho = np.zeros(len(grid2op_acts_this_obs)) + 999999.
    # test I can iterate through the actions to find a good one
    for i, act in enumerate(grid2op_acts_this_obs):
        sim_o, sim_r, sim_d, sim_i = obs.simulate(act)
        if not sim_d:
            max_rho[i] = sim_o.rho[id_interesting_lines].max()
    sorted_action = np.argsort(max_rho)
    print(sorted_action)
    
