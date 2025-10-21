import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import datetime
from expert_op4grid_recommender.utils.data_utils import StateInfo
import pandas as pd


from expert_op4grid_recommender.utils.make_assistant_env import make_grid2op_assistant_env

from expert_op4grid_recommender.utils.load_training_data import aux_prevent_asset_reconnection, load_interesting_lines

#: name of the powerline that are now removed (non existant)
#: but are in the environment because we need them when we will use
#: historical dataset
DELETED_LINE_NAME=['BXNE L32BXNE5', 'BXNE5L32MTAGN', 'BXNE5L32CORGO', 'BXNE5L31MTAGN']#dupplicating here as it causes errors otherwise when imported

def list_all_chronics(env) -> List[str]:
    res = []

    for id, sp in enumerate(env.chronics_handler.real_data.subpaths):
        res.append(os.path.basename(sp))

    return res

def search_chronic_num_from_name(scenario_name, env):
    found_id = None
    # Search scenario with provided name
    list_chronic_names = list_all_chronics(env)
    if scenario_name in list_chronic_names:
        found_id=[i  for i,scenario in enumerate(list_chronic_names) if scenario==scenario_name][0]

    return found_id

def set_thermal_limits_scenario(path,date_str,env):
    with open(os.path.join(path,"thermal_limits_" + date_str + ".json"), 'r') as fp:
        th_lim_dict_day = json.load(fp)  #
        th_lim_dict_day_arr = [th_lim_dict_day[l] for l in env.name_line]
        env.set_thermal_limit(th_lim_dict_day_arr)

def get_reconnectable_lines_disconnected_at_start(env,lines_non_reconnectable):
    maintenance_df = pd.DataFrame(env.chronics_handler.real_data.data.maintenance_handler.array, columns=env.name_line)
    lines_in_maintenance_obs_start = list(maintenance_df.iloc[0][(maintenance_df.iloc[
        0])].index)  # could use obs.time_next_maintenance instead in theory for that (but possible wrong currently for maintenance at first timestep)
    reconnectabble_line_in_maintenance_at_start = list(
        set(lines_in_maintenance_obs_start) - set(lines_non_reconnectable))
    print(f"lines in maintenance at start: {lines_in_maintenance_obs_start}")
    print(f"lines in maintenance at start and reconnectable in scenario: {reconnectabble_line_in_maintenance_at_start}")

    return reconnectabble_line_in_maintenance_at_start,maintenance_df


def get_lines_disconnected_at_start_to_reconnect(maintenance_df,lines_disconnected_at_start,timestep):

    # act to reconnect some initially disconnected lines ?
    do_reco_maintenance_at_t = ~maintenance_df[lines_disconnected_at_start].iloc[timestep]
    maintenance_to_reco_at_t = list(do_reco_maintenance_at_t[do_reco_maintenance_at_t].index)
    if len(maintenance_to_reco_at_t) != 0:
        print(f"reconnecting lines not in disconnected in maintenance data anymore {maintenance_to_reco_at_t} at timestep {timestep}")

    return maintenance_to_reco_at_t

def get_first_obs_on_chronic(date,env,path_thermal_limits=""):
    list_chronic_names=list_all_chronics(env)
    chronic_dates_str=[chronic.split("_")[0] for chronic in list_chronic_names]
    
    date_str=date.strftime('%Y%m%d')
    
    if date_str not in chronic_dates_str:
        raise("no chronic is found for this date")
    else:
        chronic_name=[list_chronic_names[i] for i,date_cr in enumerate(chronic_dates_str) if date_cr==date_str][0]
        print("we found the  chronic "+chronic_name+" for this date")
    path_chronic = [path for path in env.chronics_handler.real_data.subpaths if date.strftime('%Y%m%d') in path][0]

    ########"
    #Patch pour l'instant, voir si mieux pourrait être fait
    #On fait un premier reset pour récupérer une obs/env sain, car pour des simulation successives de défaut ou autre, l'observation pouvait être un peu vérolée
    # grid2op.Exceptions.grid2OpException.Grid2OpException: Grid2OpException "Impossible to set the thermal limit to a non initialized Environment. Have you called `env.reset()` after last game over ?"
    #

    #########

    id_chronic=search_chronic_num_from_name(chronic_name,env)
    env.set_id(id_chronic)



    if path_thermal_limits!="":
        obs = env.reset()
        set_thermal_limits_scenario(path_thermal_limits,date_str, env)
        env.set_id(id_chronic)

    #load init_state explicitly, is it needed ?
    with open(os.path.join(path_chronic,"init_state.json"), "r", encoding="utf-8") as f:
        init_state_dict = json.load(f)

    obs = env.reset()#env.reset(options = {"init state": init_state_dict})
    ##############
    # TO DO, check that the init_state is applied again ..!

    return obs


def run_contingency_on_scenario(obs,defaut,env,id_interesting_lines,reconnectabble_line_in_maintenance_at_start,maintenance_df):
    act_deco_defaut = env.action_space({"set_line_status": [(defaut, -1)]})  # + action_topo_init

    timesteps = [i for i in range(0, 48)]
    overloaded_timesteps=[]
    print("running contingency analysis over the scenario for contingency "+defaut)
    for t in timesteps[0:-1]:
        # act to reconnect some initially disconnected lines ?
        maintenance_to_reco_at_t = get_lines_disconnected_at_start_to_reconnect(maintenance_df,reconnectabble_line_in_maintenance_at_start,t)
        act_reco_maintenance = env.action_space(
            {"set_line_status": [(line_reco, 1) for line_reco in maintenance_to_reco_at_t]})

        obs_simu, reward, done, info = obs.simulate(act_deco_defaut+act_reco_maintenance, time_step=t)

        if np.any(obs_simu.rho[id_interesting_lines]>=1):
            print("there is an overload at timestep "+str(t))
            overloaded_timesteps.append(t)
    return overloaded_timesteps

def run_remedial_action(obs,defaut,lines_non_reconnectable,env,id_interesting_lines,overloaded_timesteps,action,reconnectabble_line_in_maintenance_at_start,maintenance_df):
    act_deco_defaut = env.action_space({"set_bus": {"lines_ex_id":{defaut:-1}, "lines_or_id":{defaut:-1}}})  # + action_topo_init

    for t in overloaded_timesteps:
        state = StateInfo()  # TO DO: give some relevant values when we know which ones and how to pass them
        state.should_not_reco=lines_non_reconnectable

        # act to reconnect some initially disconnected lines ?
        maintenance_to_reco_at_t = get_lines_disconnected_at_start_to_reconnect(maintenance_df,reconnectabble_line_in_maintenance_at_start,t)
        act_reco_maintenance = env.action_space(
            {"set_bus": {"lines_ex_id":{line_reco:-1 for line_reco in maintenance_to_reco_at_t}, "lines_or_id":{line_reco:-1 for line_reco in maintenance_to_reco_at_t}}})

        act_defaut_parade = aux_prevent_asset_reconnection(obs, state,act_reco_maintenance + act_deco_defaut + action)
        obs_simu, reward, done, info = obs.simulate(act_defaut_parade, time_step=t)

        if np.any(obs_simu.rho[id_interesting_lines]>=1):
            print("there is still an overload at timestep "+str(t))
    if len(overloaded_timesteps)==0:
        print("Success for the remedial action")
        print(action)
    return overloaded_timesteps
    
if __name__ == "__main__":
    path_env = '.'
    nm_env = "env_dijon_v2_assistant"
    defaut = "AISERL31RONCI"  # "AISERL31RONCI"#"P.SAOL31RONCI"#line_we_disconnect[0]
    date = datetime.datetime(2024, 8, 28) #(2024, 11, 25)#(2024, 8, 28) #we choose a date for the chronic

    # interesting lines
    lines_we_care_about = load_interesting_lines()
    line_we_disconnect = load_interesting_lines(file_name="lignes_a_deconnecter.csv")
    
    # make the environment
    env = make_grid2op_assistant_env(path_env, nm_env)
    chronics_name=list_all_chronics(env)
    
    print("chronics names are:")
    print(chronics_name)

    # id des lignes dont on doit se preoccuper des "overflows"
    id_interesting_lines = np.array([type(env).get_line_info(line_name=el)[0] for el in lines_we_care_about], dtype=int)
    
    # read all prioritized actions
    date_str = date.strftime('%Y%m%d')
    path_chronic=[path for path in env.chronics_handler.real_data.subpaths if date.strftime('%Y%m%d') in path][0]
    with open(os.path.join(path_chronic,"priorisations_sea_"+date_str+".json"), "r", encoding="utf-8") as f:
        priorisation_action_dict = json.load(f)

    # read non reconnectable lines
    lines_non_reconnectable = list(load_interesting_lines(path=path_chronic,file_name="non_reconnectable_lines.csv"))
    lines_should_not_reco_2024_and_beyond =DELETED_LINE_NAME

    lines_non_reconnectable+=lines_should_not_reco_2024_and_beyond

    #we get the first observation for the chronic at the desired date
    obs=get_first_obs_on_chronic(date,env,path_thermal_limits=path_chronic)

    print("the timestamp of the observation is: "+obs.get_time_stamp().strftime('%Y%m%d %HH:%MM'))
    assert( date.strftime('%Y%m%d')==obs.get_time_stamp().strftime('%Y%m%d'))

    # detect initially disconnected lines that might need to be reconnected
    reconnectabble_line_in_maintenance_at_start,maintenance_df =get_reconnectable_lines_disconnected_at_start(env,lines_non_reconnectable)

    #simulate contingencies
    actions=list(priorisation_action_dict[defaut].values())
    print(f"Found {len(actions)} prioritized actions for this contingency")

    overloaded_timesteps=run_contingency_on_scenario(obs,defaut,env,id_interesting_lines,reconnectabble_line_in_maintenance_at_start,maintenance_df)

    #test a remedial action
    print("applying remedial action in prioritized actions")
    action=env.action_space(actions[0])
    remaining_overloaded_timesteps=run_remedial_action(obs,defaut,lines_non_reconnectable,env,id_interesting_lines,overloaded_timesteps,action,reconnectabble_line_in_maintenance_at_start,maintenance_df)

    
