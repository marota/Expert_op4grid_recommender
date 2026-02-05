import json
from typing import List, Dict
import warnings

import pandas as pd
import pypowsybl as pp
from pandas import DataFrame
from pypowsybl.network import Network

from expert_op4grid_recommender.utils import repas

def add_local_num(df: pd.DataFrame, buses: pd.DataFrame, bus_id_attr: str, voltage_level_id_attr: str, local_num_attr: str) -> pd.DataFrame:
    df = df.merge(buses, left_on=bus_id_attr, right_index=True, how='left')
    df[voltage_level_id_attr] = df[voltage_level_id_attr].fillna('')
    df[local_num_attr] = df[local_num_attr].fillna(-1) # means disconnected
    df[local_num_attr] = df[local_num_attr].astype(int)
    return df

def add_injection_local_num(inj_df: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
    return add_local_num(inj_df, buses, 'bus_id', 'voltage_level_id', 'local_num')

def add_branch_local_num(branches: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
    branches_with_num = add_local_num(branches, buses.rename(columns=lambda x: x + '_bus1'), 'bus1_id', 'voltage_level1_id', 'local_num_bus1')
    branches_with_num = add_local_num(branches_with_num, buses.rename(columns=lambda x: x + '_bus2'), 'bus2_id', 'voltage_level2_id', 'local_num_bus2')
    return branches_with_num

def check_connectivity(buses: DataFrame, loads: DataFrame, generators: DataFrame, shunts: DataFrame, action_id: str):
    buses_out_of_cc = buses['connected_component'] >= 1
    if buses_out_of_cc.any():
        warnings.warn(f"Action {action_id} break connectivity, some buses {list(buses[buses_out_of_cc].index)} are out of main component")

    loads_out_of_cc = loads['connected_component'] >= 1
    if loads_out_of_cc.any() or (loads['local_num'] == -1).any():
        warnings.warn(f"Action {action_id} break connectivity, some loads {list(loads[loads_out_of_cc].index)} are disconnected or out of main component")

    generators_out_of_cc = generators['connected_component'] >= 1
    if generators_out_of_cc.any() or (generators['local_num'] == -1).any():
        warnings.warn(f"Action {action_id} break connectivity, some generators {list(generators[generators_out_of_cc].index)} are disconnected or out of main component")

    shunts_out_of_cc = shunts['connected_component'] >= 1
    if shunts_out_of_cc.any() or (shunts['local_num'] == -1).any():
        warnings.warn(f"Action {action_id} break connectivity, some shunts {list(shunts[shunts_out_of_cc].index)} are disconnected or out of main component")

#def old_convert_to_grid2op_action(n: Network, action: repas.Action):
#    result = {}
#
#    result["description"] = action._description
#
#    # apply topology on a variant
#    n.clone_variant("InitialState", action._id, True)
#    n.set_working_variant(action._id)
#
#    for (voltage_level_id, switches) in action._switches_by_voltage_level.items():
#        for (switch_id, open) in switches.items():
#            n.update_switches(id=switch_id, open=open)
#
#    # reload needed dataframes with up-to-date data topology and bus local numbering
#    buses = n.get_buses(attributes=['voltage_level_id', 'connected_component'])
#    buses['local_num'] = buses.groupby('voltage_level_id').cumcount() + 1
#    loads = add_injection_local_num(n.get_loads(attributes=['bus_id']), buses)
#    generators = add_injection_local_num(n.get_generators(attributes=['bus_id']), buses)
#    shunts = add_injection_local_num(n.get_shunt_compensators(attributes=['bus_id']), buses)
#    branches = add_branch_local_num(n.get_branches(attributes=['bus1_id', 'voltage_level1_id', 'bus2_id', 'voltage_level2_id']), buses)
#
#    #check_connectivity(buses, loads, generators, shunts, action._id)
#
#    lines_or_id = {}
#    lines_ex_id = {}
#    loads_id = {}
#    generators_id = {}
#    shunts_id = {}
#
#    impacted_voltage_level_ids = action._switches_by_voltage_level.keys()
#    for (index, row) in branches.iterrows():
#        if row['voltage_level1_id'] in impacted_voltage_level_ids:
#            lines_or_id[row.name] = row['local_num_bus1']
#        if row['voltage_level2_id'] in impacted_voltage_level_ids:
#            lines_ex_id[row.name] = row['local_num_bus2']
#
#    for (index, row) in loads.iterrows():
#        if row['voltage_level_id'] in impacted_voltage_level_ids:
#            loads_id[row.name] = row['local_num']
#
#    for (index, row) in generators.iterrows():
#        if row['voltage_level_id'] in impacted_voltage_level_ids:
#            generators_id[row.name] = row['local_num']
#
#    for (index, row) in shunts.iterrows():
#        if row['voltage_level_id'] in impacted_voltage_level_ids:
#            shunts_id[row.name] = row['local_num']
#
#    set_bus = {}
#    set_bus['lines_or_id'] = lines_or_id
#    set_bus['lines_ex_id'] = lines_ex_id
#    set_bus['loads_id'] = loads_id
#    set_bus['generators_id'] = generators_id
#    set_bus['shunts_id'] = shunts_id
#
#    content = {}
#    result['content'] = content
#    content['set_bus'] = set_bus
#
#    # TODO load and generation modifications
#
#    n.remove_variant(action._id)
#
#    return result


from expert_op4grid_recommender.utils import repas


def add_local_num(df: pd.DataFrame, buses: pd.DataFrame, bus_id_attr: str,
                  voltage_level_id_attr: str, local_num_attr: str) -> pd.DataFrame:
    """Optimisé avec merge inplace si possible"""
    df = df.merge(buses, left_on=bus_id_attr, right_index=True, how='left')
    df[voltage_level_id_attr] = df[voltage_level_id_attr].fillna('')
    df[local_num_attr] = df[local_num_attr].fillna(-1).astype(int)
    return df


def add_injection_local_num(inj_df: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
    return add_local_num(inj_df, buses, 'bus_id', 'voltage_level_id', 'local_num')


def add_branch_local_num(branches: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
    """Optimisé : un seul rename au lieu de deux"""
    buses_bus1 = buses.rename(columns=lambda x: x + '_bus1')
    buses_bus2 = buses.rename(columns=lambda x: x + '_bus2')

    branches_with_num = add_local_num(branches, buses_bus1, 'bus1_id',
                                      'voltage_level1_id', 'local_num_bus1')
    branches_with_num = add_local_num(branches_with_num, buses_bus2, 'bus2_id',
                                      'voltage_level2_id', 'local_num_bus2')
    return branches_with_num


def convert_to_grid2op_action(n: Network, action: repas.Action):
    """
    Version optimisée avec :
    1. Set pour impacted_voltage_level_ids (O(1) lookup)
    2. Vectorisation des filtres pandas
    3. Éviter les itérations inutiles
    4. Construction directe des dictionnaires
    """
    result = {}
    result["description"] = action._description

    # Apply topology on a variant
    n.clone_variant("InitialState", action._id, True)
    n.set_working_variant(action._id)

    # OPTIMISATION 1: Appliquer tous les switches en une seule fois si possible
    for (voltage_level_id, switches) in action._switches_by_voltage_level.items():
        switch_ids = list(switches.keys())
        open_values = list(switches.values())
        n.update_switches(id=switch_ids, open=open_values)

    # Reload needed dataframes
    buses = n.get_buses(attributes=['voltage_level_id', 'connected_component'])
    buses['local_num'] = buses.groupby('voltage_level_id').cumcount() + 1

    loads = add_injection_local_num(n.get_loads(attributes=['bus_id']), buses)
    generators = add_injection_local_num(n.get_generators(attributes=['bus_id']), buses)
    shunts = add_injection_local_num(n.get_shunt_compensators(attributes=['bus_id']), buses)
    branches = add_branch_local_num(n.get_branches(
        attributes=['bus1_id', 'voltage_level1_id', 'bus2_id', 'voltage_level2_id']
    ), buses)

    # OPTIMISATION 2: Convertir en set pour O(1) lookup au lieu de O(n)
    impacted_voltage_level_ids = set(action._switches_by_voltage_level.keys())

    # OPTIMISATION 3: Filtrage vectorisé au lieu d'itération ligne par ligne
    # Pour les branches
    branches_or_mask = branches['voltage_level1_id'].isin(impacted_voltage_level_ids)
    branches_ex_mask = branches['voltage_level2_id'].isin(impacted_voltage_level_ids)

    lines_or_id = branches.loc[branches_or_mask, 'local_num_bus1'].to_dict()
    lines_ex_id = branches.loc[branches_ex_mask, 'local_num_bus2'].to_dict()

    # Pour les injections
    loads_mask = loads['voltage_level_id'].isin(impacted_voltage_level_ids)
    loads_id = loads.loc[loads_mask, 'local_num'].to_dict()

    generators_mask = generators['voltage_level_id'].isin(impacted_voltage_level_ids)
    generators_id = generators.loc[generators_mask, 'local_num'].to_dict()

    shunts_mask = shunts['voltage_level_id'].isin(impacted_voltage_level_ids)
    shunts_id = shunts.loc[shunts_mask, 'local_num'].to_dict()

    # OPTIMISATION 4: Construction directe du dictionnaire résultat
    result['content'] = {
        'set_bus': {
            'lines_or_id': lines_or_id,
            'lines_ex_id': lines_ex_id,
            'loads_id': loads_id,
            'generators_id': generators_id,
            'shunts_id': shunts_id
            },
    }

    n.remove_variant(action._id)

    return result

def create_dict_disco_reco_lines_disco(net,filter_voltage_levels=[400,24.,  15.,  20.,  33.,  10.]):
    dict_extra_disco_reco_actions = {}

    ###
    branches_df = net.get_branches()[["voltage_level1_id", "voltage_level2_id"]]
    vl_df = net.get_voltage_levels()
    branches_df["voltage_level2"] = vl_df.loc[branches_df["voltage_level2_id"]]["nominal_v"].values
    branches_df["voltage_level1"] = vl_df.loc[branches_df["voltage_level1_id"]]["nominal_v"].values


    for line in branches_df.index:
        #line disconnection action
        voltage_level_2=branches_df.loc[line]["voltage_level2"]
        voltage_level_1=branches_df.loc[line]["voltage_level1"]
        if voltage_level_1 not in filter_voltage_levels and voltage_level_2 not in filter_voltage_levels:
            dict_key="disco_"+line
            description="deconnection de l'ouvrage "+line
            content={'set_bus': {'lines_or_id':{line:-1},'lines_ex_id':{line:-1}}}
            dict_extra_disco_reco_actions[dict_key]={"description":description,"description_unitaire":description,"content":content}

            #line reconnection action
            dict_key="reco_"+line
            description="reconnection de l'ouvrage "+line+ " aux noeuds 1 a chaque extremite"
            content={'set_bus': {'lines_or_id':{line:1},'lines_ex_id':{line:1}}}
            dict_extra_disco_reco_actions[dict_key]={"description":description,"content":content}
        else:
            print("line filtered through voltage level: "+line)

    return dict_extra_disco_reco_actions

def get_all_switch_descriptions(switches_by_voltage_level):
    """
    Génère une description unique pour tous les switches.

    Returns:
        str: Description complète avec "et" entre les actions et le poste à la fin
    """
    descriptions = []
    voltage_level = None

    for vl, switches in switches_by_voltage_level.items():
        voltage_level = vl  # Garde le nom du poste
        for switch_name, switch_value in switches.items():
            action_type = "Ouverture" if switch_value else "Fermeture"
            descriptions.append(f"{action_type} {switch_name}")

    # Joindre avec "et" et ajouter le poste à la fin
    if descriptions:
        actions_str = " et ".join(descriptions)
        return f"{actions_str} dans le poste {voltage_level}"
    return ""

def convert_repas_actions_to_grid2op_actions(n: Network, actions: List[repas.Action]):
    result = {}
    for action in actions:
        g2o_action = convert_to_grid2op_action(n, action)#convert_to_grid2op_action(n, action)
        if g2o_action is not None:
            action_key=action._id + "_" + next(iter(action._switches_by_voltage_level))
            g2o_action["description_unitaire"] = get_all_switch_descriptions(action._switches_by_voltage_level)
            result[action_key] = g2o_action

    return result


def convert_to_grid2op_actions(n: Network, actions: Dict[str, List[repas.Action]]):
    result = {}
    for (contingency, actions) in actions.items():
        result[contingency] = convert_repas_actions_to_grid2op_actions(n, actions)
    return result