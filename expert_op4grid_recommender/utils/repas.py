import json
from typing import Optional
from typing import List, Dict, Tuple, Set, Callable

import pandas as pd
from pypowsybl.network import Network


class Action:
    def __init__(self,
                 id: str,
                 horizon: str,
                 description: str,
                 switches_by_voltage_level: Dict[str, Dict[str, bool]],
                 loads_by_id: Dict[str, Tuple[float, float]],
                 generators_by_id: Dict[str, Tuple[float, bool]]):
        super().__init__()
        self._id = id
        self._horizon = horizon
        self._description = description
        self._switches_by_voltage_level = switches_by_voltage_level
        self._loads_by_id = loads_by_id
        self._generators_by_id = generators_by_id

    def __repr__(self):
        return f"Action(id={self._id}, horizon={self._horizon} switches={self._switches_by_voltage_level}, loads={self._loads_by_id}, generators={self._generators_by_id})"


def _parse_action(actions,
                  switches_by_voltage_level: Dict[str, Dict[str, bool]],
                  loads_by_id: Dict[str, Tuple[float, float]],
                  generators_by_id: Dict[str, Tuple[float, bool]],
                  voltage_level_ids: Set[str],
                  switch_ids: Set[str],
                  load_ids: Set[str],
                  generator_ids: Set[str]):
    for action in actions:
        if action['actionType'] == "CompositeAction":
            _parse_action(action['actions'], switches_by_voltage_level, loads_by_id, generators_by_id, voltage_level_ids, switch_ids, load_ids, generator_ids)
        elif action['actionType'] == "SwitchOperation":
            voltage_level_id = action['assetId']
            switch_id = action['name']
            opened = action['targetStateOpened']
            if switch_id and voltage_level_id in voltage_level_ids:
                # needed to match to switch ID of Repas DB
                cvg_switch_id = voltage_level_id + "_" + switch_id
                if cvg_switch_id in switch_ids:
                    switches_by_voltage_level.setdefault(voltage_level_id, {})
                    switches_by_voltage_level[voltage_level_id][cvg_switch_id] = opened  # because this is the way switch are built are IIDM export of CVG
        elif action['actionType'] == "LoadModification":
            load_id = action['assetId']
            if load_id in load_ids:
                p = action['activeValue']
                q = action['reactiveValue']
                loads_by_id[load_id] = (p, q)
        elif action['actionType'] == "GeneratorModification":
            gen_id = action['assetId']
            if gen_id in generator_ids:
                p = action['selectedPower']
                delta = action['delta']
                # TODO the other attributes like connection status, voltage regulation etc
                generators_by_id[gen_id] = (p, delta)
        elif action['actionType'] == "GeneratorGroupVariation":
            # TODO
            pass
        elif action['actionType'] == "LoadGroupVariation":
            # TODO
            pass
        elif action['actionType'] == "GeneratorGroupVariation":
            # TODO
            pass
        elif action['actionType'] == "LoadShedding":
            # TODO
            pass
        elif action['actionType'] == "LoadSheddingElement":
            # TODO
            pass
        elif action['actionType'] == "PSTRegulation":
            # TODO
            pass
        elif action['actionType'] == "PSTShunt":
            # TODO
            pass


def _create_action(id: str, horizon: str, description: str, actions, parsed_actions: List[Action], generators_ids: Set[str], loads_ids: Set[str], switch_ids: Set[str], voltage_level_ids: Set[str]):
    switches_by_voltage_level = {}
    loads_by_id = {}
    generators_by_id = {}
    _parse_action(actions, switches_by_voltage_level, loads_by_id, generators_by_id, voltage_level_ids,
                        switch_ids, loads_ids, generators_ids)
    if len(switches_by_voltage_level) > 0 or len(loads_by_id) > 0 or len(generators_by_id) > 0:
        parsed_actions.append(Action(id, horizon, description, switches_by_voltage_level, loads_by_id, generators_by_id))


def parse_json(file: str, n: Network, voltage_level_filter: Optional[Callable[[(str, pd.Series)], bool]]) -> List[Action]:
    # load elements ids to later filter DB on elements present in the network
    voltage_level_ids = set()
    for (index, row) in n.get_voltage_levels().iterrows():
        if voltage_level_filter is None or voltage_level_filter((index, row)):
            voltage_level_ids.add(index)
    switch_ids = set(n.get_switches(attributes=[]).index)
    load_ids = set(n.get_loads(attributes=[]).index)
    generator_ids = set(n.get_generators(attributes=[]).index)

    repas_actions = []

    with open(file, 'r') as file:
        data = json.load(file)
        contents = data['content']
        for content in contents:
            id = content['id']
            description = content['content']
            rules_list = content['rulesList']
            # TODO filter on violation elements
            for rules in rules_list:
                preventive_action = rules['preventiveAction']
                if preventive_action is not None:
                    _create_action(id, "preventive", description, preventive_action['actions'], repas_actions, generator_ids, load_ids, switch_ids, voltage_level_ids)
                curative_action = rules['curativeAction']
                if curative_action is not None:
                    _create_action(id, "curative", description, curative_action['actions'], repas_actions, generator_ids, load_ids, switch_ids, voltage_level_ids)

    return repas_actions
