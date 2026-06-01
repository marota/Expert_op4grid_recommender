"""Parser for REPAS action JSON files.

REPAS is RTE's internal format describing the remedial action catalogue. The
single entry point :func:`parse_json` walks the JSON tree, filters the
entries against elements actually present in a pypowsybl network, and
returns a list of :class:`Action` domain objects suitable for downstream
scoring.
"""

import json
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
from pypowsybl.network import Network


class Action:
    """A single REPAS action, decomposed by asset family.

    Attributes are accessed through the underscore-prefixed names to keep
    them read-only by convention; the constructor copies the dicts in as-is
    without further validation.
    """

    def __init__(
        self,
        id: str,
        horizon: str,
        description: str,
        switches_by_voltage_level: Dict[str, Dict[str, bool]],
        loads_by_id: Dict[str, Tuple[float, float]],
        generators_by_id: Dict[str, Tuple[float, bool]],
        pst_by_id: Dict[str, int],
    ) -> None:
        super().__init__()
        self._id = id
        self._horizon = horizon
        self._description = description
        self._switches_by_voltage_level = switches_by_voltage_level
        self._loads_by_id = loads_by_id
        self._generators_by_id = generators_by_id
        self._pst_by_id = pst_by_id

    def __repr__(self) -> str:
        return (
            f"Action(id={self._id}, horizon={self._horizon} "
            f"switches={self._switches_by_voltage_level}, loads={self._loads_by_id}, "
            f"generators={self._generators_by_id}, pst={self._pst_by_id})"
        )


def _parse_action(
    actions: List[dict],
    switches_by_voltage_level: Dict[str, Dict[str, bool]],
    loads_by_id: Dict[str, Tuple[float, float]],
    generators_by_id: Dict[str, Tuple[float, bool]],
    pst_by_id: Dict[str, int],
    voltage_level_ids: Set[str],
    switch_ids: Set[str],
    load_ids: Set[str],
    generator_ids: Set[str],
    pst_ids: Set[str],
    n: Network,
) -> None:
    """Recursively accumulate a REPAS action tree into per-family dicts.

    Only actions whose target element exists in the provided id sets are
    kept; unknown or unhandled action types are silently ignored — see
    marota/expert_op4grid_recommender#79 for the full backlog.
    """
    for action in actions:
        if action['actionType'] == "CompositeAction":
            _parse_action(action['actions'], switches_by_voltage_level, loads_by_id, generators_by_id, pst_by_id, voltage_level_ids, switch_ids, load_ids, generator_ids, pst_ids, n)
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
                # Missing attributes (connection status, voltage regulation, Q setpoint, ...)
                # tracked in marota/expert_op4grid_recommender#79.
                generators_by_id[gen_id] = (p, delta)
        elif action['actionType'] == "GeneratorGroupVariation":
            # Unhandled REPAS action type, tracked in marota/expert_op4grid_recommender#79.
            pass
        elif action['actionType'] == "LoadGroupVariation":
            # Unhandled REPAS action type, tracked in marota/expert_op4grid_recommender#79.
            pass
        elif action['actionType'] == "LoadShedding":
            # Unhandled REPAS action type, tracked in marota/expert_op4grid_recommender#79.
            pass
        elif action['actionType'] == "LoadSheddingElement":
            # Unhandled REPAS action type, tracked in marota/expert_op4grid_recommender#79.
            pass
        elif action['actionType'] == "PSTRegulation":
            pst_id = action['assetId']
            if pst_id in pst_ids:
                # print(f"DEBUG: Found PSTRegulation for {pst_id}")
                pass
        elif action['actionType'] == "PSTShunt":
            pst_id = action['assetId']
            if pst_id in pst_ids:
                tap_value = action['value']
                is_delta = action.get('delta', False)
                if is_delta:
                    # Variation: resolve to absolute using current tap from network
                    current_tap = n.get_phase_tap_changers().loc[pst_id, 'tap']
                    pst_by_id[pst_id] = int(current_tap + tap_value)
                else:
                    # Target: use directly
                    pst_by_id[pst_id] = int(tap_value)
            else:
                # print(f"DEBUG: PSTShunt assetId {pst_id} NOT found in pst_ids")
                pass


def _create_action(
    id: str,
    horizon: str,
    description: str,
    actions: List[dict],
    parsed_actions: List[Action],
    generators_ids: Set[str],
    loads_ids: Set[str],
    switch_ids: Set[str],
    voltage_level_ids: Set[str],
    pst_ids: Set[str],
    n: Network,
) -> None:
    """Materialise a single :class:`Action` from a REPAS node and append it to ``parsed_actions``.

    No-op if the action resolves to an empty payload after filtering
    (i.e. every target element was unknown to the network).
    """
    switches_by_voltage_level: Dict[str, Dict[str, bool]] = {}
    loads_by_id = {}
    generators_by_id = {}
    pst_by_id = {}
    _parse_action(actions, switches_by_voltage_level, loads_by_id, generators_by_id, pst_by_id, voltage_level_ids,
                  switch_ids, loads_ids, generators_ids, pst_ids, n)
    if len(switches_by_voltage_level) > 0 or len(loads_by_id) > 0 or len(generators_by_id) > 0 or len(pst_by_id) > 0:
        parsed_actions.append(Action(id, horizon, description, switches_by_voltage_level, loads_by_id, generators_by_id, pst_by_id))


def parse_json(
    file: str,
    n: Network,
    voltage_level_filter: Optional[Callable[[Tuple[str, pd.Series]], bool]],
) -> List[Action]:
    """Read a REPAS catalogue from ``file`` and return the parsed actions.

    Parameters
    ----------
    file:
        Path to a REPAS JSON file.
    n:
        Reference network used to filter out actions targeting elements that
        do not exist in this study.
    voltage_level_filter:
        Optional predicate ``(voltage_level_id, row) -> bool`` restricting
        the voltage levels whose switch actions are kept. ``None`` keeps all.
    """
    # load elements ids to later filter DB on elements present in the network
    voltage_level_ids = set()
    for (index, row) in n.get_voltage_levels().iterrows():
        if voltage_level_filter is None or voltage_level_filter((index, row)):
            voltage_level_ids.add(index)
    switch_ids = set(n.get_switches(attributes=[]).index)
    load_ids = set(n.get_loads(attributes=[]).index)
    generator_ids = set(n.get_generators(attributes=[]).index)
    pst_ids = set(n.get_phase_tap_changers(attributes=[]).index)

    repas_actions = []

    with open(file, 'r') as file:
        data = json.load(file)
        contents = data['content']
        for content in contents:
            id = content['id']
            description = content['content']
            rules_list = content['rulesList']
            # Violation-element filtering tracked in marota/expert_op4grid_recommender#79.
            for rules in rules_list:
                preventive_action = rules['preventiveAction']
                if preventive_action is not None:
                    _create_action(id, "preventive", description, preventive_action['actions'], repas_actions, generator_ids, load_ids, switch_ids, voltage_level_ids, pst_ids, n)
                curative_action = rules['curativeAction']
                if curative_action is not None:
                    _create_action(id, "curative", description, curative_action['actions'], repas_actions, generator_ids, load_ids, switch_ids, voltage_level_ids, pst_ids, n)

    return repas_actions
