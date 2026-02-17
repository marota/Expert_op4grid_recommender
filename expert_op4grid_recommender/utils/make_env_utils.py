from typing import Dict, Literal, Union
import os
from packaging import version
from importlib.metadata import version as version_importlib

import pypowsybl as pp
import pypowsybl.loadflow as lf

try:
    import lightsim2grid
    _HAS_LIGHTSIM2GRID = True
except (ImportError, Exception):
    _HAS_LIGHTSIM2GRID = False

try:
    import grid2op
    from grid2op.Parameters import Parameters
    _HAS_GRID2OP = True
except (ImportError, Exception):
    _HAS_GRID2OP = False

MIN_GLOP_VERSION = version.parse("1.11.0")
MIN_LS_VERSION = version.parse("0.10.3")
MIN_PP2GRID_VERSION = version.parse("0.3.0")
MIN_PP_VERSION = version.parse("1.11.0")


if _HAS_GRID2OP and version.parse(grid2op.__version__) < MIN_GLOP_VERSION:
    raise RuntimeError(f"grid2op minimum version needed: {MIN_GLOP_VERSION}. Please "
                       f"upgrade it from pypi with:\n"
                       f'\t `pip install "Grid2Op>={MIN_GLOP_VERSION}"`')
if _HAS_LIGHTSIM2GRID and version.parse(lightsim2grid.__version__) < MIN_LS_VERSION:
    raise RuntimeError(f"lightsim2grid minimum version needed: {MIN_LS_VERSION}. Please "
                       f"upgrade it from pypi with:\n"
                       f'\t `pip install "LightSim2Grid>={MIN_LS_VERSION}"`')
try:
    _pp2grid_version = version_importlib("pypowsybl2grid")
    if version.parse(_pp2grid_version) < MIN_PP2GRID_VERSION:
        raise RuntimeError(f"pypowsybl2grid minimum version needed: {MIN_PP2GRID_VERSION}. Please "
                           f"upgrade it from pypi with:\n"
                           f'\t `pip install "pypowsybl2grid>={MIN_PP2GRID_VERSION}"`')
except ImportError:
    pass  # pypowsybl2grid is optional (only needed for grid2op backend)
if version.parse(version_importlib("pypowsybl")) < MIN_PP_VERSION:
    raise RuntimeError(f"pypowsybl minimum version needed: {MIN_PP_VERSION}. Please "
                       f"upgrade it from pypi with:\n"
                       f'\t `pip install "pypowsybl>={MIN_PP_VERSION}"`')
    
    
N_BUSBAR_PER_SUB = 12#7#6


LOADER_KWARGS = {"use_buses_for_sub": False,
                 "use_grid2op_default_names": False,
                 "reconnect_disco_gen": False,  # TODO
                 "reconnect_disco_load": False,  # TODO
                 "n_busbar_per_sub": N_BUSBAR_PER_SUB,
                #  "dist_slack_non_renew": True
                 "gen_slack_id": 'N.SE17GROUP.1'  # TODO consistent with debug_olf_ls.py
                 }


def make_backend_kwargs_data(loader_kwargs=None, **bk_kwargs) -> Dict[Literal["loader_method", "loader_kwargs"],
                                                                      Union[str, Dict[Literal['use_buses_for_sub',
                                                                                              "double_bus_per_sub",
                                                                                              "use_grid2op_default_names",
                                                                                              "reconnect_disco_gen",
                                                                                              "reconnect_disco_load",
                                                                                              "gen_slack_id"],
                                                                                      Union[str, bool]]]]:
    if loader_kwargs is None:
        loader_kwargs = LOADER_KWARGS
    backend_kwargs_data = dict(loader_method="pypowsybl",
                               loader_kwargs=loader_kwargs,
                               max_iter=30,  # maximum iteration for the newton raphson
                               dist_slack_non_renew=True,  # slack is on all non renewable generators
                               turned_off_pv=False,  # turn off generator are not pv
                               **bk_kwargs)
    return backend_kwargs_data


def make_default_params():
    if not _HAS_GRID2OP:
        raise ImportError("grid2op is required for make_default_params()")
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    params.MAX_LINE_STATUS_CHANGED = 9999
    params.MAX_SUB_CHANGED = 9999
    params.ENV_DOES_REDISPATCHING = False
    params.STOP_EP_IF_GEN_BREAK_CONSTRAINTS = False
    # other parameters
    params.NB_TIMESTEP_COOLDOWN_SUB = 0
    params.NB_TIMESTEP_COOLDOWN_LINE = 0
    params.HARD_OVERFLOW_THRESHOLD = 9999.
    params.NB_TIMESTEP_RECONNECTION = 0
    params.IGNORE_MIN_UP_DOWN_TIME = True
    params.ENV_DC = False
    params.FORECAST_DC = False
    params.NB_TIMESTEP_OVERFLOW_ALLOWED = 9999
    params.ALLOW_DISPATCH_GEN_SWITCH_OFF = False
    return params


def create_olf_rte_parameter() -> pp.loadflow.Parameters:
    return pp.loadflow.Parameters(read_slack_bus=False,
                                  write_slack_bus=False,
                                  voltage_init_mode=pp.loadflow.VoltageInitMode.DC_VALUES,
                                  transformer_voltage_control_on=True,
                                  use_reactive_limits=True,
                                  shunt_compensator_voltage_control_on=True,
                                  phase_shifter_regulation_on=True,
                                  distributed_slack=True,
                                  dc_use_transformer_ratio=False,
                                  twt_split_shunt_admittance=True,
                                  provider_parameters={"useActiveLimits": "true",
                                                       "svcVoltageMonitoring": "false",
                                                       "voltageRemoteControl": "false",
                                                       "writeReferenceTerminals": "false",
                                                       "slackBusSelectionMode" : "NAME",
                                                       "slackBusesIds" : "N.SE1P1_0#0"})
