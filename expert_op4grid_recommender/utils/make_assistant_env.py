from typing import Dict, Literal, Union
import os

import grid2op
from grid2op.Environment import Environment
from grid2op.Backend import Backend
import logging
from pypowsybl2grid import PyPowSyBlBackend

from expert_op4grid_recommender.utils.make_env_utils import (N_BUSBAR_PER_SUB,
                                                             make_default_params,
                                                             create_olf_rte_parameter)


def create_pypowsybl_backend(n_busbar_per_sub,
                             check_isolated_and_disconnected_injections) -> Backend:
    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('powsybl').setLevel(logging.ERROR)
    logging.getLogger('pypowsybl2grid').setLevel(logging.ERROR)
    lf_parameters = create_olf_rte_parameter()
    return PyPowSyBlBackend(n_busbar_per_sub=n_busbar_per_sub,
                            check_isolated_and_disconnected_injections=check_isolated_and_disconnected_injections,
                            lf_parameters=lf_parameters)


def make_grid2op_assistant_env(path_env,
                                nm_env,
                                *,
                                allow_detachment=True,
                                params=None,
                                n_busbar=N_BUSBAR_PER_SUB) -> Environment:
    backend = create_pypowsybl_backend(n_busbar_per_sub=n_busbar,
                                       check_isolated_and_disconnected_injections=False)
    
    path = os.path.join(path_env, nm_env)
    if params is None:
        params = make_default_params()
    env = grid2op.make(path,
                       backend=backend,
                       allow_detachment=allow_detachment,
                       n_busbar=n_busbar,
                       param=params
                       )
    return env


if __name__ == "__main__":
    path_env = '.'
    nm_env = "env_dijon_v2_assistant"
    env = make_grid2op_assistant_env(path_env, nm_env)
    print(env.name_line)
    print("chronics in env")
    print(env.chronics_handler.real_data.subpaths)

    # Get the initial observation
    obs = env.reset()

    # Define the substation ID (replace with the correct one)
    substation_id = 2  # Example substation index

    # Get objects connected to the substation
    connected_objects = obs.get_obj_connect_to(substation_id=substation_id)

    # Print the result
    print("Objects connected to substation", substation_id, ":", connected_objects)
