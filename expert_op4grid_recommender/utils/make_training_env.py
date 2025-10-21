from typing import Dict, Literal, Union
import os

import lightsim2grid
from lightsim2grid import LightSimBackend

import grid2op
from grid2op.Environment import Environment

from expert_op4grid_recommender.utils.make_env_utils import (make_backend_kwargs_data,
                                                             make_default_params)


    
def make_grid2op_training_env(path_env: str,
                              nm_env: str,
                              *,
                              allow_detachment=True,
                              params=None,
                              backend_loader_kwargs=None,
                              **bk_kwargs) -> Environment:
    backend_kwargs_data = make_backend_kwargs_data(loader_kwargs=backend_loader_kwargs, **bk_kwargs)
    backend = LightSimBackend(**backend_kwargs_data)
    path = os.path.join(path_env, nm_env)

    if params is None:
        params = make_default_params()
    env = grid2op.make(path,
                       backend=backend,
                       allow_detachment=allow_detachment,
                       n_busbar=backend_kwargs_data["loader_kwargs"]["n_busbar_per_sub"],
                       param=params
                       )
    return env


if __name__ == "__main__":
    path_env = '.'
    nm_env = "env_dijon_v2_training"
    env = make_grid2op_training_env(path_env, nm_env)
    print(env.name_line)
    print([el.slack_weight for el in env.backend._grid.get_generators()])
