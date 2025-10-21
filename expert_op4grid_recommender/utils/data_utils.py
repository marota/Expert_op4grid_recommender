import copy
from datetime import datetime
from typing import Dict
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.typing_variables import RESET_OPTIONS_TYPING


class StateInfo:
    def __init__(self):
        #: the datetime (as a string) of the observation
        self.obs_datetime = None
        #: the name of the line disconnected for n-1
        self.n1_name = None
        #: the id of the line disconnected for n-1 (depends on the backend)
        self.n1_id = None
        #: the name of the "lines of interest" that should be on overflow
        self.overflow_names = None
        #: the origin bus id to which the n-1 was before being disconnected
        self.init_line_or_id = None
        #: the extremity bus id to which the n-1 was before being disconnectd
        self.init_line_ex_id = None
        #: the name of each line that cannot be reconnected 
        self.should_not_reco = None
        #: the state used to initialize the environment to the proper state
        self.init_state : RESET_OPTIONS_TYPING = None
        #: backend type
        self.backend_type = None
        
    @classmethod
    def from_init_state(cls, env: Environment, init_state: Dict, del_attr=None):
        res = cls()
        res.obs_datetime = init_state["datetime"]
        if "n1_name" in init_state:
            res.n1_name = init_state["n1_name"]
            n1_id = type(env).get_line_info(line_name=res.n1_name)[0]
            res.n1_id = n1_id
        res.overflow_names = init_state["overflow_names"]
        if "init_lines_or_id" in init_state:
            res.init_line_or_id = init_state["init_lines_or_id"]
        else:
            if "n1_name" in init_state:
                raise RuntimeError("Impossible to load a state: the init state of "
                                   "the disconnected n-1 is not saved (or side)")
        if "init_lines_ex_id" in init_state:
            res.init_line_ex_id = init_state["init_lines_ex_id"]
        else:
            if "n1_name" in init_state:
                raise RuntimeError("Impossible to load a state: the init state of "
                                   "the disconnected n-1 is not saved (ex side)")
        if "should_not_reco" in init_state:
            # information was available, so I use it
            res.should_not_reco = set(copy.deepcopy(init_state["should_not_reco"]))
        else:
            # I suppose that if a line was disconnected then it should not be reconnected
            res.should_not_reco = set([nm for nm, stat_ in init_state["init state"]["set_line_status"].items() 
                                   if int(stat_) == -1])
        if not res.n1_name in res.should_not_reco:
            res.should_not_reco.add(res.n1_name)
        if "backend_type" in init_state:
            res.backend_type = str(init_state["backend_type"])  
            
        if del_attr is None:
            # all attributes are deleted from init_state
            del init_state["datetime"]
            if "n1_id_lightsim2grid" in init_state:
                del init_state["n1_id_lightsim2grid"]
            if "n1_name" in init_state:
                del init_state["n1_name"]
            del init_state["overflow_names"]
            if "init_lines_or_id" in init_state:
                del init_state["init_lines_or_id"]
            if "init_lines_ex_id" in init_state:
                del init_state["init_lines_ex_id"]
            if "should_not_reco" in init_state:
                del init_state["should_not_reco"]
            if "backend_type" in init_state:
                del init_state["backend_type"]
            if "has_computed_n1" in init_state:
                del init_state["has_computed_n1"]
            if "proj_errors" in init_state:
                del init_state["proj_errors"]
        res.init_state = init_state
        res.init_state["init datetime"] = datetime.strptime(res.obs_datetime, "%Y%m%d-%H%M")
        return res
    
    def set_env_state(self, env : Environment) -> BaseObservation:
        """
        .. danger::
            Do not use asynchronously `state.set_env_state(env)` for different states with the same
            environment !

        Args:
            env (Environment): The grid2op environment to set

        Returns:
            BaseObservation: the grid2op observation corresponding to the input state
        """
        return env.reset(options=self.init_state)
