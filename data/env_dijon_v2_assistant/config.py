from datetime import datetime, timedelta

from grid2op.Reward import L2RPNReward
from grid2op.Rules import DefaultRules, AlwaysLegal
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Chronics.handlers import PerfectForecastHandler,CSVHandler, CSVMaintenanceHandler, JSONInitStateHandler
from grid2op.Chronics import FromHandlers

from grid2op.Action import PlayableAction, DontAct
from grid2op.Observation import CompleteObservation

from lightsim2grid import LightSimBackend


class ActionAF2024(PlayableAction):
    authorized_keys = {
        "set_line_status",
        # "change_bus",  # more than 2 busbar so deactivated
        "set_bus",
        "change_bus",
        # "redispatch",
        # "set_storage",
        "curtail",
        # "raise_alert",
        "detach_load",  # new in 1.11.0
        "detach_gen",  # new in 1.11.0
        "detach_storage",  # new in 1.11.0
    }

    attr_list_vect = [
        "_set_line_status",
        "_switch_line_status",
        "_set_topo_vect",
        "_change_bus_vect",
        # "_redispatch",
        # "_storage_power",
        "_curtail",
        # "_raise_alert",
        "_detach_load",  # new in 1.11.0
        "_detach_gen",  # new in 1.11.0
        "_detach_storage",  # new in 1.11.0
    ]
    attr_list_set = set(attr_list_vect)
    pass


class ObservationAF2024(CompleteObservation):
    attr_list_vect = [
        "year",
        "month",
        "day",
        "hour_of_day",
        "minute_of_hour",
        "day_of_week",
        "gen_p",
        "gen_q",
        "gen_v",
        "load_p",
        "load_q",
        "load_v",
        "p_or",
        "q_or",
        "v_or",
        "a_or",
        "p_ex",
        "q_ex",
        "v_ex",
        "a_ex",
        "rho",
        "line_status",
        # "timestep_overflow", #Disparait car plus de notion temporelle intrinsèque ?
        "topo_vect",
        # "time_before_cooldown_line", #Disparait car plus de notion temporelle intrinsèque ?
        # "time_before_cooldown_sub", #Disparait car plus de notion temporelle intrinsèque ?
        # "time_next_maintenance", #A enlever si on met pas les maintenances dans la class choisie pour gridvalueClass
        # "duration_next_maintenance", #A enlever si on met pas les maintenances dans la class choisie pour gridvalueClass
        # "target_dispatch", #Pas d'action dispatch
        "actual_dispatch",
        # "storage_charge", #Pas de storage
        # "storage_power_target",
        # "storage_power",
        "gen_p_before_curtail",
        "curtailment",
        "curtailment_limit",
        "curtailment_limit_effective",
        # "_shunt_p", # Pas de shunt sur la zone
        # "_shunt_q",
        # "_shunt_v",
        # "_shunt_bus",
        "current_step",
        "max_step",
        "delta_time",
        "gen_margin_up",  # A conserver pour expliquer qu'un écretement n'ai pas pu se faire complètement ?
        "gen_margin_down",  # A conserver pour expliquer qu'un écretement n'ai pas pu se faire complètement ?
        # line alert (starting grid2Op 1.9.1, for compatible envs)
        # "active_alert", #Pas d'alerte simulée
        # "attack_under_alert",
        # "time_since_last_alert",
        # "alert_duration",
        # "total_number_of_alert",
        # "time_since_last_attack",
        # "was_alert_used_after_attack",
        
        # "slack" (>= 1.11.0)
        "gen_p_delta",
        # detachment (>= 1.11.0)
        "load_detached",
        "gen_detached",
        "storage_detached",
        "load_p_detached",
        "load_q_detached",
        "gen_p_detached",
        "storage_p_detached",
    ]
    attr_list_json = [
        "_thermal_limit",
        "support_theta",
        "theta_or",
        "theta_ex",
        "load_theta",
        "gen_theta",
        # "storage_theta" #no storage
    ]
    attr_list_set = set(attr_list_vect)


th_lim = {"C.FOUL31MERVA": 252.0, "H.PAUY762": 1577.0, "COUCHL61CPVAN": 805.0, "CPVANL31RIBAU": 429.0,
          "H.PAUL61ZCUR5": 855.0, "AUXONL31TILLE": 449.0, "AISERL31MAGNY": 429.0, "GENISY612": 9999.0,
          "C.SAUL31MAGNY": 429.0, "GROSNY762": 1576.0, "GENISY615": 9999.0, "BOISSL61GEN.P": 673.0,
          "GEVREL32N.GEO": 429.0, "CHALOL32FTAIN": 395.0, "BEON L31P.SAO": 449.0, "GEN.PL62GENIS": 9999.0,
          "GROSNY761": 1576.0, "C.REGL61ZMAGN": 621.0, "NAVILL31P.SAO": 429.0, "PYMONY631": 700.0,
          "GEN.PL61IZERN": 819.0, "H.PAUL71VIELM": 3129.0, "CORGOL32MTAGN": 224.0, "BUGEYY715": 9999.0,
          "B.ROCL31VIELM": 382.0, "CREYSL72GEN.P": 2470.0, "B.ROCL32VELAR": 429.0, "FLEYRL61VOUGL": 808.0,
          "BXNE5L31VOSNE": 429.0, "COUCHY631": 1000.0, "CPVANY631": 700.0, "GEN.PL66GENIS": 9999.0, "N.SE5Y711": 9999.0,
          "N.SE5Y712": 9999.0, "BUGEYL74SSV.O": 9999.0, "CUISEL31VOUGL": 429.0, "BXNE L31BXNE5": 429.0,
          "BXNE L32BXNE5": 558.0, "MAGNYY633": 978.0, "CIZE L61FLEYR": 1006.0, "BXNE5L31MTAGN": 453.0,
          "GENISY616": 9999.0, "GENISY611": 9999.0, "GEN.PL64GENIS": 9999.0, "C.REGL61VIELM": 801.0,
          "PERRIL32VELAR": 382.0, "VIELMY763": 797.0, "BXNE5L32CORGO": 453.0, "BEON L31CPVAN": 224.0,
          "BUGEYL72SSV.O": 9999.0, "B.ROCL31VELAR": 382.0, "GEN.PL71VIELM": 2090.0, "GEN.PL61GENIS": 9999.0,
          "MEURSL32MTAGN": 429.0, "PYMONL31SAISS": 313.0, "CHAGNL31MTAGN": 224.0, "VOUGLY612": 9999.0,
          "BOCTOL72M.SEI": 3292.0, "MERVAL31SSUSU": 447.0, "MACONL61ZJOUX": 819.0, "B.ROCL32VIELM": 382.0,
          "BOCTOL71N.SE5": 3292.0, "GEN.PL65GENIS": 9999.0, "CHAGNL31MOLLE": 395.0, "GENISY614": 9999.0,
          "VIELMY771": 9499.0, "GENLIL31MAGNY": 429.0, "BUGEYY713": 9999.0, "AUXONL31COLLO": 317.0,
          "CUISEL31G.CHE": 429.0, "PYMONY632": 700.0, "GEN.PY762": 1463.0, "FTAINL32MEURS": 429.0, "GENISY613": 9999.0,
          "CPVANY633": 700.0, "CHALOL31LOUHA": 194.0, "VOUGLY611": 9999.0, "FRON5L31LOUHA": 252.0,
          "MAGNYL61ZMAGN": 805.0, "CHALOL62GROSN": 1330.0, "M.SEIL71VIELM": 2673.0, "COMMUL61H.PAU": 803.0,
          "COUCHL31ROMEL": 429.0, "COUCHL61VIELM": 1045.0, "GEN.PL73VIELM": 1174.0, "CHALOY631": 1625.0,
          "CRENEL71M.SEI": 1721.0, "GEN.PL72VIELM": 2090.0, "CHALOL61GROSN": 1330.0, "CHALOL31CHAL5": 395.0,
          "COMMUL61VIELM": 801.0, "GEN.PL63GENIS": 9999.0, "SAISSL31VOUGL": 413.0, "BUGEYL75SSV.O": 9999.0,
          "LOUHAL31PYMON": 280.0, "CHALOY632": 1625.0, "GROSNL61ZCUR5": 855.0, "COLLOL31GENLI": 317.0,
          "BOISSL61ZJOUX": 819.0, "BUGEYL73SSV.O": 9999.0, "CREYSL71SSV.O": 2799.0, "BUGEYY712": 9999.0,
          "VOUGLY631": 978.0, "C.REGY633": 1662.0, "FRON5L31G.CHE": 224.0, "COUCHL31VOSNE": 429.0,
          "CPVANL61PYMON": 713.0, "C.REGL62VIELM": 805.0, "GROSNY771": 9499.0, "C.REGY631": 1662.0,
          "CREYSL71GEN.P": 2799.0, "GUEUGL61H.PAU": 679.0, "AUXONL31RIBAU": 429.0, "AISERL31RONCI": 224.0,
          "PERRIL31ROMEL": 429.0, "TAVA5Y612": 9999.0, "CRENEL71VIELM": 1600.0, "CREYSL72SSV.O": 2799.0,
          "BXNE L31MTAGN": 429.0, "CPVANL61TAVAU": 693.0, "CHALOY633": 1625.0, "BUGEYY714": 9999.0,
          "CURTIL61GROSN": 855.0, "CHALOL61CPVAN": 609.0, "CHAL5L31MOLLE": 447.0, "CPVANL61ZMAGN": 621.0,
          "GROSNL61GUEUG": 773.0, "BXNE5L32MTAGN": 558.0, "VIELMY634": 978.0, "CURTIL61ZCUR5": 855.0,
          "CPVANY632": 700.0, "C.REGL31ZCRIM": 429.0, "KIR  L31PERRI": 382.0, "BOCTOL71M.SEI": 3292.0,
          "M.SEIL72VIELM": 2470.0, "GEN.PY771": 9499.0, "VOUGLY632": 1000.0, "GEN.PY761": 1576.0, "VIELMY762": 1576.0,
          "LOUHAL31SSUSU": 447.0, "COUCHY632": 1625.0, "GROSNL71SSV.O": 1820.0, "GROSNL61MACON": 587.0,
          "PYMONL61VOUGL": 669.0, "C.SAUL31ZCRIM": 429.0, "H.PAUY772": 9499.0, "BOCTOL72N.SE5": 3292.0,
          "GEN.PL61VOUGL": 764.0, "P.SAOL31RONCI": 224.0, "VIELMY761": 797.0, "VIELMY635": 978.0,
          "KIR  L31VELAR": 382.0, "GROSNL71VIELM": 2031.0, "COUCHL32PERRI": 429.0, "C.FOUL31NAVIL": 447.0,
          "CIZE L61IZERN": 1006.0, "TAVA5Y611": 9999.0, "H.PAUL71SSV.O": 2185.0, "CORGOL32N.GEO": 224.0,
          "COUCHL32GEVRE": 429.0}

config = {
    "backend": LightSimBackend,  # décider des paramètres +issue sur Git
    "action_class": ActionAF2024,
    "observation_class": ObservationAF2024,
    "reward_class": L2RPNReward,
    "gamerules_class": AlwaysLegal,
    # limite le nb de changements de connections powerlines/substations à 1 pas de temps -> ajouter les maintenances?
    "chronics_class": Multifolder,
    "data_feeding_kwargs": {"gridvalueClass": FromHandlers,
                            "gen_p_handler": CSVHandler("prod_p"),
                            "load_p_handler": CSVHandler("load_p"),
                            "gen_v_handler": CSVHandler("prod_v"),
                            "load_q_handler": CSVHandler("load_q"), #modifier pour les maintenances
                            "maintenance_handler": CSVMaintenanceHandler("maintenance"),
                            "init_state_handler": JSONInitStateHandler("init_state"),
                            "h_forecast": [i * 30 for i in range(1, 48)],  # Every 30 mins ahead up for 47 timesteps
                            "time_interval": timedelta(minutes=30),
                            "gen_p_for_handler": PerfectForecastHandler("prod_p_forecasted"),
                            "gen_v_for_handler": PerfectForecastHandler("prod_v_forecasted"),
                            "load_p_for_handler": PerfectForecastHandler("load_p_forecasted"),
                            "load_q_for_handler": PerfectForecastHandler("load_q_forecasted"),

                            },
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "thermal_limits": th_lim,

    'opponent_attack_cooldown': 99999999,  # deactivate opponent
    'opponent_attack_duration': 0,
    'opponent_budget_per_ts': 0.0,
    'opponent_init_budget': 0.0,
    'opponent_action_class': DontAct,
    'opponent_class': BaseOpponent,
    'opponent_budget_class': NeverAttackBudget
}
