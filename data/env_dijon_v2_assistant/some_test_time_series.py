import os
import pandas as pd
import numpy as np
import pypowsybl as pp
from some_tests_grid_iidm import OLF_PARAMS

path = os.path.abspath(os.path.split(__file__)[0])
gridnm = "grid.xiidm"
ts_dir_nm = "chronics"
NB_FORECAST = 47
USE_V = False


def check_genv_setpoit(generators, gen_v, gen_v_for):
    print("Check voltage setpoint...")
    mask_gen_v_0 = gen_v == 0.
    cols_gen_v_0 = mask_gen_v_0.columns[mask_gen_v_0.any()]
    for gen_nm in cols_gen_v_0:
        gen_el = generators.loc[gen_nm]
        if gen_el["voltage_regulator_on"] and gen_el["connected"]:
            print(f"\tERROR: generator {gen_nm} is connected, regulating voltage AND with a target_v of 0. kV")
        else:
            print(f"\t\tWARNING: generator {gen_nm} has a target_v of 0. kV which will be a problem once connected !")
    print()
    mask_gen_v_0 = gen_v_for == 0.
    cols_gen_v_0 = mask_gen_v_0.columns[mask_gen_v_0.any()]
    for gen_nm in cols_gen_v_0:
        gen_el = generators.loc[gen_nm]
        if gen_el["voltage_regulator_on"] and gen_el["connected"]:
            print(f"\tERROR: generator {gen_nm} is connected, regulating voltage AND with a target_v of 0. kV in the forecast")
        else:
            print(f"\t\tWARNING: generator {gen_nm} has a target_v of 0. kV which will be a problem once connected in the forecast!")
    print("... done\n")


def check_lf_pypowsybl_env(grid, load_p, load_q, gen_p, gen_v, use_v=False):
    print(f"Check powerflow (OLF) for the environment time serie...")
    if use_v:
        print(f"\tUsing generator voltage setpoint from the time series")
    else:
        print(f"\tREMOVING generator voltage setpoint from the time series")
        
    i = 0
    for i in range(load_p.shape[0]):
        print(f"\tStudying env time series for step {i}")
        df_load = pd.DataFrame(index=load_p.columns, data={"p0": load_p.iloc[i].values, "q0": load_q.iloc[i].values})
        grid.update_loads(df_load)
        data_ = {"target_p": gen_p.iloc[i].values}
        if use_v:
            data_["target_v"] = gen_v.iloc[i].values
        df_gen = pd.DataFrame(index=gen_p.columns, data=data_)
        grid.update_generators(df_gen)
        resdc = pp.loadflow.run_dc(grid)
        if not resdc:
            print("\t ERROR: OLF diverges in DC")
        res_standar = pp.loadflow.run_ac(grid)
        if not resdc:
            print("\t ERROR: OLF diverges in AC (with default params)")
        res_likels = pp.loadflow.run_ac(grid, parameters=OLF_PARAMS)
        if not resdc:
            print("\t ERROR: OLF diverges in AC (with 'close to lightsim')")
        mw_dc = resdc[0].slack_bus_results[0].active_power_mismatch
        mw_standar = res_standar[0].slack_bus_results[0].active_power_mismatch
        mw_likels = res_likels[0].slack_bus_results[0].active_power_mismatch
        print(f"\t\tslack absorb / produce dc: {mw_dc:.2e}MW, normal param {mw_standar:.2e}, close to ls param {mw_likels:.2e}")
    print("... done\n")
    
    
def check_lf_pypowsybl_for(grid, load_p_for, load_q_for, gen_p_for, gen_v_for, use_v=False):
    print("Check powerflow (OLF) for the forecast time serie...")
    if use_v:
        print(f"\tUsing generator voltage setpoint from the time series")
    else:
        print(f"\tREMOVING generator voltage setpoint from the time series")
    i = 0
    glob_res_dc = []
    glob_res_stand = []
    glob_res_likels = []
    for i in range(load_p_for.shape[0]):
        df_load = pd.DataFrame(index=load_p_for.columns, data={"p0": load_p_for.iloc[i].values, "q0": load_q_for.iloc[i].values})
        grid.update_loads(df_load)
        data_ = {"target_p": gen_p_for.iloc[i].values}
        if use_v:
            data_["target_v"] = gen_v_for.iloc[i].values
        df_gen = pd.DataFrame(index=gen_p_for.columns, data=data_)
        grid.update_generators(df_gen)
        resdc = pp.loadflow.run_dc(grid)
        if not resdc:
            print("\t ERROR: OLF diverges in DC")
        res_standar = pp.loadflow.run_ac(grid)
        if not resdc:
            print("\t ERROR: OLF diverges in AC (with default params)")
        res_likels = pp.loadflow.run_ac(grid, parameters=OLF_PARAMS)
        if not resdc:
            print("\t ERROR: OLF diverges in AC (with 'close to lightsim' params)")
        mw_dc = resdc[0].slack_bus_results[0].active_power_mismatch
        glob_res_dc.append(mw_dc)
        mw_standar = res_standar[0].slack_bus_results[0].active_power_mismatch
        glob_res_stand.append(mw_standar)
        mw_likels = res_likels[0].slack_bus_results[0].active_power_mismatch
        glob_res_likels.append(mw_likels)
    print(f"\t For the forecast (dc): min {np.min(glob_res_dc):.2e} MW, max: {np.max(glob_res_dc):.2e} MW")
    print(f"\t For the forecast (standard param): min {np.min(glob_res_stand):.2e} MW, max: {np.max(glob_res_stand):.2e} MW")
    print(f"\t For the forecast (close to lightsim param): min {np.min(glob_res_likels):.2e} MW, max: {np.max(glob_res_likels):.2e} MW")
    print("... done\n")
    
if __name__ == "__main__":
    all_ts = sorted([el for el in os.listdir(os.path.join(path, ts_dir_nm)) if os.path.isdir(os.path.join(path, ts_dir_nm, el))])
    grid = pp.network.load(os.path.join(path, gridnm))
    generators = grid.get_generators().sort_index()
    el = all_ts[0]
    for el in all_ts:
        print(f"Studying time series at \"{el}\"")
        this_ts_path = os.path.join(os.path.join(path, ts_dir_nm), el)
        gen_p = pd.read_csv(os.path.join(this_ts_path, "prod_p.csv"), sep=";")
        gen_v = pd.read_csv(os.path.join(this_ts_path, "prod_v.csv"), sep=";")
        load_p = pd.read_csv(os.path.join(this_ts_path, "load_p.csv"), sep=";")
        load_q = pd.read_csv(os.path.join(this_ts_path, "load_q.csv"), sep=";")
        gen_p_for = pd.read_csv(os.path.join(this_ts_path, "prod_p_forecasted.csv"), sep=";")
        gen_v_for = pd.read_csv(os.path.join(this_ts_path, "prod_v_forecasted.csv"), sep=";")
        load_p_for = pd.read_csv(os.path.join(this_ts_path, "load_p_forecasted.csv"), sep=";")
        load_q_for = pd.read_csv(os.path.join(this_ts_path, "load_q_forecasted.csv"), sep=";")
        env_size = gen_p.shape[0]
        
        print("Checking size...")
        for df, nm in zip([gen_v, load_p, load_q],
                        ["gen_v", "load_p", "load_q"]):
            if df.shape[0] != env_size:
                print(f"ERROR: {nm} has not the right size {df.shape[0]} vs {env_size}")
        for df, nm in zip([gen_p_for, gen_v_for, load_p_for, load_q_for],
                        ["gen p (forecast)", "gen_v (forecast)", "load_p (forecast)", "load_q (forecast)"]):
            if df.shape[0] != env_size * NB_FORECAST:
                print(f"ERROR: {nm} has not the right size {df.shape[0]} vs {env_size * NB_FORECAST}")
        print("... done\n")
        
        print("Checking sum gen = sum load + losses ..")
        if (gen_p.sum(axis=1) <= 1.02 * load_p.sum(axis=1)).any():
            print(f"WARNING: grid is probably not at the equilibrium: not enough generation "
                f"for some steps: "
                f"\n-total gen: {gen_p.sum(axis=1).values}"
                f"\n-total load: {load_p.sum(axis=1).values}"
                )
        print("... done\n")
        
        check_genv_setpoit(generators, gen_v, gen_v_for)

        check_lf_pypowsybl_env(grid, load_p, load_q, gen_p, gen_v, use_v=False)
        check_lf_pypowsybl_for(grid, load_p_for, load_q_for, gen_p_for, gen_v_for, use_v=False)
        check_lf_pypowsybl_env(grid, load_p, load_q, gen_p, gen_v, use_v=True)
        check_lf_pypowsybl_for(grid, load_p_for, load_q_for, gen_p_for, gen_v_for, use_v=True)
            
        print(f"End time series at \"{el}\"")
        print("=============================================================================")
