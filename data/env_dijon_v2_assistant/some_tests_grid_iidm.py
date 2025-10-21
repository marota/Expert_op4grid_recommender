
import os
import pypowsybl as pp
import pypowsybl.network as pypow_net
import pypowsybl.loadflow as pypow_lf
import networkx as nx
import numpy as np

path = os.path.abspath(os.path.split(__file__)[0])
file_name = "grid.xiidm"


OLF_PARAMS = pypow_lf.Parameters(voltage_init_mode=pp._pypowsybl.VoltageInitMode.DC_VALUES,
                                 transformer_voltage_control_on=False,
                                 use_reactive_limits=False,
                                 shunt_compensator_voltage_control_on=False,
                                 phase_shifter_regulation_on=False,
                                 distributed_slack=False,
                                 read_slack_bus = False,
                                 dc_use_transformer_ratio=False,
                                 twt_split_shunt_admittance=True,
                                 provider_parameters={"useActiveLimits": "false", 
                                                      "svcVoltageMonitoring": "false",
                                                      "voltageRemoteControl": "false",
                                                    #   "slackBusSelectionMode": "NAME",  # TODO
                                                    #   "slackBusesIds": "N.SE1P1",  # TODO
                                 }
                                 ) 


def check_grid_consistent(grid_pp, desc=None):
    start_ = ""
    if desc is not None:
        start_ = "    "
        print(f"Checking the grid {desc}")
        
    vl_df = grid_pp.get_voltage_levels().sort_index()
    bus_df = grid_pp.get_buses().sort_index()
    line_df = grid_pp.get_lines().sort_index()
    trafo_df = grid_pp.get_2_windings_transformers().sort_index()
    trafo3w_df = grid_pp.get_3_windings_transformers().sort_index()
    storage_df = grid_pp.get_batteries().sort_index()
    load_df = grid_pp.get_loads().sort_index()
    gen_df = grid_pp.get_generators().sort_index()
    shunt_df = grid_pp.get_shunt_compensators().sort_index()
    hvdc_df = grid_pp.get_hvdc_lines().sort_index()
    
    # check for unsupported element
    if trafo3w_df.shape[0] > 0:
        print()
        print(f"{start_}ERROR (UNKNOWN FEATURE): grid2op and lightsim2grid cannot handle 3 windings transformers at the moment")
        
    if hvdc_df.shape[0] > 0:
        print()
        print(f"{start_}WARNING (HUGE MODELING DIFFERENCE): grid2op cannot control HVDC, lightsim2grid does not model them as pypowsybl")

    # check for grid connex
    if (bus_df["connected_component"] != 0).any():
        which_buses = bus_df[(bus_df["connected_component"] != 0)][["voltage_level_id"]]
        print()
        print(f"{start_}ERROR: grid is not connected: some voltage_level are not in the main component, check:\n{which_buses}")
        print(f"{start_}===============================================================\n")


    el_dfs = [line_df, line_df, trafo_df, trafo_df, load_df, gen_df, shunt_df, storage_df, hvdc_df, hvdc_df]
    el_names = ["line (side 1)", "line (side 2)",
                "trafo (side 1)", "trafo (side 2)",
                "load", "generator", "shunt", "storage",
                "hvdc (side 1)", "hvdc (side 2)"]
    # check for disconnected elements
    for df, key, name in zip(el_dfs,
                            ["connected1", "connected2"] * 2 +  ["connected"] * 4 + ["connected1", "connected2"],
                            el_names):
        if (~df[key]).any():
            errors = df.loc[~df[key]].index.values
            print()
            print(f"{start_}ERROR: some {name} are disconnected. Check {name}:\n{errors}")
            if name == "load":
                print(f"\n{start_}EASY FIX: reconnect the loads and assign them a `p0` and a `q0` of 0. "
                       "In this case make sure they are always 0. in the time series (chronics)")
            elif name == "generator":
                print(f"\n{start_}POSSIBLE FIX: reconnect the generators and assign them a `target_p` and a `target_q` of 0. and set "
                       "`voltage_regulator_on` to `False`. In this case, make sure the target_p is always 0. on the time series (chronics)")
            print(f"{start_}===============================================================\n")

    # check for inconsistency between buses and voltage level
    for df, (vl_key, bus_key), name in zip(el_dfs,
                                        ([("voltage_level1_id", "bus1_id"), ("voltage_level2_id", "bus2_id")] * 2 +  
                                            [("voltage_level_id", "bus_id")] * 4 + 
                                            [("voltage_level1_id", "bus1_id"), ("voltage_level2_id", "bus2_id")]),
                                        el_names):
        if "hvdc" in name:
            continue

        vl = df[vl_key]
        bus = df[bus_key]
        mask_ko = bus == ''
        do_print_debug = False
        if mask_ko.any():
            print("")
            print(f"{start_}ERROR: as we already said, there are some issues with {name}: {mask_ko.sum()} are disconnected")
            do_print_debug = True
        vl = vl[~mask_ko]
        bus = bus[~mask_ko]
        vl_from_bus = bus_df.loc[bus, "voltage_level_id"]
        vl_from_bus.index = vl.index
        if (vl != vl_from_bus).any():
            print("")
            print(f"{start_}ERROR (INCONSISTENCY IN IIDM MODEL): some {name} are connected to voltage level in their dataframe "
                   "but they are also connected to another voltage level if you look at the voltage level to which their `bus` is connected.")
            do_print_debug = True
        if do_print_debug:
            print(f"{start_}===============================================================\n")

    # verifier que les postes sont à 1 noeud
    bbs_df = grid_pp.get_busbar_sections().sort_index()
    all_els = np.concatenate([el.index.values for el in [gen_df, load_df, line_df, trafo_df, storage_df, shunt_df, hvdc_df]])
    def get_conn_comp(topo, bbs_df, all_els):
        graph = nx.Graph()
        graph.add_nodes_from(topo.nodes.index.tolist())
        topo_switches = topo.switches
        closed_topo_switches = topo_switches[~topo_switches['open']]
        graph.add_edges_from(closed_topo_switches[['node1', 'node2']].values.tolist())
        graph.add_edges_from(topo.internal_connections[['node1', 'node2']].values.tolist())
        node_bbs = topo.nodes[np.isin(topo.nodes["connectable_id"], bbs_df.index)].index.values
        node_real_els = topo.nodes[np.isin(topo.nodes["connectable_id"], all_els)].index.values
        conn_comp = [el for el in nx.connected_components(graph) 
                    if (len(el) >= 2 and 
                        np.isin(list(el), node_bbs).any() and 
                        np.isin(list(el), node_real_els).any())]
        return conn_comp
        
    do_print_debug = False
    for vl in vl_df.index.values:
        topo = grid_pp.get_node_breaker_topology(vl)
        conn_comp = get_conn_comp(topo, bbs_df, all_els)
        if len(conn_comp) != 1:
            print(f"{start_}ERROR (voltage levels with different buses): voltage level {vl} counts {len(conn_comp)} buses")
            do_print_debug = True
    if do_print_debug:
        print(f"{start_}===============================================================\n")
    if desc is not None:
        print(f"End test for {desc}")
        print(f"===========================================================================\n")
        

if __name__ == "__main__":
    grid_pp = pypow_net.load(os.path.join(path, file_name))
    check_grid_consistent(grid_pp, desc="right after loading")
    
    # check powerflow converges
    res = pypow_lf.run_dc(grid_pp, parameters=OLF_PARAMS)
    assert res[0], "powerflow diverge in DC with 'close to lightsim2grid' parameters"
    print(f"INFO: in DC slack absorb / produce {res[0].slack_bus_results[0].active_power_mismatch:.2e}MW (should be close to 0. MW)")
    check_grid_consistent(grid_pp, desc="after a DC powerflow")

    res = pypow_lf.run_ac(grid_pp, parameters=OLF_PARAMS)
    assert res[0], "powerflow diverge in AC with 'close to lightsim2grid' parameters"
    print(f"INFO: in AC slack absorb / produce {res[0].slack_bus_results[0].active_power_mismatch:.2e}MW (should be close to 0. MW)")
    check_grid_consistent(grid_pp, desc="after an AC powerflow")
