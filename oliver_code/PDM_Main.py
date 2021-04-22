import algorithm_platform_methods as ap
import sys
import dataReader
import matplotlib as mpl
import shortestpath_nx as spnx
import passenger_simulations as ps
import shortestpath as sp
import random
import treatingZones
import graphcreator  # graphcreator_oldTransfer as
import numpy as np
import time
import datetime
import prepare_io_and_call_alns
import alns as alns
import utils
import networkx as nx
import numpy.lib.recfunctions as rfn
import pickle
import cProfile
import io
import convert
import Classes
import scipy
import re
from Classes import Parameters


import cProfile
import pstats

# apiUrl = sys.argv[1]

# call the main function, set the profile


def profile_main():

    pr = cProfile.Profile()
    pr.enable()
    # ... do something ...
    # main_test_parts()
    main()
    pr.disable()
    s = io.StringIO()

    sortby_selected = 'cumulative'  # 'totalTime'
    # ps = pstats.Stats(pr, stream=s).sort_stats('totalTime')
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby_selected)
    # ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats()
    with open('Profile_' + sortby_selected + '.txt', 'w+') as f:
        f.write(s.getvalue())
def main_test_parts():

    print('parts to test')

def main():
    # matplotlib.use('PS')
    #
    mpl.use('TkAgg')

    # initialization, zones and thresholds
    th_ZoneSelection = 8000  #[meters]
    nr_zones_to_connect_tt = 1
    nr_stations_to_connect_euclidean = 2
    min_nr_of_passenger = 0
    print('\n treshhold Zone Selection [m] : ', th_ZoneSelection)
    print(' number of min passengers  : ', min_nr_of_passenger)
    print(' nr stations connected euclidean : ', nr_stations_to_connect_euclidean)
    print(' nr stations connected TT  : ', nr_zones_to_connect_tt)

    # paths
    # path_nodesSBB = 'input/Node_SBB_ZoneNR.csv'
    path_nodesSBB = 'input/Node_SBB_AreaOfInterest.csv'

    p_allZonesOD_debug = 'input/zones_debug.csv'
    p_centroidZones = 'input/centroids_zones.csv'
    p_centroidZonesSel = 'input/zonesSelected_' + str(th_ZoneSelection) + '.csv'
    p_allZonesOD = 'input/01_DWV_OEV.txt'
    p_allZonesTT = 'input/2010_TravelTime_allZones.txt'
    p_OD_selectedZones = 'input/Demand_DWV_' + str(min_nr_of_passenger) + 'pass_' + str(th_ZoneSelection) + '_zTT_' \
                         + str(nr_zones_to_connect_tt) + '_sEu_' + str(nr_stations_to_connect_euclidean) + '.csv'
    # path_OD_selectedZones = 'input/Demand_DWV_1pass_12000_zTT_2_sEu_3.csv'

    p_OD_selectedZonesTT = 'input/zonesSelected_tt_' + str(th_ZoneSelection) + '.csv'
    # p_OD_selectedZonesTT_internal = 'input/zonesSelected_tt_' + str(th_zone_selection) + '_internal.csv'
    p_OD_reducedTT = 'input/zonesSelected_tt_reduced' + str(th_ZoneSelection) + '.csv'

    p_od_depTime = 'input/OD_desired_departure_time_' + str(th_ZoneSelection) + '.csv'
    p_od_depTime_grouped = 'input/OD_desired_departure_time_' + str(th_ZoneSelection) + '_grouped.csv'  #   '_grouped_DEBUG.csv'

    # Loading of data, define area of interest
    # Stations :
    nodesSBB, code_id_dictionary, id_code_dictionary = dataReader.process_stations(path_nodesSBB)
    time_window = ap.get_parameter('ReferenceTimeWindow') # ask viriato for the time window
    print('\nTime Window of reference day : \n', time_window)

    # infrastructure Graph, nodeSBB - id of infrastructure,
    # create_from_scratch - cache the network

    G_infra = graphcreator.explore_network(nodesSBB, create_from_scratch=True) #Oliver - false
    parameters = Classes.Parameters(G_infra)

    # Trains :
    # viriato method that gets all trains that are within selected area
    cut_trains = ap.active_trains_cut_to_time_range_driving_any_node(time_window['FromTime'],
                                                                     time_window['ToTime'],
                                                                    list(nodesSBB.Code[nodesSBB.Visited == True]))

    #cut_trains = ap.active_trains_within_time_range(time_window['FromTime'],
     #                                                                time_window['ToTime'])

    # remove trains leaving the area, those with only 1 comm stop in the area as well as cancel the parts outside area
    # cleanse the trains (when they leave the area)
    cut_trains_to_Area = ap.cut_trains_AreaOfInterest_and_adapt_Viriato_DB(cut_trains, parameters.stations_in_area, parameters)


    # same as 104, because they are now cleaned (it can be skipped, maybe)
    cut_trains_to_Area = ap.active_trains_cut_to_time_range_driving_any_node(time_window['FromTime'],
                                                                       time_window['ToTime'],
                                                                   list(nodesSBB.Code[nodesSBB.Visited==True]))
    #cut_trains_to_Area =ap.active_trains_within_time_range(time_window['FromTime'], time_window['ToTime'])

    # the creation of the graph
    create_graph_from_scratch = True
    # transfer edges Binder 17 efficient m = 4, M = 15
    m = datetime.timedelta(minutes=4)
    M = datetime.timedelta(minutes=15)
    p_graph = 'input/Graph/graph_transfer_m' + str(int(m.total_seconds() / 60)) + '_M' + str(int(M.total_seconds() / 60)) + \
              '_threshold' + str(th_ZoneSelection) + '.pickle'
    p_graph_homes_connected = 'input/Graph/graph_transfer_m' + str(int(m.total_seconds() / 60)) + '_M' + str(int(M.total_seconds() / 60)) \
                              + '_threshold' + str(th_ZoneSelection) + 'Homes_connected.pickle'
    p_odt_for_sp = 'input/Graph/graph_transfer_m' + str(int(m.total_seconds() / 60)) + '_M' + str(int(M.total_seconds() / 60)) \
                              + '_threshold' + str(th_ZoneSelection)

    G, nodes_commercial_stops = graphcreator.create_or_load_graph_with_waiting_driving_transfer_edges(M, create_graph_from_scratch,
                                                                                         cut_trains_to_Area, m, p_graph, parameters)

    # add if it has commercial stop (can passangers start their jurney here)
    nodesSBB = ap.add_field_com_stop(nodesSBB, nodes_commercial_stops)
    parameters.stations_comm_stop = nodes_commercial_stops


    # zones Centroid, demand and traveltime data selection
    select_zones = False  # if true then it will create the data from original files otherwise load reduced files
    zonesSelected, zonesSelDemand, zonesSelTT = treatingZones.treating_reading_demand_zones_tt(
        min_nr_of_passenger, nodesSBB, nr_stations_to_connect_euclidean, nr_zones_to_connect_tt, p_OD_selectedZones,
        p_OD_reducedTT, p_allZonesOD, p_allZonesTT, p_centroidZones, p_centroidZonesSel, select_zones, th_ZoneSelection)

    # time discretization for the passenger grouping and desried travel time simulation
    time_discretization = datetime.timedelta(minutes=10)
    read_od_depTime = True  # if true the file is read from input, otherwise it is created
    create_grouped_passenger = False
    if read_od_depTime:
        if create_grouped_passenger:
            od_departure_time = dataReader.dataReader_load_od_departure_time(p_od_depTime)
            # group the passengers with same o d and desired dep time
            max_group_size = 80
            od_departure_time = ps.group_passengers(od_departure_time, max_group_size, th_ZoneSelection)
        else:
            od_departure_time = dataReader.dataReader_load_od_departure_time_grouped(p_od_depTime_grouped)
            # for od in od_departure_time:
            #    od.desired_dep_time = datetime.datetime.strptime(od.desired_dep_time, "%Y-%m-%dT%H:%M")
    else:
        # simulation of the departure time of each passenger and group them according to the time discretization
        # --> od_departure_time(fromZone, toZone, passengers, desired_dep_time)
        od_departure_time = ps.od_with_departure_time(time_window, zonesSelDemand, time_discretization,
                                                      th_ZoneSelection)
        # group the passengers with same o d and desired dep time
        max_group_size = 1
        od_departure_time = ps.group_passengers(od_departure_time, max_group_size, th_ZoneSelection)

    # add the priority to od_depTime (random priority between 0-1), used for the assignement of the passangers
    od_departure_time = ps.create_priority_list(od_departure_time) 
    debug = False
    if debug:
        od_departure_time = od_departure_time[0:1000]

    create_g_home_connections_from_scratch = True

    # connect home of the passanger to the stations (similar to the transfer connection)
    G, odt_list, odt_by_origin, odt_by_dest, station_candidates, origin_name_desired_dep_time, origin_name_zone_dict = \
        graphcreator.connect_home_stations(G, nodesSBB, nr_stations_to_connect_euclidean, nr_zones_to_connect_tt, zonesSelTT,
                                           zonesSelected, id_code_dictionary, od_departure_time, p_graph_homes_connected,
                                           p_odt_for_sp, create_g_home_connections_from_scratch, parameters)
    zone_candidates = create_zone_candidates_of_stations(station_candidates)

    debug = False
    if debug:
        check_Graph_number_edges_nodes(G)

    # store in parameters, because of the high computational time
    parameters.station_candidates = station_candidates
    parameters.zone_candidates = zone_candidates
    parameters.odt_by_origin = odt_by_origin
    parameters.odt_by_destination = odt_by_dest
    parameters.odt_as_list = odt_list
    parameters.origin_name_desired_dep_time = origin_name_desired_dep_time

    # find the shortest path for all sources in selected demand
    filter_passengers = True
    if filter_passengers:
        use_nx_dijkstra = True
        use_scipy_dijkstra = False
        if use_scipy_dijkstra:
            sp.scipy_dijkstra_full_od_list(G, odt_list)

        if use_nx_dijkstra:
            find_path_forall_passengers_and_remove_unserved_demand(G, odt_list, origin_name_desired_dep_time,
                                                                   origin_name_zone_dict, parameters)



    # !! start alns
    start_alns = True
    if start_alns:

        # Get the infrastructure graph
        set_solution = prepare_io_and_call_alns.prepare_io_and_call_alns(G, G_infra, cut_trains_to_Area, parameters)

    picked_solution = pick_best_solution(set_solution)
    changes = changes_per_accepted_solution[set_solution.index(picked_solution)]
    store_solution_to_viriato(changes, parameters.initial_timetable_feasible)

    print('end of algorithm  ', '\n',
          'total running time in [sec] : see profiler')





def pick_best_solution(set_solution):

    return set_solution[0]


def store_solution_to_viriato(changes, initial_timetable):
    # todo, remember that train['EmergencyTrain] is True for ET but not defined for initial trains
    train_times_update_per_train = {}
    cancellation_from_per_train = {}
    cancellation_to_per_train = {}
    cancellation_total_per_train = set()
    for change in changes:
        if change['Action'] == 'Cancel':
            cancellation_total_per_train.add(change['train_id'])

    for change in changes:
        if change['Action'] == 'CancelFrom':
            if not cancellation_total_per_train.__contains__(change['train_id']):
                cancellation_from_per_train[change['train_id']] = change['tpn_cancel_from']

    # Todo, update und cancel to genau gleiches vorgehen
    #  updates an Viriato ausführen
    #  cancellation ausführen


def find_path_forall_passengers_and_remove_unserved_demand(G, odt_list, origin_name_desired_dep_time,
                                                           origin_name_zone_dict, parameters):
    cutoff = parameters.time_window['duration'].seconds / 60
    length, path, served_unserved_passengers, odt_withPath, odt_noPath = sp.find_sp_for_all_sources_full_graph(G, parameters)  # , cutoff = cutoff)
    print(' passengers with path : ', served_unserved_passengers[0], ', passengers without path : ',
          served_unserved_passengers[1])
    odt_list_withPath = []
    odt_withPath_by_origin = {}
    odt_withPath_by_dest = {}
    origin_name_desired_dep_time_withPath = {}
    for odt in odt_list:
        source, target, priority, groupsize = odt
        if [source, target, groupsize] in odt_withPath:
            odt_list_withPath.append(odt)
            zone = origin_name_zone_dict[source]
            if not odt_withPath_by_origin.__contains__(zone):
                odt_withPath_by_origin[zone] = {}
            odt_withPath_by_origin[zone][(source, target)] = groupsize

            if not odt_withPath_by_dest.__contains__(target):
                odt_withPath_by_dest[target] = {}
            odt_withPath_by_dest[target][(source, target)] = groupsize

            origin_name_desired_dep_time_withPath[(source, target)] = origin_name_desired_dep_time[
                (source, target)].copy()
    parameters.odt_by_origin = odt_withPath_by_origin
    parameters.odt_by_destination = odt_withPath_by_dest
    parameters.odt_as_list = odt_list_withPath
    # strange that the number of entries is so different. duplicates....
    parameters.origin_name_desired_dep_time = origin_name_desired_dep_time_withPath


def find_path_scipy_forall_passengers_and_remove_unserved_demand(G, odt_list, origin_name_desired_dep_time,
                                                                origin_name_zone_dict, parameters):
    cutoff = parameters.time_window['duration'].seconds / 60
    length, path, served_unserved_passengers, odt_withPath, odt_noPath = sp.find_sp_for_all_sources_full_graph(G,
                                                                                                               parameters)  # , cutoff = cutoff)
    print(' passengers with path : ', served_unserved_passengers[0], ', passengers without path : ',
          served_unserved_passengers[1])
    odt_list_withPath = []
    odt_withPath_by_origin = {}
    odt_withPath_by_dest = {}
    origin_name_desired_dep_time_withPath = {}
    for odt in odt_list:
        source, target, priority, groupsize = odt
        if [source, target, groupsize] in odt_withPath:
            odt_list_withPath.append(odt)
            zone = origin_name_zone_dict[source]
            if not odt_withPath_by_origin.__contains__(zone):
                odt_withPath_by_origin[zone] = {}
            odt_withPath_by_origin[zone][(source, target)] = groupsize

            if not odt_withPath_by_dest.__contains__(target):
                odt_withPath_by_dest[target] = {}
            odt_withPath_by_dest[target][(source, target)] = groupsize

            origin_name_desired_dep_time_withPath[(source, target)] = origin_name_desired_dep_time[
                (source, target)].copy()
    parameters.odt_by_origin = odt_withPath_by_origin
    parameters.odt_by_destination = odt_withPath_by_dest
    parameters.odt_as_list = odt_list_withPath
    # strange that the number of entries is so different. duplicates....
    parameters.origin_name_desired_dep_time = origin_name_desired_dep_time_withPath


def create_zone_candidates_of_stations(station_candidates):
    zone_candidates = {}

    for zone, station_candidates in station_candidates.items():
        for station, attr in station_candidates.items():
            if not zone_candidates.__contains__(station):
                zone_candidates[station] = {}
            if not zone_candidates[station].__contains__(zone):
                if 'tt_toStation' in attr.keys():
                    zone_candidates[station][zone] = attr['tt_toStation']
    return zone_candidates


def check_Graph_number_edges_nodes(G):
    nodes_origins = [x for x, y in G.nodes(data=True) if y['type'] == 'origin']
    nodes_destination = [x for x, y in G.nodes(data=True) if y['type'] == 'destination']
    train_node_types = ['arrivalNode', 'departureNode', 'arrivalNodePassing', 'departureNodePassing']
    nodes_trains = [x for x, y in G.nodes(data=True) if y['type'] in train_node_types]
    od, d, w, t = 0, 0, 0, 0
    for edge in G.edges.data():
        value = edge[2]
        if 'type' in value.keys():
            if value['type'] == 'driving':
                d += 1
            elif value['type'] == 'waiting':
                w += 1
            elif value['type'] == 'transfer':
                t += 1
        else:
            od += 1



if __name__ == "__main__":
    profile = True
    if profile:
        profile_main()
    else:
        main()
