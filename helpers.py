"""
Created on Thu Feb 25 2021

@author: BenMobility

Helpers for the master thesis main codes.
"""

# %% Imports
from py_client.aidm import *
import numpy.lib.recfunctions as rfn
import numpy as np
import random
import copy
import timetable_graph


# %% Store parameters
class Parameters:
    def __init__(self, infra_graph, time_window: TimeWindow, closed_tracks_ids, list_parameters):
        self.name = 'parameters'
        self.time_duration = time_window.to_time - time_window.from_time
        self.time_window = time_window

        # Selected zones for passengers
        self.th_zone_selection = list_parameters[0]
        self.nb_zones_to_connect = list_parameters[1]
        self.nb_stations_to_connect = list_parameters[2]
        self.min_nb_passenger = list_parameters[3]

        self.closed_tracks = closed_tracks_ids
        self.trains_on_closed_track_initial_timetable_infeasible = None

        self.initial_timetable_infeasible = None
        self.initial_timetable_feasible = None
        self.train_ids_initial_timetable_infeasible = None

        self.beta_transfer = list_parameters[10]
        self.beta_waiting = list_parameters[11]

        self.score_1 = list_parameters[13]
        self.score_2 = list_parameters[14]
        self.score_3 = list_parameters[15]
        # start temp. [z_op, z_de_reroute, z_de_cancel, z_tt]
        self.t_start = [list_parameters[16], list_parameters[17], list_parameters[56], list_parameters[18]]

        self.weight_closed_tracks = list_parameters[19]
        self.train_capacity = list_parameters[20]
        self.bus_capacity = list_parameters[21]
        self.penalty_no_path = list_parameters[22]
        # self.penalty_no_path = int(time_window['duration'].seconds / 60)  # the time window of three hours in minutes

        # Parameters for od departure desired time / group passenger formation
        self.time_discretization = list_parameters[26]
        self.group_size_passenger = list_parameters[27]
        self.earliest_time = list_parameters[54]
        self.latest_time = list_parameters[55]

        # Parameters for graph creation
        self.transfer_m = datetime.timedelta(minutes=list_parameters[4])
        self.transfer_M = datetime.timedelta(minutes=list_parameters[5])
        # Transfer edges emergency bus
        self.transfer_mBus = datetime.timedelta(minutes=list_parameters[6])
        self.transfer_MBus = datetime.timedelta(minutes=list_parameters[7])
        # Origin train departure waiting times
        self.min_wait = datetime.timedelta(minutes=list_parameters[8])
        self.max_wait = datetime.timedelta(minutes=list_parameters[9])

        # Home connections
        self.station_candidates = None
        self.zone_candidates = None
        self.stations_in_area = [n for n in infra_graph.nodes if infra_graph.nodes[n]['in_area']]
        self.stations_comm_stop = None
        self.odt_by_origin = None  # list which is the input to sp algorithm
        self.odt_by_destination = None  # dictionary with odt and number of passengers, key 1st level origin zone,
        # 2nd level origin departure node name (e.g. '1_06:15')
        self.odt_as_list = None
        self.origin_name_desired_dep_time = None

        self.set_of_trains_for_operator = {'Cancel': [], 'CancelFrom': [], 'Delay': [], 'DelayFrom': []}
        self.delayTime_to_consider_cancel = datetime.timedelta(minutes=list_parameters[23])
        self.delayTime_to_consider_partCancel = datetime.timedelta(minutes=list_parameters[24])

        self.commercial_stops = list_parameters[25]

        # Passenger files
        self.full_od_file = list_parameters[28]
        self.read_od_departure_file = list_parameters[29]
        self.create_group_passengers = list_parameters[30]
        self.max_iteration_recompute_path = list_parameters[31]
        self.read_selected_zones_demand_travel_time = list_parameters[32]
        self.capacity_constraint = list_parameters[53]

        # ALNS iterations
        self.number_iteration = list_parameters[33]
        self.number_iteration_archive = list_parameters[34]

        # Neighbourhood operations
        self.delay_options = list_parameters[35]
        self.time_delta_delayed_bus = list_parameters[36]
        self.min_headway = list_parameters[37]
        self.assign_passenger = list_parameters[38]
        self.deviation_penalty_cancel = list_parameters[39]
        self.deviation_penalty_delay = list_parameters[40]
        self.deviation_penalty_emergency = list_parameters[41]
        self.deviation_penalty_bus = list_parameters[42]
        self.deviation_penalty_rerouted = list_parameters[43]

        # Acceptance and rejection
        self.reaction_factor_operation = list_parameters[44]
        self.reaction_factor_deviation = list_parameters[45]
        self.warm_up_phase = list_parameters[46]
        self.iterations_temperature_level = list_parameters[47]
        self.number_of_temperature_change = list_parameters[48]
        self.reaction_factor_return_archive = list_parameters[49]
        self.reaction_factor_weights = list_parameters[50]

        # Greedy feasibility
        self.max_iteration_feasibility_check = list_parameters[51]
        self.max_iteration_section_check = list_parameters[52]


# Store timetables for ALNS algorithm
class Timetables:
    def __init__(self):
        self.name = 'initial_timetables'

        self.initial_timetable_infeasible = None
        self.initial_timetable_feasible = None


# Store weights for the neighbourhood operators
class Weights:
    def __init__(self):
        self.name = 'weight'

        # self.rr = 1  # Rerouting
        self.cc = 1  # complete cancel
        self.pc = 1  # partial cancel
        self.cd = 1  # complete delay
        self.pd = 1  # partial delay
        self.et = 1  # emergency train
        self.eb = 1  # emergency bus
        self.ret = 1  # return to initial train

        self.sum = self.cc + self.pc + self.cd + self.pd + self.et + self.eb + self.ret  # self.rr


# Store the score of each neighbourhood operators
class Scores:
    def __init__(self):
        self.name = 'score'
        # self.rr = 0  # Rerouting
        self.cc = 0  # complete cancel
        self.pc = 0  # partial cancel
        self.cd = 0  # complete delay
        self.pd = 0  # partial delay
        self.et = 0  # emergency train
        self.eb = 0  # emergency bus
        self.ret = 0  # return to initial train


# Store the number of usage of each neighbourhood operator
class NumberUsage:
    def __init__(self):
        self.name = 'nr_usage'
        # self.rr = 0  # Rerouting
        self.cc = 0  # complete cancel
        self.pc = 0  # partial cancel
        self.cd = 0  # complete delay
        self.pd = 0  # partial delay
        self.et = 0  # emergency train
        self.eb = 0  # emergency bus
        self.ret = 0  # return to initial train


# Store the probabilities to choose the neighbourhood operator
class Probabilities:
    def __init__(self, weights):
        self.name = 'probabilities'
        # self.rr = weights.rr / weights.sum  # Rerouting
        min_prob = 0.05
        remaining_prob = 1 - 7 * min_prob
        self.cc = min_prob + (weights.cc / weights.sum) * remaining_prob   # complete cancel
        self.pc = min_prob + self.cc + (weights.pc / weights.sum) * remaining_prob  # partial cancel
        self.cd = min_prob + self.pc + (weights.cd / weights.sum) * remaining_prob  # complete delay
        self.pd = min_prob + self.cd + (weights.pd / weights.sum) * remaining_prob  # partial delay
        self.et = min_prob + self.pd + (weights.et / weights.sum) * remaining_prob  # emergency train
        self.eb = min_prob + self.et + (weights.eb / weights.sum) * remaining_prob  # emergency bus
        self.ret = min_prob + self.eb + (weights.ret / weights.sum) * remaining_prob   # return to initial train


class EdgesOriginStationDestination:
    def __init__(self, graph, parameters):
        self.name = 'edges_o_stations_d'

        return_edges_nodes = True
        print('Generate edges from origin to station and from station to destination.')
        origin_nodes, origin_nodes_attributes, destination_nodes, destination_nodes_attributes, edges_o_stations, \
            edges_o_stations_attr, edges_stations_d, edges_stations_d_attr, edges_o_stations_dict, \
        edges_stations_d_dict = timetable_graph.generate_edges_origin_station_destination(graph, parameters,
                                                                                          return_edges_nodes)

        # Set the values in the class
        self.edges_o_stations = edges_o_stations
        self.edges_o_stations_dict = edges_o_stations_dict # Key origin, value edges connecting to train nodes
        self.origin_nodes_attributes = origin_nodes_attributes
        self.edges_stations_d = edges_stations_d
        self.edges_stations_d_dict = edges_stations_d_dict  # Key destination, value edges connecting to
        self.destination_nodes_attributes = destination_nodes_attributes


class CopyEdgesOriginStationDestination:
    def __init__(self, edges_o_stations_d):
        self.name = 'edges_o_stations_d'

        # Edges from origin to stations
        copy_edges = []
        for edge in edges_o_stations_d.edges_o_stations:
            copy_edges.append(edge.copy())
        self.edges_o_stations = copy_edges

        # Edges from stations to destination
        copy_edges = []
        for edge in edges_o_stations_d.edges_stations_d:
            copy_edges.append(edge.copy())
        self.edges_stations_d = copy_edges

        # Dictionary edges origin to stations
        copy_dict = {}
        for origin, transfers in edges_o_stations_d.edges_o_stations_dict.items():
            copy_dict[origin] = transfers.copy()
        self.edges_o_stations_dict = copy_dict  # key origin, value edges connecting to train nodes

        # Dictionary edges stations to destination
        copy_dict = {}
        for destination, transfers in edges_o_stations_d.edges_stations_d_dict.items():
            copy_dict[destination] = transfers.copy()
        self.edges_stations_d_dict = copy_dict  # key destination, value edges connecting to


class Solution:
    def __init__(self):
        self.name = 'solutions'

        self.set_of_trains_for_operator = None
        self.timetable = None
        self.total_traveltime = None
        self.total_dist_train = None
        self.deviation_reroute_timetable = None
        self.deviation_cancel_timetable = None
        self.graph = None
        self.changed_trains = None
        self.track_info = None
        self.edges_o_stations_d = None
        self.set_of_trains_for_operator = {}


def add_field_com_stop(sbb_nodes, stations_with_commercial_stop):
    """
    function that add commercial stop attributes in the sbb nodes.
    :param sbb_nodes: a recarray of sbb nodes attributes.
    :param stations_with_commercial_stop: an array of all the stations with commercial stops
    :return: sbb_nodes: a recarray of sbb nodes with now the information of all commercial stops
    """
    # Append the commercial stop field in sbb_nodes
    sbb_nodes = rfn.append_fields(sbb_nodes, 'commercial_stop', np.ones(sbb_nodes.shape[0]), dtypes='i8', usemask=False,
                                  asrecarray=True)
    # Check all the stations and spot the ones that are not commercial stop
    c = 0
    for station in sbb_nodes:
        if station.Code not in stations_with_commercial_stop:
            station.commercial_stop = 0
            c = c + 1
    # Print the number of stations that are not a commercial stop
    print(c, ' of ', sbb_nodes.shape[0], ' stations do not have commercial stops.')
    return sbb_nodes


# %% Upload centroid zones

def reading_demand_zones_travel_time(sbb_nodes, parameters):
    """
    Function that reads the text files that provides the centroid of each zones. Then, returns three arrays with
    selected zones, demand for each selected zones and the travel time between each selected zones.
    :param sbb_nodes: a recarray of sbb nodes attributes.
    :param parameters: a class object that has all the defined parameters for the main code
    :return: selected_zones, demand_selected_zones, travel_time_selected_zones: three array with selected zones, demand
    between each zones and the travel time between each zones.
    """
    # Read the files if asked, make sure the files are made with the same threshold
    if parameters.read_selected_zones_demand_travel_time:
        # Print output
        print('Read selected zones, demand selected zones and travel time selected zones files.')
        # selected_zones
        filename_zones = 'output/pickle/selected_zone_threshold_' + str(parameters.th_zone_selection) + '.pickle'
        selected_zones = np.load(filename_zones, allow_pickle=True)

        # demand_selected_zones
        filename_demand = 'output/pickle/demand_' + str(parameters.min_nb_passenger) + 'pass_' + \
                          str(parameters.th_zone_selection) + 'm_' + '_zones_' + str(parameters.nb_zones_to_connect) + \
                          '_stations_' + str(parameters.nb_stations_to_connect) + '.pickle'
        demand_selected_zones = np.load(filename_demand, allow_pickle=True)

        # travel_time_selected_zones
        filename_travel = 'output/pickle/selected_zones_tt_reduced_' + str(parameters.th_zone_selection) + 'm' + \
                          '.pickle'
        travel_time_selected_zones = np.load(filename_travel, allow_pickle=True)

    # From scratch
    else:
        # Import path
        path_centroid_zones = 'input/centroids_zones.csv'
        path_all_zones_od = 'input/01_DWV_OEV.txt'
        path_all_zones_travel_time = 'input/2010_TravelTime_allZones.txt'

        # Import from text the zone centroids
        zone_centroids = np.genfromtxt(path_centroid_zones, delimiter=',', dtype=str, skip_header=1)
        zone_centroids = np.core.records.fromarrays(zone_centroids.transpose(),
                                                    names='zone, xcoord, ycoord',
                                                    formats='i8, f8, f8')
        # Select a subset of zones
        selected_zones = find_closest_station(zone_centroids, sbb_nodes, parameters)

        # Select demand of zones (full od distance is quite heavy, if you change disruption, might think of rerun it)
        demand_selected_zones = get_demand_selected_zones(path_all_zones_od, selected_zones,
                                                          parameters.min_nb_passenger,
                                                          parameters.th_zone_selection, parameters.nb_zones_to_connect,
                                                          parameters.nb_stations_to_connect,
                                                          full_od_distance=parameters.full_od_file)

        # Select travel times
        travel_time_selected_zones = get_travel_time_selected_zones(path_all_zones_travel_time, selected_zones,
                                                                    parameters.th_zone_selection,
                                                                    parameters.nb_stations_to_connect,
                                                                    parameters.nb_zones_to_connect, sbb_nodes)

    return selected_zones, demand_selected_zones, travel_time_selected_zones


def find_closest_station(zone_centroids, sbb_nodes, parameters):
    """
    :param parameters: zone selected for the case study
    :param zone_centroids: home zones with names ; zone, x,y
    :param sbb_nodes: sbb stations with names : zone, SbbID, ViriatoID, NodeName, xcoord, ycoord, Code
    :return: all zones within threshold distance of a station
    """
    print('\nStart finding all stations within the distance of', parameters.th_zone_selection, 'meters.')

    # Calculate the distance from all zones to every station
    xy_zones = [zone_centroids.xcoord, zone_centroids.ycoord]
    xy_zones = np.asarray(xy_zones, dtype=float)  # p
    xy_stations = [sbb_nodes.xcoord, sbb_nodes.ycoord]
    xy_stations = np.asarray(xy_stations, dtype=float)  # q

    # Distance euclidean
    if xy_zones.shape[0] == 2 and xy_zones.shape[1] != 2:
        xy_zones = np.transpose(xy_zones)
    if xy_stations.shape[0] == 2 and xy_stations.shape[1] != 2:
        xy_stations = np.transpose(xy_stations)

    # provide all the indices for each station and each zones on a single array
    rows, cols = np.indices((xy_zones.shape[0], xy_stations.shape[0]))
    # distance = square root (delta x^2 + delta y^2)
    distances = ((xy_zones[rows.ravel()] - xy_stations[cols.ravel()]) ** 2)
    distances = np.round(np.sqrt(distances.sum(axis=1)), 5)
    if xy_zones.shape[0] == 1:
        pass
    else:
        # Distance matrix 2944 zones and 111 sbb nodes (2944 x 111)
        distances = np.reshape(distances, (xy_zones.shape[0], xy_stations.shape[0]))

    # For each zone, you will have the closest station and its distance
    min_distance = [np.amin(distances, axis=1), np.argmin(distances, axis=1)]
    min_distance = np.asarray(min_distance)

    # Loop through all zones
    selected_zones = zone_centroids

    de = 0
    nr = 0
    delete_indices = np.zeros(1)
    select_indices = np.zeros(1)
    select_distance = np.zeros(1)

    # min_distance.shape[1] equals the total number of zones to check
    for i in range(0, min_distance.shape[1]):  # Indices of the selected zone
        # Get the shortest distance from centroid to the zone and compare with the threshold
        if min_distance[0, i] <= parameters.th_zone_selection:
            nr = nr + 1
            if nr == 1:
                select_indices[0] = i
                select_distance[0] = min_distance[0, i]
            else:
                select_indices = np.append(select_indices, i)
                select_indices = select_indices.astype(int)
                select_distance = np.append(select_distance, min_distance[0, i])
        # Get indices of wanted delete zones when the distance is greater than the threshold
        else:
            de = de + 1
            if de == 1:
                delete_indices[0] = i
            else:
                delete_indices = np.append(delete_indices, i)

            delete_indices = delete_indices.astype(int)

    # Delete and select zones
    deleted_zones = selected_zones[delete_indices]
    selected_zones = selected_zones[select_indices]

    # Reshape the output for selected zones, we want zone, x and y coordinates
    a = np.reshape(selected_zones.zone, (selected_zones.shape[0], 1))
    b = np.reshape(selected_zones.xcoord, (selected_zones.shape[0], 1))
    c = np.reshape(selected_zones.ycoord, (selected_zones.shape[0], 1))

    # Prepare the output
    selected_zones_output = np.concatenate((a, b, c, np.reshape(select_distance, (select_distance.shape[0], 1))),
                                           axis=1)
    selected_zones_output = np.core.records.fromarrays(selected_zones_output.transpose(),
                                                       names='zone, xcoord, ycoord, distance',
                                                       formats='i8, f8, f8, f8')

    # Save the output in a text file.
    np.savetxt('output/selected_zone_threshold_' + str(parameters.th_zone_selection) + '.csv', selected_zones_output,
               delimiter=',', header="zone,x,y, distanceToStation", fmt="%s,%f,%f,%f")
    np.savetxt('output/deleted_zone_threshold' + str(parameters.th_zone_selection) + '.csv', deleted_zones,
               delimiter=',', header="zone,x,y", fmt="%s,%f,%f")

    # Save the output in a pickle file
    filename_zones = 'output/pickle/selected_zone_threshold_' + str(parameters.th_zone_selection) + '.pickle'
    selected_zones_output.dump(filename_zones)

    # Print a message to inform that the number of selected zones has been recorded
    print('Number of selected zones = ', nr, '/', zone_centroids.shape[0] + 1)
    print('The selected zones are saved in the output file in csv and in pickle.\n')

    return selected_zones_output


def get_demand_selected_zones(path, selected_zones, minimal_passengers, threshold,
                              nr_zones_to_connect_tt, nr_stations_to_connect_euclidean, full_od_distance=False):
    """
    Function that get the demand for all selected zones.
    :param full_od_distance: using the full od distance matrix available 287,088 KB. Takes a lot of time
    :param path: to the OD file with all zones
    :param selected_zones: zones to be considered, output of findClosestStation method
    :param threshold: threshold for zone selection in meters
    :param minimal_passengers: to consider demand
    :param nr_zones_to_connect_tt: number of connected zones with stations to home zones
    :param nr_stations_to_connect_euclidean: nr of stations connected by euclidean distance
    :return: OD of a subset of all zones
    """
    # Import path
    path_od_distance = 'input/2010_Distanz_Systemfahrplan_B.txt'
    path_od_distance_reduced = 'output/zonesSelected_distance_reduced8000.csv'

    # Upload the od distance matrix, two ways. if True, it takes a lot of time.
    if full_od_distance is True:
        print("Reading the full OD distance file")
        od_distance = np.genfromtxt(path_od_distance)
        od_distance = np.core.records.fromarrays(od_distance.transpose(),
                                                 names='fromZone, toZone, distance',
                                                 formats='i8, i8, f8')
        # Trips in area of interest for the od distance matrix
        in_zones = np.logical_and(np.isin(od_distance.fromZone, selected_zones.zone),
                                  np.isin(od_distance.toZone, selected_zones.zone))
        od_distance = np.unique(od_distance, axis=0)
        od_distance = od_distance[in_zones]
        filename = 'output/zonesSelected_distance_reduced' + str(threshold) + '.csv'
        np.savetxt(filename, od_distance, delimiter=',', header="fromZone, toZone, distance", fmt="%i,%i,%f")
        print('Reading the full OD done.')
    else:
        print('Load OD reduced file with threshold of 8000 meters')
        od_distance = np.genfromtxt(path_od_distance_reduced, delimiter=',', skip_header=1)
        od_distance = np.unique(od_distance, axis=0)
        od_distance = np.core.records.fromarrays(od_distance.transpose(),
                                                 names='fromZone, toZone, distance',
                                                 formats='i8, i8, f8')
        print('Loading done')

    # Upload the od all zones
    od = np.genfromtxt(path)
    od = np.core.records.fromarrays(od.transpose(),
                                    names='fromZone, toZone, passenger',
                                    formats='i8, i8, f8')

    # Print the beginning of the selection
    print('Beginning of the fetching the demand of each selected zone.')
    print('Total number of passenger trips is :', od.shape[0])
    print('Median amount of passenger per trip', np.median(od.passenger))
    total_passengers = np.sum(od.passenger)
    print('Total number of passengers is: ', total_passengers)

    # Check the od if they are in the selected zones and get the index of each od
    in_zones = np.logical_and(np.isin(od.fromZone, selected_zones.zone), np.isin(od.toZone, selected_zones.zone))
    # Get only the od in the selected zones
    od_sel = od[in_zones]
    # Get the number of trips in the area of interest
    trips_in_area = od_sel.shape[0]
    print('Trips within area of interest :', trips_in_area)
    print('Median passenger/trip within area of interest:', np.median(od_sel.passenger))
    # Get the total number of passengers in the area of interest
    passengers_area_interest = np.sum(od_sel.passenger)
    print('Number of passengers in the area of interest is :', passengers_area_interest)

    # Remove internal zone trips
    size = od_sel.shape[0]
    internal = np.logical_and(od_sel.fromZone == od_sel.toZone, od_sel.toZone == od_sel.fromZone)
    od_internal = od_sel[internal]
    od_sel = od_sel[internal == False]

    # Print the internal trips and the updated output without the internal trips
    print('Internal trips: ', size - od_sel.shape[0])
    print('Trips without internals: ', od_sel.shape[0], ' / ', trips_in_area)
    print('Median passenger/trip without internals: ', np.median(od_sel.passenger), '\n')
    print('Number of passengers without internal is: ', np.sum(od_sel.passenger), ' of ', passengers_area_interest)

    # Get distance and divide od (from zone/to zone) and create the overall output with all details
    distance = np.zeros(od_sel.shape[0], dtype=[('distance30km', 'f8')])
    od_from_zone = np.array(od_sel.fromZone, dtype=[('fromZone', 'i8')])
    od_to_zone = np.array(od_sel.toZone, dtype=[('toZone', 'i8')])
    od_pass = np.array(od_sel.passenger, dtype=[('passenger', 'f8')])
    od_sel = rfn.merge_arrays((od_from_zone, od_to_zone, od_pass, distance), asrecarray=True)

    print(f'Length of selected OD {len(od_sel)} and length of OD distance {len(od_distance)}')
    print('Start long loop now.')
    # Debug od distance if the are not found in the selected od matrix
    index_of_od_in_od_distance = 0
    for od in od_sel:
        not_found = True
        while not_found:
            if (od_distance[index_of_od_in_od_distance].fromZone == od.fromZone) & \
                    (od_distance[index_of_od_in_od_distance].toZone == od.toZone):
                if od_distance[index_of_od_in_od_distance].distance > 30000:
                    od.distance30km = 1
                    index_of_od_in_od_distance = index_of_od_in_od_distance + 1
                    not_found = False
                    continue
                else:
                    not_found = False
                    index_of_od_in_od_distance = index_of_od_in_od_distance + 1
                    continue
            else:
                index_of_od_in_od_distance = index_of_od_in_od_distance + 1

    # Save the selected od matrix
    filename = 'output/demand_' + str(minimal_passengers) + 'pass_' + str(threshold) + 'm_' + \
               '_zones_' + str(nr_zones_to_connect_tt) + '_stations_' + str(nr_stations_to_connect_euclidean) + '.csv'
    np.savetxt(filename, od_sel, delimiter=',', header="fromZone, toZone, passenger, distance > 30 km",
               fmt="%i,%i,%f, %i")

    # Save the od internal trips
    filename_int = 'output/demand_' + str(minimal_passengers) + 'pass_' + str(threshold) + 'm_' + \
                   '_zones_' + str(nr_zones_to_connect_tt) + '_stations_' + str(
        nr_stations_to_connect_euclidean) + '_internal.csv'
    np.savetxt(filename_int, od_internal, delimiter=',',
               header="zone, fromZone, toZone", fmt="%i,%i,%f")

    # Save the selected od matrix in a pickle file
    filename_demand = 'output/pickle/demand_' + str(minimal_passengers) + 'pass_' + \
                      str(threshold) + 'm_' + '_zones_' + str(nr_zones_to_connect_tt) + \
                      '_stations_' + str(nr_stations_to_connect_euclidean) + '.pickle'
    od_sel.dump(filename_demand)

    # Print a message to inform that the demand on the selected zone is done.
    print('Demand on the selected zone is done. Look in the output file for saved csv and pickle.')

    return od_sel


def get_travel_time_selected_zones(path, selected_zones, threshold, threshold_euclidean, threshold_tt, sbb_nodes):
    """
    Function that get the od for travel time in the selected zones
    :param sbb_nodes: sbb stations with names : zone, SbbID, ViriatoID, NodeName, xcoord, ycoord, Code
    :param threshold_tt: travel time threshold that equals to the number of zones to connect
    :param threshold_euclidean: euclidean threshold that equals to the number of stations to connect
    :param path: to the TT File with all zones
    :param selected_zones: zones to be considered, output of findClosestStation method
    :param threshold: distance to consider a zone or not for zone selection
    :return: OD traveltime of a subset of all zones
    """
    # Initialize parameters
    k_2 = threshold_tt + 2  # Get the traveltime to the k + 2 closest stations
    station_id = []
    tt_array_reduced = []

    # Upload the original od matrix
    print('\nStart reading the original od matrix for travel time.')
    od = np.genfromtxt(path)
    od = np.core.records.fromarrays(od.transpose(),
                                    names='fromZone, toZone, traveltime',
                                    formats='i8, i8, f8')
    od.dtype.names = 'fromZone', 'toZone', 'tt'

    # Print a message to inform the beginning of travel time for each selected zone
    print('Beginning of the fetching the travel time for each selected zone.')

    # Trips in area of interest
    in_zones = np.logical_and(np.isin(od.fromZone, selected_zones.zone), np.isin(od.toZone, selected_zones.zone))
    od_sel = od[in_zones]

    # Remove all stations without commercial stop
    sbb_nodes = sbb_nodes[sbb_nodes.commercial_stop == 1]

    # Filter the travel time matrix and select and save only the needed travel times
    initializer = True
    for zone in selected_zones.zone:
        # Add closest according to travel time
        # Select all travel time fromZone to allZones
        sub_array_travel_time = od_sel[od_sel.fromZone == zone]
        # Select all travel time fromZone to stationZone
        sub_array_travel_time = sub_array_travel_time[np.isin(sub_array_travel_time.toZone, sbb_nodes.zone)]
        # Select closest stationZones from fromZone
        index_closest_travel_time = np.argpartition(sub_array_travel_time.tt, k_2)[0:k_2]
        sub_array_travel_time = sub_array_travel_time[index_closest_travel_time]
        # Select all stations within closest zones
        stations2 = sbb_nodes.ViriatoID[
            np.isin(sbb_nodes.zone, sub_array_travel_time.toZone)]
        # Append ID
        station_id = np.append(station_id, stations2)

        # Bring it together in a nice form to save afterwards as text file
        a = np.reshape(sub_array_travel_time.fromZone, (sub_array_travel_time.shape[0], 1))
        b = np.reshape(sub_array_travel_time.toZone, (sub_array_travel_time.shape[0], 1))
        c = np.reshape(sub_array_travel_time.tt, (sub_array_travel_time.shape[0], 1))

        if initializer:
            tt_array_reduced = np.concatenate((a, b, c), axis=1)
            initializer = False
        else:
            abc = np.concatenate((a, b, c), axis=1)
            tt_array_reduced = np.append(tt_array_reduced, abc, axis=0)

    # Get also the travel time of the closest zones by euclidean distance
    closest_stations_to_zone = get_all_k_closest_stations_to_zone(sbb_nodes, selected_zones, threshold_euclidean)
    # Add travel time to the closest stations
    for close_stations in closest_stations_to_zone:
        origin_zone = close_stations[0]
        station_zone = close_stations[1]
        sub_array_travel_time = od_sel[od_sel.fromZone == origin_zone]
        sub_array_travel_time = sub_array_travel_time[sub_array_travel_time.toZone == station_zone]
        tt_origin_station_zone = sub_array_travel_time.tt
        # Bring it together in a nice form to concatenate
        a = np.reshape(origin_zone, (sub_array_travel_time.shape[0], 1))
        b = np.reshape(station_zone, (sub_array_travel_time.shape[0], 1))
        c = np.reshape(tt_origin_station_zone, (sub_array_travel_time.shape[0], 1))
        abc = np.concatenate((a, b, c), axis=1)
        tt_array_reduced = np.append(tt_array_reduced, abc, axis=0)

    tt_array_reduced = np.core.records.fromarrays(tt_array_reduced.transpose(),
                                                  names='fromZone, toZone, tt',
                                                  formats='i8, i8, f8')

    # Save the output
    filename = 'output/selected_zones_tt_reduced_' + str(threshold) + 'm' + '.csv'
    np.savetxt(filename, tt_array_reduced, delimiter=',', header="fromZone, toZone, min", fmt="%i,%i,%f")

    # Save the output in a pickle file
    filename_travel = 'output/pickle/selected_zones_tt_reduced_' + str(threshold) + 'm' + \
                      '.pickle'
    tt_array_reduced.dump(filename_travel)

    # Print message to inform that travel time is saved now
    print('Travel time within the selected zones is now saved in the output file in csv and pickle.')

    return tt_array_reduced


def get_od_departure_time(parameters, demand_selected_zones):
    """
    Function that creates the OD departure time array with the priority list
    :param parameters: class object that provides all the main code parameters
    :param demand_selected_zones: recarray that contains input from zones to zones, number of passenger and distance
    over 30 km yes or not.
    :return: OD departure time array.
    """
    # Time discretization for the passenger grouping and desired Travel time simulation
    p_od_dep_time_grouped = 'output/OD_desired_departure_time_' + str(parameters.th_zone_selection) + '_grouped.csv'
    time_discretization = datetime.timedelta(minutes=parameters.time_discretization)
    p_od_dep_time = 'output/OD_desired_departure_time_' + str(parameters.th_zone_selection) + '_' + \
                    str(parameters.time_discretization) + 'min.csv'
    read_od_dep_time = parameters.read_od_departure_file
    create_grouped_passenger = parameters.create_group_passengers

    # Read the od with departure time already computed
    print(f'Read od departure time file: {read_od_dep_time}')
    if read_od_dep_time:
        # Create passenger groups
        if create_grouped_passenger:
            # Load the od desired departure time matrix
            od_departure_time = np.genfromtxt(p_od_dep_time, delimiter=',', dtype=str, skip_header=1)
            od_departure_time = np.core.records.fromarrays(od_departure_time.transpose(),
                                                           names='fromZone, toZone, priority, desired_dep_time',
                                                           formats='i8, i8, f8,  U25')
            # Group the passengers with same od and desired dep time
            max_group_size = parameters.group_size_passenger
            od_departure_time = group_passengers(od_departure_time, max_group_size, parameters.th_zone_selection)
        else:
            # Load the od desired departure time with passengers group
            od_departure_time = np.genfromtxt(p_od_dep_time_grouped, delimiter=',', dtype=str, skip_header=1)
            od_departure_time = np.core.records.fromarrays(od_departure_time.transpose(),
                                                           names='fromZone, toZone, priority, desired_dep_time, '
                                                                 'group_size',
                                                           formats='i8, i8, f8,  U25, i8')
    # Create from scratch the od departure time
    else:
        print('Create from scratch od departure time')
        od_departure_time = od_with_departure_time(parameters, demand_selected_zones, time_discretization)
        # group the passengers with same o d and desired dep time
        max_group_size = parameters.group_size_passenger
        od_departure_time = group_passengers(od_departure_time, max_group_size, parameters.th_zone_selection)

    # Add the priority to od_departure_time (random priority between 0-1), for the passenger assignment
    od_departure_time = create_priority_list(od_departure_time, parameters)
    debug = False
    if debug:
        od_departure_time = od_departure_time[0:1000]
    return od_departure_time


def group_passengers(od_departure_time, max_group_size, threshold):
    """
    Function that groups the passengers based on their departure time with maximum group size.
    :param od_departure_time: numpy array of OD desired departure time.
    :param max_group_size: Main code parameter. Maximum number of passenger per group.
    :param threshold: Main code threshold parameter. Here it is to code track of the threshold in csv name.
    :return: od departure time with the groups.
    """
    print('Start of grouping passengers with max group size : %d ' % max_group_size)
    od_last_iteration = od_departure_time[0]
    group_size = 1
    od_list = list()

    for od in od_departure_time[1:]:
        if od == od_last_iteration:
            group_size = group_size + 1
        else:
            od_for_list = list(od_last_iteration)
            if group_size < max_group_size:
                od_for_list.append(group_size)
                od_list.append(od_for_list)
                group_size = 1
            else:
                while group_size >= max_group_size:
                    od_for_list.append(max_group_size)
                    od_list.append(od_for_list)
                    od_for_list = list(od_last_iteration)
                    group_size = group_size - max_group_size
                    if group_size == 0:
                        group_size = group_size + 1
                    elif group_size <= max_group_size:
                        od_for_list.append(group_size)
                        od_list.append(od_for_list)
                        group_size = 1
        od_last_iteration = od

    # Take care of the last od
    od_for_list = list(od_last_iteration)
    if group_size < max_group_size:
        od_for_list.append(group_size)
        od_list.append(od_for_list)
    else:
        while group_size >= max_group_size:
            od_for_list.append(max_group_size)
            od_list.append(od_for_list)
            od_for_list = list(od_last_iteration)
            group_size = group_size - max_group_size
            if group_size <= max_group_size and group_size != 0:
                od_for_list.append(group_size)
                od_list.append(od_for_list)

    passengers_before = od_departure_time.shape[0]
    passengers_after = np.asarray(od_list)
    passengers_after = passengers_after[:, 4].astype(np.int)
    passengers_after = np.sum(passengers_after)
    print('Passengers before grouping: %d' % passengers_before)
    print('Passengers after grouping: %d' % passengers_after)
    print('Size of passenger od matrix after grouping: ', len(od_list))

    od_departure_time = np.array([tuple(x) for x in od_list],
                                 dtype=[('fromZone', 'i'), ('toZone', 'i'), ('priority', 'f'),
                                        ('desired_dep_time', 'U18'), ('group_size', 'i')])

    # Transform the numpy array as a structured array
    od_departure_time = od_departure_time.view(np.recarray)

    # Save the output in csv file
    filename = 'output/OD_desired_departure_time_' + str(threshold) + '_grouped.csv'
    np.savetxt(filename, od_departure_time, delimiter=',',
               header="fromZone, toZone, priority, desired dep time, group_size",
               fmt="%i,%i,%f,%s,%i")

    return od_departure_time


def od_with_departure_time(parameters, demand_selected_zones, time_discretization):
    """
    Function that create the od array with the desired departure time.
    :param: parameters, class object that contains all the parameters of the main code
    :param: demand_selected_zones, recarray, contains the od demand selected zones
    :param: time_discretization, datetime step that comes from the main parameters
    :return: od with "fromZone, toZone, priority = 1, desired dep time"
    """
    print('Start simulation of the departure times of each passenger.')
    start_time = parameters.earliest_time
    end_time = parameters.latest_time
    time_interval = end_time - start_time
    from_zone = np.empty(0)
    to_zone = np.empty(0)
    nr_passengers = np.empty(0)
    desired_dep_time = np.empty(0)

    debug = False
    debug_n = 0
    print('Debug mode is ', str(debug))
    # Loop with all od pairs in the demand selected zone recarray
    for od in demand_selected_zones:
        debug_n = debug_n + 1
        if debug:
            if debug_n > 1000:
                break
        t = non_homo_poisson_simulator(od, start_time, time_interval, parameters)
        if len(t) == 0:  # Take care of od's with no simulated passengers
            continue
        timer_to_group = start_time + time_discretization
        # Identify the time slot of the passenger
        for time in t:
            # Increase the timer_to_group until it reaches the time of the passenger with time_discretization
            while time >= timer_to_group:
                timer_to_group = timer_to_group + time_discretization
            # Save the passenger
            from_zone = np.append(from_zone, od.fromZone)
            to_zone = np.append(to_zone, od.toZone)
            nr_passengers = np.append(nr_passengers, 1)  # Add 1 to each passenger for now as for the priority
            desired_dep_time = np.append(desired_dep_time, timer_to_group)

    # Build the od departure time array
    from_zone = np.array(from_zone, dtype=[('fromZone', 'i8')])
    to_zone = np.array(to_zone, dtype=[('toZone', 'i8')])
    nr_passengers = np.array(nr_passengers, dtype=[('priority', 'f8')])
    desired_dep_time = np.array(desired_dep_time, dtype=[('desired_dep_time', 'datetime64[m]')])
    od_departure_time = rfn.merge_arrays((from_zone, to_zone, nr_passengers, desired_dep_time), asrecarray=True)

    # Save the od desired departure time
    np.savetxt('output/OD_desired_departure_time_8000_10min.csv', od_departure_time, delimiter=',',
               header="fromZone, toZone, priority, desired dep time,", fmt="%i,%i,%f,%s")

    # Print the message
    print('OD with desired departure time is done.')

    return od_departure_time


def non_homo_poisson_simulator(od, start_time, time_interval, parameters):
    """
    function that provides the time slot for each passenger.
    :param parameters: class object with a list of parameters for the code
    :param od: od pair with all the attributes
    :param start_time: start time of the Viriato simulation time_window.from_time
    :param time_interval: equals the time window of the scenario
    :return: list of starting time for each passenger based on a non homogenous simulator.
    """
    trips_per_hour = get_correct_tph(od)
    lambda_t = {}  # passengers per hour
    for key, value in trips_per_hour.items():
        lambda_t[key] = value * od.passenger
    # Non homogeneous poisson process
    t = []
    t_return = []
    u_list = []
    w_list = []
    s = datetime.timedelta(hours=0)
    s_list = [s]  # List for debug reasons
    d_list = []
    n = 0
    m = 0
    lambda_max = max(lambda_t.values())
    while s < time_interval:
        u = random.uniform(0, 1)
        u_list.append(u)
        w = - np.log(u) / lambda_max
        w_list.append(w)
        s = s + datetime.timedelta(hours=w)
        s_list.append(s)
        d = random.uniform(0, 1)
        d_list.append(d)
        # Get the arrival rate of the simulated point
        lambda_s = start_time + s
        try:
            lambda_s = lambda_t[lambda_s.hour]
        except KeyError:
            lambda_s = 0
        # Accept or decline
        if d <= lambda_s / lambda_max:
            t.append(start_time + s)
            n = n + 1
        m = m + 1
    if n == 0:
        if od.passenger > 10:
            t_return = t
        return t_return
    # Check if the last entry is within the the time interval
    elif t[n - 1] <= start_time + time_interval:
        t_return = t
    # If the last value is outside time interval, return only n-1 values
    else:
        t_return = t[0:n - 1]
    return t_return


def get_correct_tph(od):
    """
    Function that gets the correct trips per hour.
    :param od: the od matrix with the column distance 30 km.
    :return: trips_per_hour: list of trips per hour.
    """
    if od.distance30km == 0:
        tph_short_distance = {6: 0.078814, 7: 0.137074, 8: 0.061484, 9: 0.026012}
        trips_per_hour = tph_short_distance
    else:
        tph_long_distance = {6: 0.182223, 7: 0.117423, 8: 0.051569, 9: 0.024712}
        trips_per_hour = tph_long_distance
    return trips_per_hour


def create_priority_list(od_departure_time, parameters):
    """
    function that creates the priority list for each passenger. It will be used for the passenger assignment.
    :param parameters: class object that contains all the main code parameters
    :param od_departure_time: record with all od, number of passenger and desired departure time
    :return: od_departure_time adds the random priority of each passenger
    """
    total_number_passenger = od_departure_time.shape[0]
    random_priority_values = list(np.random.uniform(0, 1, total_number_passenger))

    od_departure_time.priority = random_priority_values
    # Sort array by priorities
    od_departure_time.sort(order='priority')
    # Reverse the order
    od_departure_time = od_departure_time[::-1]
    return od_departure_time


def get_all_k_closest_stations_to_zone(sbb_nodes, selected_zones, k):
    """
    Function that get all the closest stations to zone.
    :param sbb_nodes: all the sbb nodes with attributes
    :param selected_zones: recarray of all the selected zones
    :param k: the number of stations that are the closest from the zone
    :return: list of the closest stations with zone, station distance, station id, station code
    """
    # Get coordinates for the selected zones and the sbb stations
    xy_zones = [selected_zones.xcoord, selected_zones.ycoord]
    xy_stations = [sbb_nodes.xcoord, sbb_nodes.ycoord]

    # Transform the list to an array
    xy_zones = np.asarray(xy_zones, dtype=float)
    xy_stations = np.asarray(xy_stations, dtype=float)

    # Distance euclidean
    if xy_zones.shape[0] == 2 and xy_zones.shape[1] != 2:
        xy_zones = np.transpose(xy_zones)
    if xy_stations.shape[0] == 2 and xy_stations.shape[1] != 2:
        xy_stations = np.transpose(xy_stations)

    rows, cols = np.indices((xy_zones.shape[0], xy_stations.shape[0]))
    distances = ((xy_zones[rows.ravel()] - xy_stations[cols.ravel()]) ** 2)
    distances = np.round(np.sqrt(distances.sum(axis=1)), 5)
    if xy_zones.shape[0] == 1:
        pass
    else:
        distances = np.reshape(distances, (xy_zones.shape[0], xy_stations.shape[0]))

    # Get the closest stations with the euclidean distance by a list
    list_of_stations_euclidean = []
    for i in range(0, selected_zones.shape[0]):  # loop trough all home zone
        # identify 3 closest stations euclidean distance
        idx = np.argpartition(distances[i, :], k)
        origin_zone = selected_zones.zone[i]
        stations_zone_euclidean = sbb_nodes.zone[idx[0:k]]
        station_id = sbb_nodes.ViriatoID[idx[0:k]]
        station_code = sbb_nodes.Code[idx[0:k]]
        for j in range(0, station_id.shape[0]):
            list_of_stations_euclidean.append([origin_zone, stations_zone_euclidean[j], station_id[j], station_code[j]])
    return list_of_stations_euclidean


def closest_stations_to_zone_transform_record(closest_stations_to_zone):
    """
    Function that forms the closest stations to zone as a dictionary
    :param closest_stations_to_zone: list of the closest stations with zone, station distance, station id, station code
    :return: list of the closest stations with zone, station distance, station id, station code
    """
    closest_stations_to_zone_dictionary = {}

    for close_station in closest_stations_to_zone:
        from_zone = close_station[0]
        to_station_zone = close_station[1]
        station_id = close_station[2]
        station_code = close_station[3]

        if not closest_stations_to_zone_dictionary.__contains__(from_zone):
            closest_stations_to_zone_dictionary[from_zone] = {}
        closest_stations_to_zone_dictionary[from_zone][station_code] = {'station_zone': to_station_zone,
                                                                        'station_code': station_code,
                                                                        'station_ID': station_id}

    return closest_stations_to_zone_dictionary


def identify_stations_candidates(closest_stations_to_zone, k2, sbb_nodes, zone, tt_from_origin_zone,
                                 station_candidates):
    """
    Function that keeps the station that are close to the station with respect to k2
    :param closest_stations_to_zone: list of the closest stations with zone, station distance, station id, station code
    :param k2: Threshold of number of zones to connect
    :param sbb_nodes: all the sbb nodes with attributes
    :param zone: the origin zone to check
    :param tt_from_origin_zone: travel time from origin to zone
    :param station_candidates: empty dictionary to be filled
    :return: station candidates list with the travel time to zone
    """
    # By euclidean distance
    if not station_candidates.__contains__(zone):
        # Copy is needed as we do not want to adapt the closest stations to zone dict
        station_candidates[zone] = copy.deepcopy(closest_stations_to_zone[zone])
    else:
        return station_candidates

    for station_code, value in station_candidates[zone].items():
        try:
            tt_origin_station = tt_from_origin_zone.tt[np.isin(tt_from_origin_zone.toZone, value['station_zone'])][0]
        except ValueError:
            with open('tt_to_add.txt', 'a') as file:
                file.write('\n from ' + str(tt_from_origin_zone) + ' to ' + str(value['station_zone']))
            print('Travel time is not in data, in the function identify stations candidates, helpers.py')
            continue
        station_candidates[zone][station_code].update({'tt_toStation': tt_origin_station})

    # Get all k2 closest zone with a station to origin zone according to the travel time
    if k2 < tt_from_origin_zone.tt.size:
        index_closest_travel_time = np.argpartition(tt_from_origin_zone.tt, k2)[0:k2]
        tt_from_origin_zone = tt_from_origin_zone[index_closest_travel_time]
        stations_tt = sbb_nodes[sbb_nodes.zone == tt_from_origin_zone.toZone]

        # Check each station and record the attributes for candidate
        for station in stations_tt:
            candidate_tt = {'station_zone': tt_from_origin_zone.toZone.item(), 'station_code': station.Code,
                            'station_ID': station.ViriatoID, 'tt_toStation': float(tt_from_origin_zone.tt)}
            if station.Code in station_candidates[zone].keys():
                continue
            station_candidates[zone][station.Code] = candidate_tt

    return station_candidates


def identify_dep_nodes_for_trip_at_station(timetable, parameters, time_at_station, value):
    """
    Function that identify the departure nodes in the time window to start a trip to a station
    :param timetable: Digraph of the timetable with waiting and transfers edges
    :param parameters: class object that contains all the parameters of the main code
    :param time_at_station: equals time of desired departure time plus the travel time to the station
    :param value:
    :return: list of departure nodes from a station
    """
    departure_nodes_of_station = list()

    for depart in value:
        if time_at_station + parameters.min_wait <= timetable.nodes[depart]['departureTime'] <= time_at_station + \
                parameters.max_wait:
            departure_nodes_of_station.append(depart)
        else:
            continue
    return departure_nodes_of_station


def create_zone_candidates_of_stations(station_candidates):
    """
    function that create the zones candidates for each station.
    :param station_candidates: dictionary with all the station that are the closest to each zone
    :return: zone_candidates: list of travel time zone to station with all the candidates
    """
    zone_candidates = {}

    for zone, station_candidates in station_candidates.items():
        for station, attr in station_candidates.items():
            if not zone_candidates.__contains__(station):
                zone_candidates[station] = {}
            if not zone_candidates[station].__contains__(zone):
                if 'tt_toStation' in attr.keys():
                    zone_candidates[station][zone] = attr['tt_toStation']
    return zone_candidates


def transform_odt_into_dict_key_source(odt_for_sp):
    """
    :param odt_for_sp: odt matrix with [0] origin, [1] destination, [2] priority, [3] group size
    :return: source-target dictionary with key origin, and value [(destination, group size)]
     for all destinations from specific origin
    """
    source_target_dict = dict()
    for od in odt_for_sp:
        if od[0] in source_target_dict:
            source_target_dict[od[0]].append((od[1], od[3]))
        else:
            source_target_dict[od[0]] = [(od[1], od[3])]
    return source_target_dict


def transform_edges_into_dict_key_target(list_edges):
    edges_by_target = dict()
    for edge in list_edges:
        if edge[1] in edges_by_target:
            edges_by_target[edge[1]].append(edge)
        else:
            edges_by_target[edge[1]] = [edge]
    return edges_by_target


def transform_edges_into_dict_key_source(list_edges):
    edges_by_target = dict()
    for edge in list_edges:
        if edge[0] in edges_by_target:
            edges_by_target[edge[0]].append(edge)
        else:
            edges_by_target[edge[0]] = [edge]
    return edges_by_target


def build_dict_from_viriato_object_train_id(trains_timetable):
    """
    :param trains_timetable: list of train's object.
    :return: converted dictionary keyed by key
    """
    return dict((train.id, train) for train in trains_timetable)


def build_dict_from_viriato_object_run_time_train_path_node_id(run_time):
    """
    method that create a dictionary with the key as train path node id.
    :param run_time: list of update times of train path nodes object.
    :return: converted dictionary keyed by key
    """
    return dict((node.train_path_node_id, node) for node in run_time.update_times_train_path_nodes)


def build_dict(seq, key):
    """
    :param seq: list of dictionaries
    :param key: key for the dictionaries
    :return: converted dictionary keyed by key
    """
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))


def pick_best_solution(set_solution):
    return set_solution[0]
