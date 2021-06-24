"""
Created on Sun Mar 14 2021

@author: BenMobility

Adaptive large neighboring search platform for railway repaired scheduling.
"""
# Import
import viriato_interface
import helpers
import datetime
import copy
import cProfile
import io
from operator import itemgetter
import networkx as nx
import numpy as np
import pstats
import neighbourhood_operators
import math
import timetable_graph
import pickle
import passenger_assignment


def start(timetable_initial_graph, infra_graph, trains_timetable, parameters):
    """
    function that prepare the needed input to start the alns and then call alns to get the set of solutions
    :param timetable_initial_graph: Digraph contains origin, station and destination nodes with the directed edges
    :param infra_graph: Multigraph contains nodes of the train stations and edges for tracks
    :param trains_timetable: list of trains viriato objects that contains the train path nodes with time
    :param parameters: a class object from the main code that contains all necessary parameters
    :return: set of solutions provided by the alns
    """
    print(f'Starting the ALNS with number of iterations: {parameters.number_iteration}')
    # Get the closed tracks from Viriato, the possessions are defined in Viriato
    print('Get the closed tracks.')
    possessions = viriato_interface.get_section_track_closures(parameters.time_window)
    parameters.disruption_time = (
        possessions[0].closure_time_window_from_node.from_time, possessions[0].closure_time_window_to_node.to_time)

    # Increase weight on the close track edges todo: change to remove edge after benchmark
    print('Increase the weights on the edges.')
    infra_graph, closed_section_track_ids = increase_weight_on_closed_tracks(infra_graph, possessions, parameters)

    # Get all the necessary inputs from the track occupancy
    print('Get the tracks information')
    track_info = TrackInformation(trains_timetable, closed_section_track_ids)

    # Work the magic of alns
    set_solution = alns_algorithm(timetable_initial_graph, infra_graph, trains_timetable, track_info, parameters)
    return set_solution


def increase_weight_on_closed_tracks(infra_graph, possessions, parameters):
    """
    :param infra_graph: Multigraph that contains nodes of the train stations and edges for tracks
    :param possessions: list of all closed tracks, return of viriato method closed section tracks
    :param parameters: Class object with all the parameters, here for the new weight for the closed tracks
    :return: updated infrastructure graph : closed tracks have new weight, initial_weight stored as an attribute and
    closed_section_tracks_ids is a list of all closed tracks
    """
    # Get the close section track ids and update the infrastructure graph
    closed_section_track_ids = []
    for track in possessions:
        if track.section_track_id not in closed_section_track_ids:
            closed_section_track_ids.append(track.section_track_id)
        else:
            continue

        # Get the closed track in the infrastructure graph
        tracks_between_nodes = infra_graph.adj[track.from_node_id][track.to_node_id]
        for key, value in tracks_between_nodes.items():
            if value['sectionTrackID'] != track.section_track_id:
                continue
            else:
                infra_graph[track.from_node_id][track.to_node_id][key]['initial_weight'] = \
                    infra_graph[track.from_node_id][track.to_node_id][key]['weight']
                infra_graph[track.from_node_id][track.to_node_id][key]['weight'] = parameters.weight_closed_tracks

    return infra_graph, closed_section_track_ids


class TrackInformation:
    def __init__(self, trains_timetable, closed_section_track_ids):
        self.name = 'TrackInfo'
        nr_usage_tracks, trains_on_closed_tracks, track_sequences_of_TPN, tpn_info, trainID_debugString, \
        tuple_key_value_of_tpn_ID_arrival, tuple_key_value_of_tpn_ID_departure = \
            used_tracks_all_trains(trains_timetable, closed_section_track_ids)

        self.nr_usage_tracks = nr_usage_tracks
        self.trains_on_closed_tracks = trains_on_closed_tracks
        self.track_sequences_of_TPN = track_sequences_of_TPN
        self.tpn_information = tpn_info
        self.trainID_debugString = trainID_debugString
        self.tuple_key_value_of_tpn_ID_arrival = tuple_key_value_of_tpn_ID_arrival
        self.tuple_key_value_of_tpn_ID_departure = tuple_key_value_of_tpn_ID_departure


def used_tracks_all_trains(trains_timetable, closed_section_track_ids):
    """
    function that gathers all the information into dictionary for all the trains.
    :param trains_timetable: list of viriato objects that contains all the trains timetable
    :param closed_section_track_ids: list of all the closed section track ids
    :return: multiple output, number of track used, ect.
    """
    nr_usage_tracks = dict()  # Dictionary of tracks used by train, key: train ID , value: number of usage
    train_id_debug_string = dict()  # Dictionary, key: trainID, value: debug string
    train_path_node_information = dict()  # Key : tpn_id value : {arrTime, depTime, fromNode, toNode, sectionTrackID,
    #                                                  sectionTrackID_toNextNode, nextNodeID, nextTPN_ID}
    tuple_key_value_of_train_path_node_id_arrival = dict()
    tuple_key_value_of_train_path_node_id_departure = dict()

    # Dictionary of the sequences of train path nodes, key:(sectionTrackID, fromNode, toNode) value: [TPN1, TPN2...]
    track_sequences_of_train_path_node = dict()

    # Identify all trains driving on closed tracks
    trains_on_closed_tracks = []
    for train in trains_timetable:
        try:
            train_id_debug_string[train.id] = train.debug_string

            used_tracks_single_train(closed_section_track_ids, nr_usage_tracks, train_path_node_information,
                                     track_sequences_of_train_path_node, train, trains_on_closed_tracks,
                                     tuple_key_value_of_train_path_node_id_arrival,
                                     tuple_key_value_of_train_path_node_id_departure, idx_start_delay=0)
        # Emergency bus
        except AttributeError:
            continue


    # sort the sequence tracks dict by arrival time from early to late
    for k, v in track_sequences_of_train_path_node.items():
        track_sequences_of_train_path_node[k] = sorted(v, key=itemgetter(1))

    return nr_usage_tracks, trains_on_closed_tracks, track_sequences_of_train_path_node, train_path_node_information, \
           train_id_debug_string, tuple_key_value_of_train_path_node_id_arrival, \
           tuple_key_value_of_train_path_node_id_departure


def used_tracks_single_train(closed_section_track_ids, nr_usage_tracks, train_path_node_information,
                             track_sequences_of_train_path_node, train, trains_on_closed_tracks,
                             tuple_key_value_of_train_path_node_id_arrival,
                             tuple_key_value_of_train_path_node_id_departure, idx_start_delay):
    """
    function that gathers all the information for one single train
    :param closed_section_track_ids: list of all the closed section track ids
    :param nr_usage_tracks: dict with the number tracks used
    :param train_path_node_information: dict with all the information on the train path node
    :param track_sequences_of_train_path_node: dict of the track sequences of the train path nodes
    :param train: viriato object of a train in the timetable
    :param trains_on_closed_tracks: list of trains on closed tracks
    :param tuple_key_value_of_train_path_node_id_arrival: tuple that contains key and value of train path node
    :param tuple_key_value_of_train_path_node_id_departure: tuple that contains key and value of train path node
    :param idx_start_delay: index of the starting delay equals to 0
    :return: updated parameters
    """
    # Start the loop for each node in the train path nodes
    start_train = True
    on_closed_track = False
    train_path_node_node_index = -1
    for node in train.train_path_nodes[idx_start_delay:]:
        train_path_node_node_index += 1
        if start_train:
            if node.section_track_id in closed_section_track_ids:
                idx = train.train_path_nodes.index(node)
                train_on_closed_track.idx = [idx]
                # Train found driving on closed track
                trains_on_closed_tracks.append(train_on_closed_track)

            # Fill the train path node information with the node id
            train_path_node_information[node.id] = {
                'ArrivalTime': node.arrival_time,
                'DepartureTime': node.departure_time,
                'RunTime': None, 'fromNode': None, 'toNode': node.node_id, 'SectionTrack': None,
                'nextTPN_ID': train.train_path_nodes[train_path_node_node_index + 1].id, 'TrainID': train.id}

            # Get the next section track and node id
            next_train_path_node_section_track = train.train_path_nodes[train_path_node_node_index + 1].section_track_id
            next_train_path_node_node_id = train.train_path_nodes[train_path_node_node_index + 1].node_id

            # Set the tuple key as arrival
            tuple_key = (None, None, node.node_id, 'arrival')

            # Get the track sequences of the train path node
            if not track_sequences_of_train_path_node.__contains__(tuple_key):
                track_sequences_of_train_path_node[tuple_key] = []
            track_sequences_of_train_path_node[tuple_key].append([node.id, node.arrival_time, train.id])
            tuple_key_value_of_train_path_node_id_arrival[node.id] = [tuple_key, [node.id, node.arrival_time, train.id]]

            # Set the tuple key as departure and get once again the track sequences of the train path node
            tuple_key = (next_train_path_node_section_track, node.node_id, next_train_path_node_node_id, 'departure')
            if not track_sequences_of_train_path_node.__contains__(tuple_key):
                track_sequences_of_train_path_node[tuple_key] = []
            track_sequences_of_train_path_node[tuple_key].append([node.id, node.departure_time, train.id])
            tuple_key_value_of_train_path_node_id_departure[node.id] = [tuple_key, [node.id, node.departure_time,
                                                                                    train.id]]

            # Save the node as the starting node to.
            from_node = node.node_id
            start_train = False
            continue

        if node.section_track_id in closed_section_track_ids:
            if on_closed_track:
                # Train has multiple paths on closed tracks
                train_on_closed_track.idx.extend([train.train_path_nodes.index(node)])
            else:
                train_on_closed_track = copy.deepcopy(train)
                idx = train.train_path_nodes.index(node)
                train_on_closed_track.idx = [idx]
                # Train found driving on closed track
                trains_on_closed_tracks.append(train_on_closed_track)
                on_closed_track = True

        # Create the track sequence of train path node list/dict
        to_node = node.node_id

        # This train has probably been cut to time window & AoI, leaves area and enters again the same node later
        if from_node == to_node:
            # Double check if the node is the last of the train path node.
            if node == train.train_path_nodes[-1]:
                train_path_node_information[node.id] = {
                    'ArrivalTime': node.arrival_time,
                    'DepartureTime': node.departure_time,
                    'RunTime': None, 'fromNode': None, 'toNode': node.node_id, 'SectionTrack': None,
                    'nextTPN_ID': None, 'TrainID': train.id}
            else:
                train_path_node_information[node.id] = {
                    'ArrivalTime': node.arrival_time,
                    'DepartureTime': node.departure_time,
                    'RunTime': None, 'fromNode': None, 'toNode': node.node_id, 'SectionTrack': None,
                    'nextTPN_ID': train.train_path_nodes[train_path_node_node_index + 1].id, 'TrainID': train.id}
                next_train_path_node_section_track = train.train_path_nodes[
                    train_path_node_node_index + 1].section_track_id
                next_train_path_node_node_id = train.train_path_nodes[train_path_node_node_index + 1].node_id

            # Set the tuple key as arrival
            tuple_key = (None, None, node.node_id, 'arrival')

            # Get the track sequences of the train path node
            if not track_sequences_of_train_path_node.__contains__(tuple_key):
                track_sequences_of_train_path_node[tuple_key] = []
            track_sequences_of_train_path_node[tuple_key].append([node.id, node.arrival_time, train.id])
            tuple_key_value_of_train_path_node_id_arrival[node.id] = [tuple_key, [node.id, node.arrival_time, train.id]]

            # Set the tuple key as departure and get once again the track sequences of the train path node
            tuple_key = (next_train_path_node_section_track, node.node_id, next_train_path_node_node_id, 'departure')
            if not track_sequences_of_train_path_node.__contains__(tuple_key):
                track_sequences_of_train_path_node[tuple_key] = []
            track_sequences_of_train_path_node[tuple_key].append([node.id, node.departure_time, train.id])
            tuple_key_value_of_train_path_node_id_departure[node.id] = [tuple_key, [node.id, node.arrival_time,
                                                                                    train.id]]
            from_node = to_node
            continue
        else:

            # Set the tuple key as arrival
            tuple_key = (node.section_track_id, from_node, to_node, 'arrival')
            if not track_sequences_of_train_path_node.__contains__(tuple_key):
                track_sequences_of_train_path_node[tuple_key] = []
            track_sequences_of_train_path_node[tuple_key].append([node.id, node.arrival_time, train.id])
            tuple_key_value_of_train_path_node_id_arrival[node.id] = [tuple_key, [node.id, node.arrival_time, train.id]]

            train_ends = False
            if train_path_node_node_index + 1 < len(train.train_path_nodes):
                next_train_path_node_id = train.train_path_nodes[train_path_node_node_index + 1].id
                next_train_path_node_node_id = train.train_path_nodes[train_path_node_node_index + 1].node_id
                next_train_path_node_section_track = \
                    train.train_path_nodes[train_path_node_node_index + 1].section_track_id
            else:
                next_train_path_node_id = None
                next_train_path_node_node_id = None
                next_train_path_node_section_track = None
                train_ends = True

            if next_train_path_node_node_id is not None and to_node != next_train_path_node_node_id:
                tuple_key = (next_train_path_node_section_track, to_node, next_train_path_node_node_id, 'departure')
                if not track_sequences_of_train_path_node.__contains__(tuple_key):
                    track_sequences_of_train_path_node[tuple_key] = []
                track_sequences_of_train_path_node[tuple_key].append([node.id, node.departure_time, train.id])
                tuple_key_value_of_train_path_node_id_departure[node.id] = [tuple_key, [node.id, node.departure_time,
                                                                                        train.id]]

            train_leaves_area = False
            if to_node == next_train_path_node_node_id:
                train_leaves_area = True
                next_train_path_node_id = None
                next_train_path_node_node_id = None
                next_train_path_node_section_track = None

                tuple_key = (next_train_path_node_section_track, to_node, None, 'departure')
                if not track_sequences_of_train_path_node.__contains__(tuple_key):
                    track_sequences_of_train_path_node[tuple_key] = []
                track_sequences_of_train_path_node[tuple_key].append([node.id, node.departure_time, train.id])
                tuple_key_value_of_train_path_node_id_departure[node.id] = [tuple_key, [node.id, node.departure_time,
                                                                                        train.id]]

            if not train_leaves_area and train_ends:  # End of a train
                tuple_key = (next_train_path_node_section_track, to_node, next_train_path_node_node_id, 'departure')
                if not track_sequences_of_train_path_node.__contains__(tuple_key):
                    track_sequences_of_train_path_node[tuple_key] = []
                track_sequences_of_train_path_node[tuple_key].append([node.id, node.departure_time, train.id])
                tuple_key_value_of_train_path_node_id_departure[node.id] = [tuple_key, [node.id, node.departure_time,
                                                                                        train.id]]

            train_path_node_information[node.id] = {
                'ArrivalTime': node.arrival_time,
                'DepartureTime': node.departure_time,
                'RunTime': node.minimum_run_time,
                'fromNode': from_node, 'toNode': to_node, 'SectionTrack': node.section_track_id,
                'nextTPN_ID': next_train_path_node_id, 'TrainID': train.id}

            from_node = to_node

        # Get the amount of trains running on a track
        if node.section_track_id is None:
            # Neglect the first node for the distance calculation
            continue
        if not nr_usage_tracks.__contains__(node.section_track_id):
            nr_usage_tracks[node.section_track_id] = 1
        else:
            nr_usage_tracks[node.section_track_id] += 1


def alns_algorithm(timetable_initial_graph, infra_graph, trains_timetable, track_info, parameters):
    # Set the parameters
    initial_timetable = helpers.Timetables()
    initial_timetable.initial_timetable_infeasible = copy.deepcopy(trains_timetable)
    weights = helpers.Weights()
    scores = helpers.Scores()
    number_usage = helpers.NumberUsage()
    probabilities = helpers.Probabilities(weights)
    parameters.trains_on_closed_track_initial_timetable_infeasible = \
        [train.id for train in track_info.trains_on_closed_tracks]
    parameters.train_ids_initial_timetable_infeasible = \
        [train.id for train in initial_timetable.initial_timetable_infeasible]

    # Set the number of iteration
    n_iteration = 0

    # Set the number of temperature changes and the starting temperature
    number_temperature_changes = 0
    temp_i = parameters.t_start

    # Set the iteration for the archives
    iterations_until_return_archives = parameters.number_iteration_archive
    return_to_archive_at_iteration = n_iteration + iterations_until_return_archives

    # Set the solution archive
    solution_archive = []

    # Set the multi-objective cost
    z_op_current, z_de_reroute_current, z_de_cancel_current, z_tt_current = [], [], [], []
    z_op_accepted, z_de_reroute_accepted, z_de_cancel_accepted, z_tt_accepted = [], [], [], []
    z_cur_accepted, z_cur_archived, temperature_it = [], [], []

    # Combine three different costs
    all_accepted_solutions = [z_op_accepted, z_de_reroute_accepted, z_de_cancel_accepted, z_tt_accepted]

    # Save values - current, accepted and archived
    z_for_pickle = {'z_op_cur': z_op_current,
                    'z_de_reroute_cur': z_de_reroute_current, 'z_de_cancel_cur': z_de_cancel_current,
                    'z_tt_cur': z_tt_current,
                    'z_op_acc': z_op_accepted,
                    'z_de_reroute_acc': z_de_reroute_accepted, 'z_de_cancel_acc': z_de_cancel_accepted,
                    'z_tt_acc': z_tt_accepted,
                    'z_cur_accepted': z_cur_accepted, 'z_cur_archived': z_cur_archived, 't_it': temperature_it}

    # Set the feasible timetable graphs to false
    feasible_timetable_graph, prime_feasible_timetable_graph = False, False

    # Start the iteration
    print('Start the iteration...')
    while any(t > 0 for t in temp_i) and n_iteration < parameters.number_iteration:
        print(f'\nIteration number: {n_iteration}')
        # Add an iteration in the count
        n_iteration += 1
        if feasible_timetable_graph:
            # Capture the current temperature
            temperature_it.append(temp_i.copy())
            # Get the trains timetable in a deepcopy
            trains_timetable = copy.deepcopy(timetable_solution_graph.time_table)
            # Get the track info
            track_info = TrackInformation(trains_timetable, parameters.closed_tracks)
            # Keep track of the changed trains in the current solution
            changed_trains = copy.deepcopy(timetable_solution_graph.changed_trains)
            # Identify the candidates for operators
            identify_candidates_for_operators(trains_timetable, parameters, timetable_solution_graph, changed_trains)
            # Get the number of usage and the selected operator
            number_usage, operator = select_operator(probabilities, number_usage, changed_trains, track_info,
                                                     parameters)

            # Print the selected operator
            print(f'Selected operator: {operator}')

            # Create a new timetable with the current solution but without flow
            timetable_prime_graph = timetable_solution_graph.graph

            # Create a dict of edges that combines origin to stations and stations to destination edges
            edges_o_stations_d = helpers.CopyEdgesOriginStationDestination(timetable_solution_graph.edges_o_stations_d)

            # Create a new empty solution
            timetable_solution_prime_graph = helpers.Solution()

            # Debug timetable graph
            print(f'Number of nodes in the timetable graph before operator = {len(timetable_prime_graph)}')

            # Apply the operator on the current solution
            print('Apply the operator on the current solution.')

            timetable_prime_graph, track_info, edges_o_stations_d, changed_trains, operator, \
            odt_facing_neighbourhood_operator, odt_priority_list_original = \
                apply_operator_to_timetable(operator, timetable_prime_graph, changed_trains, trains_timetable,
                                            track_info, infra_graph, edges_o_stations_d, parameters,
                                            odt_priority_list_original)

            # Debug timetable graph
            print(f'Number of nodes in the timetable graph after operator = {len(timetable_prime_graph)}')

            # Set the timetable_solution_graph parameters
            timetable_solution_prime_graph.edges_o_stations_d = edges_o_stations_d
            timetable_solution_prime_graph.timetable = trains_timetable
            timetable_solution_prime_graph.graph = timetable_prime_graph
            timetable_solution_prime_graph, timetable_prime_graph, odt_priority_list_original = \
                find_path_and_assign_pass_neighbourhood_operator(timetable_prime_graph,
                                                                 parameters,
                                                                 timetable_solution_prime_graph,
                                                                 edges_o_stations_d,
                                                                 odt_priority_list_original,
                                                                 odt_facing_neighbourhood_operator)
            timetable_solution_prime_graph.odt_priority_list_original = copy.deepcopy(odt_priority_list_original)
            timetable_solution_prime_graph.total_dist_train = distance_travelled_all_trains(trains_timetable,
                                                                                            infra_graph, parameters)
            timetable_solution_prime_graph.deviation_reroute_timetable = deviation_reroute_timetable(trains_timetable,
                                                                                                     initial_timetable,
                                                                                                     changed_trains,
                                                                                                     parameters)
            timetable_solution_prime_graph.deviation_cancel_timetable = deviation_cancel_timetable(trains_timetable,
                                                                                                   initial_timetable,
                                                                                                   changed_trains,
                                                                                                   parameters)
            timetable_solution_prime_graph.changed_trains = changed_trains
            timetable_solution_prime_graph.set_of_trains_for_operator = parameters.set_of_trains_for_operator

            # Archive the current solution
            z_op_current.append(timetable_solution_prime_graph.total_dist_train)
            z_de_reroute_current.append(timetable_solution_prime_graph.deviation_reroute_timetable)
            z_de_cancel_current.append(timetable_solution_prime_graph.deviation_cancel_timetable)
            z_tt_current.append(timetable_solution_prime_graph.total_traveltime)

            # Printout the current results
            print('z_o_current : ', timetable_solution_prime_graph.total_dist_train,
                  '\n z_d_reroute_current : ', timetable_solution_prime_graph.deviation_reroute_timetable,
                  '\n z_d_cancel_current : ', timetable_solution_prime_graph.deviation_cancel_timetable,
                  '\n z_p_current : ', timetable_solution_prime_graph.total_traveltime)

            # Archive limited to 80 solutions (memory issue, could be increased)
            timetable_solution_graph, scores, accepted_solution, archived_solution = \
                archiving_acceptance_rejection(timetable_solution_graph, timetable_solution_prime_graph, operator,
                                               parameters, scores, temp_i, solution_archive)

            # Save the current solution, the accepted solution
            z_cur_accepted.append(accepted_solution)
            z_cur_archived.append(archived_solution)

            z_op_accepted.append(timetable_solution_graph.total_dist_train)
            z_de_reroute_accepted.append(timetable_solution_graph.deviation_reroute_timetable)
            z_de_cancel_accepted.append(timetable_solution_graph.deviation_cancel_timetable)
            z_tt_accepted.append(timetable_solution_graph.total_traveltime)

            # Save the multi objective results into a pickle
            pickle_results(z_for_pickle, 'output/pickle/z_pickle.pkl')

            # Printout the accepted solution
            print('z_o_accepted : ', timetable_solution_graph.total_dist_train,
                  '\n z_d_reroute_accepted : ', timetable_solution_graph.deviation_reroute_timetable,
                  '\n z_d_cancel_accepted : ', timetable_solution_graph.deviation_cancel_timetable,
                  '\n z_p_accepted : ', timetable_solution_graph.total_traveltime)

            # Print out the temperature and the number of iteration
            print('temperature : ', temp_i, ' iteration : ', n_iteration)

            # Update the temperature
            temp_i, number_temperature_changes = update_temperature(temp_i, n_iteration, number_temperature_changes,
                                                                    all_accepted_solutions, parameters)

            # Select a solution from the archive periodically
            timetable_solution_graph, iterations_until_return_archives, return_to_archive_at_iteration = \
                periodically_select_solution(solution_archive, timetable_solution_graph, n_iteration,
                                             iterations_until_return_archives, parameters,
                                             return_to_archive_at_iteration)

            # Update the weights
            weights, scores, probabilities = update_weights(weights, scores, number_usage, number_temperature_changes,
                                                            probabilities, parameters)
        if not prime_feasible_timetable_graph:
            # Add the temperature for the iteration
            temperature_it.append(temp_i)

            # Restore the disruption infeasibility's at the first iteration
            changed_trains = neighbourhood_operators.restore_disruption_feasibility(infra_graph, track_info, parameters)

            # Get trains in the time window and area
            trains_timetable = get_all_trains_cut_to_time_window_and_area(parameters)

            # Save the initial timetable feasible graph
            parameters.initial_timetable_feasible = copy.deepcopy(trains_timetable)

            # Get all the information in one place for the tracks
            track_info = TrackInformation(trains_timetable, parameters.closed_tracks)

            # Create the restored feasibility graph
            timetable_prime_graph, edges_o_stations_d = \
                timetable_graph.create_restored_feasibility_graph(trains_timetable, parameters)

            # Record the current solution
            timetable_solution_prime_graph = helpers.Solution()
            timetable_solution_prime_graph.timetable = copy.deepcopy(trains_timetable)

            # Passenger assignment and compute the objective for passenger
            solutions, timetable_prime_graph, odt_priority_list_original =\
                find_path_and_assign_pass(timetable_prime_graph, parameters, timetable_solution_prime_graph,
                                          edges_o_stations_d)

            # Record the results of the current solution timetable
            timetable_solution_prime_graph.odt_priority_list_original = copy.deepcopy(odt_priority_list_original)
            timetable_solution_prime_graph.total_dist_train = distance_travelled_all_trains(trains_timetable,
                                                                                            infra_graph, parameters)
            timetable_solution_prime_graph.deviation_reroute_timetable = deviation_reroute_timetable(trains_timetable,
                                                                                                     initial_timetable,
                                                                                                     changed_trains,
                                                                                                     parameters)
            timetable_solution_prime_graph.deviation_cancel_timetable = deviation_cancel_timetable(trains_timetable,
                                                                                                   initial_timetable,
                                                                                                   changed_trains,
                                                                                                   parameters)

            # Check if the current solution has a value for deviation, if not, it means that the algorithm platform
            # has not been restarted for the original timetable
            if timetable_solution_prime_graph.deviation_reroute_timetable == 0 \
                    and timetable_solution_prime_graph.deviation_cancel_timetable == 0:
                raise Exception('The deviation restored feasibility is 0, we need to restart the Viriato algorithm'
                                'platform')

            # Record all the results for the current solution
            timetable_solution_prime_graph.set_of_trains_for_operator['Cancel'] = \
                copy.deepcopy(parameters.set_of_trains_for_operator['Cancel'])
            timetable_solution_prime_graph.set_of_trains_for_operator['Delay'] =\
                copy.deepcopy(parameters.set_of_trains_for_operator['Delay'])
            timetable_solution_prime_graph.graph = timetable_graph.copy_graph_with_flow(timetable_prime_graph)
            timetable_solution_prime_graph.changed_trains = copy.deepcopy(changed_trains)
            timetable_solution_prime_graph.edges_o_stations_d = \
                helpers.CopyEdgesOriginStationDestination(edges_o_stations_d)

            # Save the current and the accepted solution
            z_op_current.append(timetable_solution_prime_graph.total_dist_train)
            z_de_reroute_current.append(timetable_solution_prime_graph.deviation_reroute_timetable)
            z_de_cancel_current.append(timetable_solution_prime_graph.deviation_cancel_timetable)
            z_tt_current.append(timetable_solution_prime_graph.total_traveltime)
            z_op_accepted.append(timetable_solution_prime_graph.total_dist_train)
            z_de_reroute_accepted.append(timetable_solution_prime_graph.deviation_reroute_timetable)
            z_de_cancel_accepted.append(timetable_solution_prime_graph.deviation_cancel_timetable)
            z_tt_accepted.append(timetable_solution_prime_graph.total_traveltime)
            z_cur_accepted.append(True)
            z_cur_archived.append(True)

            # Pickle the results
            pickle_results(z_for_pickle, 'output/pickle/z_pickle.pkl')

            # Printout the results
            print(f'z_o: {timetable_solution_prime_graph.total_dist_train} \n'
                  f'z_d_reroute: {timetable_solution_prime_graph.deviation_reroute_timetable}\n'
                  f'z_d_cancel: {timetable_solution_prime_graph.deviation_cancel_timetable}\n'
                  f'z_p: {timetable_solution_prime_graph.total_traveltime}')

            # Save the solution in the archives
            solution_archive.append(timetable_solution_prime_graph)

            # Copy the solution for the next iteration
            timetable_solution_graph = get_solution_graph_with_copy_solution_prime(timetable_solution_prime_graph,
                                                                                   parameters)

            # Set the feasible graph and prime graph to true for next iteration
            feasible_timetable_graph = True
            prime_feasible_timetable_graph = True

    # Pickle the archive solutions
    pickle_archive_with_changed_trains(solution_archive)
    pickle_results(weights, 'output/pickle/weights.pkl')
    pickle_results(probabilities, 'output/pickle/probabilities.pkl')

    return solution_archive


def start_end_profiler(filename=None, start_profiling=False, end_profiling=False, pr=None):
    # Start the profiler
    if start_profiling:
        pr = cProfile.Profile()
        pr.enable()
        return pr
    # End the profiler
    if end_profiling:
        pr.disable()
        s = io.StringIO()
        sorts_selected = 'cumulative'  # or it could be 'totalTime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sorts_selected)
        ps.print_stats()
        with open('output/profiler/Profile_' + sorts_selected + filename + '.txt', 'w+') as f:
            f.write(s.getvalue())


def identify_candidates_for_operators(trains_timetable, parameters, timetable_solution_graph, changed_trains):
    timetable_prime_graph = timetable_solution_graph.graph
    parameters.set_of_trains_for_operator['Cancel'] = timetable_solution_graph.set_of_trains_for_operator['Cancel'].copy()
    parameters.set_of_trains_for_operator['Delay'] = timetable_solution_graph.set_of_trains_for_operator['Delay'].copy()
    # Loop through all the train in Viriato
    for train in trains_timetable:
        # Set commercial stop at 0
        comm_stops = 0
        try:
            # Compute the number of commercial stop in the train path
            for train_path_node in train.train_path_nodes:
                if train_path_node.stop_status == 'commercial_stop':
                    comm_stops += 1
            # If the number of commercial stop is less than 2, cancel the train
            if comm_stops <= 2:
                parameters.set_of_trains_for_operator['Cancel'].append(train.id)
        except AttributeError:
            # Compute the number of commercial stop in the train path
            for train_path_node in train['TrainPathNodes']:
                if train_path_node['StopStatus'] == 'commercial_stop':
                    comm_stops += 1
            # If the number of commercial stop is less than 2, cancel the train
            if comm_stops <= 2:
                parameters.set_of_trains_for_operator['Cancel'].append(train['ID'])
    # Set all train flows list
    all_train_flows = {}
    for (u, v, attr) in timetable_prime_graph.edges.data():
        if 'flow' in attr.keys():
            # Check only driving edges
            if attr['type'] != 'driving':
                continue
            # Check if bus in the attribute of this edge
            if 'bus_id' in attr.keys():
                # If the list does not contain yet the bus id, we will add it with the attributes
                if not all_train_flows.__contains__(attr['bus_id']):
                    all_train_flows[attr['bus_id']] = {'total_flow': sum(attr['flow']), 'nb_of_edges_with_flow': 1,
                                                       'avg_flow': sum(attr['flow'])}
                # If it contains, only add the flow, add +1 edge for this bus and compute the average again
                else:
                    train_flow = all_train_flows[attr['bus_id']]  # todo: make sure if it should not be bus id
                    train_flow['total_flow'] += sum(attr['flow'])
                    train_flow['nb_of_edges_with_flow'] += 1
                    train_flow['avg_flow'] = train_flow['total_flow'] / train_flow['nb_of_edges_with_flow']
            # If not bus, same as previously but add the attributes for the train
            else:
                if not all_train_flows.__contains__(attr['train_id']):
                    all_train_flows[attr['train_id']] = {'total_flow': sum(attr['flow']), 'nb_of_edges_with_flow': 1,
                                                         'avg_flow': sum(attr['flow'])}
                else:
                    train_flow = all_train_flows[attr['train_id']]
                    train_flow['total_flow'] += sum(attr['flow'])
                    train_flow['nb_of_edges_with_flow'] += 1
                    train_flow['avg_flow'] = train_flow['total_flow'] / train_flow['nb_of_edges_with_flow']

    overall_flow_trains = 0
    overall_flow_bus = 0
    # Check each train in all train flows and compute the overall flow.
    for train_id, attr in all_train_flows.items():
        bus = False
        if isinstance(train_id, str):  # 'EmergencyBus':
            bus = True
            max_cap = parameters.bus_capacity
            overall_flow_bus += attr['total_flow']
        else:
            max_cap = parameters.train_capacity
            overall_flow_trains += attr['total_flow']
        # If the average flow is less than 50% of the maximum capacity, the train can be cancel or delayed
        if attr['avg_flow'] <= 0.5 * max_cap:
            parameters.set_of_trains_for_operator['Cancel'].append(train_id)
            parameters.set_of_trains_for_operator['Delay'].append(train_id)
    # Check for every changed train if the action is reroute, short turn or cancel from.
    for train_id, values in changed_trains.items():
        if values['Action'] in ['Reroute', 'ShortTurn', 'CancelFrom']:
            # Put again the train_id in the set of trains operator
            if train_id in parameters.set_of_trains_for_operator['Cancel']:
                parameters.set_of_trains_for_operator['Cancel'].append(train_id)
            if train_id in parameters.set_of_trains_for_operator['Delay']:
                parameters.set_of_trains_for_operator['Delay'].append(train_id)

    print('Total flow trains', overall_flow_trains, 'Total flow bus', overall_flow_bus)


def select_operator(probabilities, number_usages, changed_trains, track_info, parameters):
    # Inform the user that there is a train on the close tracks
    if len(track_info.trains_on_closed_tracks) > 0:
        print('trains running on closed track !')

    # Set the operator to None before the loop
    operator = None
    # Loop until the operator is set
    while operator is None:
        rd = np.random.uniform(0.0, 1.0)
        if rd <= probabilities.cc:  # complete cancel
            number_usages.cc += 1
            operator = 'Cancel'

        elif probabilities.cc < rd <= probabilities.pc:  # partial cancel
            number_usages.pc += 1
            operator = 'CancelFrom'

        # elif probabilities.cc < rd <= probabilities.cd:  # delay
        #     number_usages.cd += 1
        #     operator = 'Delay'
        #
        # elif probabilities.cd < rd <= probabilities.pd:  # partial delay
        #     number_usages.pd += 1
        #     operator = 'DelayFrom'
        #
        # elif probabilities.pd < rd <= probabilities.et:  # emergency train
        #     number_usages.et += 1
        #     operator = 'EmergencyTrain'

        elif probabilities.pc < rd <= probabilities.eb:  # emergency bus
            number_usages.eb += 1
            operator = 'EmergencyBus'

        elif rd > probabilities.eb:
            list_return_candidates = candidates_for_return_operator(changed_trains, parameters)
            if len(list_return_candidates) <= 3:
                continue
            number_usages.ret += 1
            operator = 'Return'

    return number_usages, operator


def candidates_for_return_operator(changed_trains, parameters):
    # List of the possible action for return
    name_action = ['EmergencyTrain', 'EmergencyBus', 'Return', 'Reroute']
    # Built a list of possible return candidates with three conditions, no action, not on closed tracks, in infeasible
    list_return_candidates = [train_id for train_id, attributes in changed_trains.items()
                              if attributes['Action'] not in name_action
                              and train_id not in parameters.trains_on_closed_track_initial_timetable_infeasible
                              and train_id in parameters.train_ids_initial_timetable_infeasible]
    return list_return_candidates


def copy_graph_and_remove_flow(timetable_solution_graph
                               ):
    """
    method that copy the nodes and edges attributes from the current solution to a new timetable without the flow
    :param timetable_solution_graph
    : the current timetable solution digraph
    :return: a new timetable without flow (type: digraph)
    """
    # Create a new timetable graph with edges and nodes
    timetable_prime_graph = nx.DiGraph()
    edges_G = []
    attr_edges_G = {}

    # Get the node attributes from the current solution timetable
    nodes = {n: v for n, v in timetable_solution_graph
        .nodes(data=True)}

    # Get the edge attributes and applied 0 to the flow for each edge with flow in their attribute
    for u, v, attr in timetable_solution_graph\
            .edges(data=True):
        edges_G.append((u, v, {'weight': attr['weight']}))
        attr_copied = attr.copy()
        if 'flow' in attr_copied.keys():
            attr_copied['flow'] = 0
        attr_edges_G[(u, v)] = attr_copied

    # Delete the current solution timetable
    del timetable_solution_graph


    # Add the nodes and edges to prime timetable
    timetable_prime_graph.add_nodes_from(nodes.keys())
    nx.set_node_attributes(timetable_prime_graph, nodes)
    timetable_prime_graph.add_weighted_edges_from(edges_G)
    nx.set_edge_attributes(timetable_prime_graph, attr_edges_G)

    return timetable_prime_graph


def apply_operator_to_timetable(operator, timetable_prime_graph, changed_trains, trains_timetable, track_info,
                                infra_graph, edges_o_stations_d, parameters, odt_priority_list_original):
    # Apply the selected operator
    if operator == 'Cancel':
        # Cancel random train
        changed_trains, timetable_prime_graph, track_info, edges_o_stations_d, odt_facing_neighbourhood_operator,\
        odt_priority_list_original = neighbourhood_operators.operator_cancel(timetable_prime_graph,
                                                                             changed_trains,
                                                                             trains_timetable,
                                                                             track_info,
                                                                             edges_o_stations_d,
                                                                             parameters,
                                                                             odt_priority_list_original)

    elif operator == 'CancelFrom':
        changed_trains, timetable_prime_graph, train_id_to_cancel_from, track_info, edges_o_stations_d, \
        odt_facing_neighbourhood_operator, odt_priority_list_original = \
            neighbourhood_operators.operator_cancel_from(timetable_prime_graph,
                                                         changed_trains,
                                                         trains_timetable,
                                                         track_info,
                                                         infra_graph,
                                                         edges_o_stations_d,
                                                         parameters,
                                                         odt_priority_list_original)

    elif operator == 'Delay':
        # Delay random train
        print('Delay is not implemented')
        odt_facing_neighbourhood_operator = None
        # changed_trains, timetable_prime_graph, train_id_to_delay, track_info, edges_o_stations_d, \
        # odt_facing_neighbourhood_operator, odt_priority_list_original = \
        #     neighbourhood_operators.operator_complete_delay(timetable_prime_graph,
        #                                                     changed_trains,
        #                                                     trains_timetable,
        #                                                     track_info,
        #                                                     infra_graph,
        #                                                     edges_o_stations_d,
        #                                                     parameters,
        #                                                     odt_priority_list_original)

    elif operator == 'DelayFrom':

        print('Delay is not implemented')
        odt_facing_neighbourhood_operator = None

        # changed_trains, timetable_prime_graph, train_id_to_delay, track_info, edges_o_stations_d,\
        # odt_facing_neighbourhood_operator, odt_priority_list_original = \
        #     neighbourhood_operators.operator_part_delay(timetable_prime_graph,
        #                                                 changed_trains,
        #                                                 trains_timetable,
        #                                                 track_info,
        #                                                 infra_graph,
        #                                                 edges_o_stations_d,
        #                                                 parameters,
        #                                                 odt_priority_list_original)

    elif operator == 'EmergencyTrain':

        print('EmergencyTrain is not implemented')
        odt_facing_neighbourhood_operator = None

        # emergency_train = viriato_interface.get_emergency_train()
        #
        # changed_trains, timetable_prime_graph, train_id_to_delay, track_info, edges_o_stations_d, \
        # odt_facing_neighbourhood_operator, odt_priority_list_original = \
        #     neighbourhood_operators.operator_emergency_train(timetable_prime_graph,
        #                                                      changed_trains,
        #                                                      emergency_train,
        #                                                      trains_timetable,
        #                                                      track_info,
        #                                                      infra_graph,
        #                                                      edges_o_stations_d,
        #                                                      parameters,
        #                                                      odt_priority_list_original)

    elif operator == 'EmergencyBus':
        changed_trains, timetable_prime_graph, bus_id, track_info, edges_o_stations_d,\
        odt_facing_neighbourhood_operator, odt_priority_list_original = \
            neighbourhood_operators.operator_emergency_bus(timetable_prime_graph,
                                                           changed_trains,
                                                           trains_timetable,
                                                           track_info,
                                                           edges_o_stations_d,
                                                           parameters,
                                                           odt_priority_list_original)

    elif operator == 'Return':

        changed_trains, timetable_prime_graph, train_id_to_delay, track_info, edges_o_stations_d, \
        odt_facing_neighbourhood_operator, odt_priority_list_original = \
            neighbourhood_operators.operator_return_train_to_initial_timetable(timetable_prime_graph,
                                                                               changed_trains,
                                                                               trains_timetable,
                                                                               track_info,
                                                                               infra_graph,
                                                                               edges_o_stations_d,
                                                                               parameters,
                                                                               odt_priority_list_original)

    return timetable_prime_graph, track_info, edges_o_stations_d, changed_trains, operator,\
           odt_facing_neighbourhood_operator, odt_priority_list_original


def find_path_and_assign_pass(timetable_prime_graph, parameters, timetable_solution_graph, edges_o_stations_d):

    # Set the cutoff duration
    cutoff = parameters.time_duration.seconds/60

    # Add the edges origin to stations and stations to destination
    timetable_full_graph = \
        timetable_graph.create_graph_with_edges_o_stations_d(edges_o_stations_d,
                                                             timetable_graph=copy.deepcopy(timetable_prime_graph))

    # Add the origin nodes that are still missing in the full graph from the odt_list
    node_types = {}
    node_names_origin = []
    for odt in parameters.odt_as_list:
        node_types[odt[0]] = {'train': None, 'type': 'origin'}
        node_names_origin.append(odt[0])
        node_types[odt[1]] = {'train': None, 'type': 'destination'}
    timetable_full_graph.add_nodes_from(node_names_origin)
    nx.set_node_attributes(timetable_full_graph, node_types)

    # Compute the shortest path with capacity constraint
    print('Assign the passenger on the timetable graph')
    odt_facing_capacity_constraint, parameters, timetable_prime_graph = \
        passenger_assignment.capacity_constraint_1st_loop(parameters, timetable_full_graph)

    if odt_facing_capacity_constraint is None:
        odt_priority_list_original = copy.deepcopy(parameters.odt_as_list)
        assigned, unassigned = helpers.compute_assigned_not_assigned(odt_priority_list_original)
    else:
        timetable_prime_graph, assigned, unassigned, odt_facing_capacity_dict_for_iteration,\
        odt_priority_list_original = passenger_assignment.capacity_constraint_2nd_loop(parameters,
                                                                                       odt_facing_capacity_constraint,
                                                                                       timetable_prime_graph)

    total_traveltime = helpers.compute_travel_time(odt_priority_list_original, timetable_full_graph, parameters)

    # Save the total travel time for the solution
    timetable_solution_graph.total_traveltime = round(total_traveltime, 1)

    # Printout the output
    print('Passengers with path : ', assigned, ', passengers without path : ', unassigned)

    return timetable_solution_graph, timetable_prime_graph, odt_priority_list_original


def find_path_and_assign_pass_neighbourhood_operator(timetable_prime_graph, parameters, timetable_solution_graph,
                                                     edges_o_stations_d, odt_priority_list_original,
                                                     odt_facing_neighbourhood_operator):

    # Set the cutoff duration
    cutoff = parameters.time_duration.seconds/60

    # Add the edges origin to stations and stations to destination
    timetable_full_graph = \
        timetable_graph.create_graph_with_edges_o_stations_d(edges_o_stations_d,
                                                             timetable_graph=copy.deepcopy(timetable_prime_graph))

    # Add the origin nodes that are still missing in the full graph from the odt_list
    node_types = {}
    node_names_origin = []
    for odt in parameters.odt_as_list:
        node_types[odt[0]] = {'train': None, 'type': 'origin'}
        node_names_origin.append(odt[0])
        node_types[odt[1]] = {'train': None, 'type': 'destination'}
    timetable_full_graph.add_nodes_from(node_names_origin)
    nx.set_node_attributes(timetable_full_graph, node_types)

    # Debug timetable graph
    print(f'Number of nodes in the timetable graph before assigning passenger = {len(timetable_full_graph)}')

    # Compute the shortest path with capacity constraint
    print('Assign the passenger on the timetable graph')
    timetable_full_graph, assigned_disruption, unassigned_disruption, odt_facing_disruption, \
        odt_priority_list_original = \
        passenger_assignment.assignment_neighbourhood_operator(odt_priority_list_original,
                                                               odt_facing_neighbourhood_operator,
                                                               timetable_full_graph,
                                                               parameters)

    # Debug timetable graph
    print(f'Number of nodes in the timetable graph after assigning passenger = {len(timetable_full_graph)}')

    total_traveltime = helpers.compute_travel_time(odt_priority_list_original, timetable_full_graph, parameters)

    # Save the total travel time for the solution
    timetable_solution_graph.total_traveltime = round(total_traveltime, 1)

    # Printout the output
    print('Passengers with path : ', assigned_disruption, ', passengers without path : ', unassigned_disruption)

    return timetable_solution_graph, timetable_full_graph, odt_priority_list_original


def distance_travelled_all_trains(trains_timetable, infra_graph, parameters):
    # Set distance to zero
    total_distance = 0

    # Loop through all the trains in the timetable and compute the distance travelled for each
    for train in trains_timetable:
        if isinstance(train, int):
            print('Train canceled ? (method: distance_travelled_all_trains, alns_platform.py)')
            continue

        for tpn in train.train_path_nodes:
            if tpn.section_track_id is not None:
                total_distance += infra_graph.graph['cache_trackID_dist'][tpn.section_track_id]
            else:
                total_distance += parameters.deviation_penalty_bus * 40000

    # The distance is in decimeter --> divide by 10 * 1000 to have it in km
    total_distance = round(total_distance / (10*1000), 1)
    return total_distance


def deviation_reroute_timetable(trains_timetable, timetable_initial_graph, changed_trains, parameters):
    parameters.set_of_trains_for_operator = {'Cancel': [], 'CancelFrom': [], 'Delay': [], 'DelayFrom': []}
    fmt = "%Y-%m-%dT%H:%M:%S"

    # Deviation penalty
    d_rerouted = parameters.deviation_penalty_rerouted

    total_deviation = 0  # In minutes
    timetable_initial = helpers.build_dict_from_viriato_object_train_id(
        timetable_initial_graph.initial_timetable_infeasible)

    # timetable of the current solution
    timetable_prime = helpers.build_dict_from_viriato_object_train_id(trains_timetable)

    # Loop through all the trains that have been changed
    for train_id, value in changed_trains.items():
        action = value['Action']

        # Check the action, if it is reroute, update the deviation penalty
        if action == 'Reroute':
            if value['add_stop_time'] > parameters.delayTime_to_consider_cancel:
                parameters.set_of_trains_for_operator['Cancel'].append(train_id)

            # Check in the train is in the current timetable
            if train_id in timetable_prime.keys():
                try:
                    # Loop through the nodes of the train path
                    for tpn in timetable_prime[train_id].train_path_nodes:
                        if tpn.id == value['StartEndRR_tpnID']:
                            dep_time_start_rr = tpn.departure_time
                            dep_time_end_train = timetable_prime[train_id].train_path_nodes[-1].arrival_time

                            # Add the penalty for the rerouted
                            total_deviation += d_rerouted * (dep_time_end_train - dep_time_start_rr).seconds / 60
                            break
                except AttributeError:
                    continue

    return round(total_deviation, 1)


def deviation_cancel_timetable(trains_timetable, timetable_initial_graph, changed_trains, parameters):
    parameters.set_of_trains_for_operator = {'Cancel': [], 'CancelFrom': [], 'Delay': [], 'DelayFrom': []}
    fmt = "%Y-%m-%dT%H:%M:%S"

    # Deviation penalties for each operator
    d_cancel = parameters.deviation_penalty_cancel
    d_emergency = parameters.deviation_penalty_emergency

    total_deviation = 0  # In minutes
    timetable_initial = helpers.build_dict_from_viriato_object_train_id(
        timetable_initial_graph.initial_timetable_infeasible)

    # timetable of the current solution
    timetable_prime = helpers.build_dict_from_viriato_object_train_id(trains_timetable)

    # Loop through all the trains that have been changed
    for train_id, value in changed_trains.items():
        action = value['Action']

        # Check the action, if it is cancel, update the deviation penalty
        if action == 'Cancel':

            # If it is an emergency train, the deviation penalty is not compute here
            if 'EmergencyTrain' in value.keys():
                total_deviation += 0
                continue

            # If it is an emergency bus, the deviation penalty is not compute here
            elif 'EmergencyBus' in value.keys():
                total_deviation += 0
                continue

            # Add the penalty for cancel action with the constant
            total_deviation += d_cancel
            departure_first_tpn = timetable_initial[train_id].train_path_nodes[0].departure_time
            arrival_last_tpn = timetable_initial[train_id].train_path_nodes[-1].arrival_time

            # Add another penalty of the ratio time duration of the train whole itinerary
            total_deviation += d_cancel * ((arrival_last_tpn - departure_first_tpn).seconds / 60)

        # Check the action, if it is cancel from, update the deviation penalty
        elif action == 'CancelFrom':
            # If it is an emergency train, the deviation penalty is compute here
            if 'EmergencyTrain' in value.keys():
                total_deviation += deviation_emergency_train(timetable_prime, d_emergency, total_deviation, train_id,
                                                             parameters)
                continue

            # Loop through all the nodes of the train path
            for tpn in timetable_initial[train_id].train_path_nodes:
                if tpn.id == value['tpn_cancel_from']:

                    # Get the departure and arrival time
                    dep_time_canceled_from = tpn.departure_time
                    arr_time_end_train = timetable_initial[train_id].train_path_nodes[-1].arrival_time

                    # Add the deviation penalty for cancel from
                    total_deviation += d_cancel * (arr_time_end_train - dep_time_canceled_from).seconds / 60

    return round(total_deviation, 1)


def deviation_timetable(trains_timetable, timetable_initial_graph, changed_trains, parameters):
    # Parameters to penalize the operation, c - cancel, d - delay, e - emergency train, r - rerouting
    parameters.set_of_trains_for_operator = {'Cancel': [], 'CancelFrom': [], 'Delay': [], 'DelayFrom': []}
    fmt = "%Y-%m-%dT%H:%M:%S"

    # Deviation penalties for each operator
    d_cancel = parameters.deviation_penalty_cancel
    d_delay = parameters.deviation_penalty_delay
    d_emergency = parameters.deviation_penalty_emergency
    d_bus = parameters.deviation_penalty_bus
    d_rerouted = parameters.deviation_penalty_rerouted

    total_deviation = 0  # In minutes
    timetable_initial = helpers.build_dict_from_viriato_object_train_id(
        timetable_initial_graph.initial_timetable_infeasible)

    # timetable of the current solution
    timetable_prime = helpers.build_dict_from_viriato_object_train_id(trains_timetable)

    # Loop through all the trains that have been changed
    for train_id, value in changed_trains.items():
        action = value['Action']

        # Check the action, if it is cancel, update the deviation penalty
        if action == 'Cancel':

            # If it is an emergency train, the deviation penalty is not compute here
            if 'EmergencyTrain' in value.keys():
                total_deviation += 0
                continue

            # If it is an emergency bus, the deviation penalty is not compute here
            elif 'EmergencyBus' in value.keys():
                total_deviation += 0
                continue

            # Add the penalty for cancel action with the constant
            total_deviation += d_cancel
            departure_first_tpn = timetable_initial[train_id].train_path_nodes[0].departure_time
            arrival_last_tpn = timetable_initial[train_id].train_path_nodes[-1].arrival_time

            # Add another penalty of the ratio time duration of the train whole itinerary
            total_deviation += d_cancel * ((arrival_last_tpn - departure_first_tpn).seconds / 60)

        # Check the action, if it is cancel from, update the deviation penalty
        elif action == 'CancelFrom':
            # If it is an emergency train, the deviation penalty is compute here
            if 'EmergencyTrain' in value.keys():
                total_deviation += deviation_emergency_train(timetable_prime, d_emergency, total_deviation, train_id,
                                                             parameters)
                continue

            # Loop through all the nodes of the train path
            for tpn in timetable_initial[train_id].train_path_nodes:
                if tpn.id == value['tpn_cancel_from']:

                    # Get the departure and arrival time
                    dep_time_canceled_from = tpn.departure_time
                    arr_time_end_train = timetable_initial[train_id].train_path_nodes[-1].arrival_time

                    # Add the deviation penalty for cancel from
                    total_deviation += d_cancel * (arr_time_end_train - dep_time_canceled_from).seconds / 60

        # Check the action, if it is short turn, update the deviation penalty
        elif action == 'ShortTurn':
            # Like cancelled in between
            for tpn in timetable_initial[train_id].train_path_nodes:
                if tpn.id == value['tpns_cancel_from_to'][0]:
                    arrival_tpn_before_turn = tpn.arrival_time

                elif tpn.id == value['tpns_cancel_from_to'][1]:
                    depart_tpn_after_turn = tpn.departure_time

                    # Update the total deviation with the penalty of short turn
                    total_deviation += d_cancel * (depart_tpn_after_turn - arrival_tpn_before_turn)

        # Check the action, if it is reroute, update the deviation penalty
        elif action == 'Reroute':
            if value['add_stop_time'] > parameters.delayTime_to_consider_cancel:
                parameters.set_of_trains_for_operator['Cancel'].append(train_id)

            # Check in the train is in the current timetable
            if train_id in timetable_prime.keys():
                # Loop through the nodes of the train path
                for tpn in timetable_prime[train_id].train_path_nodes:
                    if tpn.id == value['StartEndRR_tpnID']:
                        dep_time_start_rr = tpn.departure_time
                        dep_time_end_train = timetable_prime[train_id].train_path_nodes[-1].arrival_time

                        # Add the penalty for the rerouted
                        total_deviation += d_rerouted * (dep_time_end_train - dep_time_start_rr).seconds / 60
                        break

        # For delay and return
        elif action == 'Delay' or action == 'Return':
            try:
                if hasattr(timetable_prime[train_id], 'emergency_train')\
                        and timetable_prime[train_id].emergency_train is True:

                    # Add penalty for emergency train
                    total_deviation += deviation_emergency_train(timetable_prime, d_emergency, total_deviation,
                                                                 train_id,
                                                                 parameters)
                    continue
                elif hasattr(timetable_prime[train_id], 'emergency_bus') \
                        and timetable_prime[train_id].emergency_bus is True:

                    # Add penalty for emergency bus
                    total_deviation += deviation_emergency_bus(timetable_prime, d_bus, total_deviation, train_id,
                                                               parameters)
                    continue
            except KeyError:
                print('Something went wrong with the deviation timetable and action delay or return.')
            max_delay_tpn, total_delay_train = deviation_delay_train(fmt, timetable_initial, timetable_prime, train_id)

            # Add the penalty for deviation
            total_deviation += total_delay_train.seconds / 60 * d_delay

            changed_trains[train_id]['total_delay'] = total_delay_train
            changed_trains[train_id]['tpn_max_delay'] = max_delay_tpn

            # If the delay is greater than the threshold, add the train id to potential for cancelling
            if total_delay_train > parameters.delayTime_to_consider_cancel:
                if train_id not in parameters.set_of_trains_for_operator['Cancel']:
                    parameters.set_of_trains_for_operator['Cancel'].append(train_id)

        # Check the action, if it is delay from, update the deviation penalty
        elif action == 'DelayFrom':
            if hasattr(timetable_prime[train_id], 'emergency_train') \
                    and timetable_prime[train_id].emergency_train is True:

                total_deviation += deviation_emergency_train(timetable_prime, d_emergency, total_deviation, train_id,
                                                             parameters)
                continue

            max_delay_tpn, total_delay_train = deviation_delay_train(fmt, timetable_initial, timetable_prime, train_id)

            total_deviation += total_delay_train.seconds / 60 * d_delay

            changed_trains[train_id]['total_delay'] = total_delay_train
            changed_trains[train_id]['tpn_max_delay'] = max_delay_tpn

            if total_delay_train > parameters.delayTime_to_consider_cancel:
                if train_id not in parameters.set_of_trains_for_operator['Cancel']:
                    parameters.set_of_trains_for_operator['Cancel'].append(train_id)

        # Check the action, if it is emergency train, update the deviation penalty
        elif action == 'EmergencyTrain':
            total_deviation += deviation_emergency_train(timetable_prime, d_emergency, total_deviation, train_id,
                                                         parameters)

        # Check the action, if it is emergency bus, update the deviation penalty
        elif action == 'EmergencyBus':
            total_deviation += deviation_emergency_bus(timetable_prime, d_bus, total_deviation, train_id, parameters)

    return round(total_deviation, 1)


def deviation_emergency_train(timetable_prime, d_e, total_deviation, train_id, parameters):
    # Get the departure and arrival time of the train
    dep_time_start = timetable_prime[train_id].train_path_nodes[0].departure_time
    arr_time_end_train = timetable_prime[train_id].train_path_nodes[-1].arrival_time

    # Add the penalty for deviation emergency train
    total_deviation += d_e * ((arr_time_end_train - dep_time_start).seconds / 60)

    return total_deviation


def deviation_emergency_bus(timetable_prime, d_b, total_deviation, train_id, parameters):

    dep_time_start = timetable_prime[train_id].train_path_nodes[0].departure_time
    arr_time_end_train = timetable_prime[train_id].train_path_nodes[-1].arrival_time

    # Add deviation penalty for bus
    total_deviation += d_b + (arr_time_end_train - dep_time_start).seconds / 60

    return total_deviation


def deviation_delay_train(fmt, timetable_initial, timetable_prime, train_id):
    # Set the parameters
    total_delay_train = datetime.timedelta(minutes=0)
    max_delay_tpn = [datetime.timedelta(minutes=0)]
    try:
        train_path_node_prime = helpers.build_dict_from_viriato_object_train_id(
            timetable_prime[train_id].train_path_nodes)
    except KeyError:
        print(f'Train {train_id} is not in the current timetable for the deviation operator. why?')

    # Loop through all the nodes of the selected train
    for train_path_node_initial in timetable_initial[train_id].train_path_nodes:
        try:
            if not train_path_node_initial.id in train_path_node_prime.keys():
                # This train has been canceled from a point, therefore this tpn is not in tpn prime anymore
                continue
            # If the departure time node is the same as the current timetable, there is no delay hence, check next node
            if train_path_node_initial.departure_time == \
                    train_path_node_prime[train_path_node_initial.id].departure_time:
                continue
            else:
                dep_tpn_initial = train_path_node_initial.departure_time
                dep_tpn_prime = train_path_node_prime[train_path_node_initial.id].departure_time
                deviation_tpn = dep_tpn_prime - dep_tpn_initial
                if max_delay_tpn[0] < deviation_tpn:
                    max_delay_tpn = [deviation_tpn, train_path_node_initial.id]

            total_delay_train += deviation_tpn
        except KeyError:
            print('Something went wrong in deviation calculation for the delay')

    return max_delay_tpn, total_delay_train


def archiving_acceptance_rejection(timetable_solution_graph, timetable_solution_prime_graph, operator, parameters,
                                   scores, temp_i, solution_archive):
    # Set the parameters
    archived_solution = False
    accepted_solution = False
    solution_prime_not_dominated_by_any_solution = True

    # if True
    if True:
        for solution in solution_archive:
            # Check if one of the multi-objective is better than the solutions in the archive with boolean
            cond_tt = timetable_solution_prime_graph.total_traveltime < solution.total_traveltime
            cond_de_reroute = \
                timetable_solution_prime_graph.deviation_reroute_timetable < solution.deviation_reroute_timetable
            cond_de_cancel = \
                timetable_solution_prime_graph.deviation_cancel_timetable < solution.deviation_cancel_timetable
            cond_op = timetable_solution_prime_graph.total_dist_train < solution.total_dist_train

            # If any of the three objectives is better. Continue the search
            if any([cond_tt, cond_op, cond_de_reroute, cond_de_cancel]):
                continue

            # If the solution does not show any improvement, it is dominated. Stop the search
            else:
                solution_prime_not_dominated_by_any_solution = False
                break

        # After all the search, if the solution is still not dominated, accept the solution
        if solution_prime_not_dominated_by_any_solution:
            archived_solution = True
            accepted_solution = True
            print('Solution added to archive')
            solution_archive.append(timetable_solution_prime_graph)
            timetable_solution_graph = get_solution_graph_with_copy_solution_prime(timetable_solution_prime_graph,
                                                                                   parameters)
            scores = update_scores(operator, parameters.score_1, scores)

    # Check if there are any solutions in the archive which are dominated by current solution
    if archived_solution:
        # Set the index
        index_dominated_solutions = []
        i = -1

        # Loop through all the solution that are archived and check if it is dominated by the current solution
        for solution in solution_archive[:-1]:
            i += 1
            cond_tt = timetable_solution_prime_graph.total_traveltime <= solution.total_traveltime
            cond_de_reroute = \
                timetable_solution_prime_graph.deviation_reroute_timetable < solution.deviation_reroute_timetable
            cond_de_cancel = \
                timetable_solution_prime_graph.deviation_cancel_timetable < solution.deviation_cancel_timetable
            cond_op = timetable_solution_prime_graph.total_dist_train <= solution.total_dist_train

            # If all the current solution objectives are smaller than the archived solution or are equals, double check
            # if it is strictly smaller in at least one of the three objectives
            if all([cond_tt, cond_op, cond_de_reroute, cond_de_cancel]):
                cond_tt = timetable_solution_prime_graph.total_traveltime < solution.total_traveltime
                cond_de_reroute = \
                    timetable_solution_prime_graph.deviation_reroute_timetable < solution.deviation_reroute_timetable
                cond_de_cancel = \
                    timetable_solution_prime_graph.deviation_cancel_timetable < solution.deviation_cancel_timetable
                cond_op = timetable_solution_prime_graph.total_dist_train < solution.total_dist_train
                # If one of the three objectives is strictly smaller than the archived solution, get the index
                if any([cond_tt, cond_op, cond_de_reroute, cond_de_cancel]):
                    index_dominated_solutions.append(i)
        # Delete all the dominated solution from the archives
        if len(index_dominated_solutions) >= 1:
            index_dominated_solutions.reverse()
            for i in index_dominated_solutions:
                del solution_archive[i]

        # Pickle the results for the archive
        pickle_results(solution_archive, 'output/pickle/solution_archive.pkl')
        print('Save solution_archive into a pickle.')

    else:
        reaction_factor_op = parameters.reaction_factor_operation
        reaction_factor_dev = parameters.reaction_factor_deviation

        # Acceptance criterion for operation objective
        z_o_prime = timetable_solution_prime_graph.total_dist_train
        z_o = timetable_solution_graph.total_dist_train
        try:
            acceptance_prob_operation = min(math.exp((-(z_o_prime - z_o) / temp_i[0])) * reaction_factor_op, 1)
        except ZeroDivisionError:
            acceptance_prob_operation = 0.001
            print('Division by zero, temperature_i', temp_i)

        # Acceptance criterion for deviation objective reroute
        z_d_reroute_prime = timetable_solution_prime_graph.deviation_reroute_timetable
        z_d_reroute = timetable_solution_graph.deviation_reroute_timetable
        try:
            acceptance_prob_deviation_reroute = \
                min(math.exp((-(z_d_reroute_prime - z_d_reroute) / temp_i[1])) * reaction_factor_dev, 1)
        except ZeroDivisionError:
            acceptance_prob_deviation_reroute = 0.001
            print('Division by zero, temperature_i', temp_i)

        # Acceptance criterion for deviation objective cancel
        z_d_cancel_prime = timetable_solution_prime_graph.deviation_cancel_timetable
        z_d_cancel = timetable_solution_graph.deviation_cancel_timetable
        try:
            acceptance_prob_deviation_cancel =\
                min(math.exp((-(z_d_cancel_prime - z_d_cancel) / temp_i[2])) * reaction_factor_dev, 1)
        except ZeroDivisionError:
            acceptance_prob_deviation_cancel = 0.001
            print('Division by zero, temperature_i', temp_i)

        # Acceptance criterion for travel time objective
        z_tt_prime = timetable_solution_prime_graph.total_traveltime
        z_tt = timetable_solution_graph.total_traveltime
        try:
            acceptance_prob_passenger = min(math.exp((-(z_tt_prime - z_tt) / temp_i[3])), 1)
        except ZeroDivisionError:
            acceptance_prob_passenger = 0.001
            print('Division by zero, temperature_i', temp_i)

        # Get the overall acceptance probability
        acceptance_prob = \
            acceptance_prob_passenger * acceptance_prob_deviation_reroute * acceptance_prob_deviation_cancel *\
            acceptance_prob_operation

        # Set the random seed
        rd = np.random.uniform(0.0, 1.0)

        # According to the random number, if the random number is lower than the acceptance probability, the solution is
        # accepted!
        if rd < acceptance_prob:
            accepted_solution = True
            timetable_solution_graph = \
                get_solution_graph_with_copy_solution_prime(timetable_solution_prime_graph, parameters)

            # Set the score for the operator, the score and the overall score for scenario 2
            scores = update_scores(operator, parameters.score_2, scores)

        # If not accepted, set the score for scenario 3 which is for not accepted solution
        else:
            scores = update_scores(operator, parameters.score_3, scores)

    # When the memory is a constraint. Limit the archive to 80
    if len(solution_archive) >= 80:
        # Throw away worst solution if archive gets to big
        idx_throw_away = 0
        current_worst = solution_archive[idx_throw_away]

        print('Max archive size reached, Worst archive solution will be deleted')
        for i in range(1, len(solution_archive)-1):
            if solution_archive[i].total_traveltime >= current_worst.total_traveltime:
                if solution_archive[i].total_traveltime > current_worst.total_traveltime:
                    idx_throw_away = i
                    continue
                else:
                    if solution_archive[i].deviation_reroute_timetable >= current_worst.deviation_reroute_timetable:
                        if solution_archive[i].deviation_reroute_timetable > current_worst.deviation_reroute_timetable:
                            idx_throw_away = i
                            continue
                        else:
                            if solution_archive[i].deviation_cancel_timetable >= \
                                    current_worst.deviation_cancel_timetable:
                                if solution_archive[i].deviation_cancel_timetable >\
                                        current_worst.deviation_cancel_timetable:
                                    idx_throw_away = i
                                    continue
                            else:
                                if solution_archive[i].total_dist_train > current_worst.total_dist_train:
                                    idx_throw_away = i

        print('Worst solution is deleted')
        del solution_archive[idx_throw_away]

    return timetable_solution_graph, scores, accepted_solution, archived_solution


def get_solution_graph_with_copy_solution_prime(timetable_solution_prime_graph, parameters):
    timetable_solution_graph = helpers.Solution()
    timetable_solution_graph.set_of_trains_for_operator =\
        copy.deepcopy(timetable_solution_prime_graph.set_of_trains_for_operator)
    timetable_solution_graph.time_table = copy.deepcopy(timetable_solution_prime_graph.timetable)
    timetable_solution_graph.graph = timetable_graph.copy_graph_with_flow(timetable_solution_prime_graph.graph)
    timetable_solution_graph.total_dist_train = timetable_solution_prime_graph.total_dist_train
    timetable_solution_graph.total_traveltime = timetable_solution_prime_graph.total_traveltime
    timetable_solution_graph.deviation_reroute_timetable = timetable_solution_prime_graph.deviation_reroute_timetable
    timetable_solution_graph.deviation_cancel_timetable = timetable_solution_prime_graph.deviation_cancel_timetable
    timetable_solution_graph.changed_trains = copy.deepcopy(timetable_solution_prime_graph.changed_trains)
    timetable_solution_graph.edges_o_stations_d = \
        helpers.CopyEdgesOriginStationDestination(timetable_solution_prime_graph.edges_o_stations_d)
    timetable_solution_graph.set_of_trains_for_operator['Cancel'] = \
        copy.deepcopy(parameters.set_of_trains_for_operator['Cancel'])
    timetable_solution_graph.set_of_trains_for_operator['Delay'] = \
        copy.deepcopy(parameters.set_of_trains_for_operator['Delay'])

    return timetable_solution_graph


def update_scores(operator, score, scores):
    if operator == 'Cancel':
        scores.cc += score

    elif operator == 'CancelFrom':
        scores.pc += score

    elif operator == 'Delay':
        scores.cd += score

    elif operator == 'DelayFrom':
        scores.pd += score

    elif operator == 'EmergencyTrain':
        scores.et += score

    elif operator == 'EmergencyBus':
        scores.eb += score

    elif operator == 'Return':
        scores.ret += score

    return scores


def pickle_results(file_for_pickle, filename):
    with open(filename, 'wb') as f:
        pickle.dump(file_for_pickle, f)


def pickle_archive_operation_travel_time_deviation(solution_archive):
    z_op, z_de_reroute, z_de_cancel, z_tt = [], [], [], []

    archive_for_pickle = {'z_op': z_op, 'z_de_reroute': z_de_reroute, 'z_de_cancel': z_de_cancel, 'z_tt': z_tt}
    for solution in solution_archive:
        z_op.append(solution.total_dist_train)
        z_de_reroute.append(solution.deviation_reroute_timetable)
        z_de_cancel.append(solution.deviation_cancel_timetable)
        z_tt.append(solution.total_traveltime)
    pickle_results(archive_for_pickle, 'output/pickle/z_archive.pkl')


def pickle_archive_with_changed_trains(solution_archive):
    z_op, z_de_reroute, z_de_cancel, z_tt, changed_trains = [], [], [], [], []

    archive_for_pickle = {'z_op': z_op, 'z_de_reroute': z_de_reroute, 'z_de_cancel': z_de_cancel, 'z_tt': z_tt,
                          'changed_trains': changed_trains}
    for solution in solution_archive:
        z_op.append(solution.total_dist_train)
        z_de_reroute.append(solution.deviation_reroute_timetable)
        z_de_cancel.append(solution.deviation_cancel_timetable)
        z_tt.append(solution.total_traveltime)
        changed_trains.append(solution.changed_trains)
    pickle_results(archive_for_pickle, 'output/pickle/z_archive.pkl')


def update_temperature(temp_i, n_iteration, nb_temperature_changes, all_accepted_solutions, parameters):
    # Set the parameters
    warm_up_phase = parameters.warm_up_phase
    iterations_at_temperature_level = parameters.iterations_temperature_level
    number_of_temperature_change = parameters.number_of_temperature_change
    p_0 = 0.999  # Initial probability
    p_f = 0.001  # Final probability

    if n_iteration >= warm_up_phase and nb_temperature_changes != number_of_temperature_change:
        if n_iteration % iterations_at_temperature_level == 0:
            nb_temperature_changes += 1
            sigma_z_op = np.std(all_accepted_solutions[0])
            sigma_z_de_reroute = np.std(all_accepted_solutions[1])
            sigma_z_de_cancel = np.std(all_accepted_solutions[2])
            sigma_z_tt = np.std(all_accepted_solutions[3])
            temp_i[0] =\
                - sigma_z_op / (np.log(p_0 + ((p_f - p_0) / number_of_temperature_change) * nb_temperature_changes))
            temp_i[1] =\
                - sigma_z_de_reroute / \
                (np.log(p_0 + ((p_f - p_0) / number_of_temperature_change) * nb_temperature_changes))
            temp_i[2] = \
                - sigma_z_de_cancel / \
                (np.log(p_0 + ((p_f - p_0) / number_of_temperature_change) * nb_temperature_changes))
            temp_i[3] =\
                - sigma_z_tt / (np.log(p_0 + ((p_f - p_0) / number_of_temperature_change) * nb_temperature_changes))

    elif nb_temperature_changes == number_of_temperature_change:
        temp_i = [0, 0, 0, 0]

    return temp_i, nb_temperature_changes


def periodically_select_solution(solution_archive, timetable_solution_graph, n_iteration,
                                 iterations_until_return_archives, parameters, return_to_archive_at_iteration):
    # Set the parameters
    reaction_factor_return_archive = parameters.reaction_factor_return_archive

    # Select a solution from the archive when the iteration number is equal to the set iteration number for returning
    if n_iteration == return_to_archive_at_iteration:

        # Printout that we are selecting a solution from the archive
        print('Selecting a solution from the archive')

        iterations_until_return_archives = reaction_factor_return_archive * iterations_until_return_archives
        if iterations_until_return_archives <= 20:
            iterations_until_return_archives = 20

        return_to_archive_at_iteration = int(n_iteration + round(iterations_until_return_archives, 0))

        # Set the random seed
        n = 8
        solutions_sorted_tt = []

        for solution in solution_archive:
            solutions_sorted_tt.append(solution.total_traveltime)
        if n > len(solutions_sorted_tt):
            n = len(solutions_sorted_tt) - 1

        # Sort by travel time
        preselect_n_best_solutions_traveltime = list(np.argsort(np.array(solutions_sorted_tt))[-n:])
        rd_index = np.random.randint(0, len(preselect_n_best_solutions_traveltime))

        timetable_solution_graph = get_solution_graph_with_copy_solution_prime(solution_archive[rd_index], parameters)
        print('Return to archive: selected solution :')
        print('z_o_accepted : ', timetable_solution_graph.total_dist_train,
              '\n z_d_reroute_accepted : ', timetable_solution_graph.deviation_reroute_timetable,
              '\n z_d_cancel_accepted : ', timetable_solution_graph.deviation_cancel_timetable,
              '\n z_p_accepted : ', timetable_solution_graph.total_traveltime)

    return timetable_solution_graph, iterations_until_return_archives, return_to_archive_at_iteration


def update_weights(weights, scores, nr_usage, nr_temperature_updates, probabilities, parameters):
    if nr_temperature_updates % 2 == 0 and nr_temperature_updates != 0:
        reaction_factor = parameters.reaction_factor_weights
        if nr_usage.cc != 0:
            weights.cc = (1 - 0.5) * weights.cc + reaction_factor * scores.cc / nr_usage.cc
            nr_usage.cc = 0
            scores.cc = 1

        if nr_usage.pc != 0:
            weights.pc = (1 - 0.5) * weights.pc + reaction_factor * scores.pc / nr_usage.pc
            nr_usage.pc = 0
            scores.pc = 1

        if nr_usage.cd != 0:
            weights.cd = (1 - 0.5) * weights.cd + reaction_factor * scores.cd / nr_usage.cd
            nr_usage.cd = 0
            scores.cd = 1

        if nr_usage.pd != 0:
            weights.pd = (1 - 0.5) * weights.pd + reaction_factor * scores.pd / nr_usage.pd
            nr_usage.pd = 0
            scores.pd = 1

        if nr_usage.et != 0:
            weights.et = (1 - 0.5) * weights.et + reaction_factor * scores.et / nr_usage.et
            nr_usage.et = 0
            scores.et = 1

        if nr_usage.eb != 0:
            weights.eb = (1 - 0.5) * weights.eb + reaction_factor * scores.eb / nr_usage.eb
            nr_usage.eb = 0
            scores.eb = 1

        if nr_usage.ret != 0:
            weights.ret = (1 - 0.5) * weights.ret + reaction_factor * scores.ret / nr_usage.ret
            nr_usage.ret = 0
            scores.ret = 1
        weights.sum = weights.cc + weights.cd + weights.eb + weights.et + weights.pc + weights.pd + weights.ret

        probabilities = helpers.Probabilities(weights)  # Changed when the weights are adapted
    return weights, scores, probabilities


def get_all_trains_cut_to_time_window_and_area(parameters):
    # Get the all the trains in the area
    trains_timetable = viriato_interface.get_trains_driving_any_node(parameters.time_window,
                                                                     parameters.stations_in_area)
    # Keep only the trains that has more than 1 station in the area of interest
    trains_timetable = viriato_interface.cut_trains_area_interest_time_window(trains_timetable,
                                                                              parameters.stations_in_area,
                                                                              parameters.time_window)

    return trains_timetable
