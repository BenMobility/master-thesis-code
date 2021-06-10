"""
Created on Tue Mar 02 2021

@author: BenMobility

Code for the neighbourhood operators for timetable search
"""
import passenger_assignment
import viriato_interface
import numpy as np
import alns_platform
import timetable_graph
import networkx as nx
import copy
import helpers
import datetime
import types
from operator import itemgetter


def short_turn_train_viriato_preselection(parameters, train_to_short_turn, train_path_node_cancel_after,
                                          train_path_node_cancel_before):
    """
    function/operator that creates two cloned trains that starts after the selected node and also run before the
    selected node. Cancel the train to short turn. Keeps only the two cloned trains.
    :param parameters: Class parameters that has station in the area and time window (type: class)
    :param train_to_short_turn: selected train to short turn (type: viriato object)
    :param train_path_node_cancel_after: a viriato object that contains all the information of a train path node
    :param train_path_node_cancel_before: a viriato object that contains all the information of a train path node
    :return: all cloned trains that make the short turn operation feasible. (type: viriato object trains)
    """
    # Clone trains
    cloned_train_1 = viriato_interface.clone_train(train_to_short_turn.id)
    cloned_train_2 = viriato_interface.clone_train(train_to_short_turn.id)

    # Get the sequence number of the train path node to cancel after for the cloned train 1
    i = 0
    while cloned_train_1.train_path_nodes[i].sequence_number != train_path_node_cancel_after.sequence_number:
        i += 1
    train_path_node_cancel_after = cloned_train_1.train_path_nodes[i].id

    # Get the sequence number of the train path node to cancel after for the cloned train 2
    i = 0
    while cloned_train_2.train_path_nodes[i].sequence_number != train_path_node_cancel_before.sequence_number:
        i += 1
    train_path_node_cancel_before = cloned_train_2.train_path_nodes[i].id

    # Cancel train 1 after a specific node
    cloned_train_1 = viriato_interface.cancel_train_after(cloned_train_1.id, train_path_node_cancel_after)
    # Cancel train 2 before a specific node
    cloned_train_2 = viriato_interface.cancel_train_before(cloned_train_2.id, train_path_node_cancel_before)
    # Get only the cloned trains inside the area of interest and time window
    cloned_trains = viriato_interface.get_trains_area_interest_time_window([cloned_train_1, cloned_train_2],
                                                                           parameters.stations_in_area,
                                                                           parameters.time_window)
    # Extract both cloned trains
    cloned_train_1 = cloned_trains[0]
    cloned_train_2 = cloned_trains[1]
    # After cloning and created to short turn trains, cancel the train
    viriato_interface.cancel_train(train_to_short_turn.id)

    return cloned_train_1, cloned_train_2


def operator_cancel(prime_timetable, changed_trains, trains_timetable, track_info, edges_o_stations_d, parameters,
                    odt_priority_list_original):
    # Initialize
    train_found = False
    list_cancel_candidates = list(set(parameters.set_of_trains_for_operator['Cancel']))
    # When the number of candidates is greater than 3, start the loop until the train is found
    if len(list_cancel_candidates) >= 3:
        n_it = 0
        while not train_found and n_it <= 5:
            n_it += 1
            train_id_to_cancel = list_cancel_candidates[np.random.randint(0, len(list_cancel_candidates))]
            i = 0
            try:
                while trains_timetable[i].id != train_id_to_cancel:
                    i += 1
                train_to_cancel = trains_timetable[i]
            except IndexError:
                continue
            comm_stops = 0
            # Check if there is more than 3 commercial stops to make partial cancel valid
            for train_path_node in train_to_cancel.train_path_nodes:
                if train_path_node.stop_status.name == 'commercial_stop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True

    if not train_found:
        # Select randomly a train from all trains
        while not train_found:
            train_id_to_cancel = trains_timetable[np.random.randint(0, len(trains_timetable))].id
            i = 0
            while trains_timetable[i].id != train_id_to_cancel:
                i += 1
            train_to_cancel = trains_timetable[i]
            comm_stops = 0
            # Check if there is more than 3 commercial stops to make partial cancel valid
            for train_path_node in train_to_cancel.train_path_nodes:
                if train_path_node.stop_status.name == 'commercial_stop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True

    # Get the train id to cancel from the train to cancel
    train_id_to_cancel = train_to_cancel.id  # Train object from viriato

    # Check if the train id is an emergency train/bus
    emergency_train = False
    bus = False
    if hasattr(train_to_cancel, 'emergency_train') and train_to_cancel.emergency_train is True:
        emergency_train = True
    if hasattr(train_to_cancel, 'emergency_bus') and train_to_cancel.emergency_bus is True:
        bus = True

    # Get the list of odt facing the neighbourhood operator
    odt_facing_neighbourhood_operator, prime_timetable, odt_priority_list_original = \
        passenger_assignment.find_passenger_affected_by_complete_cancel(prime_timetable,
                                                                        train_to_cancel,
                                                                        odt_priority_list_original)

    # Remove the edges and nodes of the canceled train
    edges_o_stations_d = remove_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_cancel, prime_timetable)
    nodes_attr_train, edges_of_train = remove_nodes_edges_of_train(prime_timetable, train_id_to_cancel)

    # If it is not a bus, you need to remove the train path node information
    if not bus:
        remove_entries_from_track_sequences(track_info, train_to_cancel)
        remove_entries_in_train_path_node_canceled_train(track_info, train_to_cancel, idx_start_cancel=0)

    # Update the attribute to cancel an emergency bus
    if bus:
        changed_trains[train_id_to_cancel] = {'train_id': train_id_to_cancel,
                                              'DebugString': train_to_cancel['DebugString'],
                                              'Action': 'Cancel',
                                              'EmergencyBus': True}

    # Update the attribute to cancel a train
    elif not emergency_train and not bus:
        changed_trains[train_id_to_cancel] = {'train_id': train_id_to_cancel,
                                              'DebugString': train_to_cancel.debug_string,
                                              'Action': 'Cancel'}

    # Update the attribute to cancel a emergency train
    elif emergency_train:
        changed_trains[train_id_to_cancel] = {'train_id': train_id_to_cancel,
                                              'DebugString': train_to_cancel.debug_string,
                                              'Action': 'Cancel',
                                              'EmergencyTrain': True}

    # Delete the train from the trains timetable
    del trains_timetable[i]

    return changed_trains, prime_timetable, track_info, edges_o_stations_d, odt_facing_neighbourhood_operator, \
           odt_priority_list_original


def operator_cancel_from(prime_timetable, changed_trains, trains_timetable, track_info, infra_graph, edges_o_stations_d,
                         parameters, odt_priority_list_original):
    # Initialize
    train_found = False
    list_cancel_candidates = list(set(parameters.set_of_trains_for_operator['Cancel']))
    # When the number of candidates is greater than 3, start the loop until the train is found
    if len(list_cancel_candidates) >= 3:
        n_it = 0
        while not train_found and n_it <= 5:
            n_it += 1
            train_id_to_cancel_from = list_cancel_candidates[np.random.randint(0, len(list_cancel_candidates))]
            i = 0
            try:
                while trains_timetable[i].id != train_id_to_cancel_from:
                    i += 1
                train_to_cancel_from = trains_timetable[i]
            except IndexError:
                continue
            comm_stops = 0
            # Check if there is more than 3 commercial stops to make partial cancel valid
            for train_path_node in train_to_cancel_from.train_path_nodes:
                if train_path_node.stop_status.name == 'commercial_stop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True
    if not train_found:
        # Select randomly a train from all trains
        while not train_found:
            train_id_to_cancel_from = trains_timetable[np.random.randint(0, len(trains_timetable))].id
            i = 0
            while trains_timetable[i].id != train_id_to_cancel_from:
                i += 1
            train_to_cancel_from = trains_timetable[i]
            comm_stops = 0
            # Check if there is more than 3 commercial stops to make partial cancel valid
            for train_path_node in train_to_cancel_from.train_path_nodes:
                if train_path_node.stop_status.name == 'commercial_stop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True

    # Check if it is an emergency train
    emergency_train = False
    if hasattr(train_to_cancel_from, 'emergency_train') and train_to_cancel_from.emergency_train is True:
        emergency_train = True

    # Cancel from last comm stop
    idx_tpn_cancel_from = identify_last_departure_train_path_node_id_of_train_to_cancel_from(train_to_cancel_from)
    train_path_node_cancel_from = train_to_cancel_from.train_path_nodes[idx_tpn_cancel_from]

    # Get the list of odt facing the neighbourhood operator
    odt_facing_neighbourhood_operator, prime_timetable, odt_priority_list_original = \
        passenger_assignment.find_passenger_affected_by_cancel_from(prime_timetable, train_to_cancel_from,
                                                                    train_path_node_cancel_from,
                                                                    odt_priority_list_original)

    # Remove edges and nodes of the canceled train
    edges_o_stations_d = remove_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_cancel_from,
                                                                 prime_timetable)
    nodes_attr_train, edges_of_train = remove_nodes_edges_of_train(prime_timetable, train_id_to_cancel_from)

    # Remove the track sequences of the canceled train
    remove_entries_from_track_sequences(track_info, train_to_cancel_from)
    remove_entries_in_train_path_node_canceled_train(track_info, train_to_cancel_from, idx_tpn_cancel_from)

    train_path_nodes_cancel = [tpn_id.id for tpn_id in train_to_cancel_from.train_path_nodes]

    # Create and add driving and waiting edges and nodes to the Graph
    nodes_edges_dict = timetable_graph.create_transit_edges_nodes_single_train(train_to_cancel_from, infra_graph, 0)

    # Add transit nodes and edges from a single train to the timetable graph
    timetable_graph.add_transit_nodes_edges_single_train_to_graph(prime_timetable, nodes_edges_dict, bus=False)

    # Create and add transfer edges to the graph
    transfer_edges, transfer_edges_attribute, \
    arrival_departure_nodes_train = timetable_graph.transfer_edges_single_train(prime_timetable,
                                                                                train_to_cancel_from,
                                                                                parameters,
                                                                                train_path_nodes_cancel)
    # Add the weight on the transfer edges
    prime_timetable.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(prime_timetable, transfer_edges_attribute)

    # Update the list of edges from origin to destination
    edges_o_stations_d = timetable_graph.add_edges_of_train_from_o_stations_d(edges_o_stations_d,
                                                                              train_to_cancel_from,
                                                                              prime_timetable, parameters, 0,
                                                                              train_path_nodes_cancel)

    # Remove the entries in the tpn_information and update the time of the delayed train
    if not emergency_train:
        changed_trains[train_id_to_cancel_from] = {'train_id': train_id_to_cancel_from,
                                                   'DebugString': train_to_cancel_from.debug_string,
                                                   'Action': 'CancelFrom',
                                                   'CancelFrom': train_to_cancel_from.id,
                                                   'tpn_cancel_from': train_path_node_cancel_from.id}

    else:
        changed_trains[train_id_to_cancel_from] = {'train_id': train_id_to_cancel_from,
                                                   'DebugString': train_to_cancel_from.debug_string,
                                                   'Action': 'CancelFrom',
                                                   'CancelFrom': train_to_cancel_from.id,
                                                   'tpn_cancel_from': train_path_node_cancel_from.id,
                                                   'EmergencyTrain': True}

    return changed_trains, prime_timetable, train_id_to_cancel_from, track_info, edges_o_stations_d,\
           odt_facing_neighbourhood_operator, odt_priority_list_original


def operator_complete_delay(prime_timetable, changed_trains, trains_timetable, track_info, infra_graph,
                            edges_o_stations_d, parameters, odt_priority_list_original):
    # Build the list of candidates
    list_delay_candidates = list(set(parameters.set_of_trains_for_operator['Delay']))
    train_id_to_delay = list_delay_candidates[np.random.randint(0, len(list_delay_candidates) - 1)]

    # Seek the train from the train timetable with the train id to delay
    i = 0
    while trains_timetable[i].id != train_id_to_delay:
        i += 1

    # Get the train to delay and the train path nodes
    train_to_delay = trains_timetable[i]
    train_path_nodes = [tpn_id.id for tpn_id in train_to_delay.train_path_nodes]

    # Set the bus and emergency train to false and check if the train to delay is one of them
    bus = False
    emergency_train = False
    if hasattr(train_to_delay, 'emergency_bus') and train_to_delay.emergency_bus is True:
        bus = True
    if hasattr(train_to_delay, 'emergency_train') and train_to_delay.emergency_train is True:
        emergency_train = True

    # Get the list of odt facing the neighbourhood operator
    odt_facing_neighbourhood_operator, prime_timetable, odt_priority_list_original = \
        passenger_assignment.find_passenger_affected_by_delay(prime_timetable,
                                                              train_to_delay,
                                                              odt_priority_list_original)

    # Remove the edges and nodes of train from origin to stations and stations to destination
    edges_o_stations_d = remove_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_delay, prime_timetable)
    nodes_attr_train, edges_of_train = remove_nodes_edges_of_train(prime_timetable, train_id_to_delay)

    # Select the time to delay
    time_to_delay = parameters.delay_options[np.random.randint(0, len(parameters.delay_options))]

    # Remove entries of train in track_sequences
    if not bus:
        remove_entries_from_track_sequences(track_info, train_to_delay)

        # Update the train times
        train_before_update = copy.deepcopy(train_to_delay)
        train_to_delay = update_train_times_feasible_path_delay_operator(train_to_delay, time_to_delay, track_info,
                                                                         infra_graph, train_path_node_index_start=0,
                                                                         parameters=parameters)
        train_update_feasible = False
        if train_to_delay.delay == 'feasible':
            train_update_feasible = True
            # Remove the entries in the tpn_information and update the time of the delayed train
            remove_entries_in_tpn_information_and_update_tpn_of_delayed_train(track_info, train_to_delay,
                                                                              idx_start_delay=0)
        else:
            train_to_delay = copy.deepcopy(train_before_update)

        # Create and add driving and waiting edges and nodes to the Graph
        nodes_edges_dict = timetable_graph.create_transit_edges_nodes_single_train(train_to_delay, infra_graph,
                                                                                   idx_start_delay=0)
    # It is a bus
    else:
        train_update_feasible = True
        train_to_delay = delay_train_path_nodes_of_bus(train_to_delay, time_to_delay, parameters)
        nodes_edges_dict = timetable_graph.create_transit_edges_nodes_emergency_bus(train_to_delay)

    # Add the transit nodes and edges to the timetable graph
    timetable_graph.add_transit_nodes_edges_single_train_to_graph(prime_timetable, nodes_edges_dict, bus)

    # Create and add transfer edges to the graph
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train = \
        timetable_graph.transfer_edges_single_train(prime_timetable, train_to_delay, parameters, train_path_nodes)

    # Add the weight on the edges
    prime_timetable.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(prime_timetable, transfer_edges_attribute)

    # Update the list of edges from origin to destination
    if not bus:
        edges_o_stations_d = timetable_graph.add_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_delay,
                                                                                  prime_timetable, parameters, 0,
                                                                                  train_path_nodes)
    # If it is a bus
    else:
        edges_o_stations_d = timetable_graph.add_edges_of_bus_from_o_stations_d(edges_o_stations_d, train_to_delay,
                                                                                prime_timetable, parameters, 0,
                                                                                train_path_nodes)

    # Update the changed trains method
    if not bus and not emergency_train and train_update_feasible:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay,
                                             'DebugString': train_to_delay.debug_string,
                                             'Action': 'Delay',
                                             'ViriatoUpdate': train_to_delay.viriato_update}

    elif bus:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay,
                                             'DebugString': train_to_delay.debug_string,
                                             'Action': 'Delay',
                                             'ViriatoUpdate': train_to_delay.viriato_update,
                                             'EmergencyBus': True}

    elif emergency_train:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay,
                                             'DebugString': train_to_delay.debug_string,
                                             'Action': 'Delay',
                                             'ViriatoUpdate': train_to_delay.viriato_update,
                                             'EmergencyTrain': True}

    elif not train_update_feasible:
        pass

    return changed_trains, prime_timetable, train_id_to_delay, track_info, edges_o_stations_d,\
           odt_facing_neighbourhood_operator, odt_priority_list_original


def operator_part_delay(prime_timetable, changed_trains, trains_timetable, track_info, infra_graph, edges_o_stations_d,
                        parameters, odt_priority_list_original):
    # Set the train search to false for starting and the random seed
    train_found = False

    # Set the list of delay candidates
    list_delay_candidates = list(set(parameters.set_of_trains_for_operator['Delay']))
    if len(list_delay_candidates) >= 3:
        n_it = 0
        while not train_found and n_it <= 3:
            n_it += 1
            train_id_to_delay = list_delay_candidates[np.random.randint(0, len(list_delay_candidates))]
            i = 0
            try:
                while trains_timetable[i].id != train_id_to_delay:
                    i += 1
                train_to_delay = trains_timetable[i]
            except IndexError:
                continue
            # Check if it is an emergency bus, if so, skip to next candidate
            if hasattr(train_to_delay, 'emergency_bus') and train_to_delay.emergency_bus is True:
                continue
            comm_stops = 0

            # For each node, compute the number of commercial nodes, if greater than 3. We found our train to delay
            for train_path_node in train_to_delay.train_path_nodes:
                if train_path_node.stop_status.name == 'commercial_stop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True

    # If there is no train found in the candidates, go get one from Viriato trains timetable
    if not train_found:
        while not train_found:
            idx = np.random.randint(0, len(trains_timetable))
            train_to_delay = copy.deepcopy(trains_timetable[idx])
            train_id_to_delay = train_to_delay.id

            # Check if it is an emergency bus, if so, skip to next candidate
            if hasattr(train_to_delay, 'emergency_bus') and train_to_delay.emergency_bus is True:
                continue
            comm_stops = 0

            # For each node, compute the number of commercial nodes, if greater than 3. We found our train to delay
            for train_path_node in train_to_delay.train_path_nodes:
                if train_path_node.stop_status.name == 'commercial_stop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True

    # Delay from last comm stop
    idx_tpn_delay_from = identify_departure_train_path_node_id_of_train_to_delay_from(train_to_delay, parameters)
    tpn_delay_from = train_to_delay.train_path_nodes[idx_tpn_delay_from]

    # Get the list of odt facing the neighbourhood operator
    odt_facing_neighbourhood_operator, prime_timetable, odt_priority_list_original = \
        passenger_assignment.find_passenger_affected_by_part_delay(prime_timetable,
                                                                   train_to_delay,
                                                                   tpn_delay_from,
                                                                   odt_priority_list_original)

    # Set emergency train to false and check that the train to delay is an emergency train
    emergency_train = False
    if hasattr(train_to_delay, 'emergency_train') and train_to_delay.emergency_train is True:
        emergency_train = True

    train_path_nodes_delay = [train_path_node.id
                              for train_path_node in train_to_delay.train_path_nodes[idx_tpn_delay_from:]]

    # Remove edges from origin to stations and stations to destination of the part delayed train
    edges_o_stations_d = remove_edges_of_part_delayed_train_from_o_stations_d(edges_o_stations_d, train_to_delay,
                                                                              prime_timetable, train_path_nodes_delay)

    # Remove the nodes and the edges of the part of the train
    nodes_attr_train, edges_of_train = remove_nodes_edges_of_part_of_train(prime_timetable, train_id_to_delay,
                                                                           train_path_nodes_delay)

    # Get time to delay
    time_to_delay = parameters.delay_options[np.random.randint(0, len(parameters.delay_options))]

    # Remove entries of train in track_sequences
    try:
        remove_entries_part_of_train_from_track_sequences(track_info,
                                                          train_to_delay.train_path_nodes[idx_tpn_delay_from:])
    except ValueError:
        print('Something went wrong when trying to remove part of the train in the track sequences')
        pass
    # Update the train times
    train_before_update = copy.deepcopy(train_to_delay)
    train_to_delay = update_train_times_feasible_path_delay_operator(train_to_delay, time_to_delay, track_info,
                                                                     infra_graph, idx_tpn_delay_from, parameters)
    train_update_feasible = False
    if train_to_delay.delay == 'feasible':
        train_update_feasible = True
        # Remove the entries in the tpn_information and update the time of the delayed train
        remove_entries_in_tpn_information_and_update_tpn_of_delayed_train(track_info, train_to_delay,
                                                                          idx_tpn_delay_from)
    else:
        train_to_delay = train_before_update

    # Create and add driving and waiting edges and nodes to the graph
    nodes_edges_dict = timetable_graph.create_transit_edges_nodes_single_train(train_to_delay, infra_graph,
                                                                               idx_tpn_delay_from)
    timetable_graph.add_transit_nodes_edges_single_train_to_graph(prime_timetable, nodes_edges_dict, bus=False)

    # Create and add transfer edges to the Graph
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train = \
        timetable_graph.transfer_edges_single_train(prime_timetable, train_to_delay, parameters, train_path_nodes_delay)
    prime_timetable.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(prime_timetable, transfer_edges_attribute)

    # Update the list of edges from origin to destination
    edges_o_stations_d = timetable_graph.add_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_delay,
                                                                              prime_timetable, parameters,
                                                                              idx_tpn_delay_from,
                                                                              train_path_nodes_delay)

    # Update the changed trains method
    if not emergency_train and train_update_feasible:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay,
                                             'DebugString': train_to_delay.debug_string,
                                             'Action': 'Delay',
                                             'ViriatoUpdate': train_to_delay.viriato_update}
    elif train_update_feasible:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay,
                                             'DebugString': train_to_delay.debug_string,
                                             'Action': 'Delay',
                                             'ViriatoUpdate': train_to_delay.viriato_update,
                                             'EmergencyTrain': True}
    elif not train_update_feasible:
        pass

    return changed_trains, prime_timetable, train_id_to_delay, track_info, edges_o_stations_d, \
           odt_facing_neighbourhood_operator, odt_priority_list_original


def operator_emergency_train(timetable_prime_graph, changed_trains, emergency_train, trains_timetable, track_info,
                             infra_graph, edges_o_stations_d, parameters, odt_priority_list_original):

    train_id_et = emergency_train.id

    # Add an attribute for emergency train (it will be checked in neighbourhood selection like cancel
    emergency_train.emergency_train = True

    odt_facing_neighbourhood_operator = None

    # Get the time window duration to add an emergency train (it will be the time window of disruption)
    time_window_duration = round((parameters.disruption_time[1] - parameters.disruption_time[0]).seconds / 60, 0)

    # template train starts at 06:00, so just delay the start of the train by random time delay
    dep_time_delay = np.random.randint(0, time_window_duration)

    tpns = [tpn_id.id for tpn_id in emergency_train.train_path_nodes]

    # update the train times and index starts at the beginning
    idx_tpn_delay_from = 0
    emergency_train = update_train_times_feasible_path_delay_operator(emergency_train,
                                                                      dep_time_delay,
                                                                      track_info,
                                                                      infra_graph,
                                                                      idx_tpn_delay_from,
                                                                      parameters)

    if emergency_train.delay == 'infeasible':
        return changed_trains, timetable_prime_graph, train_id_et, track_info, edges_o_stations_d

    # add the entries to the tpn_information and update the time of the delayed train
    add_entries_to_tpn_information_and_update_tpns_of_emergency_train(track_info, emergency_train, idx_start_delay=0)

    # create and add driving and waiting edges and nodes to the Graph
    nodes_edges_dict = timetable_graph.create_transit_edges_nodes_single_train(emergency_train,
                                                                               infra_graph,
                                                                               idx_start_delay=0)

    timetable_graph.add_transit_nodes_edges_single_train_to_graph(timetable_prime_graph, nodes_edges_dict, bus=False)

    # create and add transfer edges to the Graph
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train =\
        timetable_graph.transfer_edges_single_train(timetable_prime_graph,
                                                    emergency_train,
                                                    parameters,
                                                    tpns)

    timetable_prime_graph.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(timetable_prime_graph, transfer_edges_attribute)

    # update the list of edges from origin to destination
    edges_o_stations_d = timetable_graph.add_edges_of_train_from_o_stations_d(edges_o_stations_d,
                                                                              emergency_train,
                                                                              timetable_prime_graph,
                                                                              parameters,
                                                                              0,
                                                                              tpns)

    # update the changed trains method
    changed_trains[train_id_et] = {'train_id': train_id_et,
                                   'DebugString': emergency_train.debug_string,
                                   'Action': 'EmergencyTrain',
                                   'body_message': emergency_train['body_message'],
                                   'EmergencyTrain': True}

    trains_timetable.append(emergency_train)

    return changed_trains, timetable_prime_graph, train_id_et, track_info, edges_o_stations_d, odt_facing_neighbourhood_operator, odt_priority_list_original

def operator_emergency_bus(timetable_prime_graph, changed_trains, trains_timetable, track_info, edges_o_stations_d,
                           parameters, odt_priority_list_original):
    # Initiate the bus identification number
    bus_id_nr = 90000
    bus_id = 'Bus' + str(bus_id_nr)

    # If there is already an emergency bus in the list of changed trains, add 10 to the number until it is a new number.
    while bus_id in changed_trains.keys():
        bus_id_nr += 10
        bus_id = 'Bus' + str(bus_id_nr)

    # Get the time window and the duration of the window in minutes
    time_window_from_time = parameters.disruption_time[0]
    time_window_to_time = parameters.disruption_time[1]
    time_window_duration_minutes = round((time_window_to_time - time_window_from_time).seconds / 60, 0)

    # Generate the departure time of the emergency bus inside the time window in a randomize way
    add_time_bus = np.random.randint(0, time_window_duration_minutes - 10)
    departure_time_bus = time_window_from_time + datetime.timedelta(minutes=add_time_bus)

    # Create the emergency bus
    emergency_bus = bus_add_bus_path_nodes(bus_id, departure_time_bus, parameters)

    # Get the train path nodes of the bus for the timetable graph
    tpns_bus = [tpn_id['ID'] for tpn_id in emergency_bus['TrainPathNodes']]

    # Create and add driving and waiting edges and nodes to the timetable graph
    nodes_edges_dict = timetable_graph.create_transit_edges_nodes_emergency_bus(emergency_bus)
    timetable_graph.add_transit_nodes_edges_single_train_to_graph(timetable_prime_graph, nodes_edges_dict, bus=True)

    # Create and add transfer edges to the timetable graph
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train = \
        timetable_graph.transfer_edges_single_bus(timetable_prime_graph,
                                                  emergency_bus,
                                                  parameters.transfer_MBus,
                                                  parameters.transfer_mBus,
                                                  tpns_bus,
                                                  parameters)
    timetable_prime_graph.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(timetable_prime_graph, transfer_edges_attribute)

    # Get the list of odt facing the neighbourhood operator
    odt_facing_neighbourhood_operator, timetable_prime_graph, odt_priority_list_original = \
        passenger_assignment.find_passenger_affected_by_emergency_bus(timetable_prime_graph,
                                                                      transfer_edges,
                                                                      odt_priority_list_original)

    # Update the list of edges from origin to destination
    edges_o_stations_d = timetable_graph.add_edges_of_bus_from_o_stations_d(edges_o_stations_d,
                                                                            emergency_bus,
                                                                            timetable_prime_graph,
                                                                            parameters)

    # Update the changed trains method
    changed_trains[bus_id] = {'train_id': bus_id,
                              'DebugString': emergency_bus['DebugString'],
                              'Action': 'EmergencyBus',
                              'body_message': None,
                              'EmergencyTrain': True}

    # Add the bus in the list of changed trains list
    trains_timetable.append(emergency_bus)

    return changed_trains, timetable_prime_graph, bus_id, track_info, edges_o_stations_d,\
           odt_facing_neighbourhood_operator, odt_priority_list_original


def bus_add_bus_path_nodes(bus_id, departure_time, parameters):
    # Initiate the bus dictionary
    bus = {}

    # Get the arrival time
    arrival_time = departure_time + datetime.timedelta(minutes=10)

    # Save the bus id in the dictionary
    bus['ID'] = bus_id
    bus['EmergencyBus'] = True

    # Scenario low traffic, hence the bus needs to serve between Walisellen and Dietlikon. Direction is chosen randomly
    if np.random.uniform(0.0, 1.0) < 0.5:
        start = parameters.stations_on_closed_tracks[1]
        end = parameters.stations_on_closed_tracks[0]
    else:
        start = parameters.stations_on_closed_tracks[0]
        end = parameters.stations_on_closed_tracks[1]

    # Save the the bus path nodes
    bus['TrainPathNodes'] = [{
            "ID": bus_id + str(1),
            "SectionTrackID": None,
            "IsSectionTrackAscending": None,
            "NodeID": start,
            "NodeTrackID": None,
            "ArrivalTime": departure_time,
            "DepartureTime": departure_time,
            "MinimumRunTime": datetime.timedelta(seconds=0),
            "MinimumStopTime": datetime.timedelta(seconds=0),
            "StopStatus": "commercial_stop",
            "SequenceNumber": 0
        },
        {
            "ID": bus_id + str(2),
            "SectionTrackID": None,
            "IsSectionTrackAscending": None,
            "NodeID": end,
            "NodeTrackID": None,
            "ArrivalTime": arrival_time,
            "DepartureTime": arrival_time,
            "MinimumRunTime": datetime.timedelta(minutes=10),
            "MinimumStopTime": datetime.timedelta(seconds=0),
            "StopStatus": "commercial_stop",
            "SequenceNumber": 1
        }]
    bus['DebugString'] = "EmergencyBus"
    return bus


def remove_edges_of_train_from_o_stations_d(edges_o_stations_d, train, prime_timetable):
    # Get only the arrival and departure node of the selected train
    arr_dep_nodes_train = [(n, v) for n, v in prime_timetable.nodes(data=True) if v['type'] in
                           ['arrivalNode', 'departureNode'] and v['train'] == train.id]

    # Check for every station if it is an arrival node or departure node, then remove edge accordingly
    for station, attr in arr_dep_nodes_train:
        # Arrival node
        if attr['type'] == 'arrivalNode':
            idx_station_d = sorted([i for i, x in enumerate(edges_o_stations_d.edges_stations_d) if x[0] == station],
                                   reverse=True)
            # Remove the edge from each station to destination
            for index in idx_station_d:
                edge = edges_o_stations_d.edges_stations_d[index]
                # Remove edge from dict keyed by destination
                edges_o_stations_d.edges_stations_d_dict[edge[1]].remove(edge)
                # Once the edge is removed, delete the dict
                if len(edges_o_stations_d.edges_stations_d_dict[edge[1]]) == 0:
                    del edges_o_stations_d.edges_stations_d_dict[edge[1]]
                # Delete the edge from stations to destination from the class
                del edges_o_stations_d.edges_stations_d[index]

        # Departure node
        elif attr['type'] == 'departureNode':
            idx_o_station = sorted([i for i, x in enumerate(edges_o_stations_d.edges_o_stations) if x[1] == station],
                                   reverse=True)
            # Remove the edge from each origin to stations
            for index in idx_o_station:
                edge = edges_o_stations_d.edges_o_stations[index]
                # Remove edge from dictionary keyed by origin
                edges_o_stations_d.edges_o_stations_dict[edge[0]].remove(edge)
                # Once the edge is removed, delete the dict
                if len(edges_o_stations_d.edges_o_stations_dict[edge[0]]) == 0:
                    del edges_o_stations_d.edges_o_stations_dict[edge[0]]
                # Delete the edge from stations to destination from the class
                del edges_o_stations_d.edges_o_stations[index]

    return edges_o_stations_d


def remove_edges_of_part_delayed_train_from_o_stations_d(edges_o_stations_d, train, prime_timetable,
                                                         train_path_nodes_delay):
    # Get all the arrival and departure nodes
    arr_dep_nodes_train = [(n, v) for n, v in prime_timetable.nodes(data=True) if v['train'] == train.id
                           and v['type'] in ['arrivalNode', 'departureNode'] and n[2] in train_path_nodes_delay]

    for station, attr in arr_dep_nodes_train:
        if attr['type'] == 'arrivalNode':
            idx_station_d = sorted([i for i, x in enumerate(edges_o_stations_d.edges_stations_d) if x[0] == station],
                                   reverse=True)
            for index in idx_station_d:
                edge = edges_o_stations_d.edges_stations_d[index]
                # Remove edge from dict keyed by destination
                try:
                    edges_o_stations_d.edges_stations_d_dict[edge[1]].remove(edge)
                except ValueError:
                    print('index from station to destination is not in the list for remove edges of part delayed.')

                # Proceed with the removing edges
                if len(edges_o_stations_d.edges_stations_d_dict[edge[1]]) == 0:
                    del edges_o_stations_d.edges_stations_d_dict[edge[1]]

                del edges_o_stations_d.edges_stations_d[index]

        elif attr['type'] == 'departureNode':
            idx_o_station = sorted([i for i, x in enumerate(edges_o_stations_d.edges_o_stations) if x[1] == station],
                                   reverse=True)
            for index in idx_o_station:
                edge = edges_o_stations_d.edges_o_stations[index]
                # Remove edge from dictionary keyed by origin
                edges_o_stations_d.edges_o_stations_dict[edge[0]].remove(edge)

                # Proceed with removing edges
                if len(edges_o_stations_d.edges_o_stations_dict[edge[0]]) == 0:
                    del edges_o_stations_d.edges_o_stations_dict[edge[0]]
                del edges_o_stations_d.edges_o_stations[index]

    return edges_o_stations_d


def remove_entries_part_of_train_from_track_sequences(track_info, train_path_nodes_train_part):
    # Loop through all the nodes to get the id of all the part of the train that needs to be removed
    for train_path_node in train_path_nodes_train_part:
        tuple_key_arr, value_arr = track_info.tuple_key_value_of_tpn_ID_arrival[train_path_node.id]
        tuple_key_dep, value_dep = track_info.tuple_key_value_of_tpn_ID_departure[train_path_node.id]
        track_info.track_sequences_of_TPN[tuple_key_arr].remove(value_arr)
        track_info.track_sequences_of_TPN[tuple_key_dep].remove(value_dep)


def remove_nodes_edges_of_part_of_train(prime_timetable, train_id, train_path_nodes_part):
    # Get nodes attribute
    nodes_attr_train = {x: y for x, y in prime_timetable.nodes(data=True)
                        if 'train' in y.keys()
                        and y['train'] == train_id
                        and x[2] in train_path_nodes_part}

    # Get the edges of train
    edges_of_train = {(u, v): attr for u, v, attr in prime_timetable.edges(nodes_attr_train.keys(), data=True)}

    # Remove edges and nodes
    prime_timetable.remove_edges_from(edges_of_train)
    prime_timetable.remove_nodes_from(nodes_attr_train.keys())

    return nodes_attr_train, edges_of_train


def remove_nodes_edges_of_train(prime_timetable, train_id):
    # Get the node attributes of  the canceled train
    nodes_attr_train = {x: y for x, y in prime_timetable.nodes(data=True) if 'train' in y.keys() and
                        y['train'] == train_id}

    # Get the edges attributes of the canceled train
    edges_of_train = {}
    for u, v, attr in prime_timetable.edges(data=True):
        if u in nodes_attr_train.keys() or v in nodes_attr_train.keys():
            edges_of_train[(u, v)] = attr

    prime_timetable.remove_edges_from(edges_of_train)
    prime_timetable.remove_nodes_from(nodes_attr_train.keys())

    return nodes_attr_train, edges_of_train


def remove_entries_from_track_sequences(track_info, train_to_delay):
    # Check each node in the path for the canceled train and remove the sequence in track_info
    for train_path_node in train_to_delay.train_path_nodes:
        try:
            tuple_key_arr, value_arr = track_info.tuple_key_value_of_tpn_ID_arrival[train_path_node.id]
            tuple_key_dep, value_dep = track_info.tuple_key_value_of_tpn_ID_departure[train_path_node.id]
            track_info.track_sequences_of_TPN[tuple_key_arr].remove(value_arr)
            track_info.track_sequences_of_TPN[tuple_key_dep].remove(value_dep)
        except ValueError:
            # Whole train is not in list
            print(f'The whole train "{train_to_delay}" is not in the list. Continue.')
            continue


def remove_entries_in_train_path_node_canceled_train(track_info, train_to_cancel, idx_start_cancel):
    for train_path_node in train_to_cancel.train_path_nodes:
        tuple_key_value_arr = track_info.tuple_key_value_of_tpn_ID_arrival.pop(train_path_node.id)
        tuple_key_value_dep = track_info.tuple_key_value_of_tpn_ID_departure.pop(train_path_node.id)
        tpn_info = track_info.tpn_information.pop(train_path_node.id)

    # Update track info
    if idx_start_cancel == 0:
        train_to_cancel._AlgorithmTrain__train_path_nodes = []
    else:
        train_to_cancel._AlgorithmTrain__train_path_nodes = train_to_cancel.train_path_nodes[:idx_start_cancel + 1]

        alns_platform.used_tracks_single_train(track_info.trains_on_closed_tracks, track_info.nr_usage_tracks,
                                               track_info.tpn_information, track_info.track_sequences_of_TPN,
                                               train_to_cancel, track_info.trains_on_closed_tracks,
                                               track_info.tuple_key_value_of_tpn_ID_arrival,
                                               track_info.tuple_key_value_of_tpn_ID_departure, 0)


def identify_last_departure_train_path_node_id_of_train_to_cancel_from(train):
    # Initialize i that equals the length of the whole sequence
    i = len(train.train_path_nodes) - 2
    # Loop through all the nodes from arrival node until it reaches the last commercial stop in the sequence
    while train.train_path_nodes[i].stop_status.name != 'commercial_stop':
        i -= 1
    return i


def identify_first_arrival_train_path_node_id_of_train(train, tpn_idx_start_delay):
    i = tpn_idx_start_delay
    while train.train_path_nodes[i].stop_status.name != 'commercial_stop':
        i += 1
    return train.train_path_nodes[i].id


def identify_last_departure_train_path_node_id_of_train(train, tpn_idx_start_delay):
    i = len(train.train_path_nodes) - 1

    while train.train_path_nodes[i].stop_status.name != 'commercial_stop':
        i -= 1
    return train.train_path_nodes[i].id


def update_train_times_feasible_path_delay_operator(train_to_delay, time_to_delay, track_info, infra_graph,
                                                    train_path_node_index_start, parameters):
    # Make a copy of the train to delay
    train = copy.deepcopy(train_to_delay)

    # Node index
    node_idx = train_path_node_index_start  # Train is completely delayed from start to end
    train_path_node_start = train.train_path_nodes[node_idx]

    # Check the arrival time of the start of RR, sometimes it can be changed due to new runtime calculation
    dep_time_start = train.train_path_nodes[node_idx].departure_time
    dep_time_start += datetime.timedelta(minutes=time_to_delay)

    # Compute the stop time
    stop_time = train_path_node_start.minimum_stop_time
    train_path_node_train_added_sequences = []
    visited_train_path_nodes = []

    # Index for the remaining RR nodes until end of train path
    node_idx_delay = node_idx
    start_timetable_update = True
    runtime_delayed_train_feasible = {}
    last_node_of_train = False

    for node in train.train_path_nodes[node_idx:]:
        # Loop through all nodes beginning at first rerouted node
        if node.id not in visited_train_path_nodes and not last_node_of_train:
            j = 0
            train_path_nodes_train = dict()  # Key, TPN ID, value, all entries in initial train tpn
            train_path_node_section = [[node.id], train.id, train_path_nodes_train]

            # First train path node of section (should be a station to check dep. time feasibility)
            train_path_nodes_train[node.id] = train.train_path_nodes[node_idx_delay + j]

            j += 1
            if node_idx_delay + j == len(train.train_path_nodes) - 1:
                last_node_of_train = True

            # Find train path node of section until next stations, where a train could be parked potentially
            try:
                while infra_graph.nodes[train.train_path_nodes[node_idx_delay + j].node_id]['NodeTracks'] is None \
                        and not last_node_of_train:
                    train_path_node_section[0].extend([train.train_path_nodes[node_idx_delay + j].id])
                    train_path_nodes_train[train.train_path_nodes[node_idx_delay + j].id] = train.train_path_nodes[
                        node_idx_delay + j]

                    j += 1
                    # Check if we reach the last node of the train, probably all end of trains have nodeTracks
                    if node_idx_delay + j == len(train.train_path_nodes) - 1:
                        if infra_graph.nodes[train.train_path_nodes[node_idx_delay + j].node_id]['NodeTracks'] is None:
                            break
            except IndexError:
                pass
            train_path_node_section[0].append(train.train_path_nodes[node_idx_delay + j].id)
            train_path_nodes_train[train.train_path_nodes[node_idx_delay + j].id] = train.train_path_nodes[
                node_idx_delay + j]

            if node_idx_delay + j == len(train.train_path_nodes) - 1:
                last_node_of_train = True

            node_idx_delay += j

            # Selection of the departure Times for the inputs into the greedy feasibility check
            if start_timetable_update:
                # For this node I have to take the Departure Time of initial RR train and not from Runtime calculation
                dep_time_section_start = dep_time_start
                dep_time_train_path_node = dep_time_section_start

            else:
                try:
                    arr_time_tpn = runtime_delayed_train_feasible[node.id]['ArrivalTime']
                except KeyError:
                    pass
                arr_time_train_path_node = arr_time_tpn
                dep_time_train_path_node = arr_time_train_path_node + train_path_nodes_train[node.id].minimum_stop_time

            # Find the free capacity for all nodes until next station in tracks_used_in_section
            section_clear = False
            nr_iterations = 0
            tpn_section_added_sequences = None
            while not section_clear and nr_iterations <= parameters.max_iteration_feasibility_check:
                nr_iterations += 1
                # Last node is a tuple (dep_time_last_node, nodeID)
                if start_timetable_update:

                    # Get the departure time before the delay
                    dep_time_train_path_node_before = dep_time_train_path_node

                    # Check runtime feasibility
                    section_clear, dep_time_train_path_node, runtime_section_feasible, tpn_section_added_sequences = \
                        check_train_section_runtime_feasibility_delay_operator(dep_time_train_path_node, track_info,
                                                                               parameters, train_path_node_section,
                                                                               infra_graph)

                    # If section clear, update the added sequences and the visited train path nodes
                    if section_clear:
                        runtime_delayed_train_feasible.update(runtime_section_feasible)
                        start_timetable_update = False

                        # Update the arrival time of the last tpn of the section, is the input for the next section !!!
                        train_path_node_section[2][
                            train_path_node_section[0][-1]]._AlgorithmTrainPathNode__arrival_time = \
                            runtime_delayed_train_feasible[train_path_node_section[0][-1]]['ArrivalTime']
                        train_path_node_train_added_sequences.append(tpn_section_added_sequences)
                        visited_train_path_nodes.extend([x for x in train_path_node_section[0][0:-1]])
                else:
                    section_clear, dep_time_train_path_node, runtime_section_feasible, tpn_section_added_sequences = \
                        check_train_section_runtime_feasibility_delay_operator(dep_time_train_path_node, track_info,
                                                                               parameters, train_path_node_section,
                                                                               infra_graph)
                    if section_clear:
                        runtime_delayed_train_feasible.update(runtime_section_feasible)

                        # Update the arrival time of the last tpn of the section, is the input for the next section !!!
                        train_path_node_section[2][
                            train_path_node_section[0][-1]]._AlgorithmTrainPathNode__arrival_time = \
                            runtime_delayed_train_feasible[train_path_node_section[0][-1]]['ArrivalTime']
                        train_path_node_train_added_sequences.append(tpn_section_added_sequences)
                        visited_train_path_nodes.extend([x for x in train_path_node_section[0][0:-1]])

                # Delay the departure time of the last node by one minute and try again
                if not section_clear:
                    stop_time += dep_time_train_path_node - dep_time_train_path_node_before

            if not section_clear:
                train_updated_times = copy.deepcopy(train_to_delay)
                train_updated_times.delay = 'infeasible'

                return train_updated_times

        # If node has been visited, continue
        else:
            continue

    # Get the last node id after the loop of every node before.
    last_node_id_of_train = node.id
    runtime_delayed_train_feasible[last_node_id_of_train]['DepartureTime'] = \
        runtime_delayed_train_feasible[last_node_id_of_train]['ArrivalTime']

    # Record the train to be updated
    train_updated_times = copy.deepcopy(train_to_delay)
    train_updated_times.viriato_update = \
        viriato_interface.get_list_update_train_times_delay(runtime_delayed_train_feasible)

    # Update the train information
    train_updated_times.delay = 'feasible'
    train_updated_times.runtime_delay_feasible = runtime_delayed_train_feasible
    return train_updated_times


def check_train_section_runtime_feasibility_delay_operator(dep_time_tpn_section_start, track_info, parameters,
                                                           train_path_nodes_section, infra_graph, section_start=True):
    # Get the departure time of the section start
    dep_time_train_path_node = dep_time_tpn_section_start

    # Setting the parameters
    train_path_node_ids_section = train_path_nodes_section[0]
    train_id = train_path_nodes_section[1]
    train_path_nodes_train = train_path_nodes_section[2]
    train_path_node_sequences_added_section = []
    runtime_section_feasible = dict()  # Resulting runtime of the new path
    section_clear = False
    train_path_node_outside_area_interest = False
    next_train_path_node_outside_area_interest = False
    last_train_path_node_outside_area_interest = False
    arrival_feasibility_not_needed = False
    no_train_path_sequences_added = False  # In the case this and next node are outside of AoI, no sequences are added

    # Loop through all node ids in the section
    i = -1
    for train_path_node_id in train_path_node_ids_section:
        i += 1  # First one is equal to zero
        tpn_clear = False
        tpn = train_path_nodes_train[train_path_node_id]

        # Check if the node is outside of the area of interest
        if not infra_graph.nodes[tpn.node_id]['in_area']:
            train_path_node_outside_area_interest = True

        # If the current node is not the last one of the section
        if i + 1 < len(train_path_node_ids_section):
            next_tpn_id = train_path_nodes_train[train_path_node_ids_section[i + 1]].node_id

            # Check if the node is outside of the area of interest
            if not infra_graph.nodes[next_tpn_id]['in_area']:
                next_train_path_node_outside_area_interest = True

        # After the first node in the section, we implement the next lines
        if i != 0:
            last_tpn_node_id = train_path_nodes_train[train_path_node_ids_section[i - 1]].node_id

            # Check if the node is outside of the area of interest
            if not infra_graph.nodes[last_tpn_node_id]['in_area']:
                last_train_path_node_outside_area_interest = True

        # The first node where the section start
        if section_start:
            if not train_path_node_outside_area_interest and not next_train_path_node_outside_area_interest:
                dep_time_tpn, tpn_clear, tpn_sequences_added = check_tpn_departure_feasibility_delay_operator(
                    dep_time_tpn_section_start, i, tpn, tpn_clear, train_path_node_id, train_path_node_ids_section,
                    train_path_nodes_train, track_info, train_id, parameters)
            else:
                tpn_clear = True
            if not tpn_clear:
                return section_clear, dep_time_tpn, runtime_section_feasible, train_path_node_sequences_added_section
            else:
                if not train_path_node_outside_area_interest and not next_train_path_node_outside_area_interest:
                    train_path_node_sequences_added_section.append(tpn_sequences_added)

                    runtime_section_feasible[train_path_node_id] = \
                        {'TrackID': tpn.section_track_id,
                         'ArrivalTime': tpn.arrival_time,
                         'DepartureTime': dep_time_tpn,
                         'RunTime': train_path_nodes_train[train_path_node_id].minimum_run_time,
                         'MinimumStopTime': train_path_nodes_train[train_path_node_id].minimum_stop_time}
                section_start = False
                continue
        # Initialize the travel time
        run_time = train_path_nodes_train[train_path_node_id].arrival_time - \
                   train_path_nodes_train[train_path_node_ids_section[i - 1]].departure_time
        arrival_time_tpn = dep_time_tpn + run_time

        if train_path_node_outside_area_interest or last_train_path_node_outside_area_interest:
            tpn_clear = True
            arrival_feasibility_not_needed = True
        iter_arrival_check = 0
        while not tpn_clear and iter_arrival_check <= parameters.max_iteration_section_check:
            iter_arrival_check += 1
            # Check arrival time feasibility of tpn, if not feasible increase runtime and check again
            arrival_time_tpn_before = arrival_time_tpn
            arrival_time_tpn, tpn_clear, tpn_sequences_added, delta_t_for_departure = \
                check_arrival_feasibility_tpn_delay_operator(arrival_time_tpn, i, tpn, tpn_clear, train_path_node_id,
                                                             train_path_node_ids_section, train_path_nodes_train,
                                                             track_info, train_id, parameters)

            if not tpn_clear and delta_t_for_departure is None:
                run_time += arrival_time_tpn - arrival_time_tpn_before

            elif not tpn_clear and delta_t_for_departure is not None:
                dep_time_tpn_start_section = dep_time_tpn_section_start + delta_t_for_departure
                remove_tpn_added_to_track_info_tpn_sequences(train_path_node_sequences_added_section, track_info)
                train_path_node_sequences_added_section = None
                return section_clear, dep_time_tpn_start_section, runtime_section_feasible, \
                       train_path_node_sequences_added_section

        if not arrival_feasibility_not_needed:
            train_path_node_sequences_added_section.append(tpn_sequences_added)

        if tpn_clear and i + 1 == len(train_path_node_ids_section):
            # last arrival node of the section
            section_clear = True
            runtime_section_feasible[train_path_node_id] = \
                {'TrackID': tpn.section_track_id,
                 'ArrivalTime': arrival_time_tpn,
                 'DepartureTime': None,
                 'RunTime': train_path_nodes_train[train_path_node_id].minimum_run_time,
                 'MinimumStopTime': train_path_nodes_train[train_path_node_id].minimum_stop_time}
        else:
            # Departure feasibility of the node in section
            tpn_clear = False

            dep_time_tpn = arrival_time_tpn + train_path_nodes_train[train_path_node_id].minimum_stop_time
            if not train_path_node_outside_area_interest and not next_train_path_node_outside_area_interest:
                dep_time_tpn_before = dep_time_tpn
                dep_time_tpn, tpn_clear, tpn_sequences_added = check_tpn_departure_feasibility_delay_operator(
                    dep_time_tpn, i, tpn, tpn_clear, train_path_node_id, train_path_node_ids_section,
                    train_path_nodes_train, track_info, train_id, parameters)
            else:  # not needed to check the dep feasibility if one of both nodes is outside the area
                tpn_clear = True
                no_train_path_sequences_added = True

            if not tpn_clear:
                delta_t_for_departure = dep_time_tpn - dep_time_tpn_before
                dep_time_tpn = dep_time_tpn_section_start + delta_t_for_departure
                remove_tpn_added_to_track_info_tpn_sequences(train_path_node_sequences_added_section, track_info)
                train_path_node_sequences_added_section = None
                return section_clear, dep_time_tpn, runtime_section_feasible, train_path_node_sequences_added_section

            else:
                if not no_train_path_sequences_added:
                    train_path_node_sequences_added_section.append(tpn_sequences_added)
                runtime_section_feasible[train_path_node_id] = \
                    {'TrackID': tpn.section_track_id,
                     'ArrivalTime': arrival_time_tpn,
                     'DepartureTime': dep_time_tpn,
                     'RunTime': train_path_nodes_train[train_path_node_id].minimum_run_time,
                     'MinimumStopTime': train_path_nodes_train[train_path_node_id].minimum_stop_time}

        # Check if any other parallel track is free
        check_parallel_track = False
        if check_parallel_track:
            parallel_tracks = viriato_interface.get_parallel_section_tracks(tpn.section_track_id)
            for other_track in parallel_tracks:
                if other_track.id == tpn.section_track_id:
                    continue
                else:
                    track_to_node_used = other_track.id
                    print(' track %i : is free to use' % other_track.id)
                    break

    return section_clear, dep_time_tpn, runtime_section_feasible, train_path_node_sequences_added_section


def check_tpn_departure_feasibility_delay_operator(dep_time_tpn, i, tpn, tpn_clear, tpn_id, tpn_ids_section,
                                                   train_path_nodes_train, track_info, train_id, parameters):
    # Fix the departure time for the train path node for further manipulation
    dep_time_tpn_locked = copy.deepcopy(dep_time_tpn)

    try:
        next_tpn_node_id = train_path_nodes_train[tpn_ids_section[i + 1]].node_id
    except IndexError:
        pass

    # Get the next train path node information into a tuple key
    track_next_tpn = train_path_nodes_train[tpn_ids_section[i + 1]].section_track_id
    next_tpn_id = train_path_nodes_train[tpn_ids_section[i + 1]].id
    tuple_key = (track_next_tpn, tpn.node_id, next_tpn_node_id, 'departure')

    # Set the sequence and conditions
    tpn_sequences_added = None
    condition_pre, condition_suc, no_trains_in_opposite_direction = False, False, False

    try:
        if [tpn_id, dep_time_tpn, train_id] not in track_info.track_sequences_of_TPN[tuple_key]:
            add_tpn_to_track_sequences(tpn_id, dep_time_tpn, train_id, track_info, tuple_key)
    except KeyError:
        # No other trains on this track in this direction
        condition_pre, condition_suc, no_trains_in_opposite_direction = True, True, True

    if not condition_pre and not condition_suc:
        index_tpn_on_track = track_info.track_sequences_of_TPN[tuple_key].index([tpn_id, dep_time_tpn, train_id])

        if index_tpn_on_track == 0:
            condition_pre = True
        else:
            preceding_tpn = track_info.track_sequences_of_TPN[tuple_key][index_tpn_on_track - 1]

            delta_hw_pre = datetime.timedelta(seconds=parameters.min_headway)
            try:
                pre_tpn_info = track_info.tpn_information[preceding_tpn[0]]
                condition_pre = pre_tpn_info['DepartureTime'] + delta_hw_pre <= dep_time_tpn
            except KeyError:
                condition_pre = True
                # this train has probably been cancelled and therefore the tpn is removed

        if index_tpn_on_track + 1 == len(track_info.track_sequences_of_TPN[tuple_key]):
            condition_suc = True
        else:
            succeeding_tpn = track_info.track_sequences_of_TPN[tuple_key][index_tpn_on_track + 1]
            delta_hw_suc = datetime.timedelta(seconds=parameters.min_headway)
            try:
                suc_tpn_info = track_info.tpn_information[succeeding_tpn[0]]
                condition_suc = dep_time_tpn <= suc_tpn_info['DepartureTime'] - delta_hw_suc
            except KeyError:
                # this train has probably been cancelled and therefore the tpn is removed
                condition_suc = True

    if condition_pre and condition_suc:
        # departure feasible
        dep_time_tpn, tpn_clear = check_departure_feasibility_tpn_opposite_track_direction(dep_time_tpn,
                                                                                           next_tpn_node_id,
                                                                                           tpn.node_id, tpn_clear,
                                                                                           tpn_id, track_info,
                                                                                           track_next_tpn, train_id,
                                                                                           parameters)
        if not no_trains_in_opposite_direction:
            tpn_sequences_added = {tuple_key: [tpn_id, dep_time_tpn, train_id]}

    elif condition_pre and not condition_suc or not condition_pre and not condition_suc:
        # not feasible
        dep_time_tpn = suc_tpn_info['DepartureTime'] + delta_hw_suc
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, dep_time_tpn_locked, train_id])
    elif not condition_pre and condition_suc:
        # not feasible
        dep_time_tpn = pre_tpn_info['DepartureTime'] + delta_hw_pre
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, dep_time_tpn_locked, train_id])

    return dep_time_tpn, tpn_clear, tpn_sequences_added


def remove_entries_in_tpn_information_and_update_tpn_of_delayed_train(track_info, train_to_delay, idx_start_delay):
    for train_path_node in train_to_delay.train_path_nodes[idx_start_delay:]:
        tuple_key_value_arr = track_info.tuple_key_value_of_tpn_ID_arrival.pop(train_path_node.id)
        tuple_key_value_dep = track_info.tuple_key_value_of_tpn_ID_departure.pop(train_path_node.id)
        tpn_info = track_info.tpn_information.pop(train_path_node.id)
        train_path_node = update_delayed_train_path_node(
            train_to_delay.runtime_delay_feasible[train_path_node.id], train_path_node)

    # Update track info
    alns_platform.used_tracks_single_train(track_info.trains_on_closed_tracks,
                                           track_info.nr_usage_tracks,
                                           track_info.tpn_information,
                                           track_info.track_sequences_of_TPN,
                                           train_to_delay,
                                           track_info.trains_on_closed_tracks,
                                           track_info.tuple_key_value_of_tpn_ID_arrival,
                                           track_info.tuple_key_value_of_tpn_ID_departure,
                                           idx_start_delay)


def update_delayed_train_path_node(runtime_delay_feasible, train_path_node):
    train_path_node._AlgorithmTrainPathNode__departure_time = runtime_delay_feasible['DepartureTime']
    train_path_node._AlgorithmTrainPathNode__arrival_time = runtime_delay_feasible['ArrivalTime']
    return train_path_node


def delay_train_path_nodes_of_bus(train_to_delay, time_to_delay, parameters):
    # Update departure time
    dep_time_start = train_to_delay.train_path_nodes[0].departure_time
    dep_time_start += datetime.timedelta(minutes=time_to_delay)

    # Update arrival time
    arrival_time = dep_time_start + datetime.timedelta(minutes=parameters.time_delta_delayed_bus)

    # Add the departure time start
    train_to_delay.train_path_nodes[0].arrival_time = dep_time_start
    train_to_delay.train_path_nodes[0].departure_time = dep_time_start

    # Add the arrival time
    train_to_delay.train_path_nodes[1].arrival_time = arrival_time
    train_to_delay.train_path_nodes[1].departure_time = arrival_time

    return train_to_delay


def identify_departure_train_path_node_id_of_train_to_delay_from(train, parameters):
    # Set random seed
    idx_candidates = []
    i = 0

    # Loop through all the nodes
    for train_path_node in train.train_path_nodes:
        i += 1
        if train_path_node.stop_status.name == 'commercial_stop':
            idx_candidates.append(i)

    # Get the departure node id to delay from in a randomize fashion
    i = idx_candidates[np.random.randint(1, len(idx_candidates) - 2)]
    return i


def add_tpn_to_track_sequences(tpn_id, tpn_arr_or_dep_Time, train_id, track_info, tuple_key):
    """tpn_id, dep_time_tpn, trainID
    :param tpn_arr_or_dep_Time:
    :param tpn_id:
    :param track_info: object with the sequence on track in it
    :param train_id: of the tpn train
    :param tuple_key: key for the sequence list (sectionTrack, fromNode, toNode)
    :return: None
    """
    track_info.track_sequences_of_TPN[tuple_key].append([tpn_id, tpn_arr_or_dep_Time, train_id])
    track_info.track_sequences_of_TPN[tuple_key] = sorted(track_info.track_sequences_of_TPN[tuple_key],
                                                          key=itemgetter(1))


def check_departure_feasibility_tpn_opposite_track_direction(dep_time_tpn, next_tpn_node_id, tpn_node_id, tpn_clear,
                                                             tpn_id, track_info, track_next_tpn, train_id, parameters):
    # Locked the departure time train path node for the removal manipulation at the end
    dep_time_tpn_locked = copy.deepcopy(dep_time_tpn)

    tuple_key_opposite_direction = (track_next_tpn, next_tpn_node_id, tpn_node_id, 'arrival')
    if tuple_key_opposite_direction not in track_info.track_sequences_of_TPN.keys():
        # this track is not driven in opposite direction
        tpn_clear = True
    else:
        add_tpn_to_track_sequences(tpn_id, dep_time_tpn, train_id, track_info, tuple_key_opposite_direction)
        index_tpn_on_track_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction].index(
            [tpn_id, dep_time_tpn, train_id])

        # Preceding train
        if index_tpn_on_track_op == 0:
            condition_pre_op = True
        else:
            delta_hw_pre_op = datetime.timedelta(seconds=parameters.min_headway)
            preceding_tpn_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction][
                index_tpn_on_track_op - 1]
            try:
                pre_tpn_info_op = track_info.tpn_information[preceding_tpn_op[0]]
                condition_pre_op = pre_tpn_info_op['ArrivalTime'] + delta_hw_pre_op <= dep_time_tpn
            except KeyError:
                condition_pre_op = True

        # Succeeding train
        if index_tpn_on_track_op + 1 == len(track_info.track_sequences_of_TPN[tuple_key_opposite_direction]):
            condition_suc_op = True
        else:
            delta_hw_suc_op = datetime.timedelta(seconds=parameters.min_headway)
            succeeding_tpn_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction][
                index_tpn_on_track_op + 1]
            try:
                suc_tpn_info_op = track_info.tpn_information[succeeding_tpn_op[0]]
                condition_suc_op = dep_time_tpn <= suc_tpn_info_op['ArrivalTime'] - delta_hw_suc_op
            except KeyError:
                # this train has probably been cancelled and therefore the tpn is removed
                condition_suc_op = True

        if condition_suc_op and condition_pre_op:
            tpn_clear = True
        elif condition_pre_op and not condition_suc_op or not condition_pre_op and not condition_suc_op:
            dep_time_tpn = suc_tpn_info_op['ArrivalTime'] + delta_hw_suc_op
            tpn_clear = False
        elif not condition_pre_op and condition_suc_op:
            dep_time_tpn = pre_tpn_info_op['ArrivalTime'] + delta_hw_pre_op
            tpn_clear = False

        # Remove it from opposite direction track sequences
        track_info.track_sequences_of_TPN[tuple_key_opposite_direction].remove([tpn_id, dep_time_tpn_locked, train_id])
    return dep_time_tpn, tpn_clear


def check_arrival_feasibility_tpn_delay_operator(arr_time_tpn, i, tpn, tpn_clear, tpn_id, tpn_ids_section,
                                                 train_path_nodes_train, track_info, train_id, parameters):
    # Set the parameters
    arr_time_tpn_locked = copy.deepcopy(arr_time_tpn)
    tpn_node_id = tpn.node_id
    last_tpn_node_id = train_path_nodes_train[tpn_ids_section[i - 1]].node_id
    track_tpn = train_path_nodes_train[tpn_id].section_track_id
    tuple_key = (track_tpn, last_tpn_node_id, tpn_node_id, 'arrival')
    tpn_sequences_added = None
    delta_t_for_departure = datetime.timedelta(seconds=0)
    condition_suc, condition_pre = False, False
    no_trains_in_opposite_direction = False
    try:
        if [tpn_id, arr_time_tpn, train_id] not in track_info.track_sequences_of_TPN[tuple_key]:
            add_tpn_to_track_sequences(tpn_id, arr_time_tpn, train_id, track_info, tuple_key)
    except KeyError:
        # no other trains on this track in this direction
        condition_pre, condition_suc, no_trains_in_opposite_direction = True, True, True
    if not condition_pre and not condition_suc:
        index_tpn_on_track = track_info.track_sequences_of_TPN[tuple_key].index([tpn_id, arr_time_tpn, train_id])

        if index_tpn_on_track == 0:
            condition_pre = True
        else:
            preceding_tpn = track_info.track_sequences_of_TPN[tuple_key][index_tpn_on_track - 1]
            delta_hw_pre = datetime.timedelta(seconds=parameters.min_headway)
            try:
                pre_tpn_info = track_info.tpn_information[preceding_tpn[0]]
                condition_pre = pre_tpn_info['ArrivalTime'] + delta_hw_pre <= arr_time_tpn
            except KeyError:
                # this train has probably been cancelled and therefore the tpn is removed
                condition_pre = True

        if index_tpn_on_track + 1 == len(track_info.track_sequences_of_TPN[tuple_key]):
            condition_suc = True
        else:
            succeeding_tpn = track_info.track_sequences_of_TPN[tuple_key][index_tpn_on_track + 1]
            delta_hw_suc = datetime.timedelta(seconds=parameters.min_headway)
            try:
                suc_tpn_info = track_info.tpn_information[succeeding_tpn[0]]
                condition_suc = arr_time_tpn <= suc_tpn_info['ArrivalTime'] - delta_hw_suc
            except KeyError:
                condition_suc = True
                # this train has probably been cancelled and therefore the tpn is removed

    if condition_pre and condition_suc:
        # Departure feasible
        arr_time_tpn, delta_t_for_departure, tpn_clear = \
            check_tpn_arrival_feasibility_opposite_direction(arr_time_tpn, delta_t_for_departure, last_tpn_node_id,
                                                             tpn_node_id, tpn_clear, tpn_id, track_tpn, track_info,
                                                             train_id, parameters)
        if not no_trains_in_opposite_direction:
            tpn_sequences_added = {tuple_key: [tpn_id, arr_time_tpn, train_id]}

    elif condition_pre and not condition_suc or not condition_pre and not condition_suc:
        # not feasible
        try:
            delta_t_for_departure = suc_tpn_info['ArrivalTime'] + delta_hw_pre - arr_time_tpn
        except UnboundLocalError:
            delta_t_for_departure = datetime.timedelta(minutes=5)
        if delta_t_for_departure < datetime.timedelta(seconds=0):
            delta_t_for_departure = succeeding_tpn[1] + delta_hw_suc - arr_time_tpn
        # todo, double check this condition ! why is the succeeding tpn arrival so much different to tpn info ?
        #         if suc_tpn_info.arrival_time > arr_time_tpn:
        # 			delta_t_for_departure = suc_tpn_info.arrival_time + delta_hw_pre - arr_time_tpn
        #         else:
        # 			delta_t_for_departure = delta_hw_pre + arr_time_tpn - suc_tpn_info.arrival_time
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, arr_time_tpn_locked, train_id])

    elif not condition_pre and condition_suc:
        # not feasible
        delta_t_for_departure = None  # try do increase runtime at next iteration
        arr_time_tpn = pre_tpn_info['ArrivalTime'] + delta_hw_pre
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, arr_time_tpn_locked, train_id])

    return arr_time_tpn, tpn_clear, tpn_sequences_added, delta_t_for_departure


def check_tpn_arrival_feasibility_opposite_direction(arr_time_tpn, delta_t_for_departure, last_tpn_node_id, tpn_node_id,
                                                     tpn_clear, tpn_id, track_tpn, track_info, train_id, parameters):
    # Locked the arrival time train path node for the removal operation
    arr_time_tpn_locked = copy.deepcopy(arr_time_tpn)
    tuple_key_opposite_direction = (track_tpn, tpn_node_id, last_tpn_node_id, 'departure')
    if tuple_key_opposite_direction not in track_info.track_sequences_of_TPN.keys():
        # this track is not driven in opposite direction
        tpn_clear = True
    else:
        add_tpn_to_track_sequences(tpn_id, arr_time_tpn, train_id, track_info, tuple_key_opposite_direction)
        index_tpn_on_track_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction].index(
            [tpn_id, arr_time_tpn, train_id])

        # Preceding train
        delta_hw_pre_op = datetime.timedelta(seconds=parameters.min_headway)

        if index_tpn_on_track_op == 0:
            condition_pre_op = True
        else:
            preceding_tpn_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction][
                index_tpn_on_track_op - 1]
            try:
                pre_tpn_info_op = track_info.tpn_information[preceding_tpn_op[0]]
                condition_pre_op = pre_tpn_info_op['DepartureTime'] + delta_hw_pre_op <= arr_time_tpn
            except KeyError:
                condition_pre_op = True

        # Succeeding train
        delta_hw_suc_op = datetime.timedelta(seconds=parameters.min_headway)
        if index_tpn_on_track_op + 1 == len(track_info.track_sequences_of_TPN[tuple_key_opposite_direction]):
            condition_suc_op = True
        else:
            succeeding_tpn_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction][
                index_tpn_on_track_op + 1]
            try:
                suc_tpn_info_op = track_info.tpn_information[succeeding_tpn_op[0]]
                condition_suc_op = arr_time_tpn <= suc_tpn_info_op['DepartureTime'] - delta_hw_suc_op
            except KeyError:
                # this train has probably been cancelled and therefore the tpn is removed
                condition_suc_op = True

        if condition_suc_op and condition_pre_op:
            tpn_clear = True

        elif condition_pre_op and not condition_suc_op or not condition_pre_op and not condition_suc_op:
            delta_t_for_departure = suc_tpn_info_op['DepartureTime'] + delta_hw_pre_op - arr_time_tpn
            if delta_t_for_departure < datetime.timedelta(seconds=0):
                delta_t_for_departure = succeeding_tpn_op[1] + delta_hw_suc_op - arr_time_tpn

        elif not condition_pre_op and condition_suc_op:
            # not feasible
            delta_t_for_departure = None  # try do increase runtime at next iteration
            arr_time_tpn = pre_tpn_info_op['DepartureTime'] + delta_hw_pre_op
            tpn_clear = False

        # remove it from opposite direction track sequences
        track_info.track_sequences_of_TPN[tuple_key_opposite_direction].remove([tpn_id, arr_time_tpn_locked, train_id])
    return arr_time_tpn, delta_t_for_departure, tpn_clear


def remove_tpn_added_to_track_info_tpn_sequences(tpn_sequences_added_section, track_info):
    for node_sequence in tpn_sequences_added_section:
        if node_sequence is None:
            continue
        for key, value in node_sequence.items():
            try:
                track_info.track_sequences_of_TPN[key].remove(value)
            except ValueError:
                pass


def restore_disruption_feasibility(infra_graph, track_info, parameters):
    # Restore disruption feasibility
    print('\nStart of the restore disruption feasibility')

    # Initialized the list for changed trains
    changed_trains = {}

    # Loop through all the trains on closed tracks
    for train_disruption_infeasible in track_info.trains_on_closed_tracks:
        # Copy the train
        train_original = copy.deepcopy(train_disruption_infeasible)

        # Get the train id
        train_id = train_disruption_infeasible.id

        # If a train is disrupted to the last node, cancel train from the last commercial stop
        if train_disruption_infeasible.train_path_nodes[train_disruption_infeasible.idx[0]] == \
                train_disruption_infeasible.train_path_nodes[-1] or \
                train_disruption_infeasible.train_path_nodes[train_disruption_infeasible.idx[0]] == \
                train_disruption_infeasible.train_path_nodes[-2]:

            # Find the last commercial stop before the node which is reached by the disrupted track
            i = 1
            while not train_disruption_infeasible.train_path_nodes[
                          train_disruption_infeasible.idx[0] - i].stop_status.name == 'commercial_stop':
                i += 1
            # Once the last node with comm stop reached, cancel train from here
            train_disruption_infeasible.cancel_train_after = train_disruption_infeasible.train_path_nodes[
                train_disruption_infeasible.idx[0] - i].id

            dbg_string = train_disruption_infeasible.debug_string
            remove_train_path_node_info_canceled_train(track_info, train_disruption_infeasible,
                                                       train_disruption_infeasible.cancel_train_after)
            changed_trains[train_id] = {'train_id': train_id, 'DebugString': dbg_string, 'Action': 'CancelFrom',
                                        'tpn_cancel_from': train_disruption_infeasible.cancel_train_after}

            train_disruption_infeasible = \
                viriato_interface.cancel_train_after(train_disruption_infeasible.id,
                                                     train_disruption_infeasible.cancel_train_after)
            continue

        # Try to reroute the train, if it is not possible, apply cancel cancel from or short turn, depending on case
        train_disruption_infeasible = try_to_reroute_train_via_different_nodes(train_disruption_infeasible, infra_graph,
                                                                               parameters)

        if hasattr(train_disruption_infeasible, 'reroute_path'):
            rerouted_train = viriato_interface.reroute_train(train_disruption_infeasible,
                                                             train_disruption_infeasible.reroute_path,
                                                             train_disruption_infeasible.index_reroute, track_info,
                                                             infra_graph)
            # Get the cut train indices for the time window
            cut_train_idx = None
            if rerouted_train is not None:
                cut_train_idx = rerouted_train.cut_indices
            else:
                pass

            # Get the run time of the new rerouted train
            run_time = viriato_interface.calculate_run_times(rerouted_train.id)

            # Update time from start of rerouting
            train_disruption_infeasible = update_reroute_train_times_feasible_path(rerouted_train, run_time, track_info,
                                                                                   infra_graph, parameters)

            # If we have to cancel the train, cancel train and go for the next train
            if hasattr(train_disruption_infeasible, 'cancel_train_id') \
                    and train_disruption_infeasible.cancel_train_id is not None:
                dbg_string = train_disruption_infeasible.debug_string
                remove_train_path_node_info_canceled_train(track_info, train_disruption_infeasible)

                # Cancel the train on Viriato
                train_disruption_infeasible = \
                    viriato_interface.cancel_train(train_disruption_infeasible.cancel_train_id)

                # Update the changed trains list
                changed_trains[train_id] = {'train_id': train_id, 'DebugString': dbg_string, 'Action': 'Cancel'}
                continue

            # If we have to partially cancel a train, cancel it partially and go to the next train
            elif hasattr(train_disruption_infeasible, 'cancel_train_after') \
                    and train_disruption_infeasible.cancel_train_after is not None:
                dbg_string = train_disruption_infeasible.debug_string
                remove_train_path_node_info_canceled_train(track_info, train_disruption_infeasible,
                                                           train_path_node_id_cancel_from=
                                                           train_disruption_infeasible.cancel_train_after)

                # Cancel the train after on Viriato
                train_disruption_infeasible = \
                    viriato_interface.cancel_train_after(train_disruption_infeasible.cancel_train_after)

                # Update the changed trains list
                changed_trains[train_id] = {'train_id': train_id, 'DebugString': dbg_string, 'Action': 'CancelFrom',
                                            'tpn_cancel_from': train_disruption_infeasible.cancel_train_after}
                continue

            # Update the track_info information
            update_train_path_node_information(0, track_info, train_disruption_infeasible, train_original)

            # Update the changed trains list with the current train
            changed_trains[train_id] = \
                {'train_id': train_id,
                 'DebugString': train_disruption_infeasible.debug_string,
                 'Action': 'Reroute',
                 'StartEndRR_tpnID':
                     train_disruption_infeasible.train_path_nodes[
                         train_disruption_infeasible.start_end_reroute_idx[0]].id,
                 'add_stop_time': train_disruption_infeasible.add_stop_time}

            # Cut the train with the given index
            if cut_train_idx is not None:
                train_disruption_infeasible = \
                    train_disruption_infeasible._AlgorithmTrain__train_path_nodes[cut_train_idx[0]:cut_train_idx[1]]
        # Cancel the train if not feasible
        elif hasattr(train_disruption_infeasible, 'cancel_train_id') \
                and train_disruption_infeasible.cancel_train_id is not None:
            dbg_string = train_disruption_infeasible.debug_string
            remove_train_path_node_info_canceled_train(track_info, train_disruption_infeasible)

            # Cancel the train on Viriato
            train_disruption_infeasible = viriato_interface.cancel_train(train_disruption_infeasible.cancel_train_id)

            # Update the changed trains list
            changed_trains[train_id] = {'DebugString': dbg_string,
                                        'Action': 'Cancel'}

        # Cancel the train after if not feasible
        elif hasattr(train_disruption_infeasible, 'cancel_train_after') \
                and train_disruption_infeasible.cancel_train_after is not None:
            dbg_string = train_disruption_infeasible.debug_string
            train_before_cancel = train_disruption_infeasible
            remove_train_path_node_info_canceled_train(track_info, train_disruption_infeasible,
                                                       train_path_node_id_cancel_from=
                                                       train_disruption_infeasible.cancel_train_after)

            # Cancel the train partially on Viriato
            train_canceled_partially = \
                viriato_interface.cancel_train_after(train_id, train_disruption_infeasible.cancel_train_after)

            # Update the changed trains list
            changed_trains[train_id] = {'train_id': train_id,
                                        'DebugString': dbg_string,
                                        'Action': 'CancelFrom',
                                        'tpn_cancel_from': train_disruption_infeasible.cancel_train_after}

        # If short turned train
        elif hasattr(train_disruption_infeasible, 'short_turn'):
            dbg_string = train_disruption_infeasible.debug_string
            tpn_initial_train = copy.deepcopy(train_disruption_infeasible.train_path_nodes)
            cancel_idx = train_disruption_infeasible.idx
            cancel_tpn_indices = [train_disruption_infeasible.train_path_nodes[cancel_idx[0] - 1],
                                  train_disruption_infeasible.train_path_nodes[cancel_idx[1]]]

            remove_tpn_info_short_turned_train(track_info, train_disruption_infeasible)
            train_before_disruption, train_after_disruption = short_turn_train(parameters, train_disruption_infeasible)
            changed_trains[train_id] = {'train_id': train_id,
                                        'DebugString': dbg_string,
                                        'Action': 'ShortTurn',
                                        'initial_tpn': tpn_initial_train,
                                        'tpns_cancel_from_to': cancel_tpn_indices,
                                        'train_before': train_before_disruption,
                                        'train_after': train_after_disruption}

    return changed_trains


def remove_train_path_node_info_canceled_train(track_info, train_disruption_infeasible,
                                               train_path_node_id_cancel_from=None):
    # Remove the train path nodes
    if train_path_node_id_cancel_from is None:
        # Loop through all the nodes
        for node in train_disruption_infeasible.train_path_nodes:
            try:
                del track_info.tpn_information[node.id]
            except KeyError:
                pass
    else:
        for node in train_disruption_infeasible.train_path_nodes:
            if node.id != train_path_node_id_cancel_from:
                continue
            else:
                try:
                    del track_info.tpn_information[node.id]
                except KeyError:
                    pass


def try_to_reroute_train_via_different_nodes(train, infra_graph, parameters):
    """
    :param parameters:
    :param train:
    :param infra_graph:
    :return: Returns the train as well the options which should be considered to apply
    """
    # Set the new parameters of the train
    train.cancel_train_id = None
    train.cancel_train_after = None
    train.reroute = False
    train.short_turn = False

    if len(train.idx) == 1:
        idx = train.idx[0]
    else:
        train.short_turn = True
        return train

    # Get the sequence number that is not reached
    sequence_number_not_reached = train.train_path_nodes[idx].sequence_number

    # Identify the last node which can be reached and has commercial stop
    i = 1
    while train.train_path_nodes[idx - i].stop_status.name != 'commercial_stop':
        if train.train_path_nodes[idx - i] == train.train_path_nodes[0]:
            # print(' start of train reached before commercial stop with node tracks')
            train.cancel_train_id = train.id
            return train
        i += 1
    reroute_from_node = train.train_path_nodes[idx - i]
    if len(viriato_interface.get_node_info(reroute_from_node.node_id).node_tracks) == 0:
        # Check if this node is the first node of the train
        if reroute_from_node.sequence_number == train.train_path_nodes[0].sequence_number:
            # todo Double check if this if is even necessary, seems like a second check of same condition
            train.cancel_train_id = train.id
            return train
        else:
            cancel_train_from_node = True
            train.cancel_train_after = reroute_from_node.id
            return train

    index_reroute = [idx - i, None]

    # Identify the node which can not be reached and has commercial stop
    i = 0
    while train.train_path_nodes[idx + i].stop_status.name != 'commercial_stop':
        if train.train_path_nodes[idx + i] == train.train_path_nodes[-1]:
            train.cancel_train_after = reroute_from_node.id
            return train
        i += 1
    reroute_to_node = train.train_path_nodes[idx + i]
    if len(viriato_interface.get_node_info(reroute_to_node.node_id).node_tracks) == 0:
        if reroute_from_node.id == train.train_path_nodes[0].id:
            # If it is the first node of train path, cancel train
            train.cancel_train_id = train.id
        else:
            train.cancel_train_after = reroute_from_node.id
        return train

    index_reroute[1] = idx + i
    # Try to find a path to the node not reached
    try:
        # With the cutoff of the weight_closed_tracks we make sure the path via this edges is not considered
        path = nx.single_source_dijkstra(infra_graph, reroute_from_node.node_id, reroute_to_node.node_id,
                                         cutoff=100000)
        path_found = True
        train.reroute = True
        train.reroute_path = path
        train.index_reroute = index_reroute
        return train
    except nx.NetworkXNoPath:
        path_found = False

    # If no path found identify the node after the closed section track
    if sequence_number_not_reached != train.train_path_nodes[-1].sequence_number and not path_found:
        # Look in train path nodes until the next commercial stop is found
        n = index_reroute[1] + 1
        while train.train_path_nodes[n].stop_status.name != 'commercial_stop':
            if train.train_path_nodes[n] == train.train_path_nodes[-1]:
                train.cancel_train_after = reroute_from_node.id
                return train
            n += 1
        first_node_after_closure = train.train_path_nodes[n].node_id
        try:
            path = nx.single_source_dijkstra(infra_graph, reroute_from_node.node_id, first_node_after_closure,
                                             cutoff=100000)
            index_reroute[1] = n
            viriato_node_rerouted_to = viriato_interface.get_node_info(first_node_after_closure)
            path_found = True
            train.reroute = True
            train.reroute_path = path
            train.index_reroute = index_reroute
            return train

        except nx.NetworkXNoPath:
            path_found = False
            train.cancel_train_after = reroute_from_node.id
            return train

    elif sequence_number_not_reached == train.train_path_nodes[-1].sequence_number and not path_found:
        train.cancel_train_after = reroute_from_node.id
        return train


def update_reroute_train_times_feasible_path(rerouted_train, rerouted_train_run_times, track_info, infra_graph,
                                             parameters):
    # Make a copy of the train
    train = copy.deepcopy(rerouted_train)

    # Check for the specific train
    if train.debug_string == 'RVZH S 20 tt_(S)':
        print('wait')

    # Create the run time dictionary for all the train path nodes of the rerouted train
    rerouted_train_run_times_dict = \
        helpers.build_dict_from_viriato_object_run_time_train_path_node_id(rerouted_train_run_times)

    # Get the node index for the first rerouted node
    node_idx = train.start_end_reroute_idx[0]

    # Copy the node on the train path where the rerouting starts
    train_path_node_start_reroute = copy.deepcopy(train.train_path_nodes[node_idx])

    # Check the arrival time of the start of reroute, sometimes it can be changed due to new runtime calculation
    dep_time_last_node_before_reroute = train.train_path_nodes[node_idx - 1].departure_time  # Previous node dep. time
    arrival_time_start_reroute_initial = train_path_node_start_reroute.arrival_time  # Current node arrival time

    # Check the arrival time: Get the run time from previous node to the first rerouted node
    if rerouted_train_run_times.update_times_train_path_nodes[node_idx].minimum_run_time is not None:
        runtime_to_starting_node_reroute = \
            rerouted_train_run_times.update_times_train_path_nodes[node_idx].minimum_run_time
    else:
        runtime_to_starting_node_reroute = None
        rerouted_train_run_times_dict[train_path_node_start_reroute.id].minimum_run_time = None

    # Check the arrival time: if first node to rerouting arrival time is sooner than the departure + runtime,
    # If so, update the new arrival time at the first node in the rerouting path
    if dep_time_last_node_before_reroute + runtime_to_starting_node_reroute > arrival_time_start_reroute_initial:
        # Get the new arrival time
        new_arrival_time_start_reroute = dep_time_last_node_before_reroute + runtime_to_starting_node_reroute

        # Compute departure time at the first rerouting node by adding the new arrival time plus the stopping time
        new_departure_time_start_reroute = \
            new_arrival_time_start_reroute + \
            rerouted_train_run_times.update_times_train_path_nodes[node_idx].minimum_stop_time

        # Update the departure time, the arrival time and the run time for the first rerouting node.
        train.train_path_nodes[node_idx]._AlgorithmTrainPathNode__arrival_time = new_arrival_time_start_reroute
        train.train_path_nodes[node_idx]._AlgorithmTrainPathNode__departure_time = new_departure_time_start_reroute

    # Get the stop time from the beginning of the rerouting path
    stop_time = train_path_node_start_reroute.minimum_stop_time
    tpn_train_added_sequences = []
    visited_train_path_nodes = []

    # Index for the remaining reroute nodes until end of train path and set the parameters for the loop
    node_idx_reroute = node_idx
    start_travel_time_update = True
    runtime_reroute_train_feasible = {}
    last_node_of_train = False
    add_stop_time = datetime.timedelta(minutes=0)

    # Loop through all the nodes on the train path
    for node in train.train_path_nodes[node_idx:]:
        # Loop through all nodes beginning at first rerouted node
        if node.id not in visited_train_path_nodes and not last_node_of_train:
            # Set the parameters for the loop
            j = 0
            run_times_of_section = dict()  # Key NodeID, value runtime from calculate runtime
            tpns_reroute_train = dict()  # Key, TPN ID, value, all entries in initial train tpn
            tpns_section = [run_times_of_section, [node.id], train.id, tpns_reroute_train]  # Nodes of the section
            # Save the run times of the first tpn node of section (should be a station to check dep. time feasibility)
            run_times_of_section[node.id] = \
                rerouted_train_run_times_dict[train.train_path_nodes[node_idx_reroute + j].id]
            # Save the first train path node of the section
            tpns_reroute_train[node.id] = train.train_path_nodes[node_idx_reroute + j]
            j += 1

            # Check if it is the last node of the train path
            if node_idx_reroute + j == len(train.train_path_nodes) - 1:
                last_node_of_train = True

            # Identify the section by getting all the tpns of section until next station,
            # where a train could be parked potentially
            while infra_graph.nodes[train.train_path_nodes[node_idx_reroute + j].node_id]['NodeTracks'] is None:
                tpns_section[1].extend([train.train_path_nodes[node_idx_reroute + j].id])
                tpns_reroute_train[train.train_path_nodes[node_idx_reroute + j].id] = \
                    train.train_path_nodes[node_idx_reroute + j]

                # Find the run_time to next node
                run_times_of_section[train.train_path_nodes[node_idx_reroute + j].id] = \
                    rerouted_train_run_times_dict[train.train_path_nodes[node_idx_reroute + j].id]
                j += 1

                # Check if we reach the last node of the train, probably all end of trains have nodeTracks
                if node_idx_reroute + j == len(train.train_path_nodes) - 1:
                    if infra_graph.nodes[train.train_path_nodes[node_idx_reroute + j].node_id]['NodeTracks'] is None:
                        print('This train ends at a station without station tracks')
                        break

            tpns_section[1].append(train.train_path_nodes[node_idx_reroute + j].id)
            run_times_of_section[train.train_path_nodes[node_idx_reroute + j].id] = \
                rerouted_train_run_times_dict[train.train_path_nodes[node_idx_reroute + j].id]
            tpns_reroute_train[train.train_path_nodes[node_idx_reroute + j].id] = \
                train.train_path_nodes[node_idx_reroute + j]

            # Check if it is the last node of the train path
            if node_idx_reroute + j == len(train.train_path_nodes) - 1:
                last_node_of_train = True

            # Get the index of the current node
            node_idx_reroute += j

            # Selection of the departure Times for the inputs into the greedy feasibility check
            if start_travel_time_update:
                # For this node I have to take the Departure Time of initial RR train and not from Runtime calculation
                dep_time_tpn = tpns_reroute_train[node.id].departure_time

            else:
                try:
                    arr_time_tpn = runtime_reroute_train_feasible[node.id]['ArrivalTime']
                except KeyError:
                    print(f'Key error: how that node id: {node.id} does not have runtime feasible?')
                # Get the departure time from the arrival time plus the minimum stop time
                dep_time_tpn = arr_time_tpn + run_times_of_section[node.id].minimum_stop_time

            # Find the free capacity for all nodes until next station in tracks_used_in_section
            section_clear = False
            nr_iterations = 0

            # Start the greedy feasibility
            while not section_clear and nr_iterations <= parameters.max_iteration_feasibility_check:
                nr_iterations += 1
                # Steps for first node
                if start_travel_time_update:

                    # Get the departure time before the rerouting
                    dep_time_tpn_before = dep_time_tpn

                    # Check the train section runtime feasibility
                    section_clear, dep_time_tpn, runtime_section_feasible, tpn_section_added_sequences = \
                        check_train_section_runtime_feasibility(dep_time_tpn, track_info, tpns_section, infra_graph,
                                                                parameters)

                    # If the section is clear, update the runtime reroute feasible train with the new train path node
                    if section_clear:
                        runtime_reroute_train_feasible.update(runtime_section_feasible)
                        start_travel_time_update = False

                        # Update the arrival time of the last tpn of the section, is the input for the next section
                        tpns_section[3][tpns_section[1][-1]]._AlgorithmTrainPathNode__arrival_time = \
                            runtime_reroute_train_feasible[tpns_section[1][-1]]['ArrivalTime']
                        tpn_train_added_sequences.append(tpn_section_added_sequences)
                        visited_train_path_nodes.extend([x for x in tpns_section[1][0:-1]])
                else:
                    section_clear, dep_time_tpn, runtime_section_feasible, tpn_section_added_sequences = \
                        check_train_section_runtime_feasibility(dep_time_tpn, track_info, tpns_section, infra_graph,
                                                                parameters)
                    if section_clear:
                        runtime_reroute_train_feasible.update(runtime_section_feasible)

                        # Update the arrival time of the last tpn of the section, is the input for the next section
                        tpns_section[3][tpns_section[1][-1]]._AlgorithmTrainPathNode__arrival_time = \
                            runtime_reroute_train_feasible[tpns_section[1][-1]]['ArrivalTime']
                        tpn_train_added_sequences.append(tpn_section_added_sequences)
                        visited_train_path_nodes.extend([x for x in tpns_section[1][0:-1]])

                # Delay the departure time of the last node by one minute and try again
                if not section_clear:
                    add_stop_time += dep_time_tpn - dep_time_tpn_before

            # If the section is not clear after feasibility check, cancel the train from the last feasible node
            if not section_clear:
                train_updated_times = copy.deepcopy(train)
                if rerouted_train.start_end_reroute_idx[0] > 10:
                    train_updated_times.cancel_train_from = \
                        rerouted_train.train_path_nodes[rerouted_train.start_end_reroute_idx[0]].id
                    train_updated_times.cancel_train_id = None
                else:
                    train_updated_times.cancel_train_from = None
                    train_updated_times.cancel_train_id = train_updated_times.id
                return train_updated_times
        # If node has been visited, continue
        else:
            continue

    # Get the last node information
    last_node_id_of_train = node.id
    stop_time_last_node = runtime_reroute_train_feasible[last_node_id_of_train]['MinimumStopTime']
    arrival_time_last_node = runtime_reroute_train_feasible[last_node_id_of_train]['ArrivalTime']
    departure_time_last_node = arrival_time_last_node + stop_time_last_node
    runtime_reroute_train_feasible[last_node_id_of_train]['DepartureTime'] = departure_time_last_node

    # Update the train train path node on Viriato
    train_updated_times = copy.deepcopy(train)
    train_updated_times.viriato_update = \
        viriato_interface.get_list_update_train_times_rerouting(runtime_reroute_train_feasible)
    train_updated_times = viriato_interface.viriato_update_train_times(train_updated_times.id,
                                                                       train_updated_times.viriato_update)

    # Update the train information
    train_updated_times.cut_indices = train.cut_indices
    train_updated_times.start_end_reroute_idx = train.start_end_reroute_idx
    train_updated_times.cancel_train_id = None
    train_updated_times.cancel_train_from = None
    train_updated_times.add_stop_time = add_stop_time
    return train_updated_times


def check_train_section_runtime_feasibility(dep_time_tpn_section_start, track_info, train_path_nodes_section,
                                            infra_graph, parameters, section_start=True):
    # Get the departure time for the train path node
    dep_time_tpn = dep_time_tpn_section_start

    # Get the runtime section and the indices of the train path nodes
    run_time_section = train_path_nodes_section[0]
    tpn_ids_section = train_path_nodes_section[1]

    # Get the train id and the train path node rerouted and set the section
    train_id = train_path_nodes_section[2]
    tpn_rr_train = train_path_nodes_section[3]
    tpn_sequences_added_section = []

    # Set the parameters for the greedy feasibility check
    runtime_section_feasible = dict()  # Resulting runtime of the new path
    section_clear = False
    train_path_node_outside_area_of_interest = False
    next_tpn_outside_area_of_interest = False
    last_tpn_outside_area_of_interest = False
    arrival_feasibility_not_needed = False
    no_tpn_sequences_added = False  # in the case this and next node are outside of AoI, no sequences are added
    i = -1

    # Loop through all the train path node section indices
    for tpn_id in tpn_ids_section:
        i += 1

        # Get the train path node id and the run time
        tpn_clear = False
        tpn = tpn_rr_train[tpn_id]
        tpn_runtime = run_time_section[tpn_id]

        # Check if the node is outside of the area of interest
        if not infra_graph.nodes[tpn.node_id]['in_area']:
            train_path_node_outside_area_of_interest = True

        # If the iteration is lower than the length of the whole train path node indices, get the next id
        if i + 1 < len(tpn_ids_section):
            next_tpn_id = tpn_rr_train[tpn_ids_section[i + 1]].node_id

            # Check if the node is outside of the area of interest
            if not infra_graph.nodes[next_tpn_id]['in_area']:
                next_tpn_outside_area_of_interest = True

        # If the iteration is not the first of the loop, get the previous node as the last node id
        if i != 0:
            last_tpn_id = tpn_rr_train[tpn_ids_section[i - 1]].node_id

            # Check if the node is not inside of the area of interest
            if not infra_graph.nodes[last_tpn_id]['in_area']:
                last_tpn_outside_area_of_interest = True

        # Go through this part if it is the starting section
        if section_start:

            # Check if the current node and the next node are outside of area, if not, check departure feasibility
            if not train_path_node_outside_area_of_interest and not next_tpn_outside_area_of_interest:
                dep_time_tpn, tpn_clear, tpn_sequences_added = check_tpn_departure_feasibility(
                    dep_time_tpn_section_start, i, tpn, tpn_clear, tpn_id, tpn_ids_section, tpn_rr_train, track_info,
                    train_id, parameters)

            # If the node is outside the area of interest, hence tpn is clear
            else:
                tpn_clear = True

            # If the train path node is not clear, return a false section clear with no runtime section
            if not tpn_clear:
                return section_clear, dep_time_tpn, runtime_section_feasible, tpn_sequences_added_section

            # If the train path node is clear, get the input on the runtime section
            else:
                if not train_path_node_outside_area_of_interest and not next_tpn_outside_area_of_interest:
                    tpn_sequences_added_section.append(tpn_sequences_added)
                runtime_section_feasible[tpn_id] = {'TrackID': tpn.section_track_id, 'ArrivalTime': tpn.arrival_time,
                                                    'DepartureTime': dep_time_tpn,
                                                    'RunTime': tpn_runtime.minimum_run_time,
                                                    'MinimumStopTime': tpn_runtime.minimum_stop_time,
                                                    'OriginalStopStatus': tpn.stop_status}

                # Set the section start to false and go to the next node
                section_start = False
                continue

        # Initialize the travel time
        min_runtime_tpn = run_time_section[tpn_id].minimum_run_time
        arrival_time_tpn = dep_time_tpn + min_runtime_tpn

        # If the train path node is outside of the area of interest, we do not need to verify the feasibility
        if train_path_node_outside_area_of_interest or last_tpn_outside_area_of_interest:
            tpn_clear = True
            arrival_feasibility_not_needed = True

        # Check arrival node
        iter_arrival_check = 0
        while not tpn_clear and iter_arrival_check <= parameters.max_iteration_section_check:
            iter_arrival_check += 1
            # Check arrival time feasibility of tpn, if not feasible increase runtime and check again
            arrival_time_tpn_before = arrival_time_tpn
            arrival_time_tpn, tpn_clear, tpn_sequences_added, delta_t_for_departure = check_arrival_feasibility(
                arrival_time_tpn, i, tpn, tpn_clear, tpn_id, tpn_ids_section, tpn_rr_train, track_info, train_id,
                parameters)

            # Increase the runtime if the tpn is not clear and the delta departure is none
            if not tpn_clear and delta_t_for_departure is None:
                min_runtime_tpn += arrival_time_tpn - arrival_time_tpn_before

            # Change the departure time of the start section with the computed delta t for departure time
            elif not tpn_clear and delta_t_for_departure is not None:
                dep_time_tpn_start_section = dep_time_tpn_section_start + delta_t_for_departure
                remove_tpn_added_to_track_info_tpn_sequences(tpn_sequences_added_section, track_info)
                tpn_sequences_added_section = None
                return section_clear, dep_time_tpn_start_section, runtime_section_feasible, tpn_sequences_added_section

        # If arrival feasibility check is needed, add the sequence in the list to check it later
        if not arrival_feasibility_not_needed:
            tpn_sequences_added_section.append(tpn_sequences_added)

        # If the train path node is clear and it is equal to the last arrival node of the section, get the info
        if tpn_clear and i + 1 == len(tpn_ids_section):
            # last arrival node of the section
            section_clear = True
            min_runtime_tpn_duration = min_runtime_tpn
            runtime_section_feasible[tpn_id] = {'TrackID': tpn.section_track_id,
                                                'ArrivalTime': arrival_time_tpn,
                                                'DepartureTime': None,
                                                'RunTime': min_runtime_tpn_duration,
                                                'MinimumStopTime': tpn_runtime.minimum_stop_time,
                                                'OriginalStopStatus': tpn.stop_status}

        # If not the last node of the train path, check the departure feasibility
        else:
            tpn_clear = False

            # Get the departure time of the current node
            dep_time_tpn = arrival_time_tpn + tpn_runtime.minimum_stop_time

            # If the node is not outside of the area of interest, check departure feasibility
            if not train_path_node_outside_area_of_interest and not next_tpn_outside_area_of_interest:
                dep_time_tpn_before = dep_time_tpn
                dep_time_tpn, tpn_clear, tpn_sequences_added = check_tpn_departure_feasibility(dep_time_tpn, i, tpn,
                                                                                               tpn_clear, tpn_id,
                                                                                               tpn_ids_section,
                                                                                               tpn_rr_train, track_info,
                                                                                               train_id, parameters)

            # Not needed to check the dep feasibility if one of both nodes is outside the area
            else:
                tpn_clear = True
                no_tpn_sequences_added = True

            # If the train path node is not clear, compute delta t for departure
            if not tpn_clear:
                delta_t_for_departure = dep_time_tpn - dep_time_tpn_before
                dep_time_tpn = dep_time_tpn_section_start + delta_t_for_departure
                remove_tpn_added_to_track_info_tpn_sequences(tpn_sequences_added_section, track_info)
                tpn_sequences_added_section = None
                return section_clear, dep_time_tpn, runtime_section_feasible, tpn_sequences_added_section

            # If the train path node is clear and it is not added to the sequence, add it, and save the information
            else:
                if not no_tpn_sequences_added:
                    tpn_sequences_added_section.append(tpn_sequences_added)
                min_runtime_tpn_duration = min_runtime_tpn
                runtime_section_feasible[tpn_id] = {'TrackID': tpn.section_track_id,
                                                    'ArrivalTime': arrival_time_tpn,
                                                    'DepartureTime': dep_time_tpn,
                                                    'RunTime': min_runtime_tpn_duration,
                                                    'MinimumStopTime': tpn_runtime.minimum_stop_time,
                                                    'OriginalStopStatus': tpn.stop_status}

    return section_clear, dep_time_tpn, runtime_section_feasible, tpn_sequences_added_section


def check_tpn_departure_feasibility(dep_time_tpn, i, tpn, tpn_clear, tpn_id, tpn_ids_section, tpn_rr_train, track_info,
                                    train_id, parameters):
    # Get the node id of the train path node
    tpn_node_id = tpn.node_id

    # Fix the departure time for the train path node for further manipulation
    dep_time_tpn_locked = copy.deepcopy(dep_time_tpn)
    # Get the next train path node id
    try:
        next_train_path_node_node_id = tpn_rr_train[tpn_ids_section[i + 1]].node_id
    except IndexError:
        print('wait')

    # Get the next section track id and the next node id
    track_next_train_path_node = tpn_rr_train[tpn_ids_section[i + 1]].section_track_id
    next_train_path_node_id = tpn_rr_train[tpn_ids_section[i + 1]].id

    # Create the tuple key for the departure node
    tuple_key = (track_next_train_path_node, tpn_node_id, next_train_path_node_node_id, 'departure')
    tpn_sequences_added = None

    # Add the train path node in the track sequences
    try:
        if [tpn_id, dep_time_tpn, train_id] not in track_info.track_sequences_of_TPN[tuple_key]:
            add_tpn_to_track_sequences(tpn_id, dep_time_tpn, train_id, track_info, tuple_key)
    except KeyError:
        print('Tuple Key Error in check train path node departure feasibility ')
        print(tuple_key)
        print([tpn_id, dep_time_tpn, train_id])

    # Get the index of the track in the current train path node
    index_train_path_node_on_track = track_info.track_sequences_of_TPN[tuple_key].index([tpn_id, dep_time_tpn,
                                                                                         train_id])

    if index_train_path_node_on_track == 0:
        condition_pre = True
    else:
        # Get the preceding train on the same node
        preceding_train_path_node = track_info.track_sequences_of_TPN[tuple_key][index_train_path_node_on_track - 1]
        try:
            # Compute the minimum headway between the current train and the preceding train on the same node
            min_headway_preceding = viriato_interface.get_headway_time(track_next_train_path_node, tpn_node_id,
                                                                       next_train_path_node_node_id,
                                                                       preceding_train_path_node[0],
                                                                       next_train_path_node_id)
        except Exception:
            min_headway_preceding = datetime.timedelta(seconds=parameters.min_headway)

        # Check the feasibility of the departure on the current node with the preceding node + the min headway
        try:
            pre_tpn_info = track_info.tpn_information[preceding_train_path_node[0]]
            condition_pre = pre_tpn_info['DepartureTime'] + min_headway_preceding <= dep_time_tpn
        except KeyError:
            condition_pre = True
            # this train has probably been cancelled and therefore the tpn is removed

    # Check if there is a succeeding node to the current
    if index_train_path_node_on_track + 1 == len(track_info.track_sequences_of_TPN[tuple_key]):
        condition_suc = True

    # If there is a succeeding node, proceed with the feasibility check
    else:
        succeeding_train_path_node = track_info.track_sequences_of_TPN[tuple_key][index_train_path_node_on_track + 1]

        # Get the minimum headway for the next node
        try:
            min_headway_succeeding = viriato_interface.get_headway_time(track_next_train_path_node, tpn_node_id,
                                                                        next_train_path_node_node_id,
                                                                        next_train_path_node_id,
                                                                        succeeding_train_path_node[0])

        except Exception:
            min_headway_succeeding = datetime.timedelta(seconds=parameters.min_headway)

        # Check the feasibility of the departure time at the next node with the minimum headway
        try:
            suc_tpn_info = track_info.tpn_information[succeeding_train_path_node[0]]
            condition_suc = dep_time_tpn <= suc_tpn_info['DepartureTime'] - min_headway_succeeding
        except KeyError:
            # this train has probably been cancelled and therefore the tpn is removed
            condition_suc = True

    # If both condition preceding and succeeding are true, the departure is feasible
    if condition_pre and condition_suc:

        # Check opposite track direction departure
        dep_time_tpn, tpn_clear = check_departure_feasibility_tpn_opposite_track_direction(dep_time_tpn,
                                                                                           next_train_path_node_node_id,
                                                                                           tpn_node_id, tpn_clear,
                                                                                           tpn_id, track_info,
                                                                                           track_next_train_path_node,
                                                                                           train_id, parameters)

        tpn_sequences_added = {tuple_key: [tpn_id, dep_time_tpn, train_id]}

    # Not feasible, hence compute the needed departure time and remove the current track sequence of the train path node
    elif condition_pre and not condition_suc or not condition_pre and not condition_suc:
        dep_time_tpn = suc_tpn_info['DepartureTime'] + min_headway_succeeding
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, dep_time_tpn_locked, train_id])

    # Not feasible because only the preceding node.
    elif not condition_pre and condition_suc:
        dep_time_tpn = pre_tpn_info['DepartureTime'] + min_headway_preceding
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, dep_time_tpn_locked, train_id])

    return dep_time_tpn, tpn_clear, tpn_sequences_added


def check_arrival_feasibility(arr_time_tpn, i, tpn, tpn_clear, tpn_id, tpn_ids_section, tpn_rr_train, track_info,
                              train_id, parameters):
    # Set the parameters for the feasibility check
    arr_time_tpn_locked = copy.deepcopy(arr_time_tpn)
    tpn_node_id = tpn.node_id
    last_train_path_node_node_id = tpn_rr_train[tpn_ids_section[i - 1]].node_id
    track_train_path_node = tpn_rr_train[tpn_ids_section[i]].section_track_id
    tuple_key = (track_train_path_node, last_train_path_node_node_id, tpn_node_id, 'arrival')
    tpn_sequences_added = None
    delta_t_for_departure = datetime.timedelta(seconds=0)

    # Check if the track sequence is already in the track info database, if not, add it
    try:
        if [tpn_id, arr_time_tpn, train_id] not in track_info.track_sequences_of_TPN[tuple_key]:
            add_tpn_to_track_sequences(tpn_id, arr_time_tpn, train_id, track_info, tuple_key)
    except KeyError:
        pass

    # Get the index number of the current node on track
    index_train_path_node_on_track = track_info.track_sequences_of_TPN[tuple_key].index([tpn_id, arr_time_tpn,
                                                                                         train_id])

    # If it is the first node, the preceding condition is feasible
    if index_train_path_node_on_track == 0:
        condition_pre = True

    # If not, check the feasibility
    else:
        preceding_train_path_node = track_info.track_sequences_of_TPN[tuple_key][index_train_path_node_on_track - 1]

        # Get the minimum headway with the preceding node
        try:
            min_headway_preceding = viriato_interface.get_headway_time(track_train_path_node,
                                                                       last_train_path_node_node_id, tpn_node_id,
                                                                       preceding_train_path_node[0],
                                                                       tpn_id)

        except Exception:
            min_headway_preceding = datetime.timedelta(seconds=parameters.min_headway)

        # Check the feasibility of the arrival time plus the headway if it is lower than the original arrival time
        try:
            pre_tpn_info = track_info.tpn_information[preceding_train_path_node[0]]
            condition_pre = pre_tpn_info['ArrivalTime'] + min_headway_preceding <= arr_time_tpn
        except KeyError:
            # this train has probably been cancelled and therefore the tpn is removed
            condition_pre = True

    # Check if it is the last node of the sequence, if yes, the succeeding condition is true
    if index_train_path_node_on_track + 1 == len(track_info.track_sequences_of_TPN[tuple_key]):
        condition_suc = True

    # If not, check the feasibility
    else:
        succeeding_train_path_node = track_info.track_sequences_of_TPN[tuple_key][index_train_path_node_on_track + 1]
        try:
            min_headway_succeeding = viriato_interface.get_headway_time(track_train_path_node,
                                                                        last_train_path_node_node_id,
                                                                        tpn_node_id,
                                                                        tpn_id,
                                                                        succeeding_train_path_node[0])

        except Exception:
            min_headway_succeeding = datetime.timedelta(seconds=parameters.min_headway)

        # Check the feasibility of arrival time with the minimum headway
        try:
            suc_tpn_info = track_info.tpn_information[succeeding_train_path_node[0]]
            condition_suc = arr_time_tpn <= suc_tpn_info['ArrivalTime'] - min_headway_succeeding
        except KeyError:
            condition_suc = True
            # This train has probably been cancelled and therefore the tpn is removed

    # If both preceding and succeeding are true, it is feasible, need to check opposite direction
    if condition_pre and condition_suc:
        # Arrival feasible
        # Check arrival feasibility opposite track direction
        arr_time_tpn, delta_t_for_departure, tpn_clear = \
            check_tpn_arrival_feasibility_opposite_direction(arr_time_tpn, delta_t_for_departure,
                                                             last_train_path_node_node_id, tpn_node_id, tpn_clear,
                                                             tpn_id, track_train_path_node, track_info, train_id,
                                                             parameters)

        tpn_sequences_added = {tuple_key: [tpn_id, arr_time_tpn, train_id]}

    # If both condition are not true or the succeeding condition is not true, compute delta time for departure
    elif condition_pre and not condition_suc or not condition_pre and not condition_suc:
        # not feasible
        delta_t_for_departure = suc_tpn_info['ArrivalTime'] + min_headway_preceding - arr_time_tpn

        # If delta is less than zero, get the next node arrival time + headway - the current node arrival time
        if delta_t_for_departure < datetime.timedelta(seconds=0):
            delta_t_for_departure = succeeding_train_path_node[1] + min_headway_succeeding - arr_time_tpn

        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, arr_time_tpn_locked, train_id])

    # If the only the preceding condition is not respect, increase the runtime for the next iteration
    elif not condition_pre and condition_suc:
        # not feasible
        delta_t_for_departure = None  # try do increase runtime at next iteration
        arr_time_tpn = pre_tpn_info['ArrivalTime'] + min_headway_preceding
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, arr_time_tpn_locked, train_id])

    return arr_time_tpn, tpn_clear, tpn_sequences_added, delta_t_for_departure


def update_train_path_node_information(node_idx, track_info, train_updated_times, train_original):
    # Remove all tpn of original train
    for tpn in train_original.train_path_nodes:

        # Delete the train path node from track info
        try:
            del track_info.tpn_information[tpn.id]
        except KeyError:
            continue

    # Update the tpn information
    from_tpn_index = node_idx
    if train_updated_times.train_path_nodes[from_tpn_index - 1].sequence_number >= 0:
        from_node = train_updated_times.train_path_nodes[from_tpn_index - 1].node_id
    else:
        from_node = None

    # Loop through all the nodes in the train path nodes
    tpn_node_index = from_tpn_index
    for node in train_updated_times.train_path_nodes[from_tpn_index:]:
        to_node = node.node_id
        if tpn_node_index + 1 < len(train_updated_times.train_path_nodes):
            next_train_path_node_id = train_updated_times.train_path_nodes[tpn_node_index + 1].id
        else:
            next_train_path_node_id = None

        if node.minimum_run_time is not None:
            min_run_time = node.minimum_run_time
        else:
            min_run_time = None

        track_info.tpn_information[node.id] = {
            'ArrivalTime': node.arrival_time,
            'DepartureTime': node.departure_time,
            'RunTime': min_run_time,
            'from_node': from_node,
            'to_node': to_node,
            'SectionTrack': node.section_track_id,
            'next_train_path_node_id': next_train_path_node_id,
            'TrainID': train_updated_times.id}

        from_node = to_node
        tpn_node_index += 1


def remove_tpn_info_short_turned_train(track_info, train_disruption_infeasible):
    # Get the index for cancel the train
    idx_to_cancel = train_disruption_infeasible.idx

    # Loop through the nodes that have to be canceled
    for node in train_disruption_infeasible.train_path_nodes[idx_to_cancel[0]:idx_to_cancel[1] + 1]:
        # Delete the node information on track info
        try:
            del track_info.tpn_information[node.id]
        except KeyError:
            pass


def short_turn_train(parameters, train_disruption_infeasible):
    # Clone the train twice for the short turn and keep only the path inside the area of interest and time window
    cloned_train1 = viriato_interface.clone_train(train_disruption_infeasible.id)
    cloned_train2 = viriato_interface.clone_train(train_disruption_infeasible.id)
    cloned_trains = viriato_interface.cut_trains_area_interest_time_window([cloned_train1, cloned_train2],
                                                                           parameters.stations_in_area,
                                                                           parameters.time_window)
    # Separate the two cloned trains
    cloned_train1 = cloned_trains[0]
    cloned_train2 = cloned_trains[1]
    idx_tpn_closed_track = train_disruption_infeasible.idx

    # Cancel train 1 from
    cloned_train1 = viriato_interface.cancel_train_after(cloned_train1.train_path_nodes[idx_tpn_closed_track[0] - 1].id)
    # cancel train 2 to
    cloned_train2 = viriato_interface.cancel_train_before(cloned_train2.train_path_nodes[idx_tpn_closed_track[1]].id)

    # Make sure they are in the area of interest and time window after partially cancel it
    cloned_trains = viriato_interface.cut_trains_area_interest_time_window([cloned_train1, cloned_train2],
                                                                           parameters.stations_in_area,
                                                                           parameters.time_window)

    # Separate the two cloned trains
    cloned_train1 = cloned_trains[0]
    cloned_train2 = cloned_trains[1]

    # Cancel the original train
    train_disruption_infeasible = viriato_interface.cancel_train(train_disruption_infeasible.id)
    return cloned_train1, cloned_train2


def add_entries_to_tpn_information_and_update_tpns_of_emergency_train(track_info, train_to_delay, idx_start_delay):
    for train_path_node in train_to_delay.train_path_nodes[idx_start_delay:]:
        train_path_node = update_delayed_train_path_node(
            train_to_delay.runtime_delay_feasible[train_path_node.id], train_path_node)

    # Update track info
    alns_platform.used_tracks_single_train(track_info.trains_on_closed_tracks,
                                           track_info.nr_usage_tracks,
                                           track_info.tpn_information,
                                           track_info.track_sequences_of_TPN,
                                           train_to_delay,
                                           track_info.trains_on_closed_tracks,
                                           track_info.tuple_key_value_of_tpn_ID_arrival,
                                           track_info.tuple_key_value_of_tpn_ID_departure,
                                           idx_start_delay)
