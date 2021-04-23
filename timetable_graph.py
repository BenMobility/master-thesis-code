"""
Created on Thu Feb 25 2021

@author: BenMobility

Build timetable graphs for the master thesis main codes.
"""
import networkx as nx
import viriato_interface
import neighbourhood_operators
import numpy as np
import datetime
import helpers
import pickle
import copy


def get_trains_timetable(time_window, sbb_nodes, parameters, debug_mode_train):
    """
    function that get the timetable with all the trains inside the time window and the area of interest.
    :param debug_mode_train: boolean value
    :param time_window: time window of the study. (type: a viriato object)
    :param sbb_nodes: a list of nodes/stations in the area of interest. (type: recarray)
    :param parameters: a class with all the recorded parameters for this study. (type: class)
    :return: set of trains in the time window and the area of interest (type: a viriato object)
    """
    # First, we need to keep only trains in the selected time window
    trains_timetable = \
        viriato_interface.get_trains_cut_time_range_driving_any_node(time_window,
                                                                     list(sbb_nodes.Code[
                                                                              sbb_nodes.Visited == True]))

    # If debug mode is one, only keep the ten first trains
    if debug_mode_train:
        trains_timetable = trains_timetable[0:10]

    print(f'Number of trains in the time window is: {len(trains_timetable)}')

    # Second, we want to keep only trains in the area of interest
    for train in trains_timetable:  # Check all the trains one at a time
        train_outside_area = False
        train_inside_area = False
        nodes_in_area = list()
        nodes_outside_area = list()
        nodes_idx_entering_area = list()
        nodes_idx_leaving_area = list()
        comm_stops = 0
        # Check every nodes the train drives through
        for j in range(0, len(train.train_path_nodes)):
            #  Check if the node is in the area of interest
            if train.train_path_nodes[j].node_id in parameters.stations_in_area:
                # Check if the node is actually a commercial stop
                if train.train_path_nodes[j].stop_status.name == 'commercial_stop':
                    comm_stops += 1  # If it is a commercial stop, add one
                nodes_in_area.append(train.train_path_nodes[j])  # Add node if it is in area of interest
                # Since train nodes are in chronological order, the first node tells that the train enters the area
                if not train_inside_area:
                    train_enters_area = True  # Train enters the area
                    train_inside_area = True  # Train is inside the area
                else:
                    train_enters_area = False  # Since the train is already entered. It is inside.
                # Add the entering area node ids only
                if train_enters_area:
                    nodes_idx_entering_area.append(j)
            # If the node is not in the area
            else:
                nodes_outside_area.append(train.train_path_nodes[j])
                # Since train nodes are in chronological order, the first node outside means the train leaves the area
                if not train_outside_area:
                    train_leaves_area = True  # Train leaves area
                else:
                    train_leaves_area = False  # Train is already outside, so it does not leave the area
                # Add leaving area node ids
                if train_leaves_area:
                    nodes_idx_leaving_area.append(j)
                train_outside_area = True
        # After checking all the nodes of the train path, if there is less than x (1) commercial stop, cancel train
        if comm_stops <= parameters.commercial_stops:
            viriato_interface.cancel_train(train.id)  # Cancelling a train that has less commercial stop than threshold
            continue
        # Check output : train leaving the area, but it has never entered
        if len(nodes_idx_leaving_area) == 1 and len(nodes_idx_entering_area) == 0:
            print(f'Train {train.id} leaves area, yet never entered.')
        # Check output : train never leaves, never enters in the area
        elif len(nodes_idx_leaving_area) == 0 and len(nodes_idx_entering_area) == 0:
            print(f'Train {train.id} never leaves, never enters in the area. Short service?')
        # Train that enters and leaves the area once
        elif len(nodes_idx_leaving_area) == 1 and len(nodes_idx_entering_area) == 1:
            # Make sure the train enters before it leaves
            if nodes_idx_entering_area[0] < nodes_idx_leaving_area[0]:  # The value represent the chronological order
                # Check if the first entering area node is the first node in the train sequence path
                if train.train_path_nodes[nodes_idx_entering_area[0]].sequence_number == 0:
                    # Check if the first leaving node is greater than 1 in the sequence of train path nodes
                    if nodes_idx_leaving_area[0] > 1:
                        # Cancel the train right after the last node in the area
                        viriato_interface.cancel_train_after(train.id, train.train_path_nodes[nodes_idx_leaving_area[0]
                                                                                              - 1].id)
                    else:
                        # If the leaving area node is the second node and the first node is entering, cancel train
                        viriato_interface.cancel_train(train.id)
                # Train starts outside the time window and finish also outside time window, need to cancel before&after
                else:
                    try:
                        train_path_node_cancel_before = train.train_path_nodes[nodes_idx_entering_area[0]].id
                        train_path_node_cancel_after = train.train_path_nodes[nodes_idx_leaving_area[0] - 1].id
                        viriato_interface.cancel_train_before(train.id, train_path_node_cancel_before)
                        viriato_interface.cancel_train_after(train.id, train_path_node_cancel_after)
                    except Exception:
                        print('wait, train starts outside the time window and finish also outside.')
            # If the entering node index is greater than the leaving node index
            else:
                # Train starts outside area and enters the area, cancel before entering
                viriato_interface.cancel_train_before(train.id, train.train_path_nodes[nodes_idx_entering_area[0]].id)
        # When train starts outside the area, runs through the area and ends outside
        elif len(nodes_idx_leaving_area) == 2 and len(nodes_idx_entering_area) == 1:
            # Keep only the train path node after the entering node and before the second leaving node
            train_path_node_cancel_before = train.train_path_nodes[nodes_idx_entering_area[0]].id
            train_path_node_cancel_after = train.train_path_nodes[nodes_idx_leaving_area[1] - 1].id
            viriato_interface.cancel_train_before(train.id, train_path_node_cancel_before)
            viriato_interface.cancel_train_after(train.id, train_path_node_cancel_after)
        # When train starts inside the area, leaves the area and comes back in the area
        elif len(nodes_idx_leaving_area) == 1 and len(nodes_idx_entering_area) == 2:
            train_path_node_cancel_after = train.train_path_nodes[nodes_idx_leaving_area[0] - 1]
            train_path_node_cancel_before = train.train_path_nodes[nodes_idx_entering_area[1]]
            neighbourhood_operators.short_turn_train_viriato_preselection(parameters, train,
                                                                          train_path_node_cancel_after,
                                                                          train_path_node_cancel_before)
        # When trains starts inside the area, leaves the area, enters back in the area and leaves again the area
        elif len(nodes_idx_leaving_area) == 2 and len(nodes_idx_entering_area) == 2:
            train_path_node_cancel_after = train.train_path_nodes[nodes_idx_leaving_area[0] - 1]
            train_path_node_cancel_before = train.train_path_nodes[nodes_idx_entering_area[1]]
            neighbourhood_operators.short_turn_train_viriato_preselection(parameters, train,
                                                                          train_path_node_cancel_after,
                                                                          train_path_node_cancel_before)
        # Check if there is more than 2 leaving and 2 entering areas
        elif len(nodes_idx_leaving_area) > 2 or len(nodes_idx_entering_area) > 2:
            print(f'There are {len(nodes_idx_leaving_area)} leaving area nodes and there are '
                  f'{len(nodes_idx_entering_area)} entering area nodes')
        else:
            pass

    # Third, we need to do a second pass on the cut time range
    trains_timetable_2 = \
        viriato_interface.get_trains_cut_time_range_driving_any_node(time_window,
                                                                     list(sbb_nodes.Code[
                                                                              sbb_nodes.Visited == True]))

    # If debug mode is one, only keep the ten first trains
    if debug_mode_train:
        trains_timetable_2 = trains_timetable_2[0:10]
    print(f'Number of trains in the time window and in the area of interest is: {len(trains_timetable_2)}')

    return trains_timetable_2


def get_timetable_with_waiting_transfer_edges(trains_timetable, parameters):
    """
    function that takes as parameters the train timetable and create the edges for transfer and waiting
    :param trains_timetable: all trains travelling on nodes (type: a viriato object)
    :param parameters: class of all parameters for this study. (type: class)
    :return:
    """
    stations_with_commercial_stop = []
    arrival_nodes = []
    arrival_nodes_attributes = {}
    arrival_nodes_passing_attributes = {}
    departure_nodes = []
    departure_nodes_attributes = {}
    departure_nodes_passing_attributes = {}
    driving_edges = list()
    driving_edges_attributes = {}
    waiting_edges = list()
    waiting_edges_attributes = {}

    n = 0
    debug = False

    # Loop through all trains
    for train in trains_timetable:
        n = n + 1
        s = 0  # Number of stop of a train
        total_train_length = len(train.train_path_nodes)
        train_id = train.id
        start_of_train = True

        if n > 100 and debug:
            break
        # Loop through all path nodes of a train
        for train_path_nodes in train.train_path_nodes:
            s = s + 1

            # Check if the train path goes within the stations in the area
            if train_path_nodes.node_id not in parameters.stations_in_area:
                # If not in the area, do not consider this node and go for the next one.
                continue

            # Consider only the trains where have commercial stops
            if train_path_nodes.stop_status.name == "commercial_stop":
                stations_with_commercial_stop.append(train_path_nodes.node_id)
                # Update time and node
                arrival_time_this_node = train_path_nodes.arrival_time
                arrival_node_this_node = train_path_nodes.node_id
                departure_time_this_node = train_path_nodes.departure_time
                departure_node_this_node = train_path_nodes.node_id

                # First node in the train path is the starting node, does not have an edge
                if start_of_train:
                    # Combine information of the departure node of the train path in one list
                    node_name_dep_this = (
                        departure_node_this_node, train_path_nodes.departure_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'd')
                    # Add the departure node into the list of all departure nodes
                    departure_nodes.append(node_name_dep_this)
                    # Record the attributes, train id, departure time, stop status
                    attributes = {'train': train_id, 'type': 'departureNode',
                                  'departureTime': departure_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    departure_nodes_attributes[node_name_dep_this] = attributes
                    departure_time_last_node = departure_time_this_node
                    departure_node_last_node_name = node_name_dep_this
                    start_of_train = False
                    continue

                # If the stop is equal to the total train path length, it means it is at the end of the train path
                elif s == total_train_length:
                    # Combine information of the arrival node in one list
                    node_name_arr_this = (
                        arrival_node_this_node, train_path_nodes.arrival_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'a')
                    # Add the arrival node into the list of all arrival nodes
                    arrival_nodes.append(node_name_arr_this)
                    # Record the attributes, train id, arrival time, stop status
                    attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    arrival_nodes_attributes[node_name_arr_this] = attributes
                    # Driving edge to this node, running time = arrival time - departure time
                    run_time = arrival_time_this_node - departure_time_last_node
                    # Add this edge in the driving edges for the timetable graph
                    driving_edges.append(
                        [departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                    # Add the attributes of this edge
                    driving_edges_attributes[(departure_node_last_node_name,
                                              node_name_arr_this)] = {'flow': 0,
                                                                      'type': 'driving',
                                                                      'train_id': train_id}
                    # Reset the departure and arrival nodes
                    del node_name_dep_this
                    del node_name_arr_this
                    del departure_time_last_node

                # Nodes in between two transit nodes (on the first iteration, departure time of the last node is
                # recorded from the starting node at line 188, then it will be recorded here below for the next
                # iterations)
                else:
                    # Arrival nodes
                    # Combine information of the arrival node in one list
                    node_name_arr_this = (
                        arrival_node_this_node, train_path_nodes.arrival_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'a')
                    # Add the arrival node into the list of all arrival nodes
                    arrival_nodes.append(node_name_arr_this)
                    # Record the attributes, train id, arrival time, stop status
                    attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    arrival_nodes_attributes[node_name_arr_this] = attributes
                    # Driving edge between last node and this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    # Add this edge in the driving edges for the timetable graph
                    driving_edges.append(
                        [departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                    # Add the attributes of this edge
                    driving_edges_attributes[(departure_node_last_node_name,
                                              node_name_arr_this)] = {'flow': 0,
                                                                      'type': 'driving',
                                                                      'train_id': train_id}

                    # Departure nodes
                    # Combine information of the departure node in one list
                    node_name_dep_this = (
                        departure_node_this_node, train_path_nodes.departure_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'd')
                    # Add the departure node into the list of all departure nodes
                    departure_nodes.append(node_name_dep_this)
                    # Record the attributes, train id, departure time, stop status
                    attributes = {'train': train_id, 'type': 'departureNode',
                                  'departureTime': departure_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    departure_nodes_attributes[node_name_dep_this] = attributes
                    # Waiting edge between the departure and the arrival time of this node
                    wait_time = (departure_time_this_node - arrival_time_this_node)
                    # Add the waiting time on the waiting edge lists
                    waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds / 60)])
                    waiting_edges_attributes[(node_name_arr_this, node_name_dep_this)] = {'flow': 0,
                                                                                          'type': 'waiting',
                                                                                          'train_id': train_id}

                    # Update the departure node for next iteration
                    departure_node_last_node_name = node_name_dep_this
                    departure_time_last_node = departure_time_this_node

            # If the node is not a commercial stop, it is a passing node
            elif train_path_nodes.stop_status.name == 'passing':
                arrival_time_this_node = train_path_nodes.arrival_time
                arrival_node_this_node = train_path_nodes.node_id
                departure_time_this_node = train_path_nodes.departure_time
                departure_node_this_node = train_path_nodes.node_id

                # If it is the first node of the train path
                if start_of_train:
                    # Combine information of the departure node of the train path in one list
                    node_name_dep_this = (
                        departure_node_this_node, train_path_nodes.departure_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'dp')
                    departure_nodes.append(node_name_dep_this)
                    # Add the departure node into the list of all departure nodes
                    attributes = {'train': train_id, 'type': 'departureNodePassing',
                                  'departureTime': departure_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    departure_nodes_passing_attributes[node_name_dep_this] = attributes

                    # Update the departure node for the next iteration
                    departure_time_last_node = departure_time_this_node
                    departure_node_last_node_name = node_name_dep_this
                    start_of_train = False
                    continue

                # When s is equal to the length of the train path nodes, it means we have reached the end of the path
                elif s == total_train_length:
                    # Combine information of the arrival node in one list
                    node_name_arr_this = (
                        arrival_node_this_node, train_path_nodes.arrival_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'ap')
                    # Add the arrival node into the list of all arrival nodes
                    arrival_nodes.append(node_name_arr_this)
                    # Record the attributes, train id, arrival time, stop status
                    attributes = {'train': train_id, 'type': 'arrivalNodePassing',
                                  'arrivalTime': arrival_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    arrival_nodes_passing_attributes[node_name_arr_this] = attributes
                    # Driving edge between last node and this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    # Add the attributes of this edge
                    driving_edges.append(
                        [departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                    driving_edges_attributes[(departure_node_last_node_name,
                                              node_name_arr_this)] = {'flow': 0,
                                                                      'type': 'driving',
                                                                      'train_id': train_id}
                    # Reset the departure and arrival nodes
                    del node_name_dep_this
                    del node_name_arr_this
                    del departure_time_last_node

                # Node in between two transit nodes (on the first iteration, departure time of the last node is
                # recorded from the starting node at line 188, then it will be recorded here below for the next
                # iterations)
                else:
                    # Arrival nodes
                    # Combine information of the arrival node in one list
                    node_name_arr_this = (
                        arrival_node_this_node, train_path_nodes.arrival_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'ap')
                    # Add the arrival node into the list of all arrival nodes
                    arrival_nodes.append(node_name_arr_this)
                    # Record the attributes, train id, arrival time, stop status
                    attributes = {'train': train_id, 'type': 'arrivalNodePassing',
                                  'arrivalTime': arrival_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    arrival_nodes_passing_attributes[node_name_arr_this] = attributes
                    # Driving edge between last node and this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    # Add this edge in the driving edges for the timetable graph
                    driving_edges.append(
                        [departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                    # Add the attributes of this edge
                    driving_edges_attributes[(departure_node_last_node_name,
                                              node_name_arr_this)] = {'flow': 0,
                                                                      'type': 'driving',
                                                                      'train_id': train_id}

                    # Departure nodes
                    # Combine information of the departure node in one list
                    node_name_dep_this = (
                        departure_node_this_node, train_path_nodes.departure_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'dp')
                    departure_nodes.append(node_name_dep_this)
                    # Add the departure node into the list of all departure nodes
                    attributes = {'train': train_id, 'type': 'departureNodePassing',
                                  'departureTime': departure_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    # Record the attributes, train id, departure time, stop status
                    departure_nodes_passing_attributes[node_name_dep_this] = attributes
                    # Waiting edge between the departure and the arrival time of this node
                    wait_time = (departure_time_this_node - arrival_time_this_node)
                    # Add the waiting time on the waiting edge lists
                    waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds / 60)])
                    waiting_edges_attributes[(node_name_arr_this, node_name_dep_this)] = {'flow': 0,
                                                                                          'type': 'waiting',
                                                                                          'train_id': train_id}

                    # Update the departure node for next iteration
                    departure_node_last_node_name = node_name_dep_this
                    departure_time_last_node = departure_time_this_node

    # Print message to be updated of the progress
    print('Transit nodes are created.')

    # Nodes for commercial stops
    stations_with_commercial_stop = np.unique(np.array(stations_with_commercial_stop))

    # Create the timetable with the driving, waiting edges
    timetable_graph = nx.DiGraph()
    timetable_graph.add_nodes_from(arrival_nodes_attributes)
    nx.set_node_attributes(timetable_graph, arrival_nodes_attributes)

    timetable_graph.add_nodes_from(arrival_nodes_passing_attributes)
    nx.set_node_attributes(timetable_graph, arrival_nodes_passing_attributes)

    timetable_graph.add_nodes_from(departure_nodes_attributes)
    nx.set_node_attributes(timetable_graph, departure_nodes_attributes)

    timetable_graph.add_nodes_from(departure_nodes_passing_attributes)
    nx.set_node_attributes(timetable_graph, departure_nodes_passing_attributes)

    timetable_graph.add_weighted_edges_from(driving_edges)
    nx.set_edge_attributes(timetable_graph, driving_edges_attributes)

    timetable_graph.add_weighted_edges_from(waiting_edges)
    nx.set_edge_attributes(timetable_graph, waiting_edges_attributes)

    # Save the nodes in a text file
    np.savetxt('output/nodes_commercial_stops' + '_m' + str(int(parameters.transfer_m.total_seconds() / 60)) + '_M'
               + str(int(parameters.transfer_M.total_seconds() / 60)) + '.csv', stations_with_commercial_stop)

    # Print message when the output of commercial stops is saved
    print('Nodes with commercial stops is saved in csv in the output file.')
    # create di graph with transit nodes and edges (waiting, driving)

    # Up until this line there are only trains in the graph
    print('Add transfer edges')
    timetable_graph = add_transfer_edges_to_graph(timetable_graph, parameters)
    # Save timetable with only trains graph as pickle file
    nx.write_gpickle(timetable_graph, 'output/pickle/timetable_graph_only_trains.pickle')

    return timetable_graph, stations_with_commercial_stop


def add_transfer_edges_to_graph(timetable_graph, parameters):
    # Zurich station has multiple lines, thus has transfer edges inside the node.
    zh_hb = ['85ZMUS', '85ZUE', '45ZSZU']  # Zurich codes to Node IDs '85ZMUS' --> 611, '85ZUE --> 638', '45ZSZU --> 13'
    zh_hb_code = [606, 633, 13]
    # todo: get all nodes, viriato interface, open dict, order node_id(list number code) by node_codes(keys)

    trains_by_node = dict()

    # Create trains by node
    for x, y in timetable_graph.nodes(data=True):
        # Check the type of node it is
        if y['type'] in ['arrivalNode', 'arrivalNodePassing', 'departureNodePassing']:
            continue
        # Get station number
        station = x[0]
        # Check if the train nodes has the station
        if not trains_by_node.__contains__(station):
            trains_by_node[station] = list()
        # Record the inputs for the station in a dictionary form
        station_in_dict = trains_by_node[station]
        station_in_dict.append([y['train'], x])

    # Create transfer edges
    transfer_edges = list()
    transfer_edges_attributes = dict()
    # Identify all arrival nodes
    arrival_nodes = [(x, y) for x, y in timetable_graph.nodes(data=True) if y['type'] == 'arrivalNode']
    for node, z in arrival_nodes:
        try:
            trains_in_same_node = copy.deepcopy(trains_by_node[node[0]])
            # Link all trains in Zurich main station, because there were three different stations for Zurich
            if node[0] in zh_hb_code:
                for zh_node in zh_hb_code:
                    if zh_node == node[0]:
                        continue
                    else:
                        trains_in_same_node.extend(trains_by_node[zh_node])
        except KeyError:
            print('Node ', node[0], ' has an arrival but no departure, probably nodeIDs of ZH_HB has changed.')
            continue

        # Check for trains that are departing from the same node
        dep_trains_same_node = [[x[0], x[1]] for x in trains_in_same_node if x[0] != z['train']]

        # Compute the weight of the transfer edges
        for train_id, label in dep_trains_same_node:
            try:
                delta_t = timetable_graph.nodes[label]['departureTime'] - timetable_graph.nodes[node]['arrivalTime']
            except KeyError:
                print(
                    'Error in the transfer edge creation, check selection of departing trains of same node')
                continue
            # Compute only the transfer edges that are inside the wanted time window
            if parameters.transfer_m < delta_t < parameters.transfer_M:
                weight = delta_t * parameters.beta_waiting + datetime.timedelta(minutes=parameters.beta_transfer)
                transfer_edges.append([node, label, float(weight.seconds / 60)])
                transfer_edges_attributes[(node, label)] = {'type': 'transfer'}

    # Add transfer edges to the timetable graph
    timetable_graph.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(timetable_graph, transfer_edges_attributes)
    return timetable_graph


def connect_home_stations(timetable_waiting_transfer, sbb_nodes, travel_time_selected_zones, selected_zones, id_nodes,
                          od_departure_time, parameters, create_timetable_home_connections_from_scratch):
    """
    Function the adds the home to station edges to the timetable_waiting_transfer.
    :param timetable_waiting_transfer: Digraph of the timetable with waiting and transfers edges
    :param sbb_nodes: a recarray of sbb nodes attributes.
    :param travel_time_selected_zones: recarray with from Zone to Zone travel time only for selected zones
    :param selected_zones: recarray with the selected zones with the coordinates
    :param id_nodes: nodes id the Viriato database inside the time window
    :param od_departure_time: od array with the desired departure time for each passenger
    :param parameters: class object with all the main parameters of the main code
    :param create_timetable_home_connections_from_scratch: Boolean parameter to create from scratch or not
    :return: timetable initial graph (type: Digraph)
    """
    # filename path to get the pickles
    filename_timetable = 'output/pickle/initial_timetable_m' + str(int(parameters.transfer_m.total_seconds() / 60)) \
                         + '_M' + str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
                         str(parameters.th_zone_selection) + '.pickle'
    filename_odt_list = 'output/pickle/odt_list_m' + str(int(parameters.transfer_m.total_seconds() / 60)) \
                        + '_M' + str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
                        str(parameters.th_zone_selection) + '.pickle'
    filename_odt_by_origin = 'output/pickle/odt_by_origin_m' + str(int(parameters.transfer_m.total_seconds() / 60)) \
                             + '_M' + str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
                             str(parameters.th_zone_selection) + '.pickle'
    filename_odt_by_dest = 'output/pickle/odt_by_dest_m' + str(int(parameters.transfer_m.total_seconds() / 60)) \
                           + '_M' + str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
                           str(parameters.th_zone_selection) + '.pickle'
    filename_station_candidates = 'output/pickle/station_candidates_m' + \
                                  str(int(parameters.transfer_m.total_seconds() / 60)) + '_M' + \
                                  str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
                                  str(parameters.th_zone_selection) + '.pickle'
    filename_origin_name_desired_dep_time = 'output/pickle/origin_name_desired_dep_time_m' + \
                                            str(int(parameters.transfer_m.total_seconds() / 60)) + '_M' + \
                                            str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
                                            str(parameters.th_zone_selection) + '.pickle'
    filename_origin_name_zone_dict = 'output/pickle/origin_name_zone_dict_m' + \
                                     str(int(parameters.transfer_m.total_seconds() / 60)) + '_M' + \
                                     str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
                                     str(parameters.th_zone_selection) + '.pickle'

    print('Start creating the edges home to stations')
    print(f'From scratch: {create_timetable_home_connections_from_scratch}')

    # Load the pickle files
    if not create_timetable_home_connections_from_scratch:
        # Unpickling timetable
        timetable_initial_graph = nx.read_gpickle(filename_timetable)

        # Unpickling odt_list
        with open(filename_odt_list, "rb") as path_list:
            odt_list = pickle.load(path_list)

        # Unpickling odt_by_origin
        with open(filename_odt_by_origin, 'rb') as path_origin:
            odt_by_origin = pickle.load(path_origin)

        # Unpickling odt_by_dest
        with open(filename_odt_by_dest, 'rb') as path_dest:
            odt_by_dest = pickle.load(path_dest)

        # Unpickling station_candidates
        with open(filename_station_candidates, 'rb') as path_candidate:
            station_candidates = pickle.load(path_candidate)

        # Unpickling origin_name_desired_dep_time
        with open(filename_origin_name_desired_dep_time, 'rb') as path_desired:
            origin_name_desired_dep_time = pickle.load(path_desired)

        # Unpickling origin_name_zone_dict
        with open(filename_origin_name_zone_dict, 'rb') as path_dict:
            origin_name_zone_dict = pickle.load(path_dict)

    # Create home connections from scratch
    else:
        destination_nodes = []
        destination_nodes_attributes = {}
        edges_d_stations = list()
        k2 = parameters.nb_zones_to_connect
        k = parameters.nb_stations_to_connect

        # Remove all stations without commercial stop
        sbb_nodes = sbb_nodes[sbb_nodes.commercial_stop == 1]

        # Identify zones where trips start
        all_origin_zones = list(np.unique(od_departure_time.fromZone))
        # Identify zones where trips end
        all_destination_zones = list(np.unique(od_departure_time.toZone))

        # Get all the k closest euclidean stations to a zone and transform it into a record array
        closest_stations_to_zone = helpers.get_all_k_closest_stations_to_zone(sbb_nodes, selected_zones, k)
        closest_stations_to_zone = helpers.closest_stations_to_zone_transform_record(closest_stations_to_zone)

        # Connect all origins of an odt with the departure nodes and create odt list for sp input
        timetable_with_origin, odt_list, odt_by_origin, odt_by_dest, station_candidates, origin_name_desired_dep_time, \
        origin_name_zone_dict = connect_origins_with_stations_for_all_odt(timetable_waiting_transfer, all_origin_zones,
                                                                          k2, parameters, sbb_nodes, od_departure_time,
                                                                          travel_time_selected_zones,
                                                                          closest_stations_to_zone)

        # Connect all destinations of an odt with the arrival nodes and create odt list for sp input
        timetable_initial_graph, station_candidates = \
            connect_all_destinations_with_stations(timetable_with_origin, all_destination_zones,
                                                   closest_stations_to_zone, destination_nodes,
                                                   destination_nodes_attributes, edges_d_stations, k2, sbb_nodes,
                                                   travel_time_selected_zones, station_candidates)

        # Save the output in pickle files
        # Timetable
        nx.write_gpickle(timetable_initial_graph, filename_timetable)

        # odt_list
        with open(filename_odt_list, 'wb') as f:
            pickle.dump(odt_list, f)

        # odt_by_origin
        with open(filename_odt_by_origin, 'wb') as f:
            pickle.dump(odt_by_origin, f)

        # odt_by_dest
        with open(filename_odt_by_dest, 'wb') as f:
            pickle.dump(odt_by_dest, f)

        # station_candidates
        with open(filename_station_candidates, 'wb') as f:
            pickle.dump(station_candidates, f)

        # origin_name_desired_dep_time
        with open(filename_origin_name_desired_dep_time, 'wb') as f:
            pickle.dump(origin_name_desired_dep_time, f)

        # origin_name_zone_dict
        with open(filename_origin_name_zone_dict, 'wb') as f:
            pickle.dump(origin_name_zone_dict, f)

        print('Initial timetable graph with odt lists are saved in pickle files.')

    return timetable_initial_graph, odt_list, odt_by_origin, odt_by_dest, station_candidates, \
           origin_name_desired_dep_time, origin_name_zone_dict


def connect_origins_with_stations_for_all_odt(timetable_waiting_transfer, all_origin_zones, k2, parameters, sbb_nodes,
                                              od_departure_time, travel_time_selected_zones, closest_stations_to_zone):
    """
    Function the connects the origin zones with stations for all od pairs
    :param timetable_waiting_transfer: Digraph of the timetable with waiting and transfers edges
    :param all_origin_zones: Zones where trips start
    :param k2: Threshold of number of zones to connect
    :param parameters: class object with all the parameters in the main code
    :param sbb_nodes: a recarray of sbb nodes attributes.
    :param od_departure_time: od array with the desired departure time for each passenger
    :param travel_time_selected_zones: recarray with from Zone to Zone travel time only for selected zones
    :param closest_stations_to_zone: list of the closest stations with zone, station distance, station id, station code
    :return: timetable graph with origins to stations
    """
    # for sp algorithm, odt_dictionary with information about passengers on a trip
    origin_node_name_zone_dict = dict()
    origin_nodes = []
    origin_nodes_attributes = {}
    edges_o_stations = list()
    odt_list = list()
    odt_by_origin = {}
    odt_by_dest = {}

    station_id_zone_dict = {}
    for station in sbb_nodes:
        station_id_zone_dict[station.Code] = station.zone

    # Create dictionary with key : station & value : all departing nodes
    departing_trains_by_node = dict()

    for x, y in timetable_waiting_transfer.nodes(data=True):
        if y['type'] in ['arrivalNode', 'arrivalNodePassing', 'departureNodePassing']:
            continue
        try:
            zone = station_id_zone_dict[x[0]]
        except KeyError:
            print('Node', x[0], ' is not in Area of Interest. Check that all train stops are in Area of Interest')
        if not departing_trains_by_node.__contains__(x[0]):
            departing_trains_by_node[x[0]] = list()
        dep_nodes_in_dict = departing_trains_by_node[x[0]]
        dep_nodes_in_dict.append(x)

    zone_station_dep_nodes = dict()
    station_candidates = dict()
    origin_name_desired_dep_time = {}

    # Loop trough all home zones where trips are starting
    for origin_zone in all_origin_zones:
        # tt fom origin zones to all stations
        tt_from_origin_zone = travel_time_selected_zones[travel_time_selected_zones.fromZone == origin_zone]
        tt_from_origin_zone = tt_from_origin_zone[np.isin(tt_from_origin_zone.toZone, sbb_nodes.zone)]

        if not zone_station_dep_nodes.__contains__(origin_zone):
            # Identify stations candidates of origin zone
            stations_candidates_origin = helpers.identify_stations_candidates(closest_stations_to_zone, k2, sbb_nodes,
                                                                              origin_zone, tt_from_origin_zone,
                                                                              station_candidates)
            zone_station_dep_nodes[origin_zone] = dict()
            if origin_zone in stations_candidates_origin.keys():
                for station_code, v in stations_candidates_origin[origin_zone].items():
                    if station_code in departing_trains_by_node.keys():
                        zone_station_dep_nodes[origin_zone][station_code] = departing_trains_by_node[station_code]

        # Get all odt starting in the origin zone
        odt_of_origin_zone = od_departure_time[np.isin(od_departure_time.fromZone, origin_zone)]

        # Create the dictionary for the group size
        odt_dict_od_group_size = {}

        # Check every od pair with the desired departure time and create node and edges to station for graph
        for odt in odt_of_origin_zone:
            origin_node_name = str(odt.fromZone) + '_' + odt.desired_dep_time[-5:]
            odt_list.append([origin_node_name, odt.toZone, odt.priority, odt.group_size])
            origin_name_desired_dep_time[(origin_node_name, odt.toZone)] = odt.desired_dep_time
            origin_node_name_zone_dict[origin_node_name] = origin_zone

            # Add the node to the list of nodes if it is not in the origin nodes list
            if origin_node_name not in origin_nodes:
                origin_nodes.append(origin_node_name)
                attributes = {'zone': origin_zone, 'departure_time': odt.desired_dep_time, 'type': 'origin'}
                origin_nodes_attributes[origin_node_name] = attributes

            # Check all the possible edges from origin zone to the closest departure station node
            if origin_zone in zone_station_dep_nodes.keys():
                for station_code, value in zone_station_dep_nodes[origin_zone].items():
                    try:
                        tt_to_station = stations_candidates_origin[origin_zone][station_code]['tt_toStation']
                    except KeyError:
                        tt_to_station = 50
                        print('Travel time is not in data, hence manually set to 50 min ', origin_zone, ' ',
                              stations_candidates_origin[origin_zone][station_code]['station_zone'])

                    # Identify train departure nodes candidates at stations
                    time_at_station = datetime.datetime.strptime(odt.desired_dep_time, "%Y-%m-%dT%H:%M") + \
                                      datetime.timedelta(minutes=int(tt_to_station))

                    departure_nodes_of_station = helpers.identify_dep_nodes_for_trip_at_station(
                        timetable_waiting_transfer,
                        parameters, time_at_station,
                        value)

                    # Add the edges of origin to departures at the station with weight of travel time
                    for d in departure_nodes_of_station:
                        edge = [origin_node_name, d, tt_to_station]
                        if edge not in edges_o_stations:
                            edges_o_stations.append(edge)

                # Transform odt matrix into a list with adapted node names
                odt_dict_od_group_size[origin_node_name, odt.toZone] = odt.group_size
                if not odt_by_dest.__contains__(odt.toZone):
                    odt_by_dest[odt.toZone] = {}
                odt_by_dest[odt.toZone].update({(origin_node_name, odt.toZone): odt.group_size})

            odt_by_origin[origin_zone] = odt_dict_od_group_size

    # Add origin nodes and attributes
    timetable_waiting_transfer.add_nodes_from(origin_nodes)
    nx.set_node_attributes(timetable_waiting_transfer, origin_nodes_attributes)
    # Add the edges of an origin node to all departing nodes of the selected stations
    # and the same for stations to destination
    timetable_waiting_transfer.add_weighted_edges_from(edges_o_stations)

    return timetable_waiting_transfer, odt_list, odt_by_origin, odt_by_dest, stations_candidates_origin, \
           origin_name_desired_dep_time, origin_node_name_zone_dict


def connect_all_destinations_with_stations(timetable_with_origin, all_destination_zones, closest_stations_to_zone,
                                           destination_nodes, destination_nodes_attributes, edges_d_stations, k2,
                                           sbb_nodes, travel_time_selected_zones, station_candidates):
    """
    Function that connects all the destination zone to the station in the timetable graph
    :param timetable_with_origin: Timetable graph with origins to stations
    :param all_destination_zones: All destination zones from the od matrix
    :param closest_stations_to_zone: List of the closest stations with zone, station distance, station id, station code
    :param destination_nodes: Empty array to be filled in this function
    :param destination_nodes_attributes: Empty list to be filled in this function
    :param edges_d_stations: Empty list to be filled in this function
    :param k2: Threshold of number of zones to connect
    :param sbb_nodes: A recarray of sbb nodes attributes.
    :param travel_time_selected_zones: Recarray with from Zone to Zone travel time only for selected zones
    :param station_candidates: List with the travel time to zone
    :return: timetable_initial_graph: timetable graph with all the edges needed to start the ALNS
    """
    # Dictionary with key : station & value : all arriving nodes
    arriving_trains_by_node = dict()

    for x, y in timetable_with_origin.nodes(data=True):
        if y['type'] in ['departureNode', 'arrivalNodePassing', 'departureNodePassing', 'origin']:
            continue
        if not arriving_trains_by_node.__contains__(x[0]):
            arriving_trains_by_node[x[0]] = list()
        arr_nodes_in_dict = arriving_trains_by_node[x[0]]
        arr_nodes_in_dict.append(x)

    # Connect all destinations with train arrival nodes
    for destination_zone in all_destination_zones:
        # Add the destination to the list of nodes for the graph
        destination_nodes.append(destination_zone)
        attributes = {'zone': destination_zone, 'type': 'destination'}
        destination_nodes_attributes[destination_zone] = attributes

        # Travel time fom destination zones to all stations
        tt_from_destination_zone = travel_time_selected_zones[travel_time_selected_zones.fromZone == destination_zone]
        tt_from_destination_zone = tt_from_destination_zone[np.isin(tt_from_destination_zone.toZone, sbb_nodes.zone)]

        stations_candidates = helpers.identify_stations_candidates(closest_stations_to_zone, k2, sbb_nodes,
                                                                   destination_zone, tt_from_destination_zone,
                                                                   station_candidates)
        # Identify stations candidates of destination zone
        # Loop through station candidates and link station with departure
        for station_code, value in stations_candidates[destination_zone].items():
            # Identify train arrival nodes and add the edges
            try:
                tt_to_station = stations_candidates[destination_zone][station_code]['tt_toStation']
            except KeyError:
                tt_to_station = 50
                print('Travel time is not in data, hence manually set to 50 min. ', destination_zone, ' ',
                      stations_candidates[destination_zone][station_code]['station_zone'])
            try:
                for a in arriving_trains_by_node[station_code]:
                    edge = [a, destination_zone, tt_to_station]
                    if edge not in edges_d_stations:
                        edges_d_stations.append(edge)
            except KeyError:
                pass
    timetable_with_origin.add_nodes_from(destination_nodes)
    nx.set_node_attributes(timetable_with_origin, destination_nodes_attributes)
    # Add the edges of an origin node to all arriving nodes of the selected station
    timetable_with_origin.add_weighted_edges_from(edges_d_stations)
    return timetable_with_origin, station_candidates


def create_transit_edges_nodes_single_train(train, infra_graph, idx_start_delay):
    # Set the parameters
    total_train_length = len(train.train_path_nodes)
    train_id = train.id

    arrival_nodes = []
    arrival_nodes_attributes = dict()
    departure_nodes = []
    departure_nodes_attributes = dict()
    driving_edges = list()
    driving_edges_attributes = dict()
    waiting_edges = list()
    waiting_edges_attributes = dict()

    if idx_start_delay == 0:
        start_of_train = True
    else:
        start_of_train = False
        departure_time_last_node = train.train_path_nodes[idx_start_delay - 1].departure_time
        departure_node_last_node = train.train_path_nodes[idx_start_delay - 1].node_id
        departure_node_last_node_name = (departure_node_last_node,
                                         train.train_path_nodes[idx_start_delay - 1].departure_time,
                                         train.train_path_nodes[idx_start_delay - 1].id, 'd')
        attributes = {'train': train_id, 'type': 'departureNode', 'departureTime': departure_time_last_node,
                      'StopStatus': train.train_path_nodes[idx_start_delay - 1].stop_status.name}
        departure_nodes_attributes[departure_node_last_node_name] = attributes

    # Loop trough all path nodes of a train
    # Initialize the number of stop of the train
    s = 0
    for train_path_node in train.train_path_nodes[idx_start_delay:]:
        s = s + 1
        # Skip node if it is not in the area
        if not infra_graph.nodes[train_path_node.node_id]['in_area']:
            continue

        # Consider when only train stops
        if train_path_node.stop_status.name == 'commercial_stop':
            # Update time and node
            arrival_time_this_node = train_path_node.arrival_time
            arrival_node_this_node = train_path_node.node_id
            departure_time_this_node = train_path_node.departure_time
            departure_node_this_node = train_path_node.node_id

            if start_of_train:
                node_name_dep_this = (departure_node_this_node, train_path_node.departure_time, train_path_node.id, 'd')
                departure_nodes.append(node_name_dep_this)
                attributes = {'train': train_id, 'type': 'departureNode', 'departureTime': departure_time_this_node,
                              'StopStatus': train_path_node.stop_status.name}
                departure_nodes_attributes[node_name_dep_this] = attributes
                departure_time_last_node = departure_time_this_node
                departure_node_last_node = departure_node_this_node
                departure_node_last_node_name = node_name_dep_this
                start_of_train = False
                continue

            # If it is the end of the train
            elif s == total_train_length:
                node_name_arr_this = (arrival_node_this_node, train_path_node.arrival_time, train_path_node.id, 'a')
                arrival_nodes.append(node_name_arr_this)
                attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node,
                              'StopStatus': train_path_node.stop_status.name}
                arrival_nodes_attributes[node_name_arr_this] = attributes

                # Driving edge to this node
                run_time = arrival_time_this_node - departure_time_last_node
                driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = {'flow': 0,
                                                                                               'type': 'driving',
                                                                                               'train_id': train_id}

                # Reset the parameters for next iteration
                del departure_node_last_node
                del node_name_dep_this
                del node_name_arr_this
                del departure_time_last_node

            # Node in between twp transit nodes
            else:

                # Arrival nodes
                node_name_arr_this = (arrival_node_this_node, train_path_node.arrival_time, train_path_node.id, 'a')
                arrival_nodes.append(node_name_arr_this)
                attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node,
                              'StopStatus': train_path_node.stop_status.name}
                arrival_nodes_attributes[node_name_arr_this] = attributes

                # Driving edge between last node and this node
                run_time = arrival_time_this_node - departure_time_last_node
                driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = {'flow': 0,
                                                                                               'type': 'driving',
                                                                                               'train_id': train_id}

                # Departure nodes
                node_name_dep_this = (departure_node_this_node, train_path_node.departure_time, train_path_node.id, 'd')
                departure_nodes.append(node_name_dep_this)
                attributes = {'train': train_id, 'type': 'departureNode', 'departureTime': departure_time_this_node,
                              'StopStatus': train_path_node.stop_status.name}

                departure_nodes_attributes[node_name_dep_this] = attributes

                # Waiting edge between last node and this node
                wait_time = (departure_time_this_node - arrival_time_this_node)
                waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds / 60)])
                waiting_edges_attributes[node_name_arr_this, node_name_dep_this] = {'flow': 0, 'type': 'waiting',
                                                                                    'train_id': train_id}

                # Update the departure node for next iteration
                departure_node_last_node = departure_node_this_node
                departure_node_last_node_name = node_name_dep_this
                departure_time_last_node = departure_time_this_node

        # Passing nodes
        elif train_path_node.stop_status.name == 'passing':
            arrival_time_this_node = train_path_node.arrival_time
            arrival_node_this_node = train_path_node.node_id
            departure_time_this_node = train_path_node.departure_time
            departure_node_this_node = train_path_node.node_id

            if start_of_train:
                node_name_dep_this = (departure_node_this_node, train_path_node.departure_time, train_path_node.id,
                                      'dp')
                departure_nodes.append(node_name_dep_this)
                attributes = {'train': train_id, 'type': 'departureNodePassing',
                              'departureTime': departure_time_this_node, 'StopStatus': train_path_node.stop_status.name}
                departure_nodes_attributes[node_name_dep_this] = attributes
                departure_time_last_node = departure_time_this_node
                departure_node_last_node = departure_node_this_node
                departure_node_last_node_name = node_name_dep_this
                start_of_train = False
                continue

            # End of the train
            elif s == total_train_length:
                node_name_arr_this = (arrival_node_this_node, train_path_node.arrival_time, train_path_node.id, 'ap')
                arrival_nodes.append(node_name_arr_this)
                attributes = {'train': train_id, 'type': 'arrivalNodePassing', 'arrivalTime': arrival_time_this_node,
                              'StopStatus': train_path_node.stop_status.name}
                arrival_nodes_attributes[node_name_arr_this] = attributes

                # Driving edge to this node
                run_time = arrival_time_this_node - departure_time_last_node
                driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = {'flow': 0,
                                                                                               'type': 'driving',
                                                                                               'train_id': train_id}

                # Reset the parameters for the next iteration
                del departure_node_last_node
                del node_name_dep_this
                del node_name_arr_this
                del departure_time_last_node

            # Node in between two transit nodes
            else:
                node_name_arr_this = (arrival_node_this_node, train_path_node.arrival_time, train_path_node.id, 'ap')
                arrival_nodes.append(node_name_arr_this)
                attributes = {'train': train_id, 'type': 'arrivalNodePassing', 'arrivalTime': arrival_time_this_node,
                              'StopStatus': train_path_node.stop_status.name}

                arrival_nodes_attributes[node_name_arr_this] = attributes

                # Driving edge between last node and this node
                run_time = arrival_time_this_node - departure_time_last_node
                driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = {'flow': 0,
                                                                                               'type': 'driving',
                                                                                               'train_id': train_id}

                node_name_dep_this = (departure_node_this_node, train_path_node.departure_time, train_path_node.id,
                                      'dp')
                departure_nodes.append(node_name_dep_this)
                attributes = {'train': train_id, 'type': 'departureNodePassing',
                              'departureTime': departure_time_this_node, 'StopStatus': train_path_node.stop_status.name}
                departure_nodes_attributes[node_name_dep_this] = attributes

                # Waiting edge between last node and this node
                wait_time = (departure_time_this_node - arrival_time_this_node)
                waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds / 60)])
                waiting_edges_attributes[node_name_arr_this, node_name_dep_this] = {'flow': 0, 'type': 'waiting',
                                                                                    'train_id': train_id}

                # Update the departure node for next iteration
                departure_node_last_node = departure_node_this_node
                departure_node_last_node_name = node_name_dep_this
                departure_time_last_node = departure_time_this_node

    nodes_edges_dict = {'arrival_nodes': arrival_nodes,
                        'departure_nodes': departure_nodes,
                        'arrival_nodes_attr': arrival_nodes_attributes,
                        'departure_nodes_attr': departure_nodes_attributes,
                        'driving_edges': driving_edges,
                        'waiting_edges': waiting_edges,
                        'driving_attr': driving_edges_attributes,
                        'waiting_attr': waiting_edges_attributes}

    return nodes_edges_dict


def add_transit_nodes_edges_single_train_to_graph(prime_timetable, nodes_edges_dict, bus=False):
    prime_timetable.add_nodes_from(nodes_edges_dict['arrival_nodes_attr'])
    nx.set_node_attributes(prime_timetable, nodes_edges_dict['arrival_nodes_attr'])
    prime_timetable.add_nodes_from(nodes_edges_dict['departure_nodes_attr'])
    nx.set_node_attributes(prime_timetable, nodes_edges_dict['departure_nodes_attr'])
    prime_timetable.add_weighted_edges_from(nodes_edges_dict['driving_edges'])
    nx.set_edge_attributes(prime_timetable, nodes_edges_dict['driving_attr'])
    if not bus:
        prime_timetable.add_weighted_edges_from(nodes_edges_dict['waiting_edges'])
        nx.set_edge_attributes(prime_timetable, nodes_edges_dict['waiting_attr'])


def transfer_edges_single_train(prime_timetable, train, parameters, train_path_nodes_delay):
    # Zurich main station for transfers todo: need to change to not specific and more generic (Get the node id)
    zh_hb_code = ['85ZMUS', '85ZUE', '45ZSZU']
    zh_hb_code = [606, 633, 13]

    # Get the commercial stops of the train
    comm_stops_of_train = {node.id: node.node_id for node in train.train_path_nodes
                           if node.stop_status.name == 'commercial_stop' and node.id in train_path_nodes_delay}

    # Get the zurich node codes if it is in the commercial stops of the train
    zh_station_helper = []
    if any(x in zh_hb_code for x in comm_stops_of_train.values()):
        zh_station_helper = zh_hb_code

    # Initialize candidates
    departure_train_candidates_by_node = dict()
    arrival_train_candidates_by_node = dict()
    departure_nodes_delayed_train = dict()
    arrival_nodes_delayed_train = dict()
    for x, y in prime_timetable.nodes(data=True):
        try:
            if y['train'] == train.id:
                if y['type'] == 'arrivalNode' and x[2] in train_path_nodes_delay:
                    arrival_nodes_delayed_train[x[2]] = (x, y)
                elif y['type'] == 'departureNode' and x[2] in train_path_nodes_delay:
                    departure_nodes_delayed_train[x[2]] = (x, y)
                else:
                    continue
            elif x[0] not in comm_stops_of_train.values() and x[0] not in zh_station_helper:
                continue
            elif y['type'] in ['arrivalNodePassing', 'departureNodePassing']:
                continue
            elif y['type'] == 'arrivalNode':
                station = x[0]
                if not arrival_train_candidates_by_node.__contains__(station):
                    arrival_train_candidates_by_node[station] = list()
                arr_station_in_dict = arrival_train_candidates_by_node[station]
                arr_station_in_dict.append([x, y])
            elif y['type'] == 'departureNode':
                station = x[0]
                if not departure_train_candidates_by_node.__contains__(station):
                    departure_train_candidates_by_node[station] = list()
                dep_station_in_dict = departure_train_candidates_by_node[station]
                dep_station_in_dict.append([x, y])
            else:
                print('Check transfer edges single train, somehow a not train node appears in the timetable')
        except KeyError:
            print(x, y)

    # Set transfer edges and attributes list
    transfer_edges = list()
    transfer_edges_attributes = dict()

    # Identify all arrival nodes
    for train_path_node, station in comm_stops_of_train.items():
        # Select dep and arr candidates of trains if there are any
        arrivals_at_station = []
        departures_at_station = []

        if station in arrival_train_candidates_by_node.keys():
            arrivals_at_station = arrival_train_candidates_by_node[station]
            if station in zh_station_helper:
                for zh in zh_station_helper:
                    if station != zh and zh in arrival_train_candidates_by_node.keys():
                        arrivals_at_station.extend(arrival_train_candidates_by_node[zh])

        # Departures
        if station in departure_train_candidates_by_node.keys():
            departures_at_station = departure_train_candidates_by_node[station]
            if station in zh_station_helper:
                for zh in zh_station_helper:
                    if station != zh and zh in departure_train_candidates_by_node.keys():
                        departures_at_station.extend(departure_train_candidates_by_node[zh])

        # Skip the starting node of a train for arrival of train to departure transfer
        if not train_path_node == train.train_path_nodes[0].id:
            # Transfer edges from arrival of the delayed train to all departures of other trains
            for label, attr in departures_at_station:
                delta_t = attr['departureTime'] - arrival_nodes_delayed_train[train_path_node][1]['arrivalTime']
                if parameters.transfer_m < delta_t < parameters.transfer_M:
                    weight = delta_t * parameters.beta_waiting + datetime.timedelta(minutes=parameters.beta_transfer)
                    transfer_edges.append([arrival_nodes_delayed_train[train_path_node][0], label,
                                           float(weight.seconds / 60)])
                    transfer_edges_attributes[arrival_nodes_delayed_train[train_path_node][0],
                                              label] = {'type': 'transfer'}

        # Transfer edges from arrival of any train to departure of delayed train
        if not train_path_node == train.train_path_nodes[-1].id:
            for label, attr in arrivals_at_station:
                delta_t = departure_nodes_delayed_train[train_path_node][1]['departureTime'] - attr['arrivalTime']
                if parameters.transfer_m < delta_t < parameters.transfer_M:
                    weight = delta_t * parameters.beta_waiting + datetime.timedelta(minutes=parameters.beta_transfer)
                    transfer_edges.append([label, departure_nodes_delayed_train[train_path_node][0],
                                           float(weight.seconds / 60)])
                    transfer_edges_attributes[label,
                                              departure_nodes_delayed_train[train_path_node][0]] = {'type': 'transfer'}

    return transfer_edges, transfer_edges_attributes, [arrival_nodes_delayed_train, departure_nodes_delayed_train]


def add_edges_of_train_from_o_stations_d(edges_o_stations_d, train, prime_timetable, parameters, tpn_idx_start_delay,
                                         train_path_nodes_delay):
    arr_dep_nodes_train = [(n, v) for n, v in prime_timetable.nodes(data=True)
                           if v['type'] in ['arrivalNode', 'departureNode']
                           and v['train'] == train.id
                           and n[2] in train_path_nodes_delay]

    # Get the information from the parameters
    odt_by_origin = parameters.odt_by_origin
    zone_candidates = parameters.zone_candidates

    # Get the first arrival and the last departure of the train
    first_arrival_of_train = \
        neighbourhood_operators.identify_first_arrival_train_path_node_id_of_train(train, tpn_idx_start_delay)
    last_departure_of_train = \
        neighbourhood_operators.identify_last_departure_train_path_node_id_of_train(train, tpn_idx_start_delay)

    # Initialize parameters
    fmt = "%Y-%m-%dT%H:%M"
    min_wait = parameters.min_wait
    max_wait = parameters.max_wait

    for transit_node, attr in arr_dep_nodes_train:
        station = transit_node[0]
        if attr['type'] == 'arrivalNode':
            if not transit_node[2] == first_arrival_of_train and tpn_idx_start_delay == 0:
                zones_to_connect = zone_candidates[station]
                for zone, tt_to_zone in zones_to_connect.items():
                    edge = [transit_node, zone, tt_to_zone]
                    if zone in edges_o_stations_d.edges_stations_d_dict.keys():
                        edges_o_stations_d.edges_stations_d_dict[zone].append(edge)
                    else:
                        edges_o_stations_d.edges_stations_d_dict[zone] = [edge]
                    edges_o_stations_d.edges_stations_d.append(edge)

        elif attr['type'] == 'departureNode':
            if not transit_node[2] == last_departure_of_train:
                zones_to_connect = zone_candidates[station]
                for zone, tt_to_zone in zones_to_connect.items():
                    if zone not in odt_by_origin.keys():
                        continue
                    for trip in odt_by_origin[zone]:
                        desired_dep_time = parameters.origin_name_desired_dep_time[(trip[0], trip[1])]
                        desired_dep_time = datetime.datetime.strptime(desired_dep_time, fmt)
                        time_delta = attr['departureTime'] - desired_dep_time
                        if min_wait < time_delta < max_wait:
                            edge = [trip[0], transit_node, tt_to_zone]
                            if trip[0] in edges_o_stations_d.edges_o_stations_dict.keys():
                                edges_o_stations_d.edges_o_stations_dict[trip[0]].append(edge)
                            else:
                                edges_o_stations_d.edges_o_stations_dict[trip[0]] = [edge]
                            edges_o_stations_d.edges_o_stations.append(edge)

    return edges_o_stations_d


def add_edges_of_bus_from_o_stations_d(edges_o_stations_d, train, prime_timetable, parameters, tpn_idx_start_delay,
                                       tpn_delay):
    # Get the arrival and departure node of the bus
    arr_dep_nodes_bus = [(n, v) for n, v in prime_timetable.nodes(data=True) if v['train'] == train.id
                         and v['type'] in ['arrivalNode', 'departureNode'] and n[2] in tpn_delay]

    # Get the OD time from the parameters and the zone candidates as well
    odt_by_origin = parameters.odt_by_origin
    zone_candidates = parameters.zone_candidates

    fmt = "%Y-%m-%dT%H:%M"

    # Loop through all nodes of the bus path
    for transit_node, attr in arr_dep_nodes_bus:
        station = transit_node[0]

        # For arrival nodes
        if attr['type'] == 'arrivalNode':
            zones_to_connect = zone_candidates[station]
            for zone, tt_to_zone in zones_to_connect.items():
                edge = [transit_node, zone, tt_to_zone]
                if zone in edges_o_stations_d.edges_stations_d_dict.keys():
                    edges_o_stations_d.edges_stations_d_dict[zone].append(edge)
                else:
                    edges_o_stations_d.edges_stations_d_dict[zone] = [edge]
                edges_o_stations_d.edges_stations_d.append(edge)

        # For departure nodes
        elif attr['type'] == 'departureNode':
            zones_to_connect = zone_candidates[station]
            for zone, tt_to_zone in zones_to_connect.items():
                if zone not in odt_by_origin.keys():
                    continue
                for trip in odt_by_origin[zone]:
                    desired_dep_time = parameters.origin_name_desired_dep_time[(trip[0], trip[1])]
                    desired_dep_time = datetime.datetime.strptime(desired_dep_time, fmt)
                    time_delta = attr['departureTime'] - desired_dep_time
                    if parameters.min_wait < time_delta < parameters.max_wait:
                        edge = [trip[0], transit_node, tt_to_zone]
                        if trip[0] in edges_o_stations_d.edges_o_stations_dict.keys():
                            edges_o_stations_d.edges_o_stations_dict[trip[0]].append(edge)
                        else:
                            edges_o_stations_d.edges_o_stations_dict[trip[0]] = [edge]
                        edges_o_stations_d.edges_o_stations.append(edge)

    return edges_o_stations_d


def create_transit_edges_nodes_emergency_bus(bus):
    # Set the parameters
    arrival_nodes = []
    arrival_nodes_attributes = dict()
    departure_nodes = []
    departure_nodes_attributes = dict()
    driving_edges = list()
    driving_edges_attributes = dict()

    # Loop through all path nodes of a bus
    n = 0
    for train_path_node in bus.train_path_nodes:
        n += 1

        # Update time and node
        arrival_time_this_node = train_path_node.arrival_time
        arrival_node_this_node = train_path_node.node_id
        departure_time_this_node = train_path_node.departure_time
        departure_node_this_node = train_path_node.node_id

        # First node
        if n == 1:
            node_name_dep_this = (departure_node_this_node, train_path_node.departure_time, train_path_node.id, 'd')
            departure_nodes.append(node_name_dep_this)
            attributes = {'train': bus.id, 'type': 'departureNode', 'departureTime': departure_time_this_node,
                          'StopStatus': train_path_node.stop_status.name, 'bus': 'EmergencyBus'}
            departure_nodes_attributes[node_name_dep_this] = attributes
            departure_time_last_node = departure_time_this_node
            departure_node_last_node = departure_node_this_node
            departure_node_last_node_name = node_name_dep_this

        # End of the bus
        elif n == 2:
            node_name_arr_this = (arrival_node_this_node, train_path_node.arrival_time, train_path_node.id, 'a')
            arrival_nodes.append(node_name_arr_this)
            attributes = {'train': bus.id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node,
                          'StopStatus': train_path_node.stop_status.name, 'bus': 'EmergencyBus'}

            arrival_nodes_attributes[node_name_arr_this] = attributes

            # Driving edge to this node
            run_time = arrival_time_this_node - departure_time_last_node
            driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
            driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = {'flow': 0,
                                                                                           'type': 'driving',
                                                                                           'bus_id': bus.id,
                                                                                           'bus': True}

    # Create the nodes and edges dictionary
    nodes_edges_dict = {'arrival_nodes': arrival_nodes,
                        'departure_nodes': departure_nodes,
                        'arrival_nodes_attr': arrival_nodes_attributes,
                        'departure_nodes_attr': departure_nodes_attributes,
                        'driving_edges': driving_edges,
                        'driving_attr': driving_edges_attributes}

    return nodes_edges_dict


def create_graph_with_edges_o_stations_d(edges_o_stations_d, od=None, timetable_graph=None):
    if timetable_graph is None:
        timetable_graph = nx.DiGraph()

    edges_o_stations = edges_o_stations_d.edges_o_stations
    edges_stations_d = edges_o_stations_d.edges_stations_d

    # Edges_o_stations_dict: key origin, value edges connecting to train nodes
    edges_o_stations_dict = edges_o_stations_d.edges_o_stations_dict
    edges_stations_d_dict = edges_o_stations_d.edges_stations_d_dict

    if od is None:
        timetable_graph.add_weighted_edges_from(edges_o_stations)
        timetable_graph.add_weighted_edges_from(edges_stations_d)

    else:
        origin = od[0]
        dest = od[1]
        timetable_graph.add_weighted_edges_from(edges_o_stations_dict[origin])
        timetable_graph.add_weighted_edges_from(edges_stations_d_dict[dest])

    return timetable_graph


def copy_graph_with_flow(timetable_graph):
    # Create a new graph from networkx
    timetable_graph_prime = nx.DiGraph()
    edges_graph = []
    attr_edges_graph = {}

    # For each edge on the graph, record the attributes
    nodes = {n: v for n, v in timetable_graph.nodes(data=True)}
    for u, v, attr in timetable_graph.edges(data=True):
        edges_graph.append((u, v, {'weight': attr['weight']}))
        attr_edges_graph[(u, v)] = attr

    # Delete the graph
    del timetable_graph

    # Add the nodes, the edges and attributes on the new timetable graph
    timetable_graph_prime.add_nodes_from(nodes.keys())
    nx.set_node_attributes(timetable_graph_prime, nodes)
    timetable_graph_prime.add_weighted_edges_from(edges_graph)
    nx.set_edge_attributes(timetable_graph_prime, attr_edges_graph)

    return timetable_graph_prime


def copy_graph_and_remove_flow(timetable_graph):

    # Set the parameters
    timetable_graph_prime = nx.DiGraph()
    edges_graph = []
    attr_edges_graph = {}

    # Get the attributes, the nodes and the edges
    nodes = {n: v for n, v in timetable_graph.node(data=True)}
    for u, v, attr in timetable_graph.edges(data=True):
        edges_graph.append((u, v, {'weight': attr['weight']}))
        # Copy the attributes
        attr_copied = attr.copy()
        # Set the flow to zero
        if 'flow' in attr_copied.keys():
            attr_copied['flow'] = 0
        attr_edges_graph[(u, v)] = attr_copied

    # Delete the timetable graph
    del timetable_graph

    # Add the nodes, edges and attributes to the new graph
    timetable_graph_prime.add_nodes_from(nodes.keys())
    nx.set_node_attributes(timetable_graph_prime, nodes)
    timetable_graph_prime.add_weighted_edges_from(edges_graph)
    nx.set_edge_attributes(timetable_graph_prime, attr_edges_graph)

    # For debugging
    debug = False
    if debug:
        for u, v, attr in edges_graph:
            if 'weight' not in attr.keys():
                print(u, v, attr)

    return timetable_graph_prime


def remove_flow_on_graph(timetable_graph):
    # Search on all edges of the graph and remove the flow
    for (u, v, attr) in timetable_graph.edges.data():
        if 'flow' in attr.keys():
            if attr['flow'] != 0:
                attr['flow'] = 0
                if 'initial_weight' in attr.keys():
                    attr['weight'] = attr['initial_weight']
    return timetable_graph


def create_restored_feasibility_graph(trains_timetable, parameters):
    # Get the timetable graph and the stations with commercial stops
    timetable_graph, parameters.stations_comm_stop = create_timetable_with_waiting_transfer_edges(trains_timetable,
                                                                                                  parameters)

    # Add the transfers edges to the graph
    timetable_graph = add_transfer_edges_to_graph(timetable_graph, parameters)

    # Connect the homes with the stations candidates
    edges_o_stations_d = connections_homes_with_station_candidates(timetable_graph, parameters)

    return timetable_graph, edges_o_stations_d


def create_timetable_with_waiting_transfer_edges(trains_timetable, parameters):
    """
    function that takes as parameters the train timetable and create the edges for transfer and waiting
    :param trains_timetable: all trains travelling on nodes (type: a viriato object)
    :param parameters: class of all parameters for this study. (type: class)
    :return: timetable_graph and stations with commercial stops
    """
    stations_with_commercial_stop = []
    arrival_nodes = []
    arrival_nodes_attributes = {}
    arrival_nodes_passing_attributes = {}
    departure_nodes = []
    departure_nodes_attributes = {}
    departure_nodes_passing_attributes = {}
    driving_edges = list()
    driving_edges_attributes = {}
    waiting_edges = list()
    waiting_edges_attributes = {}

    n = 0
    debug = False

    # Loop through all trains
    for train in trains_timetable:
        n = n + 1
        s = 0  # Number of stop of a train
        total_train_length = len(train.train_path_nodes)
        train_id = train.id
        start_of_train = True

        if n > 100 and debug:
            break
        # Loop through all path nodes of a train
        for train_path_nodes in train.train_path_nodes:
            s = s + 1

            # Check if the train path goes within the stations in the area
            if train_path_nodes.node_id not in parameters.stations_in_area:
                # If not in the area, do not consider this node and go for the next one.
                continue

            # Consider only the trains where have commercial stops
            if train_path_nodes.stop_status.name == "commercial_stop":
                stations_with_commercial_stop.append(train_path_nodes.node_id)
                # Update time and node
                arrival_time_this_node = train_path_nodes.arrival_time
                arrival_node_this_node = train_path_nodes.node_id
                departure_time_this_node = train_path_nodes.departure_time
                departure_node_this_node = train_path_nodes.node_id

                # First node in the train path is the starting node, does not have an edge
                if start_of_train:
                    # Combine information of the departure node of the train path in one list
                    node_name_dep_this = (
                        departure_node_this_node, train_path_nodes.departure_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'd')
                    # Add the departure node into the list of all departure nodes
                    departure_nodes.append(node_name_dep_this)
                    # Record the attributes, train id, departure time, stop status
                    attributes = {'train': train_id, 'type': 'departureNode',
                                  'departureTime': departure_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    departure_nodes_attributes[node_name_dep_this] = attributes
                    departure_time_last_node = departure_time_this_node
                    departure_node_last_node_name = node_name_dep_this
                    start_of_train = False
                    continue

                # If the stop is equal to the total train path length, it means it is at the end of the train path
                elif s == total_train_length:
                    # Combine information of the arrival node in one list
                    node_name_arr_this = (
                        arrival_node_this_node, train_path_nodes.arrival_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'a')
                    # Add the arrival node into the list of all arrival nodes
                    arrival_nodes.append(node_name_arr_this)
                    # Record the attributes, train id, arrival time, stop status
                    attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    arrival_nodes_attributes[node_name_arr_this] = attributes
                    # Driving edge to this node, running time = arrival time - departure time
                    run_time = arrival_time_this_node - departure_time_last_node
                    # Add this edge in the driving edges for the timetable graph
                    driving_edges.append(
                        [departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                    # Add the attributes of this edge
                    driving_edges_attributes[(departure_node_last_node_name,
                                              node_name_arr_this)] = {'flow': 0,
                                                                      'type': 'driving',
                                                                      'train_id': train_id}
                    # Reset the departure and arrival nodes
                    del node_name_dep_this
                    del node_name_arr_this
                    del departure_time_last_node

                # Nodes in between two transit nodes (on the first iteration, departure time of the last node is
                # recorded from the starting node at line 188, then it will be recorded here below for the next
                # iterations)
                else:
                    # Arrival nodes
                    # Combine information of the arrival node in one list
                    node_name_arr_this = (
                        arrival_node_this_node, train_path_nodes.arrival_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'a')
                    # Add the arrival node into the list of all arrival nodes
                    arrival_nodes.append(node_name_arr_this)
                    # Record the attributes, train id, arrival time, stop status
                    attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    arrival_nodes_attributes[node_name_arr_this] = attributes
                    # Driving edge between last node and this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    # Add this edge in the driving edges for the timetable graph
                    driving_edges.append(
                        [departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                    # Add the attributes of this edge
                    driving_edges_attributes[(departure_node_last_node_name,
                                              node_name_arr_this)] = {'flow': 0,
                                                                      'type': 'driving',
                                                                      'train_id': train_id}

                    # Departure nodes
                    # Combine information of the departure node in one list
                    node_name_dep_this = (
                        departure_node_this_node, train_path_nodes.departure_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'd')
                    # Add the departure node into the list of all departure nodes
                    departure_nodes.append(node_name_dep_this)
                    # Record the attributes, train id, departure time, stop status
                    attributes = {'train': train_id, 'type': 'departureNode',
                                  'departureTime': departure_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    departure_nodes_attributes[node_name_dep_this] = attributes
                    # Waiting edge between the departure and the arrival time of this node
                    wait_time = (departure_time_this_node - arrival_time_this_node)
                    # Add the waiting time on the waiting edge lists
                    waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds / 60)])
                    waiting_edges_attributes[(node_name_arr_this, node_name_dep_this)] = {'flow': 0,
                                                                                          'type': 'waiting',
                                                                                          'train_id': train_id}

                    # Update the departure node for next iteration
                    departure_node_last_node_name = node_name_dep_this
                    departure_time_last_node = departure_time_this_node

            # If the node is not a commercial stop, it is a passing node
            elif train_path_nodes.stop_status.name == 'passing':
                arrival_time_this_node = train_path_nodes.arrival_time
                arrival_node_this_node = train_path_nodes.node_id
                departure_time_this_node = train_path_nodes.departure_time
                departure_node_this_node = train_path_nodes.node_id

                # If it is the first node of the train path
                if start_of_train:
                    # Combine information of the departure node of the train path in one list
                    node_name_dep_this = (
                        departure_node_this_node, train_path_nodes.departure_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'dp')
                    departure_nodes.append(node_name_dep_this)
                    # Add the departure node into the list of all departure nodes
                    attributes = {'train': train_id, 'type': 'departureNodePassing',
                                  'departureTime': departure_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    departure_nodes_passing_attributes[node_name_dep_this] = attributes

                    # Update the departure node for the next iteration
                    departure_time_last_node = departure_time_this_node
                    departure_node_last_node_name = node_name_dep_this
                    start_of_train = False
                    continue

                # When s is equal to the length of the train path nodes, it means we have reached the end of the path
                elif s == total_train_length:
                    # Combine information of the arrival node in one list
                    node_name_arr_this = (
                        arrival_node_this_node, train_path_nodes.arrival_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'ap')
                    # Add the arrival node into the list of all arrival nodes
                    arrival_nodes.append(node_name_arr_this)
                    # Record the attributes, train id, arrival time, stop status
                    attributes = {'train': train_id, 'type': 'arrivalNodePassing',
                                  'arrivalTime': arrival_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    arrival_nodes_passing_attributes[node_name_arr_this] = attributes
                    # Driving edge between last node and this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    # Add the attributes of this edge
                    driving_edges.append(
                        [departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                    driving_edges_attributes[(departure_node_last_node_name,
                                              node_name_arr_this)] = {'flow': 0,
                                                                      'type': 'driving',
                                                                      'train_id': train_id}
                    # Reset the departure and arrival nodes
                    del node_name_dep_this
                    del node_name_arr_this
                    del departure_time_last_node

                # Node in between two transit nodes (on the first iteration, departure time of the last node is
                # recorded from the starting node at line 188, then it will be recorded here below for the next
                # iterations)
                else:
                    # Arrival nodes
                    # Combine information of the arrival node in one list
                    node_name_arr_this = (
                        arrival_node_this_node, train_path_nodes.arrival_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'ap')
                    # Add the arrival node into the list of all arrival nodes
                    arrival_nodes.append(node_name_arr_this)
                    # Record the attributes, train id, arrival time, stop status
                    attributes = {'train': train_id, 'type': 'arrivalNodePassing',
                                  'arrivalTime': arrival_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    arrival_nodes_passing_attributes[node_name_arr_this] = attributes
                    # Driving edge between last node and this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    # Add this edge in the driving edges for the timetable graph
                    driving_edges.append(
                        [departure_node_last_node_name, node_name_arr_this, float(run_time.seconds / 60)])
                    # Add the attributes of this edge
                    driving_edges_attributes[(departure_node_last_node_name,
                                              node_name_arr_this)] = {'flow': 0,
                                                                      'type': 'driving',
                                                                      'train_id': train_id}

                    # Departure nodes
                    # Combine information of the departure node in one list
                    node_name_dep_this = (
                        departure_node_this_node, train_path_nodes.departure_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        train_path_nodes.id, 'dp')
                    departure_nodes.append(node_name_dep_this)
                    # Add the departure node into the list of all departure nodes
                    attributes = {'train': train_id, 'type': 'departureNodePassing',
                                  'departureTime': departure_time_this_node,
                                  'StopStatus': train_path_nodes.stop_status.name}
                    # Record the attributes, train id, departure time, stop status
                    departure_nodes_passing_attributes[node_name_dep_this] = attributes
                    # Waiting edge between the departure and the arrival time of this node
                    wait_time = (departure_time_this_node - arrival_time_this_node)
                    # Add the waiting time on the waiting edge lists
                    waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds / 60)])
                    waiting_edges_attributes[(node_name_arr_this, node_name_dep_this)] = {'flow': 0,
                                                                                          'type': 'waiting',
                                                                                          'train_id': train_id}

                    # Update the departure node for next iteration
                    departure_node_last_node_name = node_name_dep_this
                    departure_time_last_node = departure_time_this_node

    # Print message to be updated of the progress
    print('Transit nodes are created.')

    # Nodes for commercial stops
    stations_with_commercial_stop = np.unique(np.array(stations_with_commercial_stop))

    # Create the timetable with the driving, waiting edges
    timetable_graph = nx.DiGraph()
    timetable_graph.add_nodes_from(arrival_nodes_attributes)
    nx.set_node_attributes(timetable_graph, arrival_nodes_attributes)

    timetable_graph.add_nodes_from(arrival_nodes_passing_attributes)
    nx.set_node_attributes(timetable_graph, arrival_nodes_passing_attributes)

    timetable_graph.add_nodes_from(departure_nodes_attributes)
    nx.set_node_attributes(timetable_graph, departure_nodes_attributes)

    timetable_graph.add_nodes_from(departure_nodes_passing_attributes)
    nx.set_node_attributes(timetable_graph, departure_nodes_passing_attributes)

    timetable_graph.add_weighted_edges_from(driving_edges)
    nx.set_edge_attributes(timetable_graph, driving_edges_attributes)

    timetable_graph.add_weighted_edges_from(waiting_edges)
    nx.set_edge_attributes(timetable_graph, waiting_edges_attributes)

    return timetable_graph, stations_with_commercial_stop


def connections_homes_with_station_candidates(timetable_graph, parameters):
    # Connect all origins of an odt with the departure nodes and create odt list for sp input
    edges_o_stations_d = helpers.EdgesOriginStationDestination(timetable_graph, parameters)

    return edges_o_stations_d


def generate_edges_origin_station_destination(timetable_graph, parameters, return_edges_nodes=True):
    """
    :param timetable_graph: Graph with all train arcs and nodes
    :param parameters: class object with all the parameters for the code
    :param return_edges_nodes: default True, otherwise add the edges to the graph and return G
    :return: edges and nodes or fully loaded graph
    """
    # Set the parameters
    origin_nodes = []
    origin_nodes_attributes = {}
    destination_nodes = []
    destination_nodes_attributes = {}
    edges_o_stations = list()
    edges_o_stations_attr = {}
    edges_stations_d = list()
    edges_stations_d_attr = {}

    station_candidates = parameters.station_candidates
    odt = parameters.odt_by_origin
    origin_name_desired_dep_time = parameters.origin_name_desired_dep_time

    # Get the departing and arrival nodes separately
    departing_trains_by_node, arriving_trains_by_node = get_all_departing_and_arriving_nodes_at_stations(
        timetable_graph)

    # Loop trough all home zones where trips are starting
    for origin_zone, trips_origin in odt.items():

        # Loop through all trips origin to get the origin node name, the destination and the desired departure time
        for od_pair, group_size in trips_origin.items():
            origin_node_name = od_pair[0]
            destination_node_name = od_pair[1]
            desired_dep_time = origin_name_desired_dep_time[(origin_node_name, destination_node_name)]

            # If the node name does not appear the origin nodes, add to the list
            if origin_node_name not in origin_nodes:
                origin_nodes.append(origin_node_name)
                attributes = {'zone': origin_zone, 'departure_time': desired_dep_time, 'type': 'origin'}
                origin_nodes_attributes[origin_node_name] = attributes

            # Edges for departure at origin
            for station_code, station_values in station_candidates[origin_zone].items():
                try:
                    tt_to_station = station_values['tt_toStation']
                except KeyError:
                    tt_to_station = 50
                    print('Travel time is not in data, set to 50 minutes ', origin_zone, ' ',
                          station_values[station_code]['station_zone'])

                # Identify train departure nodes candidates at stations
                time_at_station = datetime.datetime.strptime(desired_dep_time, "%Y-%m-%dT%H:%M") + \
                                  datetime.timedelta(minutes=int(tt_to_station))

                departure_nodes_of_station = identify_departure_nodes_for_trip_at_station(timetable_graph, parameters,
                                                                                          time_at_station,
                                                                                          departing_trains_by_node[
                                                                                              station_code])

                # Add the edges of origin to departures at the station with weight of travel time
                for d in departure_nodes_of_station:
                    edge = [origin_node_name, d, tt_to_station]
                    attributes = {'type': 'origin_station'}
                    if edge not in edges_o_stations:
                        edges_o_stations.append(edge)
                        edges_o_stations_attr[(origin_node_name, d)] = attributes

            # Add the destination nodes and edges connecting with stations if not added yet
            if destination_node_name in destination_nodes:
                continue
            else:  # Add the node to the list of nodes
                destination_nodes.append(destination_node_name)
                attributes = {'zone': destination_node_name, 'type': 'destination'}
                destination_nodes_attributes[destination_node_name] = attributes
            for station_code, station_values in station_candidates[destination_node_name].items():
                try:
                    tt_to_station = station_values['tt_toStation']
                except KeyError:
                    tt_to_station = 50
                    print('Travel time is not in data, set to 50 minutes ', destination_node_name, ' ',
                          station_values[station_code]['station_zone'])

                # Add the edges of origin to departures at the station with weight of travel time
                try:
                    for a in arriving_trains_by_node[station_code]:  # All trains arriving station candidate
                        edge = [a, destination_node_name, tt_to_station]
                        attributes = {'type': 'station_destination'}
                        if edge not in edges_stations_d:
                            edges_stations_d.append(edge)
                            edges_stations_d_attr[(a, destination_node_name)] = attributes
                except KeyError:
                    pass

    if not return_edges_nodes:
        # Add origin nodes and attributes
        timetable_graph.add_nodes_from(origin_nodes)
        nx.set_node_attributes(timetable_graph, origin_nodes_attributes)
        timetable_graph.add_nodes_from(destination_nodes)
        nx.set_node_attributes(timetable_graph, destination_nodes_attributes)

        # Add the edges of an origin node to all departing nodes of the selected stations
        # and the same for stations to destination
        timetable_graph.add_weighted_edges_from(edges_o_stations)
        nx.set_edge_attributes(timetable_graph, edges_o_stations_attr)
        timetable_graph.add_weighted_edges_from(edges_stations_d)
        nx.set_edge_attributes(timetable_graph, edges_stations_d_attr)
        return timetable_graph

    else:
        edges_o_stations_dict = helpers.transform_edges_into_dict_key_source(edges_o_stations)
        edges_stations_d_dict = helpers.transform_edges_into_dict_key_target(edges_stations_d)

        return origin_nodes, origin_nodes_attributes, destination_nodes, destination_nodes_attributes,\
               edges_o_stations, edges_o_stations_attr, edges_stations_d, edges_stations_d_attr, edges_o_stations_dict,\
               edges_stations_d_dict


def get_all_departing_and_arriving_nodes_at_stations(timetable_graph):
    # Set the dictionary
    departing_trains_by_node = dict()  # Dictionary with key : station & value : all departing nodes
    arriving_trains_by_node = dict()  # Dictionary with key : station & value : all departing nodes

    # Check all the nodes in the timetable graph and get only the departure nodes and arrival nodes
    for x, y in timetable_graph.nodes(data=True):
        # If any of this type, skip to the next node
        if y['type'] in ['arrivalNodePassing', 'departureNodePassing', 'origin', 'destination']:
            continue
        # If the type is arrival node, then add to the arrival list plus the dict.
        elif y['type'] == 'arrivalNode':
            if not arriving_trains_by_node.__contains__(x[0]):
                arriving_trains_by_node[x[0]] = list()
            arr_nodes_in_dict = arriving_trains_by_node[x[0]]
            arr_nodes_in_dict.append(x)

        # If the type is departure node, then add to the departure list plus the dict.
        elif y['type'] == 'departureNode':
            if not departing_trains_by_node.__contains__(x[0]):
                departing_trains_by_node[x[0]] = list()
            dep_nodes_in_dict = departing_trains_by_node[x[0]]
            dep_nodes_in_dict.append(x)

    return departing_trains_by_node, arriving_trains_by_node


def identify_departure_nodes_for_trip_at_station(timetable_graph, parameters, time_at_station, value):
    # Set the list for e nodes of the station
    departure_nodes_of_station = list()

    # Loop through all the departure trains from the station
    for depart in value:
        # Check the time at the station is in the bracket of possible waiting time to get in the train
        if time_at_station + parameters.min_wait <= timetable_graph.nodes[depart]['departureTime'] <= time_at_station \
                + parameters.max_wait:
            # If feasible, add the train in the departure nodes of the station
            departure_nodes_of_station.append(depart)
        else:
            continue
    return departure_nodes_of_station
