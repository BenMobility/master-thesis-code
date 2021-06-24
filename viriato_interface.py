"""
Created on Tue Feb  23 2021

@author: BenMobility

Python files with all method to get the needed inputs

Update:
1- All csv and text files
2- Viriato files
"""

# %% Imports
from py_client import algorithm_interface
from py_client.aidm import *
import numpy as np
import numpy.lib.recfunctions as rfn
import copy


# %% paths

# Viriato
base_url = 'http://localhost:8080'  # Viriato localhost

path_nodesSBB = 'input/Node_SBB_AreaOfInterest.csv'
path_allZonesOD_debug = 'input/zones_debug.csv'
path_centroidZones = 'input/centroids_zones.csv'


# path_OD_depTime = 'input/OD_desired_departure_time_' + str(th_zone_selection) + '.csv'
# path_OD_depTime_grouped = 'input/OD_desired_departure_time_' + str(th_zone_selection) + '_grouped.csv'

# %% Viriato
with algorithm_interface.create(base_url) as algorithm_interface:
    def get_time_window():
        """
        method that returns the time window value for the specified key. Here is timeWindowParameter.
        :return: time_window: an object that contains information about the time window of the scenario
        """
        return algorithm_interface.get_time_window_algorithm_parameter("timeWindowParameter")


    def get_node_info(node_id):
        """
        method to get the info from any node
        :param node_id: node identification in order to find to node in Viriato (type: int)
        :return: node_info: object that contains code, id, debug string and node tracks.
        """
        return algorithm_interface.get_node(node_id)


    def get_neighboring_nodes_from_node(from_node_id):
        """
        method that returns the list of all neighbor nodes that can be visited from the node with id from the node
        from_node_id.
        :param from_node_id: node identification of the origin node (type: integer)
        :return: list of all neighbor nodes from the node.
        """
        return algorithm_interface.get_nodes_with_section_track_from(from_node_id)


    def get_neighboring_nodes_to_node(to_node_id):
        """
        method that returns a list of all neighbor nodes from which a train can run to the node with id.
        :param to_node_id: node_id: node identification in order to find to node in Viriato (type: int)
        :return: list of all neighbor nodes to the node.
        """
        return algorithm_interface.get_nodes_with_section_track_to(to_node_id)


    def get_neighboring_nodes_between(from_node_id, to_node_id):
        """
        method that returns a list of all neighbor nodes of the nodes with ids from node_id to node_id
        :param from_node_id: node identification of the origin node (type: integer)
        :param to_node_id:  node identification of the destination node (type: integer)
        :return: list of all neighbor nodes in between ()
        """
        return algorithm_interface.get_neighboring_nodes_between(from_node_id, to_node_id)


    def get_section_track(section_track_id):
        """
        method that returns an object for the given section track ID.
        :param section_track_id: Integer of the section track id
        :return: object with all this information of the section track
        """
        return algorithm_interface.get_section_track(section_track_id)


    def get_section_track_from(from_node_id):
        """
        method that returns a list of all section tracks which a train can use to any neighboring node of the node
        with id from node.
        :param from_node_id: node identification of the origin node (type: integer)
        :return: list of all section tracks which a train can use to any neighboring node
        """
        return algorithm_interface.get_section_tracks_from(from_node_id)


    def get_section_track_to(to_node_id):
        """
        method that returns a list of all section tracks which a train can use to visit a node with id from a
        neighboring node.
        :param to_node_id: node identification of the destination node (type: integer)
        :return:  list of all section tracks to visit a node id
        """
        return algorithm_interface.get_section_tracks_to(to_node_id)


    def get_section_track_closures(time_window):
        """
        method the returns all section track closures anywhere on the network in time_window
        :param time_window: a Viriato object that contains the time_window of the project
        :return: track_closures: all section track closures anywhere on the network (type: Viriato object)
        """
        return algorithm_interface.get_section_track_closures(time_window)


    def get_node_track_closures(time_window):
        """
        method that returns all node track closures anywhere on the network in time window.
        :param time_window: time window from Viriato.
        :return: A list of all node on track closures
        """
        return algorithm_interface.get_node_track_closures(time_window)


    def get_section_track_closure_ids(time_window):
        """
        method the returns all section track closures anywhere on the network in time_window
        :param time_window: a Viriato object that contains the time_window of the project
        :return: track_closure_ids: list of all track section ids that are closed
        """
        closed_tracks = algorithm_interface.get_section_track_closures(time_window)
        closed_section_track_ids = []
        for track in closed_tracks:
            if track.section_track_id not in closed_section_track_ids:
                closed_section_track_ids.append(track.section_track_id)
            else:
                continue
        return closed_section_track_ids


    def get_section_tracks_between(from_node_id, to_node_id):
        """
        method that return a list of all section tracks a train can use to run from node_id to node_id.
        :param from_node_id: node identification of the destination node (type: integer)
        :param to_node_id: node identification of the destination node (type: integer)
        :return: return a list of all section tracks a train can use to run from node_id to node_id
        """
        return algorithm_interface.get_section_tracks_between(from_node_id, to_node_id)


    def get_parallel_section_tracks(section_track_id):
        """
        method that return a list of all section tracks starting and ending at the same nodes.
        :param section_track_id: section track identification (type: integer)
        :return: return a list of all section tracks starting and ending at the same nodes.
        """
        return algorithm_interface.get_parallel_section_tracks(section_track_id)


    def get_all_visited_nodes_in_time_window(time_window):
        """
        method to get all the visited nodes with its code from Viriato. It gets all the trains in the given
        time window and then extract all the node ids from the trains. From the node ids, the method will
        get Viriato node codes.
        :parameter: time_window: a Viriato object called previously in the main code.
        :return: nodes_code: all visited node codes (type : list)
        :return: id_nodes: all visited node ids (type: list)
        """
        # All trains traveling in the given time window
        trains_cut_to_time_range = algorithm_interface.get_trains_cut_to_time_range(time_window)
        # Initialize the list of all visited nodes in time window
        node_ids = list()
        # Loop to seek each nodes for each train in the time window
        for train in trains_cut_to_time_range:
            for train_path_node in train.train_path_nodes:
                node_ids.append(train_path_node.node_id)
        # Keep only the unique ids
        node_ids = set(node_ids)
        # Initialize the list to store the codes of all the visited nodes
        nodes_code = {}
        id_nodes = {}
        # Store all the unique codes
        for node_id in node_ids:
            node = algorithm_interface.get_node(node_id)  # Gets the node from Viriato
            code = node.code
            nodes_code[node_id] = code
            id_nodes[code] = node_id
        return nodes_code, id_nodes


    def get_all_viriato_node_ids_to_code_dictionary(all_node_ids):
        code_id_dictionary = {}
        id_code_dictionary = {}
        all_node_ids = set(all_node_ids)  # set only stores unique values
        for node_id in all_node_ids:
            node = all_node_ids(node_id)
            code = node['Code']
            code_id_dictionary[node_id] = code
            id_code_dictionary[code] = node_id

        return code_id_dictionary, id_code_dictionary

    def send_status_message(message):
        """
        Method that update a status message to Viriato
        :param message: str type
        """

        algorithm_interface.show_status_message(message)

    def send_status_log_message(message):
        algorithm_interface.show_status_message(message, message)

    def notify_users(title, message):
        algorithm_interface.notify_user(title, description=message)

    def get_trains_cut_time_range_driving_any_node(time_window, node_ids):
        """
        method that returns a list containing all trains that run through a node being in a list.
        :param time_window: a Viriato object that contains the starting and ending time window.
        :param node_ids: list of node ids.
        :return: a list containing all trains that run through a node for a given node ids list.
        """
        return algorithm_interface.get_trains_cut_to_time_range_driving_any_node(time_window, node_ids)


    def get_trains_area_interest_time_window(trains, stations_in_area, time_window):
        """
        method that get all trains in the area of interest and in the selected time_window
        :param trains: list of trains (type: list of viriato object of trains)
        :param stations_in_area: list of station ids (type: integer)
        :param time_window: time window selected previously on Viriato (type: viriato object)
        :return: list of trains inside the area of interest and in the time window
        """
        last_train_path_node_departure_time = None
        get_trains_to_area_time = list()
        last_train_path_node = None
        first_node_before_set = False
        start = True
        # Select train at a time and check if it is inside the area and time window
        for train in trains:
            train_path_node_in_area = list()
            # Check every train path node for each train if it is in the area and the time window
            for train_path_node in train.train_path_nodes:
                # Check if the node is in the area of interest
                if train_path_node.node_id in stations_in_area:
                    # Check if the train path node is inside the time window
                    if time_window.from_time <= train_path_node.arrival_time <= time_window.to_time:
                        if not first_node_before_set and not start:
                            first_node_before_set = True
                            train_path_node_in_area.append(last_train_path_node)
                        train_path_node_in_area.append(train_path_node)
                    else:
                        # Check if the last path node is inside the area
                        if last_train_path_node in train_path_node_in_area and not start:
                            # Check if the last path node is inside the time window
                            if last_train_path_node_departure_time < time_window.to_time:
                                # Add the last path node in the train path node in area
                                train_path_node_in_area.append(train_path_node)
                    last_train_path_node = train_path_node
                    last_train_path_node_departure_time = train_path_node.departure_time
                start = False
            # Modify the train path nodes of the train if only the path node in area length is bigger than 1
            if len(train_path_node_in_area) > 1:
                train.train_path_nodes = train_path_node_in_area
            else:
                continue
            # Record the train with its new train path nodes inside the area of interest and time window
            get_trains_to_area_time.append(train)
        return get_trains_to_area_time


    def get_trains_driving_any_node(time_window, node_ids):
        """
        method that returns a list containing all trains that run through a node with node id being in the
        list of node_ids.
        :param time_window: Viriato object that contains the arrival and departure time of a train
        :param node_ids: List of integers that contains the node ids
        :return: a list containing all trains (Viriato object)
        """
        return algorithm_interface.get_trains_driving_any_node(time_window, node_ids)

    def get_emergency_train():
        """
        Methods that calls an emergency train from Viriato
        :return: an emergency train object from viriato
        """
        emergency_train = algorithm_interface.get_algorithm_train_parameter('emergency_train')
        emergency_train = copy.deepcopy(emergency_train)

        # deepcopy instead of clone_train... at the end of the alns only
        # emergency_train = algorithm_interface.clone_train(emergency_train.id)
        return emergency_train

    def cancel_train(train_id):
        """
        method that cancel an existing train on the timetable
        :param train_id: the train id that wished to be canceled (type: integer)
        :return: the id of the cancelled train. (type: integer)
        """
        return algorithm_interface.cancel_train(train_id)


    def cancel_train_after(train_id, train_path_node_id):
        """
        method that cancel an existing train partially and return the resulting train
        :param train_id: the ID of the train. (type: integer)
        :param train_path_node_id: the ID of the train path node that start the cancellation. Has to be at least the
        second train path node of the train. (type: integer)
        :return: The resulting train (type: Viriato object)
        """
        return algorithm_interface.cancel_train_after(train_id, train_path_node_id)


    def cancel_train_before(train_id, train_path_node_id):
        """
        method that cancel an existing train partially and return the resulting train
        :param train_id: the ID of the train. (type: integer)
        :param train_path_node_id: the ID of the train path node that start the cancellation. Has to be at most second
        to last train path node of the train. (type: integer)
        :return: The resulting train (type: Viriato object)
        """
        return algorithm_interface.cancel_train_before(train_id, train_path_node_id)


    def clone_train(train_id):
        """
        method that create an identical copy of an existing train and return the copied train. The newly created train
        is part of the timetable and will be found in subsequent queries
        :param train_id: the ID of the train to copy (type: integer)
        :return: The copied train. (type: viriato object of the copied train)
        """
        return algorithm_interface.clone_train(train_id)


    def get_headway_time(section_track_id, from_node_id, to_node_id, preceding_train_path_node_id,
                         succeeding_train_path_node_id):
        """
        methode that returns the headway time between two trains if they were using the given section track
        :param section_track_id: Identification number of the section track
        :param from_node_id: Identification number of the node id the train is coming from
        :param to_node_id: Identification number of the node id the train is going to
        :param preceding_train_path_node_id: Identification number of the train path node the train is coming from
        :param succeeding_train_path_node_id: Identification number of the train path node the train is going to
        :return: the duration of the headway
        """
        return algorithm_interface.get_headway_time(section_track_id, from_node_id, to_node_id,
                                                    preceding_train_path_node_id, succeeding_train_path_node_id)


    def reroute_train(train_to_reroute, path, idx_start_end, track_info, infra_graph):
        """
        section_tracks_between(node1, node2, tpn_track_sequences, infra_graph)
        :param infra_graph:
        :param track_info:
        :param train_to_reroute: the train which has to be rerouted
        :param idx_start_end: list of indices of TrainPathNodes in the train to reroute
        :param path: the shortest path with closed track, first/last node are the start/end of rerouting
        :return: rerouted train on viriato platform
        """
        # Get the indices for start and end nodes of the train path
        idx_start_end_initial = idx_start_end.copy()
        number_nodes_added_new_path = \
            len(path[1]) - len(train_to_reroute.train_path_nodes[idx_start_end_initial[0]:idx_start_end_initial[1] + 1])

        # Get the index of the node after the reroute
        idx_1st_node_after_reroute = idx_start_end_initial[1] + 1

        # Get the sequence number of the starting and ending node of the train path node of the train to reroute
        sequence_number_start = train_to_reroute.train_path_nodes[0].sequence_number
        sequence_number_end = train_to_reroute.train_path_nodes[-1].sequence_number

        # Get the 1st node information from Viriato
        viriato_node = copy.deepcopy(get_node_info(path[1][0]))

        # Get the node track start to reroute
        node_track_start_reroute = viriato_node.node_tracks[0].id

        # Get the node id of starting node of the reroute
        path_node_id_start_reroute = train_to_reroute.train_path_nodes[idx_start_end_initial[0]].id
        path_node_id_end_reroute = train_to_reroute.train_path_nodes[idx_start_end_initial[1]].id

        # Get the section track id for the starting of the reroute path
        end_section_track = select_section_track(path[1][0], path[1][1], 'departure', track_info, infra_graph)

        # Get the track id, End section track of 1st node is start_section_track of next node if a station
        track_id_from_last_node = end_section_track
        track_id_to_next_node = track_id_from_last_node

        # Set the list of routing edges
        routing_edges = []

        # Check the node track at the first node where rerouting begins if it is set
        algorithm_interface.update_node_track(train_to_reroute.id, path_node_id_start_reroute,
                                              node_track_start_reroute)

        # First outgoing edge
        routing_edges.append(OutgoingRoutingEdge(path[1][0], end_section_track, node_track_start_reroute))

        # Iteration through all the nodes in the shortest path of the alternative path (reroute)
        j = 1  # Index in path list
        for node in path[1][1:-1]:  # Keep first and last node the same for both original and reroute path
            # Get the information of the current node
            viriato_node = get_node_info(node)
            track_id_from_last_node = track_id_to_next_node
            track_id_to_next_node = select_section_track(path[1][j], path[1][j + 1], 'departure', track_info,
                                                         infra_graph)

            # Check if it is a junction
            if len(viriato_node.node_tracks) == 0:
                routing_edges.append(CrossingRoutingEdge(viriato_node.id, track_id_from_last_node,
                                                         track_id_to_next_node))

            # If not junction, append incoming and outgoing edges
            else:
                node_track = viriato_node.node_tracks[0].id  # Assign the node track
                routing_edges.append(IncomingRoutingEdge(viriato_node.id, track_id_from_last_node, node_track))
                routing_edges.append(OutgoingRoutingEdge(viriato_node.id, track_id_to_next_node, node_track))

            # Add one to j and continue the loop
            j += 1

        # Last rerouted node, last node of the path to the next node in the train path
        viriato_node = get_node_info(path[1][-1])
        node_track = viriato_node.node_tracks[0].id
        track_id_from_last_node = track_id_to_next_node

        # Last incoming edge
        routing_edges.append(IncomingRoutingEdge(viriato_node.id, node_track, track_id_from_last_node))

        # Check the node track at the last node of the rerouting path if it is set
        algorithm_interface.update_node_track(train_to_reroute.id, path_node_id_end_reroute, node_track)

        # Update viriato train to reroute
        rerouted_train = algorithm_interface.reroute_train(train_to_reroute.id,
                                                           UpdateTrainRoute(train_to_reroute.id,
                                                                            end_train_path_node_id=
                                                                            path_node_id_end_reroute,
                                                                            routing_edges=routing_edges,
                                                                            start_train_path_node_id=
                                                                            path_node_id_start_reroute))

        #  Track down the first rerouted node and the last rerouted node tracks, and change them to the previous
        # id from the original train

        # Identify if the rerouted train has a node outside the time window, adapt the indices if so
        length_train_before_reroute = len(train_to_reroute.train_path_nodes)  # Get the length for comparison
        rerouted_train_not_cut_to_time_window = False
        if sequence_number_start != rerouted_train.train_path_nodes[0].sequence_number \
                or sequence_number_end != rerouted_train.train_path_nodes[-1].sequence_number:
            # Rerouted train has changed, not cut to time window, find the index of train path nodes up to where to cut
            nr_nodes_added_before = sequence_number_start - rerouted_train.train_path_nodes[0].sequence_number
            # New indices to cut train path nodes so that all trains before / afterwards are the same
            cut_before_idx = nr_nodes_added_before
            cut_after_idx = length_train_before_reroute + nr_nodes_added_before + number_nodes_added_new_path
            rerouted_train_not_cut_to_time_window = True
        else:
            cut_before_idx = 0
            cut_after_idx = 0

        # If the rerouted train is not cut to the time window, we need to cut it. Hence, get the indices
        if rerouted_train_not_cut_to_time_window:
            rerouted_train.cut_indices = [cut_before_idx, cut_after_idx]
        else:
            rerouted_train.cut_indices = None

        # For update trains, we need the start and end reroute indices saved in the rerouted train
        rerouted_train.start_end_reroute_idx = [idx_start_end_initial[0] + cut_before_idx,
                                                idx_start_end_initial[1] + cut_after_idx]
        return rerouted_train


    def select_section_track(from_node, to_node, identifier, track_info, infra_graph):
        # Set the parameters
        edges_between = infra_graph.get_edge_data(from_node, to_node)
        all_track_id = list()
        all_frequency = list()

        # Loop through all the edges
        for track_id, attributes in edges_between.items():
            tuple_key = (track_id, from_node, to_node, identifier)
            all_track_id.extend([track_id])

            if tuple_key in track_info.track_sequences_of_TPN.keys():
                len_sequences = len(track_info.track_sequences_of_TPN[tuple_key])
                all_frequency.extend([len_sequences])
            else:
                len_sequences = 0
                all_frequency.extend([len_sequences])

        # Sort the list by the number of trains on that track from low to high
        max_value = max(all_frequency)
        max_index = all_frequency.index(max_value)
        section_track_selected = all_track_id[max_index]

        return section_track_selected


    def calculate_run_times(train_id):
        """
        method that calculate the run times of given algorithm train and return the times
        :param train_id: The ID of the algorithm train to be calculated (integer)
        :return: the times
        """
        return algorithm_interface.calculate_run_times(train_id)


    def get_list_update_train_times_rerouting(runtime_reroute_train_feasible):
        """
        method that update the times of an existing Viriato train and return the updated result.
        :param runtime_reroute_train_feasible: dictionary of all the updated train path nodes
        :return: list to apply on viriato after when the solution is kept.
        """
        list_update_times_train_path_node = []
        for node_id, run_times in runtime_reroute_train_feasible.items():
            updated_stop_status = None
            if run_times['ArrivalTime'] != run_times['DepartureTime'] \
                    and run_times['OriginalStopStatus'] == StopStatus.passing:
                updated_stop_status = StopStatus.operational_stop
            list_update_times_train_path_node.append(UpdateTimesTrainPathNode(node_id, run_times['ArrivalTime'],
                                                                              run_times['DepartureTime'],
                                                                              run_times['RunTime'],
                                                                              run_times['MinimumStopTime'],
                                                                              updated_stop_status))
        return list_update_times_train_path_node


    def get_list_update_train_times_delay(runtime_reroute_train_feasible):
        """
        method that update the times of an existing Viriato train and return the updated result.
        :param runtime_reroute_train_feasible: dictionary of all the updated train path nodes
        :return: list to apply on viriato after when the solution is kept.
        """
        list_update_times_train_path_node = []
        for node_id, run_times in runtime_reroute_train_feasible.items():
            list_update_times_train_path_node.append(UpdateTimesTrainPathNode(node_id, run_times['ArrivalTime'],
                                                                              run_times['DepartureTime'],
                                                                              run_times['RunTime'],
                                                                              run_times['MinimumStopTime']))
        return list_update_times_train_path_node


    def viriato_update_train_times(train_id, list_update_times_train_path_node):
        """
        method that update the times of an existing Viriato train and return the updated result.
        :param train_id: the ID of the train to be updated.
        :param list_update_times_train_path_node: list of update times train path node
        :return: updated train on Viriato
        """
        return algorithm_interface.update_train_times(train_id, list_update_times_train_path_node)

# %% SBB files


def get_sbb_nodes(node_codes):
    """
    method that extract all the stations from the SBB files and convert them to current form
    nodes SBB for the infrastructure graph.
    :param node_codes: all the visited node codes in Viriato
    :return: sbb_nodes: all the sbb nodes with zone, sbb ID, Viriato ID, node name and coordinates (type: Rec array)
    """
    # Get all the sbb nodes in the area of interest
    stations = np.genfromtxt(path_nodesSBB, delimiter=',', dtype=str, skip_header=1)
    # Keep only sbb ID
    station_zone_number = stations[:, 2]
    # Define the station number as an integer
    station_zone_number = station_zone_number.astype(int).reshape(station_zone_number.shape[0], 1)
    # Add SBB ID and SBB ID Viriato
    station_zone_number = np.concatenate(
        (station_zone_number, np.asarray(stations[:, 0:2], dtype=str).reshape(station_zone_number.shape[0], 2)), axis=1)
    # Add Name in strings
    station_zone_number = np.concatenate(
        (station_zone_number, np.asarray(stations[:, 3], dtype=str).reshape(station_zone_number.shape[0], 1)), axis=1)
    # Add x and y coordinate in floats
    station_zone_number = np.concatenate(
        (station_zone_number, np.asarray(stations[:, 4:6], dtype=float).reshape(station_zone_number.shape[0], 2)),
        axis=1)
    # Convert nd array into structured array
    sbb_nodes = np.core.records.fromarrays(station_zone_number.transpose(),
                                           names='zone, SbbID, ViriatoID, NodeName, xcoord, ycoord',
                                           formats='i8, U13, U13, U13, f8, f8')

    # Add Viriato node codes wit SSB nodes names : 'zone, SbbID, ViriatoID, NodeName, xcoord, ycoord, Code'
    sbb_nodes = rfn.append_fields(sbb_nodes, 'Code', np.ones(sbb_nodes.shape[0]) * -1, dtypes='i8', usemask=False,
                                  asrecarray=True)
    sbb_nodes = rfn.append_fields(sbb_nodes, 'Visited', np.zeros(sbb_nodes.shape[0]), dtypes=bool, usemask=False,
                                  asrecarray=True)  # todo: can try to remove the bool and see if it works
    # Add the code from viriato to the list of nodes
    print(f'Number of Viriato nodes is {len(node_codes)}.')
    print(f'Number of SBB nodes is {len(sbb_nodes)}.')
    for key, value in node_codes.items():
        idx = np.where(sbb_nodes.ViriatoID == value)
        sbb_nodes.Code[idx] = key
        sbb_nodes.Visited[idx] = True
    # Check if all of the Viriato scenario nodes are outside of the area of interest
    if all(x == -1 for x in sbb_nodes.Code):
        print('None of the Viriato nodes are in the area of interest')

    # Remove Horgen & Rüti kilometer from area of interest
    print('* Removing node Horgen and node Rüti from SBB nodes.')
    sbb_nodes = sbb_nodes[sbb_nodes.ViriatoID != '85HG']
    sbb_nodes = sbb_nodes[sbb_nodes.ViriatoID != '85RUEKM']
    print(f'New Number of SBB nodes is {len(sbb_nodes)}.')

    # Compute the number of visited nodes
    print(f'Number of SBB nodes visited by Viriato is: {len([x for x in sbb_nodes if x.Code != -1])}')
    print('*Removing unvisited nodes by Viriato trains on the SBB nodes list.')
    sbb_nodes = sbb_nodes[sbb_nodes.Code != -1]
    print(f'New Number of SSB nodes is {len(sbb_nodes)}.')
    return sbb_nodes


def cut_trains_area_of_interest(trains_timetable, stations_in_area):
    # Set the list
    train_index = 0
    trains_timetable_area = list()

    # Check for each train in the timetable if they go in the station of the area of interest
    for train in trains_timetable:
        # Set the list of nodes in the area
        nodes_in_area = list()
        # Loop through all the nodes in the train path
        for j in range(0, len(train.train_path_nodes)):
            # If the node is in the area append in the list
            if train.train_path_nodes[j].node_id in stations_in_area:
                nodes_in_area.append(copy.deepcopy(train.train_path_nodes[j]))
        # If there is more than 1 node in the area, get the train in the timetable area
        if len(nodes_in_area) > 1:
            trains_timetable_area.append(copy.deepcopy(train))
            trains_timetable_area[train_index]._AlgorithmTrain__train_path_nodes = nodes_in_area
            train_index += 1
        # If there is less than 1 or equal, do not consider the train
        else:
            continue
    return trains_timetable_area


def cut_trains_area_interest_time_window(trains_timetable, stations_in_area, time_window):
    # Set the parameters
    from_time = time_window.from_time
    to_time = time_window.to_time
    cut_trains_to_area_time = list()
    last_tpn = None
    first_node_before_set = False
    last_node_after_set = False
    start = True

    # Loop through all the trains in the timetable
    for train in trains_timetable:
        tpn_in_area = list()

        # Loop through all the nodes in the train path node
        for tpn in train.train_path_nodes:

            # If the node is inside the area of interest
            if tpn.node_id in stations_in_area:
                tpn_departure_time = tpn.departure_time
                tpn_arrival_time = tpn.arrival_time

                # If the arrival time is inside the time window, append the train path node
                if from_time <= tpn_arrival_time <= to_time:
                    if not first_node_before_set and not start:
                        first_node_before_set = True
                        tpn_in_area.append(last_tpn)
                    tpn_in_area.append(tpn)

                # If not in the time window, check if it is the last train path node, check departure time last node
                else:
                    if last_tpn in tpn_in_area and not start and not last_node_after_set:

                        # Keep the node if the last departure time node is inside the time window
                        if last_tpn_departure_time < to_time:
                            tpn_in_area.append(tpn)

                # Keep track of the precedent node for the iteration
                last_tpn = tpn
                last_tpn_departure_time = tpn_departure_time
            start = False

        # Keep the train if there is more than one node in the area of interest
        if len(tpn_in_area) > 1:
            train._AlgorithmTrain__train_path_nodes = tpn_in_area

        # If not, check the next train
        else:
            continue

        # Keep the train in the timetable
        cut_trains_to_area_time.append(train)
    return cut_trains_to_area_time
