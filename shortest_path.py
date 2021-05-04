"""
Created on Tue Mar 10 2021

@author: BenMobility

Shortest path computation for the master thesis main codes.
"""
import networkx as nx
import convert
import scipy.sparse
import helpers
import time
from heapq import heappush, heappop
from itertools import count
import datetime
import timetable_graph
import copy
import numpy as np


def scipy_dijkstra_full_od_list(timetable_graph, odt_list):
    """
    Function that takes the timetable graph and will create a sparse matrix to compute the shortest path of each od
    pair in the list provided
    :param timetable_graph: Digraph that equals to the timetable
    :param odt_list: List of all od pairs
    :return: compute the shortest path for all od pairs
    """
    compare_with_nx = False
    nb_unequal_paths = 0
    sparse_matrix, index, index_names = convert.to_scipy_sparse_matrix(timetable_graph)
    dist_matrix, precedents = scipy.sparse.csgraph.dijkstra(sparse_matrix, directed=True, return_predecessors=True)

    # For each od pair in the list, compute the shortest path
    for odt in odt_list:
        index_source = index[odt[0]]
        index_target = index[odt[1]]
        length = dist_matrix[index_source, index_target]
        path = get_path(precedents, index_source, index_target, index_names)

        # If we want to compare with the Dijkstra in networkx
        if compare_with_nx:
            try:
                l, p = nx.single_source_dijkstra(timetable_graph, odt[0], odt[1])
                if p != path and abs(l - length) > 0.2:
                    nb_unequal_paths += 1
            except nx.exception.NetworkXNoPath:
                pass

    # Print the number of unequal paths in all the od list
    print(f'The number of unequal path in the computation of the shortest path is {nb_unequal_paths}.')


def get_path(precedent, i, j, index_names):
    """
    get the path from source node and target node
    :param precedent: list of nodes of the shortest path from dijkstra
    :param i: source node
    :param j: target node
    :param index_names: names of the nodes
    :return: the shortest path for each od pairs
    """
    path = [index_names[j]]
    k = j
    while precedent[i, k] != -9999:
        path.append(index_names[precedent[i, k]])
        k = precedent[i, k]
    return path[::-1]


def find_path_for_all_passengers_and_remove_unserved_demand(timetable_graph, odt_list, origin_name_desired_dep_time,
                                                            origin_name_zone_dict, parameters):
    """
    Function that finds the path for all passengers and remove the unserved demand from the list
    :param timetable_graph: Digraph that equals to the timetable
    :param odt_list: List of all od pairs
    :param origin_name_desired_dep_time: zone with the desired departure time
    :param origin_name_zone_dict: origin zone for each passengers
    :param parameters: class object with all the parameters in the main code
    :return: updated parameters in the class parameters (i.e: odt_list_with_path)
    """
    length, path, served_unserved_passengers, odt_with_path, odt_no_path = find_sp_for_all_sources_full_graph(
        timetable_graph, parameters)

    print('Passengers with path: ', served_unserved_passengers[0], ', passengers without path: ',
          served_unserved_passengers[1])

    odt_list_with_path = []
    odt_with_path_by_origin = {}
    odt_with_path_by_dest = {}
    odt_list_group_size = {}
    origin_name_desired_dep_time_with_path = {}

    # Check all the od pairs and see if there is a path
    for odt in odt_list:
        source, target, priority, group_size, odt_path = odt
        if [source, target, group_size] in odt_with_path:
            odt_list_with_path.append(odt)
            zone = origin_name_zone_dict[source]
            if not odt_with_path_by_origin.__contains__(zone):
                odt_with_path_by_origin[zone] = {}
            odt_with_path_by_origin[zone][(source, target)] = group_size

            if not odt_with_path_by_dest.__contains__(target):
                odt_with_path_by_dest[target] = {}
            odt_with_path_by_dest[target][(source, target)] = group_size

            origin_name_desired_dep_time_with_path[(source, target)] = origin_name_desired_dep_time[
                (source, target)].copy()

            # To compute after the total travel time for the initial timetable
            odt_list_group_size[(source, target)] = group_size

    # Save the parameters in the class parameters
    parameters.odt_by_origin = odt_with_path_by_origin
    parameters.odt_by_destination = odt_with_path_by_dest
    parameters.odt_as_list = odt_list_with_path
    parameters.origin_name_desired_dep_time = origin_name_desired_dep_time_with_path

    # Save the total initial travel time
    initial_total_travel_time = 0
    for source_target, number_passengers_in_group in odt_list_group_size.items():
        initial_total_travel_time = initial_total_travel_time + length[source_target] * number_passengers_in_group
    parameters.initial_total_travel_time = initial_total_travel_time

    # Get the odt on the closed tracks
    odt_closed_track = []
    for node_closed_track in parameters.path_nodes_on_closed_track:
        for key, values in path.items():
            if not isinstance(values, np.int32) and values is not None:
                if any([node_closed_track == c for c in values]):
                    odt_closed_track.append(key)
    parameters.odt_closed_track = odt_closed_track


def find_sp_for_all_sources_full_graph(timetable_graph, parameters, cutoff=None):
    """
    function that finds the shortest path for all source nodes in the full timetable graph
    :param cutoff: boolean parameters set to None
    :param parameters: class object with all the parameters in the main code
    :param timetable_graph: space time graph
    :return: length {(o_t, d): trip time in min} and path {(o_t, d):[sp nodes]}, served_unserved_p = [served, unserved]
    """
    odt_for_sp = parameters.odt_as_list
    ods_no_path_initial_timetable = []
    ods_with_path_initial_timetable = []
    length = dict()
    path = dict()
    served_unserved_pass = [0, 0]
    i = 0
    # Transform source targets into dictionary, key source, value [(target, group size)]
    source_targets_dict = helpers.transform_odt_into_dict_key_source(odt_for_sp)

    # Compute the time
    tic = tic1 = time.time()
    for source, target_group_size in source_targets_dict.items():
        i += 1
        if len(target_group_size) == 1:
            try:
                l, p = single_source_dijkstra(timetable_graph, source, target_group_size[0][0], cutoff=cutoff)
                length = {**length, **{(source, target_group_size[0][0]): l}}
                path = {**path, **{(source, target_group_size[0][0]): p}}
                served_unserved_pass[0] += target_group_size[0][1]
                ods_with_path_initial_timetable.append([source, target_group_size[0][0], target_group_size[0][1]])
            except nx.exception.NetworkXNoPath:
                length = {**length, **{(source, target_group_size[0][0]): None}}
                path = {**path, **{(source, target_group_size[0][0]): None}}
                served_unserved_pass[1] += target_group_size[0][1]
                ods_no_path_initial_timetable.append([source, target_group_size[0][0], target_group_size[0][1]])
        else:
            l, p = single_source_dijkstra(timetable_graph, source, cutoff=cutoff)
            untuple_target_group_size = [[*x] for x in zip(*target_group_size)]
            target = untuple_target_group_size[0]
            length = {**length, **{(source, y): l[y] if y in l.keys() else None for y in target}}
            path = {**path, **{(source, y): p[y] if y in p.keys() else None for y in target}}
            for target_group in target_group_size:
                if target_group[0] in l.keys():
                    served_unserved_pass[0] += target_group[1]
                    ods_with_path_initial_timetable.append([source, target_group[0], target_group[1]])
                else:
                    served_unserved_pass[1] += target_group[1]
                    ods_no_path_initial_timetable.append([source, target_group[0], target_group[1]])

        if i % 100 == 0:
            print(' iterations completed : ', i, ' | ', len(source_targets_dict), ' in ', time.time() - tic, ' [s]')
            tic = time.time()
    print('\n Shortest path done for ', len(source_targets_dict), ' sources in ', (time.time() - tic1) / 60, '[min]')
    return length, path, served_unserved_pass, ods_with_path_initial_timetable, ods_no_path_initial_timetable


def single_source_dijkstra(timetable_graph, source, target=None, cutoff=None,
                           weight='weight'):
    """Find shortest weighted paths and lengths from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Uses Dijkstra's algorithm to compute shortest paths and lengths
    between a source and all other reachable nodes in a weighted graph.

    Parameters
    ----------
    timetable_graph : NetworkX graph

    source : node label
       Starting node for path


    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list.
       If target is None, paths and lengths to all nodes are computed.
       The return value is a tuple of two dictionaries keyed by target nodes.
       The first dictionary stores distance to each target node.
       The second stores the path to each target node.
       If target is not None, returns a tuple (distance, path), where
       distance is the distance from source to target and path is a list
       representing the path from source to target.
    """
    return multi_source_dijkstra(timetable_graph, {source}, target=target, cutoff=cutoff, weight=weight)


def multi_source_dijkstra(timetable_graph, sources, target=None, cutoff=None, weight='weight'):
    """Find shortest weighted paths and lengths from a given set of
    source nodes.

    Parameters
    ----------
    timetable_graph : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list
       If target is None, returns a tuple of two dictionaries keyed by node.
       The first dictionary stores distance from one of the source nodes.
       The second stores the path from one of the sources to that node.
       If target is not None, returns a tuple of (distance, path) where
       distance is the distance from source to target and path is a list
       representing the path from source to target.
    """

    if not sources:
        raise ValueError('sources must not be empty')
    if target in sources:
        return 0, [target]
    weight = _weight_function(timetable_graph, weight)
    paths = {source: [source] for source in sources}  # dictionary of paths
    # dist is a tuple
    dist = _dijkstra_multisource(timetable_graph, sources, weight, paths=paths, cutoff=cutoff, target=target)

    if target is None:
        return dist, paths
    try:
        return dist[target], paths[target]
    except KeyError:
        raise nx.NetworkXNoPath("No path to {}.".format(target))


def _weight_function(timetable_graph, weight):
    """Returns a function that returns the weight of an edge.

    Parameters
    ----------
    timetable_graph : NetworkX graph.

    weight : string or function
        If it is callable, `weight` itself is returned. If it is a string,
        it is assumed to be the name of the edge attribute that represents
        the weight of an edge. In that case, a function is returned that
        gets the edge weight according to the specified edge attribute.

    Returns
    -------
    function
        This function returns a callable that accepts exactly three inputs:
        a node, an node adjacent to the first one, and the edge attribute
        dictionary for the eedge joining those nodes. That function returns
        a number representing the weight of an edge.
    """

    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if timetable_graph.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


def _dijkstra_multisource(timetable_graph, sources, weight, pred=None, paths=None,
                          cutoff=None, target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    timetable_graph : NetworkX graph

    sources : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.

    weight: function
        Function with (u, v, data) input that returns that edges weight

    pred: dict of lists, optional(default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored.

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node.
        If None, paths are not stored.

    target : node label, optional
        Ending node for path. Search is halted when target is found.

    cutoff : integer or float, optional
        Depth to stop the search. Only return paths with length <= cutoff.

    Returns
    -------
    distance : dictionary
        A mapping from node to shortest distance to that node from one
        of the source nodes.
    """

    # get successor of source nodes
    timetable_graph_successor = timetable_graph._succ if timetable_graph.is_directed() else timetable_graph._adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # Use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        if source not in timetable_graph:
            raise nx.NodeNotFound("Source {} not in timetable graph".format(source))
        seen[source] = 0
        push(fringe, (0, next(c), source))

    while fringe:
        (d, _, v) = pop(fringe)  # Pop and return smallest item from the heap, remove node from queue
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            print(target, v)
            break
        for u, e in timetable_graph_successor[v].items():  # u is  successor node with weight e
            cost = weight(v, u, e)  # cost to reach actual nodes
            if cost is None:
                continue
            if isinstance(cost, datetime.timedelta):
                cost = cost.seconds/60  # weights in minutes
            vu_dist = dist[v] + cost
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                if vu_dist < dist[u]:
                    if timetable_graph.nodes[u]['type'] == 'origin' or timetable_graph.nodes[u]['type'] != 'destination':
                        raise ValueError('Contradictory paths found:',
                                         'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist

                if target is not None:
                    if timetable_graph.nodes[u]['type'] != 'origin' or timetable_graph.nodes[u]['type'] != 'destination':
                        push(fringe, (vu_dist, next(c), u))  # add the node to the queue, if its no home
                    elif u == target:
                        push(fringe, (vu_dist, next(c), u))  # add the node to the queue, if it is target
                else:
                    if timetable_graph.nodes[u]['type'] != 'origin' or timetable_graph.nodes[u]['type'] != 'destination':
                        push(fringe, (vu_dist, next(c), u))  # add the node to the queue, if its no home
                    elif timetable_graph.nodes[u]['type'] == 'origin' or timetable_graph.nodes[u]['type'] == 'destination':
                        dist[u] = vu_dist
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    return dist


def find_sp_for_all_ods_full_graph_scipy(timetable_prime_graph, parameters, edges_o_stations_d, cutoff=None):
    """
    :param cutoff: Set a cutoff for the assignment
    :param edges_o_stations_d: list of edges from origin to stations and stations to destination
    :param timetable_prime_graph: space time graph without connections of origins or destinations
    :param parameters: that contains odt_for_sp: demand matrix with all trips to calculate
    :return: length {(o_t, d): trip time in min} and path {(o_t, d):[sp nodes]}, served_unserved_p = [served, unserved]
    """
    # Print out the results
    print_out = False

    # Create the full graph with origin to stations and stations to destination
    timetable_full_graph = \
        timetable_graph.create_graph_with_edges_o_stations_d(edges_o_stations_d,
                                                             timetable_graph=copy.deepcopy(timetable_prime_graph))

    sparse_matrix, index, index_names = convert.to_scipy_sparse_matrix(timetable_full_graph)
    dist_matrix, predecessors = scipy.sparse.csgraph.dijkstra(sparse_matrix, directed=True, return_predecessors=True)

    # Set the parameters
    total_traveltime = 0
    odt_for_sp = parameters.odt_as_list
    served_unserved_pass = [0, 0]
    i = 0
    nr_source_not_connected = 0
    nr_target_not_connected = 0

    # Compute the time it takes
    tic = tic1 = time.time()

    # Loop through all the OD departure time for the shortest path
    for odt in odt_for_sp:
        i += 1
        source = odt[0]
        target = odt[1]
        group_size = odt[3]
        # Only one target for this source
        try:
            index_source = index[odt[0]]
        except KeyError:
            # Source not connected to graph
            nr_source_not_connected += 1
            served_unserved_pass[1] += group_size
            total_traveltime += parameters.penalty_no_path * group_size
            continue
        try:
            index_target = index[odt[1]]
        except KeyError:
            # Target not connected to graph
            served_unserved_pass[1] += group_size
            total_traveltime += parameters.penalty_no_path * group_size
            nr_target_not_connected += 1
            continue

        length = dist_matrix[index_source, index_target]
        path = get_path(predecessors, index_source, index_target, index_names)

        # If length is infinity, there is no path, than a threshold could also be added (1m)
        if length == float('inf'):
            served_unserved_pass[1] += group_size

            # Assign penalty
            total_traveltime += parameters.penalty_no_path * group_size

        else:
            served_unserved_pass[0] += group_size
            total_traveltime += group_size * length

            # Passenger assignment and capacity constraint
            if parameters.assign_passenger:
                timetable_prime_graph = \
                    assign_pass_and_remove_arcs_exceeded_capacity(timetable_prime_graph, parameters, path,
                                                                  (target, group_size))

        # Print out the iteration with the time computation
        if i % 100 == 0:
            if print_out:
                print('Iterations completed : ', i, ' | ', len(odt_for_sp), ' in ', time.time() - tic, ' [s]')
            tic = time.time()

    if print_out:
        print('Shortest path done for ', len(odt_for_sp), ' ODs in ', (time.time() - tic1) / 60, '[min]')
        print(nr_source_not_connected, 'unconnected sources')
        print(nr_target_not_connected, 'unconnected targets')
    return timetable_prime_graph, served_unserved_pass, total_traveltime


def assign_pass_and_remove_arcs_exceeded_capacity(timetable_prime_graph, parameters, path_odt, target_group_size):
    # Loop through all the assigned path odt and assign the flow
    for idx in range(1, len(path_odt) - 2):
        if timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['type'] in ['driving', 'waiting']:
            timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['flow'] += target_group_size[1]

            # Check if use of capacity constraint, if not, go to next path
            if not parameters.capacity_constraint:
                continue

            # Check if they use the bus in the path
            if 'bus' in timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]].keys():

                # Check if the flow exceed the bus capacity, if so, update weight to weight_closed_tracks
                if timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['flow'] >= parameters.bus_capacity:
                    timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['initial_weight'] = \
                        timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['weight']
                    timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['weight'] = parameters.weight_closed_tracks

            # Check if the flow of the train exceed the capacity, if so, add the weight of the closed track
            elif timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['flow'] >= parameters.train_capacity:
                timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['initial_weight'] = \
                    timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['weight']
                timetable_prime_graph[path_odt[idx]][path_odt[idx + 1]]['weight'] = parameters.weight_closed_tracks
    return timetable_prime_graph
