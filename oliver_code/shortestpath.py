from collections import deque
from heapq import heappush, heappop
from itertools import count
import networkx as nx
import numpy as np
import datetime
import time
import utils
import alns
import convert
import scipy

from networkx.utils import generate_unique_node


def _weight_function(G, weight):
    """Returns a function that returns the weight of an edge.

    The returned function is specifically suitable for input to
    functions :func:`_dijkstra` and :func:`_bellman_ford_relaxation`.

    Parameters
    ----------
    G : NetworkX graph.

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

    If `G` is a multigraph, and `weight` is not callable, the
    minimum edge weight over all parallel edges is returned. If any edge
    does not have an attribute with key `weight`, it is assumed to
    have weight one.

    """

    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


def single_source_dijkstra(G, source, target=None, cutoff=None,
                           weight='weight'):
    """Find shortest weighted paths and lengths from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Uses Dijkstra's algorithm to compute shortest paths and lengths
    between a source and all other reachable nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    stations_frequencies: dictionary key (from node, toNode) [trains per hour]

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

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

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
     G = nx.path_graph(5)
     length, path = nx.single_source_dijkstra(G, 0)
     print(length[4])
    4
     for node in [0, 1, 2, 3, 4]:
    ...     print('{}: {}'.format(node, length[node]))
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4
     path[4]
    [0, 1, 2, 3, 4]
     length, path = nx.single_source_dijkstra(G, 0, 1)
     length
    1
     path
    [0, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    single_source_dijkstra_path()
    single_source_dijkstra_path_length()
    single_source_bellman_ford()
    """
    return multi_source_dijkstra(G, {source}, cutoff=cutoff, target=target,
                                 weight=weight)


def single_source_dijkstra_with_nx(G, source, target=None, cutoff=None,
                                   weight='weight'):
    """Find shortest weighted paths and lengths from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Uses Dijkstra's algorithm to compute shortest paths and lengths
    between a source and all other reachable nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    stations_frequencies: dictionary key (from node, toNode) [trains per hour]

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

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

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
     G = nx.path_graph(5)
     length, path = nx.single_source_dijkstra(G, 0)
     print(length[4])
    4
     for node in [0, 1, 2, 3, 4]:
    ...     print('{}: {}'.format(node, length[node]))
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4
     path[4]
    [0, 1, 2, 3, 4]
     length, path = nx.single_source_dijkstra(G, 0, 1)
     length
    1
     path
    [0, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    single_source_dijkstra_path()
    single_source_dijkstra_path_length()
    single_source_bellman_ford()
    """
    return multi_source_dijkstra_nx(G, {source}, cutoff=cutoff, target=target,
                                 weight=weight)


def single_source_dijkstra_path(G, source, cutoff=None, weight='weight'):
    """Find shortest weighted paths in G from a source node.

    Compute shortest path between source and all other reachable
    nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path.

    stations_frequencies: dictionary key (from node, toNode) [trains per hour]

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    paths : dictionary
       Dictionary of shortest path lengths keyed by target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
     G=nx.path_graph(5)
     path=nx.single_source_dijkstra_path(G,0)
     path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    single_source_dijkstra(), single_source_bellman_ford()

    """

    return multi_source_dijkstra_path(G, {source},  cutoff=cutoff,
                                      weight=weight)


def multi_source_dijkstra_path(G, sources, cutoff=None, weight='weight'):
    """Find shortest weighted paths in G from a given set of source
    nodes.

    Compute shortest path between any of the source nodes and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    stations_frequencies :

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    paths : dictionary
       Dictionary of shortest paths keyed by target.

    Examples
    --------
     G = nx.path_graph(5)
     path = nx.multi_source_dijkstra_path(G, {0, 4})
     path[1]
    [0, 1]
     path[3]
    [4, 3]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    multi_source_dijkstra(), multi_source_bellman_ford()

    """
    length, path = multi_source_dijkstra(G, sources, cutoff=cutoff,
                                         weight=weight)

    return path


def multi_source_dijkstra(G, sources, target=None, cutoff=None, weight='weight'):
    """Find shortest weighted paths and lengths from a given set of
    source nodes.

    Uses Dijkstra's algorithm to compute the shortest paths and lengths
    between one of the source nodes and the given `target`, or all other
    reachable nodes if not specified, for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    stations_frequencies :

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

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
        return (0, [target])
    weight = _weight_function(G, weight)
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = _dijkstra_multisource(G, sources, weight, paths=paths,
                                 cutoff=cutoff, target=target)

    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError:
        raise nx.NetworkXNoPath("No path to {}.".format(target))


def multi_source_dijkstra_nx(G, sources, target=None, cutoff=None, weight='weight'):
    """Find shortest weighted paths and lengths from a given set of
    source nodes.

    Uses Dijkstra's algorithm to compute the shortest paths and lengths
    between one of the source nodes and the given `target`, or all other
    reachable nodes if not specified, for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    stations_frequencies :

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

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
        return (0, [target])
    weight = _weight_function(G, weight)
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = _dijkstra_multisource_nx(G, sources, weight, paths=paths,
                                    cutoff=cutoff, target=target)

    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError:
        raise nx.NetworkXNoPath("No path to {}.".format(target))


def _dijkstra_multisource(G, sources, weight, pred=None, paths=None,
                          cutoff=None, target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.

    weight: function
        Function with (u, v, data) input that returns that edges weight

    stations_frequencies:

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

    Raises
    ------
    NodeNotFound
        If any of `sources` is not in `G`.

    Notes
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original pred and paths objects passed
    as arguments. No need to explicitly return pred or paths.

    """

    # get successor of source nodes
    G_succ = G._succ if G.is_directed() else G._adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        sourceForSuccessor = False  # to make sure that for the source, it is looking for successor nodes
        if source not in G:
            raise nx.NodeNotFound("Source {} not in G".format(source))
        seen[source] = 0
        push(fringe, (0, next(c), source))

    while fringe:
        (d, _, v) = pop(fringe)  # Pop and return smallest item from the heap, remove node from queue
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():  # u is  successor node with weight e
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
                    if G.node[u]['type'] == 'origin' or G.node[u]['type'] != 'destination':  # G.node[u]['type'] != 'origin_destination':  # as the actual shortest distance of homeZones is stored in here
                        raise ValueError('Contradictory paths found:',
                                         'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist

                if target is not None:
                    if G.node[u]['type'] != 'origin' or G.node[u]['type'] != 'destination':  # G.node[u]['type'] != 'origin_destination':
                        push(fringe, (vu_dist, next(c), u))  # add the node to the queue, if its no home
                    elif u == target:
                        push(fringe, (vu_dist, next(c), u))  # add the node to the queue, if it is target
                else:
                    # if G.node[u]['type'] == 'origin' or G.node[u]['type'] == 'destination':
                    #   print(G.node[u]['type'])
                    if G.node[u]['type'] != 'origin' or G.node[u]['type'] != 'destination':  # G.node[u]['type'] != 'destination':
                        push(fringe, (vu_dist, next(c), u))  # add the node to the queue, if its no home
                    elif G.node[u]['type'] == 'origin' or G.node[u]['type'] == 'destination':  # G.node[u]['type'] == 'origin_destination':  # add vu_dist to homeZone to dist, as its never in the queue
                        dist[u] = vu_dist
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return dist


def _dijkstra(G, source, weight, pred=None, paths=None, cutoff=None,
              target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths from a
    single source.

    This is a convenience function for :func:`_dijkstra_multisource`
    with all the arguments the same, except the keyword argument
    `sources` set to ``[source]``.

    """
    return _dijkstra_multisource(G, [source], weight, pred=pred, paths=paths,
                                 cutoff=cutoff, target=target)


def dijkstra_predecessor_and_distance(G, source, cutoff=None, weight='weight'):
    """Compute weighted shortest path length and predecessors.

    Uses Dijkstra's Method to obtain the shortest weighted paths
    and return dictionaries of predecessors for each node and
    distance for each node from the `source`.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only return paths with length <= cutoff.

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    pred, distance : dictionaries
       Returns two dictionaries representing a list of predecessors
       of a node and the distance to each node.
       Warning: If target is specified, the dicts are incomplete as they
       only contain information for the nodes along a path to target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The list of predecessors contains more than one element only when
    there are more than one shortest paths to the key node.

    Examples
    --------
     import networkx as nx
     G = nx.path_graph(5, create_using = nx.DiGraph())
     pred, dist = nx.dijkstra_predecessor_and_distance(G, 0)
     sorted(pred.items())
    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
     sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

     pred, dist = nx.dijkstra_predecessor_and_distance(G, 0, 1)
     sorted(pred.items())
    [(0, []), (1, [0])]
     sorted(dist.items())
    [(0, 0), (1, 1)]
    """

    weight = _weight_function(G, weight)
    pred = {source: []}  # dictionary of predecessors
    return (pred, _dijkstra(G, source, weight, pred=pred, cutoff=cutoff))


def find_sp_for_all_sources_full_graph(G, parameters, cutoff = None):
    '''
    :param G: space time graph
    :param odt_for_sp: demand matrix with all trips to calculate
    :return: length {(o_t, d): trip time in min} and path {(o_t, d):[sp nodes]}, served_unserved_p = [served, unserved]
    '''
    odt_for_sp = parameters.odt_as_list
    ods_no_path_initialTimetable = []
    ods_with_path_initialTimetable = []
    length = dict()
    path = dict()
    served_unserved_pass = [0, 0]
    i = 0
    # transform source targets into dictionary, key source, value [(target, group size)]
    source_targets_dict = utils.transform_odt_into_dict_key_source(odt_for_sp)
    tic = tic1 = time.time()
    for source, target_groupsize in source_targets_dict.items():
        i += 1
        if len(target_groupsize) == 1:
            try:
                l, p = single_source_dijkstra(G, source, target_groupsize[0][0], cutoff=cutoff)
                length = {**length, **{(source, target_groupsize[0][0]): l}}
                path = {**path, **{(source, target_groupsize[0][0]): p}}
                served_unserved_pass[0] += target_groupsize[0][1]
                ods_with_path_initialTimetable.append([source, target_groupsize[0][0], target_groupsize[0][1]])
            except nx.exception.NetworkXNoPath:
                length = {**length, **{(source, target_groupsize[0][0]): None}}
                path = {**path, **{(source, target_groupsize[0][0]): None}}
                served_unserved_pass[1] += target_groupsize[0][1]
                ods_no_path_initialTimetable.append([source, target_groupsize[0][0], target_groupsize[0][1]])
                # print('not served passenger with od: ', source, target_groupsize[0][0])
        else:
            l, p = single_source_dijkstra(G, source, cutoff=cutoff)
            # try:
            untuple_target_groupsize = [[*x] for x in zip(*target_groupsize)]
            target = untuple_target_groupsize[0]
            length = {**length, **{(source, y): l[y] if y in l.keys() else None for y in target}}
            path = {**path, **{(source, y): p[y] if y in p.keys() else None for y in target}}
            for target_group in target_groupsize:
                if target_group[0] in l.keys():
                    served_unserved_pass[0] += target_group[1]
                    ods_with_path_initialTimetable.append([source, target_group[0], target_group[1]])
                else:
                    served_unserved_pass[1] += target_group[1]
                    ods_no_path_initialTimetable.append([source, target_group[0], target_group[1]])
                    # print('not served passenger with od: ', source, target_group[0])

            # except KeyError:
            #    length[source] = list(map(l.get, target))
        if i % 100 == 0:
            print(' iterations completed : ', i, ' | ', len(source_targets_dict), ' in ', time.time() - tic, ' [s]')
            tic = time.time()
    print(' sp done for ', len(source_targets_dict), ' sources in ', (time.time() - tic1) / 60, '[min]')
    return length, path, served_unserved_pass, ods_with_path_initialTimetable, ods_no_path_initialTimetable


def find_sp_for_all_sources_sparse_graph(G_prime, G, parameters, edges_o_stations_d, cutoff=None, assign_passenger=True):
    '''
    :param G: space time graph without connections of origins or destinations
    :param odt_for_sp: demand matrix with all trips to calculate
    :return: length {(o_t, d): trip time in min} and path {(o_t, d):[sp nodes]}, served_unserved_p = [served, unserved]
    '''
    edges_o_s = edges_o_stations_d.edges_o_stations_dict
    edges_o_d = edges_o_stations_d.edges_stations_d_dict
    total_traveltime = 0
    odt_for_sp = parameters.odt_as_list
    length = dict()
    path = dict()
    served_unserved_pass = [0, 0]
    i = 0
    # transform source targets into dictionary, key source, value [(target, group size)]
    # tic_transform = time.time()
    source_targets_dict = utils.transform_odt_into_dict_key_source(odt_for_sp)
    # print('transformations took : ', str(time.time()-tic_transform), ' sec')
    tic = tic1 = time.time()
    for source, targets_groupsize in source_targets_dict.items():
        i += 1
        if source in edges_o_s.keys():
            edges_o_station = edges_o_s[source]
        else:
            # source is not connected to a departure node
            for target_group in targets_groupsize:
                served_unserved_pass[1] += target_group[1]
            continue

        edges_station_d = []
        for target in targets_groupsize:
            edges_station_d.extend(edges_o_d[target[0]])
        # edges_station_d = [edges_o_d[j[0]] for j in targets_groupsize]
        G_prime.add_weighted_edges_from(edges_o_station)
        G_prime.add_weighted_edges_from(edges_station_d)

        # only one target for this source
        if len(targets_groupsize) == 1:
            try:
                l, p = single_source_dijkstra_with_nx(G_prime, source, targets_groupsize[0][0], cutoff=cutoff)
                # l2, p2 = single_source_dijkstra_with_nx(G, source, targets_groupsize[0][0])

                length = {**length, **{(source, targets_groupsize[0][0]): l}}
                path = {**path, **{(source, targets_groupsize[0][0]): p}}
                served_unserved_pass[0] += targets_groupsize[0][1]
                total_traveltime += l * targets_groupsize[0][1]

                # for edges connecting zones with stations, no flow has to be assigned
                path_odt = path[(source, targets_groupsize[0][0])]
                if assign_passenger:
                    G_prime = assign_pass_and_remove_arcs_exceeded_capacity(G_prime, parameters, path_odt, targets_groupsize[0], remove_arcs_exceeding=False)

            except nx.exception.NetworkXNoPath:
                length = {**length, **{(source, targets_groupsize[0][0]): None}}
                path = {**path, **{(source, targets_groupsize[0][0]): None}}
                served_unserved_pass[1] += targets_groupsize[0][1]
                total_traveltime += parameters.penalty_no_path
                # print('not served passenger with od: ', source, targets_groupsize[0][0])
        else:
            l, p = single_source_dijkstra_with_nx(G_prime, source, cutoff=cutoff)
            # l2, p2 = single_source_dijkstra_with_nx(G, source)

            untuple_target_groupsize = [[*x] for x in zip(*targets_groupsize)]
            target = untuple_target_groupsize[0]
            length = {**length, **{(source, y): l[y] if y in l.keys() else None for y in target}}
            path = {**path, **{(source, y): p[y] if y in p.keys() else None for y in target}}
            for target_group in targets_groupsize:
                if target_group[0] in l.keys():
                    served_unserved_pass[0] += target_group[1]
                    total_traveltime += l[target_group[0]]*target_group[1]
                    # assign passengers to the Graph
                    path_odt = path[(source, target_group[0])]
                    if assign_passenger:
                        assign_pass_and_remove_arcs_exceeded_capacity(G_prime, parameters, path_odt, target_group, remove_arcs_exceeding=False)
                else:
                    served_unserved_pass[1] += target_group[1]
                    total_traveltime += parameters.penalty_no_path * target_group[1]

        # remove edges and nodes
        G_prime.remove_edges_from(edges_o_station)
        G_prime.remove_edges_from(edges_station_d)
        G_prime.remove_nodes_from(np.unique(np.array(edges_o_station)[:, 0]))
        G_prime.remove_nodes_from(np.unique(np.array(edges_station_d)[:, 1]))

        if i % 100 == 0:
            print(' iterations completed : ', i, ' | ', len(source_targets_dict), ' in ', time.time() - tic, ' [s]')
            tic = time.time()
    print(' sp done for ', len(source_targets_dict), ' sources in ', (time.time() - tic1) / 60, '[min]')

    return G_prime, length, path, served_unserved_pass, total_traveltime


def find_sp_for_sources_targets_graph(G_prime, G, parameters, edges_o_stations_d, cutoff=None,
                                         assign_passenger=True):
    print_out = False
    '''
    :param G: space time graph without connections of origins or destinations
    :param odt_for_sp: demand matrix with all trips to calculate
    :return: length {(o_t, d): trip time in min} and path {(o_t, d):[sp nodes]}, served_unserved_p = [served, unserved]
    '''

    edges_o_s = edges_o_stations_d.edges_o_stations_dict
    edges_o_d = edges_o_stations_d.edges_stations_d_dict
    total_traveltime = 0
    odt_for_sp = parameters.odt_as_list
    # length = dict()
    # path = dict()
    served_unserved_pass = [0, 0]
    i = 0
    # transform source targets into dictionary, key source, value [(target, group size)]
    # tic_transform = time.time()
    # source_targets_dict = utils.transform_odt_into_dict_key_source(odt_for_sp)
    # print('transformations took : ', str(time.time()-tic_transform), ' sec')
    tic = tic1 = time.time()
    for odt in odt_for_sp:
        i += 1
        source = odt[0]
        target = odt[1]
        groupsize = odt[3]
        if source in edges_o_s.keys():
            edges_o_station = edges_o_s[source]
        else:
            # source is not connected to a departure node
            served_unserved_pass[1] += groupsize
            continue

        edges_station_d = []
        if target in edges_o_d.keys():
            edges_station_d.extend(edges_o_d[target])
        else:
            served_unserved_pass[1] += groupsize
            continue
        # edges_station_d = [edges_o_d[j[0]] for j in targets_groupsize]
        G_prime.add_weighted_edges_from(edges_o_station)
        G_prime.add_weighted_edges_from(edges_station_d)

        # only one target for this source
        try:
            l, p = single_source_dijkstra_with_nx(G_prime, source, target, cutoff=cutoff)
            # l2, p2 = single_source_dijkstra_with_nx(G, source, targets_groupsize[0][0])

            # length = {**length, **{(source, target): l}}
            # path = {**path, **{(source, target): p}}
            served_unserved_pass[0] += groupsize
            total_traveltime += l * groupsize

            # for edges connecting zones with stations, no flow has to be assigned
            # path_odt = path[(source, target)]
            if assign_passenger:
                G_prime = assign_pass_and_remove_arcs_exceeded_capacity(G_prime, parameters, p, (target, groupsize), remove_arcs_exceeding=False)

        except nx.exception.NetworkXNoPath:
            # length = {**length, **{(source, targets_groupsize[0][0]): None}}
            # path = {**path, **{(source, targets_groupsize[0][0]): None}}
            served_unserved_pass[1] += groupsize
            total_traveltime += parameters.penalty_no_path
            # print('not served passenger with od: ', source, targets_groupsize[0][0])

        # remove edges and nodes
        G_prime.remove_edges_from(edges_o_station)
        G_prime.remove_edges_from(edges_station_d)
        G_prime.remove_nodes_from(np.unique(np.array(edges_o_station)[:, 0]))
        G_prime.remove_nodes_from(np.unique(np.array(edges_station_d)[:, 1]))


        if i % 100 == 0:
            if print_out:
                print(' iterations completed : ', i, ' | ', len(odt_for_sp), ' in ', time.time() - tic, ' [s]')
            tic = time.time()

    if print_out:
        print(' sp done for ', len(odt_for_sp), ' ODs in ', (time.time() - tic1) / 60, '[min]')

    return G_prime, served_unserved_pass, total_traveltime


# look for the path and calculate distance
def find_sp_for_all_ods_full_graph_scipy(G_prime, G, parameters, edges_o_stations_d, cutoff=None,
                                         assign_passenger=True):
    print_out = False
    '''
    :param G: space time graph without connections of origins or destinations
    :param odt_for_sp: demand matrix with all trips to calculate
    :return: length {(o_t, d): trip time in min} and path {(o_t, d):[sp nodes]}, served_unserved_p = [served, unserved]
    '''

    G_fullGraph = alns.create_graph_with_edges_o_stations_d(edges_o_stations_d, G=G_prime.copy())
    M, index, index_names = convert.to_scipy_sparse_matrix(G_fullGraph)
    dist_matrix, preds = scipy.sparse.csgraph.dijkstra(M, directed=True, return_predecessors=True)

    total_traveltime = 0
    odt_for_sp = parameters.odt_as_list
    # length = dict()
    # path = dict()
    served_unserved_pass = [0, 0]
    i = 0
    nr_src_not_connected = 0
    nr_tgt_not_connected = 0

    tic = tic1 = time.time()
    for odt in odt_for_sp:
        i += 1
        source = odt[0]
        target = odt[1]
        groupsize = odt[3]
        # only one target for this source
        try:
            index_source = index[odt[0]]
        except KeyError:
            # source not connected to graph
            nr_src_not_connected += 1
            served_unserved_pass[1] += groupsize
            total_traveltime += parameters.penalty_no_path * groupsize
            continue
        try:
            index_target = index[odt[1]]
        except KeyError:
            # target not connected to graph
            served_unserved_pass[1] += groupsize
            total_traveltime += parameters.penalty_no_path * groupsize
            nr_tgt_not_connected += 1
            continue

        length = dist_matrix[index_source, index_target]
        path = get_path(preds, index_source, index_target, index_names)
        if length == float('inf'): # no path, treshold could also be added (1m)
            served_unserved_pass[1] += groupsize
            total_traveltime += parameters.penalty_no_path * groupsize
            # print('not served passenger with od: ', source, targets_groupsize[0][0])
        else:
            served_unserved_pass[0] += groupsize
            total_traveltime += groupsize * length

            if assign_passenger:
                G_prime = assign_pass_and_remove_arcs_exceeded_capacity(G_prime, parameters, path, (target, groupsize), remove_arcs_exceeding=False) # disregarded train capacity

        if i % 100 == 0:
            if print_out:
                print(' iterations completed : ', i, ' | ', len(odt_for_sp), ' in ', time.time() - tic, ' [s]')
            tic = time.time()

    if print_out:
        print(' sp done for ', len(odt_for_sp), ' ODs in ', (time.time() - tic1) / 60, '[min]')
        print(nr_src_not_connected, 'unconnected srcs')
        print(nr_tgt_not_connected, 'unconnected tgts')
    return G_prime, served_unserved_pass, total_traveltime



def scipy_dijkstra_full_od_list(G, odt_list):
    compare_with_nx = False
    nr_unequal_paths = 0
    M, index, index_names = convert.to_scipy_sparse_matrix(G)
    dist_matrix, preds = scipy.sparse. .dijkstra(M, directed=True, return_predecessors=True)
    for odt in odt_list:
        index_source = index[odt[0]]
        index_target = index[odt[1]]
        length = dist_matrix[index_source, index_target]
        path = get_path(preds, index_source, index_target, index_names)
        # preds[index_target]
        if compare_with_nx:
            try:
                l, p = nx.single_source_dijkstra(G, odt[0], odt[1])
                if p != path and abs(l - length) > 0.2:
                    nr_unequal_paths += 1
                    # print('nx')
                    # print(l, p)
                    # print('scipy')
                    # print(length, path)
            except nx.exception.NetworkXNoPath:
                pass


    print(nr_unequal_paths)




def get_path(Pr, i, j, index_names):
    path = [index_names[j]]
    k = j
    while Pr[i, k] != -9999:
        path.append(index_names[Pr[i, k]])
        k = Pr[i, k]
    return path[::-1]


def assign_pass_and_remove_arcs_exceeded_capacity(G_prime, parameters, path_odt, target_groupsize, remove_arcs_exceeding=False):
    for idx in range(1, len(path_odt) - 2):
        if G_prime[path_odt[idx]][path_odt[idx + 1]]['type'] in ['driving', 'waiting']:
            G_prime[path_odt[idx]][path_odt[idx + 1]]['flow'] += target_groupsize[1]
            if not remove_arcs_exceeding:
                continue
            if 'bus' in G_prime[path_odt[idx]][path_odt[idx + 1]].keys():
                if G_prime[path_odt[idx]][path_odt[idx + 1]]['flow'] >= parameters.bus_capacity:
                    G_prime[path_odt[idx]][path_odt[idx + 1]]['initial_weight'] = \
                        G_prime[path_odt[idx]][path_odt[idx + 1]]['weight']
                    G_prime[path_odt[idx]][path_odt[idx + 1]]['weight'] = parameters.weight_closed_tracks

            elif G_prime[path_odt[idx]][path_odt[idx + 1]]['flow'] >= parameters.train_capacity:
                G_prime[path_odt[idx]][path_odt[idx + 1]]['initial_weight'] = G_prime[path_odt[idx]][path_odt[idx + 1]]['weight']
                G_prime[path_odt[idx]][path_odt[idx + 1]]['weight'] = parameters.weight_closed_tracks
    return G_prime


def shortest_paths_lengths_dijkstra(graph,  sources, target, weight='weight'):
    lengths = {}
    paths = {}
    print('\nStart of shortest paths calculation')
    tic = time.time()
    for i in range(0, sources.shape[0]):
        source = sources[i]
        lengths[source], paths[source] = single_source_dijkstra(graph, source)
        if i % 100 == 0:
            print(i, ' of ', sources.shape[0], ' sources dijkstra completed in ', time.time()-tic, ' seconds')
            tic = time.time()
    return paths, lengths


def separate_reachable_and_unreachable_trips(paths, zonesSelDemand):
    unreached_trips = []
    reached_trips = []
    for trip in zonesSelDemand:
        try:
            path = paths[trip[0]][trip[1]]
            reached_trips.append(list(trip))
        except:
            unreached_trips.append(list(trip))
    print('\nNumber of unservable trips : ', len(unreached_trips), ' / ', zonesSelDemand.shape[0], ' total trips')
    return reached_trips, unreached_trips


def check_to_reach_a_neighbor(frequency_dictionary, graph, unreached_trips):
    unreachable_nb = {}
    c = 0
    d = 0
    for trip in unreached_trips:
        origin = trip[0]
        destination = trip[1]
        x = []
        for nb in graph.neighbors(destination):
            x.append(nb)
        unreachable_nb[destination] = x
        for nb in unreachable_nb[destination]:  # try to reach a neighbor node of destination
            d = d + 1
            try:
                single_source_dijkstra(graph, origin, frequency_dictionary, nb)
            except nx.exception.NetworkXNoPath:
                # print('KeyError: not reachable neighbour node')
                # print('From: ', origin, '  to', nb)
                c = c + 1
    print('unreached trips are tried to reach its neighbour, unsuccessfull for ', c, ' / ', d, 'times')


def assign_path_load(paths, reached_trip):
    path_load = {}
    for trip in reached_trip:
        path = paths[trip[0]][trip[1]]
        for i in np.arange(1, len(path)):
            if (path[i - 1], path[i]) in path_load:
                path_load[(path[i - 1], path[i])] = path_load[(path[i - 1], path[i])] + trip[2]
            else:
                path_load[(path[i - 1], path[i])] = trip[2]

    for k, v in path_load.items():
        path_load[k] = int(round(v))

    return path_load


def _dijkstra_multisource_Backups(G, sources, weight, pred=None, paths=None,
                                  cutoff=None, target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

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

    Raises
    ------
    NodeNotFound
        If any of `sources` is not in `G`.

    Notes
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original pred and paths objects passed
    as arguments. No need to explicitly return pred or paths.

    """

    # get successor of source nodes
    G_succ = G._succ if G.is_directed() else G._adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        sourceForSuccessor = False  # to make sure that for the source, it is looking for successor nodes
        if source not in G:
            raise nx.NodeNotFound("Source {} not in G".format(source))
        seen[source] = 0
        push(fringe, (0, next(c), source))

    while fringe:
        (d, _, v) = pop(fringe)  # Pop and return smallest item from the heap, remove node from queue
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():  # u is  successor node with weight e, cond u = '85WKM'
            cost = weight(v, u, e)  # cost to reach actual nodes
            if cost is None:
                continue
            vu_dist = dist[v] + cost
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                if target is not None:
                    if G.node[u]['name'] != 'homeZones':
                        push(fringe, (vu_dist, next(c), u))  # add the node to the queue, if its no home
                    elif u == target:
                        push(fringe, (vu_dist, next(c), u))  # add the node to the queue, if its no home
                else:
                    push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return dist


def _dijkstra_multisource_nx(G, sources, weight, pred=None, paths=None,
                          cutoff=None, target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

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

    Raises
    ------
    NodeNotFound
        If any of `sources` is not in `G`.

    Notes
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original pred and paths objects passed
    as arguments. No need to explicitly return pred or paths.

    """
    G_succ = G._succ if G.is_directed() else G._adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        if source not in G:
            raise nx.NodeNotFound("Source {} not in G".format(source))
        seen[source] = 0
        push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():
            cost = weight(v, u, e)
            if cost is None:
                continue
            # if isinstance(cost, datetime.timedelta):
            #     cost = cost.seconds/60  # weights in minutes
            # try:
            vu_dist = dist[v] + cost
            # except TypeError:
            #     print('hi')
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return dist

