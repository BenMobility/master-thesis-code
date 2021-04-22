import \
    networkx as nx
import utils
import numpy as np
import datetime
import treatingZones
import pickle
import algorithm_platform_methods as ap
import copy
import Classes

'''
idx_zones_for_nodes = np.argpartition(subArrayTravelTime.tt, threshold_tt)[0:threshold_tt]
zones_for_nodes = subArrayTravelTime[idx_zones_for_nodes]
stations_for_edges = nodesSBB.ViriatoID[
    np.isin(nodesSBB.zone, subArrayTravelTime.toZone)]

for j in zones_for_nodes:
    zStation = nodesSBB[nodesSBB.zone == j.toZone]
    # get travel time and add it to the weights
    for station in zones_for_nodes:
        tt = tt_array_reduced.tt[tt_array_reduced.toZone == zStation]
        weights.append(weights, tt)
'''

def create_restored_feasibility_graph(trains, parameters):
    stations_in_area = parameters.stations_in_area

    G, parameters.stations_comm_stop = create_graph_with_transit_nodes_and_edges(trains, stations_in_area)

    G = add_transfer_edges_to_graph(G, parameters)
    # pr = cProfile.Profile()
    # pr.enable()
    # ... do something ...
    # main_test_parts()
    edges_o_stations_d = connections_homes_with_station_candidates(G, parameters)

    # pr.disable()
    # s = io.StringIO()

    # sortby_selected = 'cumulative'  # 'totalTime'
    # ps = pstats.Stats(pr, stream=s).sort_stats('totalTime')
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby_selected)
    # ps = pstats.Stats(pr).sort_stats(sortby)
    # ps.print_stats()
    # with open('Profile_connect_home_station.txt', 'w+') as f:
    #    f.write(s.getvalue())
    return G, edges_o_stations_d


def create_or_load_graph_with_waiting_driving_transfer_edges(M, create_graph_from_scratch, cut_trains, m, p_graph,
                                                             parameters):
    if create_graph_from_scratch:
        # transit nodes (trains arriving, departing) and edges (waiting and driving at stations)
        G, nodes_commercial_stops = create_graph_with_transit_nodes_and_edges(cut_trains, parameters.stations_in_area)

        np.savetxt('Input/Graph/nodes_commercial_stops' + str(int(m.total_seconds() / 60)) + '_M' + str(
        int(M.total_seconds() / 60)) + '.csv', nodes_commercial_stops)
        # create di graph with transit nodes and edges (waiting, driving)

        # up until this line there are only trains in the graph
        G = add_transfer_edges_to_graph(G, parameters) 
        # save graph as pickle file
        nx.write_gpickle(G, p_graph)
    else:
        G = nx.read_gpickle(p_graph)
        nodes_commercial_stops = np.genfromtxt('Input/Graph/nodes_commercial_stops' + str(int(m.total_seconds() / 60)) + '_M' + str(
        int(M.total_seconds() / 60)) + '.csv', dtype='i')
    return G, nodes_commercial_stops


def connect_home_stations(G, nodesSBB, thNRofStations, thNrofZones, zonesSelTT, zonesSelected, id_code_dictionary,
                          od_departure_time, p_graph_homes_connected, p_odt_for_sp, create_from_scratch, parameters):
    '''
    :param G: Graph
    :param nodesSBB: stations
    :param thNRofStations: ...
    :param thNrofZones: ...
    :param zonesSelTT: ...
    :param zonesSelTT_internal:
    :param zonesSelected:
    :return: adds a list edges (origin, destinations) to the graph, connection between the stations and zones with tt as weight in minutes
    '''
    debug = True
    # Todo remove debug mode
    if not create_from_scratch:
        if debug:
            p_graph_homes_connected = p_graph_homes_connected[:-7] + '_DEBUG' + p_graph_homes_connected[-7:]
        G = nx.read_gpickle(p_graph_homes_connected)
        # Unpickling odt for sp
        if debug:
            filename = p_odt_for_sp + '_odt_for_sp_DEBUG.pickle'
        else:
            filename = p_odt_for_sp + '_odt_for_sp.pickle'
        with open(filename, "rb") as fp:
            odt_list = pickle.load(fp)
        # Unpickling odt for sp passengers
        if debug:
            filename = p_odt_for_sp + '_odt_for_sp_passengers_DEBUG.pickle'
        else:
            filename = p_odt_for_sp + '_odt_for_sp_passengers.pickle'
        with open(filename, "rb") as fp:
            odt_by_origin = pickle.load(fp)

    else:

        destination_nodes = []
        destination_nodes_attributes = {}

        edges_d_stations = list()


        k2 = thNrofZones
        k = thNRofStations

        # remove all stations without commercial stop
        nodesSBB = nodesSBB[nodesSBB.commercial_stop == 1]

        # identify zones where trips start
        all_origin_zones = list(np.unique(od_departure_time.fromZone))
        # identify zones where trips end
        all_destination_zones = list(np.unique(od_departure_time.toZone))

        # get all the k closest euclidean stations to a zone and transform it into a record array
        closest_stations_to_zone = treatingZones.get_all_k_closest_stations_to_zone(nodesSBB, zonesSelected, k)
        closest_stations_to_zone = closest_stations_to_zone_transform_record(closest_stations_to_zone)

        # connect all origins of an odt with the departure nodes and create odt list for sp input
        G, odt_list, odt_by_origin, odt_by_dest, station_candidates, origin_name_desired_dep_time, origin_name_zone_dict = \
            connect_origins_with_stations_for_all_odt(G, all_origin_zones, k2, parameters, nodesSBB,
                                                      od_departure_time, zonesSelTT, closest_stations_to_zone)

        G, station_candidates = connect_all_destinations_with_stations(G, all_destination_zones, closest_stations_to_zone, destination_nodes,
                                                   destination_nodes_attributes, edges_d_stations, id_code_dictionary, k2,
                                                   nodesSBB, zonesSelTT, station_candidates)

    return G, odt_list, odt_by_origin, odt_by_dest, station_candidates, origin_name_desired_dep_time, origin_name_zone_dict


def connections_homes_with_station_candidates(G, parameters):

    # odt_as_list = parameters.odt_as_list

    min_wait = parameters.min_wait
    max_wait = parameters.max_wait
    # remove all stations without commercial stop

    # connect all origins of an odt with the departure nodes and create odt list for sp input
    edges_o_stations_d = Classes.Edges_origin_station_destionation(G, max_wait, min_wait, parameters)

    return edges_o_stations_d



def generate_edges_origin_station_destination(G, max_wait, min_wait, parameters, return_edges_nodes = True):
    '''
    :param G: Graph with all train arcs and nodes
    :param max_wait: to connect origin with station departure nodes
    :param min_wait: to connect origin with station departure nodes
    :param parameters: ...
    :param return_edges_nodes: default True, otherwise add the edges to the graph and return G
    :return: edges and nodes or fully loaded graph
    '''
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

    departing_trains_by_node, arriving_trains_by_node = get_all_departing_and_arriving_nodes_at_stations(G)

    for origin_zone, trips_origin in odt.items():  # loop trough all home zones where trips are starting

        for od_pair, group_size in trips_origin.items():
            origin_node_name = od_pair[0]
            destination_node_name = od_pair[1]  # Is the same as the zone of the destination

            desired_dep_time = origin_name_desired_dep_time[(origin_node_name, destination_node_name)]
            if origin_node_name not in origin_nodes:  # add the node to the list of nodes
                origin_nodes.append(origin_node_name)
                attributes = {'zone': origin_zone, 'departure_time': desired_dep_time, 'type': 'origin'}
                origin_nodes_attributes[origin_node_name] = attributes

            # edges for departure at origin
            for station_code, station_values in station_candidates[origin_zone].items():
                try:
                    tt_to_station = station_values['tt_toStation']
                except KeyError:
                    tt_to_station = 50
                    print('tt not in data, set to 50m ', origin_zone, ' ', station_values[station_code]['station_zone'])

                # identify train departure nodes candidates at stations
                time_at_station = datetime.datetime.strptime(desired_dep_time, "%Y-%m-%dT%H:%M") + \
                                  datetime.timedelta(minutes=int(tt_to_station))

                departure_nodes_of_station = identify_dep_nodes_for_trip_at_station(G, parameters, time_at_station,
                                                                                    departing_trains_by_node[station_code])

                # add the edges of origin to departures at the station with weight of travel time
                for d in departure_nodes_of_station:
                    edge = [origin_node_name, d, tt_to_station]
                    attributes = {'type': 'origin_station'}
                    if edge not in edges_o_stations:
                        edges_o_stations.append(edge)
                        edges_o_stations_attr[(origin_node_name, d)] = attributes

            # add the destination nodes and edges connecting with stations if not added yet
            if destination_node_name in destination_nodes:
                continue
            else:  # add the node to the list of nodes
                destination_nodes.append(destination_node_name)
                attributes = {'zone': destination_node_name, 'type': 'destination'}
                destination_nodes_attributes[destination_node_name] = attributes
            for station_code, station_values in station_candidates[destination_node_name].items():
                try:
                    tt_to_station = station_values['tt_toStation']
                except KeyError:
                    tt_to_station = 50
                    print('tt not in data, set to 50m ', destination_node_name, ' ', station_values[station_code]['station_zone'])

                # add the edges of origin to departures at the station with weight of travel time
                try:
                    for a in arriving_trains_by_node[station_code]:  # all trains arriving station candidate
                        edge = [a, destination_node_name, tt_to_station]
                        attributes = {'type': 'station_destination'}
                        if edge not in edges_stations_d:
                            edges_stations_d.append(edge)
                            edges_stations_d_attr[(a, destination_node_name)] = attributes
                except KeyError:
                    # print('no arrival at station : ', station_code)
                    # no train arrives at this station
                    pass

    if not return_edges_nodes:
        # add origin nodes and attributes
        G.add_nodes_from(origin_nodes)
        nx.set_node_attributes(G, origin_nodes_attributes)
        G.add_nodes_from(destination_nodes)
        nx.set_node_attributes(G, destination_nodes_attributes)

        # add the edges of an origin node to all departing nodes of the selected stations and the same for stations to destination
        G.add_weighted_edges_from(edges_o_stations)
        nx.set_edge_attributes(G, edges_o_stations_attr)
        G.add_weighted_edges_from(edges_stations_d)
        nx.set_edge_attributes(G, edges_stations_d_attr)
        return G
    else:
        edges_o_stations_dict = utils.transform_edges_into_dict_key_source(edges_o_stations)
        edges_stations_d_dict = utils.transform_edges_into_dict_key_target(edges_stations_d)

        return origin_nodes, origin_nodes_attributes, destination_nodes, destination_nodes_attributes, edges_o_stations,\
                edges_o_stations_attr, edges_stations_d, edges_stations_d_attr, edges_o_stations_dict, edges_stations_d_dict



def connect_origins_with_stations_for_all_odt(G, all_origin_zones, k2, parameters, nodesSBB, od_departure_time,
                                              zonesSelTT, closest_stations_to_zone):
    '''
    :param G:
    :param all_origin_zones:
    :param edges_o_stations:
    :param k:
    :param k2:
    :param max_wait:
    :param min_wait:
    :param nodesSBB:
    :param od_departure_time:
    :param odt_list:
    :param odt_by_origin:
    :param origin_nodes:
    :param origin_nodes_attributes:
    :param zonesSelTT:
    :param zonesSelected:
    :return: Graph, odt for sp algorithm, odt_dictionary with information about passengers on a trip
    '''

    origin_node_name_zone_dict = dict()
    origin_nodes = []
    origin_nodes_attributes = {}
    edges_o_stations = list()
    odt_list = list()
    odt_by_origin = {}
    odt_by_dest = {}

    station_id_zone_dict = {}
    for station in nodesSBB:
        # id = code_id_dictionary[station.Code]
        station_id_zone_dict[station.Code] = station.zone
    departing_trains_by_node = dict()  # dictionary with key : station & value : all departing nodes

    for x, y in G.nodes(data=True):
        if y['type'] in ['arrivalNode', 'arrivalNodePassing', 'departureNodePassing']:
            continue
        try:
            zone = station_id_zone_dict[x[0]]
        except KeyError:
            print('Node', x[0], ' is not in Area of Interest, pls doublecheck that all train stops are in Area of Interest')

        if not departing_trains_by_node.__contains__(x[0]):
            departing_trains_by_node[x[0]] = list()
        dep_nodes_in_dict = departing_trains_by_node[x[0]]
        dep_nodes_in_dict.append(x)

    zone_station_dep_nodes = dict()
    station_candidates = dict()
    zone_candidates = dict()
    origin_name_desired_dep_time = {}

    for origin_zone in all_origin_zones:  # loop trough all home zones where trips are starting
        # tt fom origin zones to all stations
        tt_from_origin_zone = zonesSelTT[zonesSelTT.fromZone == origin_zone]
        tt_from_origin_zone = tt_from_origin_zone[np.isin(tt_from_origin_zone.toZone, nodesSBB.zone)]

        if not zone_station_dep_nodes.__contains__(origin_zone):
            # identify stations candidates of origin zone
            stations_candidates_origin = identify_stations_candidates(closest_stations_to_zone, k2, nodesSBB,
                                                                      origin_zone, tt_from_origin_zone, station_candidates)
            zone_station_dep_nodes[origin_zone] = dict()
            if origin_zone in stations_candidates_origin.keys():
                for station_code, v in stations_candidates_origin[origin_zone].items():
                    if station_code in departing_trains_by_node.keys():
                        zone_station_dep_nodes[origin_zone][station_code] = departing_trains_by_node[station_code]



        # todo tt may be not in the matrix
        # get all odt starting in the origin zone
        odt_of_origin_zone = od_departure_time[np.isin(od_departure_time.fromZone, origin_zone)]

        odt_dict_o_d_groupSize = {}  # dictionary for the group size
        for odt in odt_of_origin_zone:
            origin_node_name = str(odt.fromZone) + '_' + odt.desired_dep_time[-5:]
            odt_list.append([origin_node_name, odt.toZone, odt.priority, odt.group_size])  # used as an input to the sp algorithm
            origin_name_desired_dep_time[(origin_node_name, odt.toZone)] = odt.desired_dep_time
            origin_node_name_zone_dict[origin_node_name] = origin_zone

            if origin_node_name not in origin_nodes:  # add the node to the list of nodes
                origin_nodes.append(origin_node_name)
                attributes = {'zone': origin_zone, 'departure_time': odt.desired_dep_time, 'type': 'origin'}
                origin_nodes_attributes[origin_node_name] = attributes

        if origin_zone in zone_station_dep_nodes.keys():
            for station_code, value in zone_station_dep_nodes[origin_zone].items():
                try:
                    tt_to_station = stations_candidates_origin[origin_zone][station_code]['tt_toStation']
                except KeyError:
                    tt_to_station = 50
                    print('tt not in data, manually set to 50 min ', origin_zone, ' ', stations_candidates_origin[origin_zone][station_code]['station_zone'])

                # identify train departure nodes candidates at stations
                time_at_station = datetime.datetime.strptime(odt.desired_dep_time, "%Y-%m-%dT%H:%M") +\
                                                            datetime.timedelta(minutes=int(tt_to_station))

                departure_nodes_of_station = identify_dep_nodes_for_trip_at_station(G, parameters, time_at_station,
                                                                                    value)

                # add the edges of origin to departures at the station with weight of travel time
                for d in departure_nodes_of_station:
                    edge = [origin_node_name, d, tt_to_station]
                    if edge not in edges_o_stations:
                        edges_o_stations.append(edge)

            # transform odt matrix into a list with adapted node names
             #odt_for_sp.append([origin_node_name, odt.toZone])
            odt_dict_o_d_groupSize[origin_node_name, odt.toZone] = odt.group_size
            if not odt_by_dest.__contains__(odt.toZone):
                odt_by_dest[odt.toZone] = {}
            odt_by_dest[odt.toZone].update({(origin_node_name, odt.toZone): odt.group_size})

        odt_by_origin[origin_zone] = odt_dict_o_d_groupSize

    # add the edges and nodes to the graph, therefore connect the origins to all departing nodes of a station
    # add origin nodes and attributes
    # todo, uncomment to add the nodes and edges to full graph
    G.add_nodes_from(origin_nodes)
    nx.set_node_attributes(G, origin_nodes_attributes)
    # add the edges of an origin node to all departing nodes of the selected stations and the same for stations to destination
    G.add_weighted_edges_from(edges_o_stations)

    return G, odt_list, odt_by_origin, odt_by_dest, stations_candidates_origin, origin_name_desired_dep_time, origin_node_name_zone_dict


def connect_all_destinations_with_stations(G, all_destination_zones, closest_stations_to_zone, destination_nodes,
                                           destination_nodes_attributes, edges_d_stations, id_code_dictionary, k2,
                                           nodesSBB, zonesSelTT, station_candidates):
    '''
        for x, y in G.nodes(data=True):
        if y['type'] in ['arrivalNode', 'arrivalNodePassing', 'departureNodePassing']:
            continue
        try:
            zone = station_id_zone_dict[x[0]]
        except KeyError:
            print('Node', x[0], ' is not in Area of Interest, pls doublecheck that all train stops are in Area of Interest')

        if not departing_trains_by_node.__contains__(x[0]):
            departing_trains_by_node[x[0]] = list()
        dep_nodes_in_dict = departing_trains_by_node[x[0]]
        dep_nodes_in_dict.append(x)

    '''
    arriving_trains_by_node = dict()  # dictionary with key : station & value : all arriving nodes

    for x, y in G.nodes(data=True):
        if y['type'] in ['departureNode', 'arrivalNodePassing', 'departureNodePassing']:
            continue
        if not arriving_trains_by_node.__contains__(x[0]):
            arriving_trains_by_node[x[0]] = list()
        arr_nodes_in_dict = arriving_trains_by_node[x[0]]
        arr_nodes_in_dict.append(x)

    # connect all destinations with train arrival nodes
    for destination_zone in all_destination_zones:
        # add the destination to the list of nodes for the graph
        destination_nodes.append(destination_zone)
        attributes = {'zone': destination_zone, 'type': 'destination'}
        destination_nodes_attributes[destination_zone] = attributes

        # tt fom destination zones to all stations
        tt_from_destination_zone = zonesSelTT[zonesSelTT.fromZone == destination_zone]
        tt_from_destination_zone = tt_from_destination_zone[np.isin(tt_from_destination_zone.toZone, nodesSBB.zone)]

        stations_candidates = identify_stations_candidates(closest_stations_to_zone, k2, nodesSBB, destination_zone,
                                                           tt_from_destination_zone, station_candidates)
        # identify stations candidates of destination zone

        # loop through station candidates and link station with departure
        for station_code, value in stations_candidates[destination_zone].items():
            # identify train arrival nodes and add the edges
            # arrival_nodes_of_station = [x for x, y in G.nodes(data=True) if y['type'] == 'arrivalNode' and
            #                            x[0] == station_code]

            try:
                tt_to_station = stations_candidates[destination_zone][station_code]['tt_toStation']
            except KeyError:
                tt_to_station = 50
                print('tt not in data, manually set to 50 min ', destination_zone, ' ', stations_candidates[destination_zone][station_code]['station_zone'])
            try:
                for a in arriving_trains_by_node[station_code]:
                    edge = [a, destination_zone, tt_to_station]
                    if edge not in edges_d_stations:
                        edges_d_stations.append(edge)
            except KeyError:
                # print('no arrival at station : ', station_code)
                # no train arrives at this station
                pass
    G.add_nodes_from(destination_nodes)
    nx.set_node_attributes(G, destination_nodes_attributes)
    # add the edges of an origin node to all arriving nodes of the selected station
    G.add_weighted_edges_from(edges_d_stations)
    # G.has_edge('9a22031', 2)
    return G, station_candidates


def identify_dep_nodes_for_trip_at_station(G, parameters, time_at_station, value):
    departure_nodes_of_station = list()

    for depart in value:
        if time_at_station + parameters.min_wait <= G.nodes[depart]['departureTime'] <= time_at_station + parameters.max_wait:
            departure_nodes_of_station.append(depart)
        else:
            continue
    return departure_nodes_of_station


def identify_stations_candidates(closest_stations_to_zone, k2, nodesSBB, zone, tt_from_origin_zone, station_candidates):
    # by euclidean distance
    if not station_candidates.__contains__(zone):
        # copy is needed as we do not want to adapt the closest stations to zone dict
        station_candidates[zone] = copy.deepcopy(closest_stations_to_zone[zone])
    else:
        return station_candidates

    for station_code, value in station_candidates[zone].items():
        try:
            tt_origin_station = tt_from_origin_zone.tt[np.isin(tt_from_origin_zone.toZone, value['station_zone'])].item()
        except ValueError:
            with open('tt_to_add.txt', 'a') as file:
                file.write('\n from ' + str(tt_from_origin_zone) + ' to ' + str(value['station_zone']))
            print('tt no in data, in identify stations candidates, line 292 in graphcreator')
            continue
        station_candidates[zone][station_code].update({'tt_toStation': tt_origin_station})

        # stations_candidates.append([origin_zone, station['station_zone'], station['station_ID'], station['station_code'], tt_origin_station])


    # get all k2 closest zone with a station to origin zone according to the travel time
    if k2 < tt_from_origin_zone.tt.size:
        index_closestTravelTime = np.argpartition(tt_from_origin_zone.tt, k2)[0:k2]

        tt_from_origin_zone = tt_from_origin_zone[index_closestTravelTime]
        stations_tt = nodesSBB[nodesSBB.zone == tt_from_origin_zone.toZone]
        for station in stations_tt:
            candidate_tt = {'station_zone': tt_from_origin_zone.toZone.item(), 'station_code': station.Code,
                        'station_ID': station.ViriatoID, 'tt_toStation': float(tt_from_origin_zone.tt)}
            if station.Code in station_candidates[zone].keys():
                continue
            station_candidates[zone][station.Code] = candidate_tt
        # stations_candidates.append([tt_from_origin_zone.fromZone.item(), tt_from_origin_zone.toZone.item(), station,
        #                            tt_from_origin_zone.tt.item()])



    return station_candidates


# access and egress edges
def create_edges_to_connect_od_with_stations(G, edges_od_stations, id_code_dictionary):
    '''
    :param G: Graph
    :param edges_od_stations: edges with the o/d and all the closest stations according to tt and euclidean
    :param id_code_dictionary: converts the id of a sbb node into the code used in viriato
    :return: edges from origin to all departure node stations and arrival nodes of a station with the destinations
    '''
    edges_origins_stations_departing_nodes = list()
    edges_stations_arrival_nodes_destination = list()

    for edges_of_origin in edges_od_stations:
        origin = edges_of_origin[0]
        station = edges_of_origin[1]
        station_code = id_code_dictionary[station]
        tt_weight = edges_of_origin[2]
        # identify all departure nodes in the graph
        departure_nodes_of_station = [x for x, y in G.nodes(data=True) if y['type'] == 'departureNode' and
                                      y['station'] == station_code]

        arrival_nodes_of_station = [x for x, y in G.nodes(data=True) if y['type'] == 'arrivalNode' and
                                      y['station'] == station_code]
        # connect origins with departure nodes
        edges = list(zip(np.ones(len(departure_nodes_of_station), dtype=int) * origin, departure_nodes_of_station,
                         np.ones(len(departure_nodes_of_station), dtype=int) * tt_weight))
        edges_origins_stations_departing_nodes.extend(edges)

        # connect arrival nodes of a station with the destinations
        edges = list(zip(arrival_nodes_of_station, np.ones(len(arrival_nodes_of_station), dtype=int) * origin,
                         np.ones(len(arrival_nodes_of_station), dtype=int) * tt_weight))
        edges_stations_arrival_nodes_destination.extend(edges)
    return edges_origins_stations_departing_nodes, edges_stations_arrival_nodes_destination


def closest_stations_to_zone_transform_record(closest_stations_to_zone):

    closest_stations_to_zone_dictionary = {}
    closest_zones_to_station_dictionary = {}

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


# create transit nodes (arrival, departure) and edges (waiting, driving)
def create_graph_with_transit_nodes_and_edges(cut_trains, stations_in_area):

    '''
    :param cut_trains: all trains travelling on nodes
    takes all driving trains and creates the transit nodes arrival and departures at a station
    N (s,t,k) with s station, t time for arrival / departure, k trainID
    :param stations_in_area:
    :return: Nodes and attributes of the transit nodes for the creation of the graph
    '''
    stations_with_commercial_stop = []
    arrival_nodes = []
    arrival_nodes_attributes = {}
    arrival_nodes_passing = []
    arrival_nodes_passing_attributes = {}
    departure_nodes = []
    departure_nodes_attributes = {}
    departure_nodes_passing = []
    departure_nodes_passing_attributes = {}
    driving_edges = list()
    driving_edges_attributes = {}
    waiting_edges = list()
    waiting_edges_attributes = {}

    # train_debugString_id_dictionary = {}  # key debugString, value ID
    train_information = {}
    n = 0  # nr of trains
    debug = False

    for train in cut_trains:  # loop through all trains

        n = n + 1
        # print(n)
        s = 0  # nr of stop of a train
        total_train_length = len(train['TrainPathNodes'])
        train_id = train['ID']
        start_of_train = True  #
        # train_left_area = False
        # print(train['DebugString'], train['ID'])
        if n > 100 and debug:
            break  # for debugging, exit loop after n trains
        # departure_node = train['TrainPathNodes'][0]['NodeID']
        # print(train)
        for train_path_nodes in train['TrainPathNodes']:  # loop trough all path nodes of a train
            s = s + 1
            if train_path_nodes['NodeID'] not in stations_in_area:
                # train_left_area = True
                continue

            if train_path_nodes['StopStatus'] == 'commercialStop':  # consider only when train stops
                stations_with_commercial_stop.append(train_path_nodes['NodeID'])
                # update time and node
                arrival_time_this_node = datetime.datetime.strptime(train_path_nodes['ArrivalTime'], "%Y-%m-%dT%H:%M:%S")
                arrival_node_this_node = train_path_nodes['NodeID']
                departure_time_this_node = datetime.datetime.strptime(train_path_nodes['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
                departure_node_this_node = train_path_nodes['NodeID']

                if start_of_train:
                    # node_name_dep_this = str(departure_node_this_node) + 'd' + str(train_id)
                    node_name_dep_this = (departure_node_this_node, train_path_nodes['DepartureTime'], train_path_nodes['ID'], 'd')
                    departure_nodes.append(node_name_dep_this)
                    attributes = {'train': train_id, 'type': 'departureNode', 'departureTime': departure_time_this_node, 'StopStatus': train_path_nodes['StopStatus']}
                    departure_nodes_attributes[node_name_dep_this] = attributes
                    departure_time_last_node = departure_time_this_node
                    departure_node_last_node = departure_node_this_node
                    departure_node_last_node_name = node_name_dep_this
                    start_of_train = False
                    continue

                elif s == total_train_length:  # end of a train
                    # node_name_arr_this = str(arrival_node_this_node) + 'a' + str(train_id)
                    node_name_arr_this = (arrival_node_this_node, train_path_nodes['ArrivalTime'], train_path_nodes['ID'], 'a')
                    # arrival_nodes.append(node_name_arr_this)
                    arrival_nodes.append(node_name_arr_this)
                    attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node, 'StopStatus': train_path_nodes['StopStatus']}
                    arrival_nodes_attributes[node_name_arr_this] = attributes
                    # driving edge to this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds/60)])
                    driving_edges_attributes[(departure_node_last_node_name, node_name_arr_this)] = {'flow': 0, 'type': 'driving', 'train_id': train_id}

                    del departure_node_last_node
                    del node_name_dep_this
                    del node_name_arr_this
                    del departure_time_last_node

                else:  # node in between two transit nodes

                    # arrival nodes
                    # node_name_arr_this = str(arrival_node_this_node) + 'a' + str(train_id)
                    node_name_arr_this = (arrival_node_this_node, train_path_nodes['ArrivalTime'], train_path_nodes['ID'], 'a')
                    arrival_nodes.append(node_name_arr_this)
                    attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node, 'StopStatus': train_path_nodes['StopStatus']}
                    arrival_nodes_attributes[node_name_arr_this] = attributes
                    # driving edge between last node and this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds/60)])
                    driving_edges_attributes[(departure_node_last_node_name, node_name_arr_this)] = {'flow': 0, 'type': 'driving', 'train_id': train_id}

                    # departure nodes
                    # node_name_dep_this = str(departure_node_this_node) + 'd' + str(train_id)
                    node_name_dep_this = (departure_node_this_node, train_path_nodes['DepartureTime'], train_path_nodes['ID'], 'd')
                    departure_nodes.append(node_name_dep_this)
                    attributes = {'train': train_id, 'type': 'departureNode', 'departureTime': departure_time_this_node, 'StopStatus': train_path_nodes['StopStatus']}
                    # attributes = {'station': departure_node_this_node, 'departureTime': departure_time_this_node, 'train': train_id, 'type': 'departureNode'}
                    departure_nodes_attributes[node_name_dep_this] = attributes
                    # waiting edge between last node and this node
                    wait_time = (departure_time_this_node - arrival_time_this_node)
                    waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds/60)])
                    waiting_edges_attributes[(node_name_arr_this, node_name_dep_this)] = {'flow': 0, 'type': 'waiting', 'train_id': train_id}

                    # update the departure node for next iteration
                    departure_node_last_node = departure_node_this_node
                    departure_node_last_node_name = node_name_dep_this
                    departure_time_last_node = departure_time_this_node
            # passing nodes
            elif train_path_nodes['StopStatus'] == 'passing':
                arrival_time_this_node = datetime.datetime.strptime(train_path_nodes['ArrivalTime'], "%Y-%m-%dT%H:%M:%S")
                arrival_node_this_node = train_path_nodes['NodeID']
                departure_time_this_node = datetime.datetime.strptime(train_path_nodes['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
                departure_node_this_node = train_path_nodes['NodeID']
                if start_of_train:  #
                    # node_name_dep_this = str(departure_node_this_node) + 'dp' + str(train_id)
                    # departure_nodes_passing.append(node_name_dep_this)
                    # attributes = {'station': departure_node_this_node, 'departureTime': departure_time_this_node, 'train': train_id, 'type': 'departureNodePassing'}
                    node_name_dep_this = (departure_node_this_node, train_path_nodes['DepartureTime'], train_path_nodes['ID'], 'dp')
                    departure_nodes.append(node_name_dep_this)
                    attributes = {'train': train_id, 'type': 'departureNodePassing', 'departureTime': departure_time_this_node, 'StopStatus': train_path_nodes['StopStatus']}
                    departure_nodes_passing_attributes[node_name_dep_this] = attributes
                    departure_time_last_node = departure_time_this_node
                    departure_node_last_node = departure_node_this_node
                    departure_node_last_node_name = node_name_dep_this
                    start_of_train = False
                    continue

                elif s == total_train_length:  # end of a train
                    # node_name_arr_this = str(arrival_node_this_node) + 'ap' + str(train_id)
                    # arrival_nodes.append(node_name_arr_this)
                    # attributes = {'station': arrival_node_this_node, 'arrivalTime': arrival_time_this_node, 'train': train_id, 'type': 'arrivalNodePassing'}
                    node_name_arr_this = (arrival_node_this_node, train_path_nodes['ArrivalTime'], train_path_nodes['ID'], 'ap')
                    arrival_nodes.append(node_name_arr_this)
                    attributes = {'train': train_id, 'type': 'arrivalNodePassing', 'arrivalTime': arrival_time_this_node, 'StopStatus': train_path_nodes['StopStatus']}
                    arrival_nodes_passing_attributes[node_name_arr_this] = attributes
                    # driving edge to this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds/60)])
                    driving_edges_attributes[(departure_node_last_node_name, node_name_arr_this)] = {'flow': 0, 'type': 'driving', 'train_id': train_id}

                    del departure_node_last_node
                    del node_name_dep_this
                    del node_name_arr_this
                    del departure_time_last_node

                else:  # node in between two transit nodes
                    node_name_arr_this = (arrival_node_this_node, train_path_nodes['ArrivalTime'], train_path_nodes['ID'], 'ap')
                    arrival_nodes.append(node_name_arr_this)
                    attributes = {'train': train_id, 'type': 'arrivalNodePassing', 'arrivalTime': arrival_time_this_node, 'StopStatus': train_path_nodes['StopStatus']}

                    arrival_nodes_passing_attributes[node_name_arr_this] = attributes
                    # driving edge between last node and this node
                    run_time = arrival_time_this_node - departure_time_last_node
                    driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds/60)])
                    driving_edges_attributes[(departure_node_last_node_name, node_name_arr_this)] = {'flow': 0, 'type': 'driving', 'train_id': train_id}

                    node_name_dep_this = (departure_node_this_node, train_path_nodes['DepartureTime'], train_path_nodes['ID'], 'dp')
                    departure_nodes.append(node_name_dep_this)
                    attributes = {'train': train_id, 'type': 'departureNodePassing', 'departureTime': departure_time_this_node, 'StopStatus': train_path_nodes['StopStatus']}
                    departure_nodes_passing_attributes[node_name_dep_this] = attributes
                    # waiting edge between last node and this node
                    wait_time = (departure_time_this_node - arrival_time_this_node)
                    waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds/60)])
                    waiting_edges_attributes[(node_name_arr_this, node_name_dep_this)] = {'flow': 0, 'type': 'waiting', 'train_id': train_id}

                    # update the departure node for next iteration
                    departure_node_last_node = departure_node_this_node
                    departure_node_last_node_name = node_name_dep_this
                    departure_time_last_node = departure_time_this_node

    print('\nTransit nodes created  \n'
          'arrival_nodes_attributes[key = NodeID + a + trainID]\n'
          'departure_nodes_attributes[key = NodeID + d + trainID]\n')

    stations_with_commercial_stop = np.unique(np.array(stations_with_commercial_stop))

    G = create_graph_transit_nodes_edges_driving_waiting(arrival_nodes_attributes,
                                                         departure_nodes_attributes,
                                                         arrival_nodes_passing_attributes,
                                                         departure_nodes_passing_attributes,
                                                         driving_edges, driving_edges_attributes,
                                                         waiting_edges, waiting_edges_attributes)

    return G, stations_with_commercial_stop


def create_graph_transit_nodes_edges_driving_waiting(arrival_nodes_attributes, departure_nodes_attributes,
                                                     arrival_nodes_passing_attributes, departure_nodes_passing_attributes,
                                                     driving_edges, driving_edges_attributes,
                                                     waiting_edges, waiting_edges_attributes):

    G = nx.DiGraph()
    G.add_nodes_from(arrival_nodes_attributes)
    nx.set_node_attributes(G, arrival_nodes_attributes)

    G.add_nodes_from(arrival_nodes_passing_attributes)
    nx.set_node_attributes(G, arrival_nodes_passing_attributes)

    G.add_nodes_from(departure_nodes_attributes)
    nx.set_node_attributes(G, departure_nodes_attributes)

    G.add_nodes_from(departure_nodes_passing_attributes)
    nx.set_node_attributes(G, departure_nodes_passing_attributes)

    G.add_weighted_edges_from(driving_edges)
    nx.set_edge_attributes(G, driving_edges_attributes)

    G.add_weighted_edges_from(waiting_edges)
    nx.set_edge_attributes(G, waiting_edges_attributes)
    return G


def add_transfer_edges_to_graph(G, parameters):
    print('start identification of transfer graphs')
    # '85ZMUS' --> 611, '85ZUE --> 638', '45ZSZU --> 13'
    zhHB = ['85ZMUS', '85ZUE', '45ZSZU']
    zhHB_code = [611, 638, 13]

    trains_by_node = dict()

    # create trains by node
    for x, y in G.nodes(data=True):
        if y['type'] in ['arrivalNode', 'arrivalNodePassing', 'departureNodePassing']:
            continue
        # station = y['station'] with old labels!
        station = x[0]
        if not trains_by_node.__contains__(station):
            trains_by_node[station] = list()

        station_in_dict = trains_by_node[station]
        station_in_dict.append([y['train'], x])

    #generalized travel time, penalizing...
    beta_transfer = 10  # minutes
    beta_waiting = 2.5  # min/min

    transfer_edges = list()
    transfer_edges_attributes = dict()
    # identify all arrival nodes
    arrival_nodes = [(x, y) for x, y in G.nodes(data=True) if y['type'] == 'arrivalNode']
    for node, z in arrival_nodes:
        try:
            trains_in_same_node = trains_by_node[node[0]].copy()
            # link all trains in Zurich Mainstation, because there were three different stations for zh
            if node[0] in zhHB_code:
                for zh_node in zhHB_code:
                    if zh_node == node[0]:
                        continue
                    else:
                        trains_in_same_node.extend(trains_by_node[zh_node])
        except KeyError:
            print('This node, ', node[0], ' has an arrival but no departure, probably nodeIDs of ZH_HB has changed')
            continue

        # check for trains that are departing from the same node
        dep_trains_same_node = [[x[0], x[1]] for x in trains_in_same_node if x[0] != z['train']]

        for train_id, label in dep_trains_same_node:
            try:
                delta_t = G.nodes[label]['departureTime'] - G.nodes[node]['arrivalTime']
            except KeyError:
                print('Error in the transfer edge creation, check selection of departing trains of same node, graphcreator.py')
                continue
            if delta_t > parameters.transfer_m and delta_t < parameters.transfer_M:
                weight = delta_t*beta_waiting + datetime.timedelta(minutes=beta_transfer)
                transfer_edges.append([node, label, float(weight.seconds/60)])
                transfer_edges_attributes[(node, label)] = {'type': 'transfer'}

    # print(transfer_edges)
    G.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(G, transfer_edges_attributes)
    return G


def transfer_edges_single_train(G, train, M, m, tpn_index_start_delay, tpns_delay):
    # print('start identification of transfer graphs')
    # '85ZMUS' --> 611, '85ZUE --> 638', '45ZSZU --> 13'
    # (G, train_to_cancelFrom, parameters.transfer_M, parameters.transfer_m, tpn_idx_start_Cancel, 0)
    zhHB = ['85ZMUS', '85ZUE', '45ZSZU']
    zhHB_code = [611, 638, 13]

    comm_stops_of_train = {node['ID']: node['NodeID'] for node in train['TrainPathNodes']
                           if node['StopStatus'] == 'commercialStop' and node['ID'] in tpns_delay}
    # tpns_by_nodeID = utils.build_dict(train['TrainPathNodes'], 'ID')


    zh_station_helper = []
    if any(x in zhHB_code for x in comm_stops_of_train.values()):
        zh_station_helper = zhHB_code

    # n = [x[2] for x,y in G.nodes(data=True) if x[2] in tpns_delay if y['type'] == 'arrivalNode']
    departure_train_candidates_by_node = dict()
    arrival_train_candidates_by_node = dict()
    departure_nodes_delayed_train = dict()
    arrival_nodes_delayed_train = dict()
    for x, y in G.nodes(data=True):
        try:
            if y['train'] == train['ID']:
                if y['type'] == 'arrivalNode' and x[2] in tpns_delay:
                    arrival_nodes_delayed_train[x[2]] = (x, y)
                elif y['type'] == 'departureNode' and x[2] in tpns_delay:
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
                print('Check transfer edges single train, somehow a not train node appears in G')
        except KeyError:
            print(x, y)

    beta_transfer = 10  # minutes
    beta_waiting = 2.5  # min/min
    transfer_edges = list()
    transfer_edges_attributes = dict()

    # identify all arrival nodes
    for tpn, station in comm_stops_of_train.items():
        # select dep and arr candidates of trains if there are any
        arrivals_at_station = []
        departures_at_station = []

        if station in arrival_train_candidates_by_node.keys():
            arrivals_at_station = arrival_train_candidates_by_node[station]
            if station in zh_station_helper:
                for zh in zh_station_helper:
                    if station != zh and zh in arrival_train_candidates_by_node.keys():
                        arrivals_at_station.extend(arrival_train_candidates_by_node[zh])

        # departures
        if station in departure_train_candidates_by_node.keys():
            departures_at_station = departure_train_candidates_by_node[station]
            if station in zh_station_helper:
                for zh in zh_station_helper:
                    if station != zh and zh in departure_train_candidates_by_node.keys():
                        departures_at_station.extend(departure_train_candidates_by_node[zh])

        # skip the starting node of a train for arrival of train to departure transfer
        if not tpn == train['TrainPathNodes'][0]['ID']:
            # transfer edges from arrival of the delayed train to all departures of other trains
            for label, attr in departures_at_station:
                delta_t = attr['departureTime'] - arrival_nodes_delayed_train[tpn][1]['arrivalTime']
                if m < delta_t < M:
                    weight = delta_t * beta_waiting + datetime.timedelta(minutes=beta_transfer)
                    transfer_edges.append([arrival_nodes_delayed_train[tpn][0], label, float(weight.seconds / 60)])
                    transfer_edges_attributes[arrival_nodes_delayed_train[tpn][0], label] = {'type': 'transfer'}

        # transfer edges from arrival of any train to departure of delayed train
        if not tpn == train['TrainPathNodes'][-1]['ID']:
            for label, attr in arrivals_at_station:
                delta_t = departure_nodes_delayed_train[tpn][1]['departureTime'] - attr['arrivalTime']
                if m < delta_t < M:
                    weight = delta_t * beta_waiting + datetime.timedelta(minutes=beta_transfer)
                    transfer_edges.append([label, departure_nodes_delayed_train[tpn][0], float(weight.seconds / 60)])
                    transfer_edges_attributes[label, departure_nodes_delayed_train[tpn][0]] = {'type': 'transfer'}

    # print(transfer_edges)

    return transfer_edges, transfer_edges_attributes, [arrival_nodes_delayed_train, departure_nodes_delayed_train]


def transfer_edges_single_bus(G, bus, M, m, tpn_index_start_delay, tpns_bus):
    # print('start identification of transfer graphs')
    # '85ZMUS' --> 611, '85ZUE --> 638', '45ZSZU --> 13'
    zhHB = ['85ZMUS', '85ZUE', '45ZSZU']
    zhHB_code = [611, 638, 13]

    comm_stops_of_bus = {node['ID']: node['NodeID'] for node in bus['TrainPathNodes']
                           if node['StopStatus'] == 'commercialStop' and node['ID'] in tpns_bus}
    # tpns_by_nodeID = utils.build_dict(train['TrainPathNodes'], 'ID')


    zh_station_helper = []

    if any(x in zhHB_code for x in comm_stops_of_bus.values()):
        zh_station_helper = zhHB_code

    # n = [x[2] for x,y in G.nodes(data=True) if x[2] in tpns_bus if y['type'] == 'arrivalNode']
    departure_train_candidates_by_node = dict()
    arrival_train_candidates_by_node = dict()
    departure_nodes_bus = dict()
    arrival_nodes_bus = dict()
    for x, y in G.nodes(data=True):
        try:
            if y['train'] == bus['ID']:
                if y['type'] == 'arrivalNode' and x[2] in tpns_bus:
                    arrival_nodes_bus[x[2]] = (x, y)
                elif y['type'] == 'departureNode' and x[2] in tpns_bus:
                    departure_nodes_bus[x[2]] = (x, y)
                else:
                    continue
            elif x[0] not in comm_stops_of_bus.values() and x[0] not in zh_station_helper:
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
                print('Check transfer edges single train, somehow a not train node appears in G')
        except KeyError:
            print(x, y)

    beta_transfer = 10  # minutes
    beta_waiting = 2.5  # min/min
    transfer_edges = list()
    transfer_edges_attributes = dict()

    # identify all arrival nodes
    for tpn in bus['TrainPathNodes']:
        station = tpn['NodeID']

        # select dep and arr candidates of trains if there are any
        arrivals_at_station = []
        departures_at_station = []

        if station in arrival_train_candidates_by_node.keys():
            arrivals_at_station = arrival_train_candidates_by_node[station]
            if station in zh_station_helper:
                for zh in zh_station_helper:
                    if station != zh:
                        arrivals_at_station.extend(arrival_train_candidates_by_node[zh])

        # departures
        if station in departure_train_candidates_by_node.keys():
            departures_at_station = departure_train_candidates_by_node[station]
            if station in zh_station_helper:
                for zh in zh_station_helper:
                    if station != zh:
                        departures_at_station.extend(departure_train_candidates_by_node[zh])

        # skip the starting node of a train for arrival of train to departure transfer

        # transfer edges from arrival of the bus to all departures of other trains
        if tpn['SequenceNumber'] == 1:
            for label, attr in departures_at_station:
                delta_t = attr['departureTime'] - arrival_nodes_bus[tpn['ID']][1]['arrivalTime']
                if m < delta_t < M:
                    weight = delta_t * beta_waiting + datetime.timedelta(minutes=beta_transfer)
                    transfer_edges.append([arrival_nodes_bus[tpn['ID']][0], label, float(weight.seconds / 60)])
                    transfer_edges_attributes[arrival_nodes_bus[tpn['ID']][0], label] = {'type': 'transfer'}

        elif tpn['SequenceNumber'] == 0:
            # only edges from arrival of trains to departure of bus
            for label, attr in arrivals_at_station:
                delta_t = departure_nodes_bus[tpn['ID']][1]['departureTime'] - attr['arrivalTime']
                if m < delta_t < M:
                    weight = delta_t * beta_waiting + datetime.timedelta(minutes=beta_transfer)
                    transfer_edges.append([label, departure_nodes_bus[tpn['ID']][0], float(weight.seconds / 60)])
                    transfer_edges_attributes[label, departure_nodes_bus[tpn['ID']][0]] = {'type': 'transfer'}

    # print(transfer_edges)

    return transfer_edges, transfer_edges_attributes, [arrival_nodes_bus, departure_nodes_bus]


def explore_network(nodesSBB, create_from_scratch=True):

    '''
    :param nodesSBB: stations in the area of interest
    :param create_from_scratch: default False, if False load the pickled graph
    :return: infrastructure Graph as undirected graph with viriato code as nodes and weighted edges between
    all neighbors.
    '''

    p_graph_infra = 'input/Graph/graph_infrastructure.pickle'
    if not create_from_scratch:
        return nx.read_gpickle(p_graph_infra)

    # explore network node by node
    viriato_stations = []  # list of stations for infrastructure graph
    viriato_stations_attributes = {}  # dictionary of attributes
    edges = {}# list of links between stations for infrastructure graph
    edges_attributes = {}
    added_sections = []  # list of sections already added to the graph
    # undirected multigraph

    cache_section_track_id_distance = {}
    G_infra = nx.MultiGraph(cache_trackID_dist=cache_section_track_id_distance)
    nodes_to_explore = [nodesSBB.Code[0]]  # list of nodes to explore
    # viriato_stations[nodes_to_explore[0]] = nodes(nodes_to_explore[0])

    while len(nodes_to_explore) != 0:
        actual_node = nodes_to_explore.pop(0)
        # check if node is already explored
        if actual_node not in viriato_stations:
            viriato_stations.append(int(actual_node))
            node_info = ap.nodes(actual_node)
            # print(node_info['DebugString'])
            # create attributes, depending on if its in the area or not
            attributes = dict()
            attributes['Code'] = node_info['Code']
            attributes['DebugString'] = node_info['DebugString']
            attributes['in_area'] = False
            if actual_node in nodesSBB.Code:
                attributes['in_area'] = True   # node is not in the area of interest
            if len(node_info['NodeTracks']) > 0:
                attributes['NodeTracks'] = node_info['NodeTracks']
            else:
                attributes['NodeTracks'] = None
            viriato_stations_attributes[actual_node] = attributes

        #  find all neighbors of actual node
        neighbors = ap.neighbor_nodes(actual_node)
        # loop through neighbors to identify the sections between the stations
        for neighbor in neighbors:
            # add the neighbor to the list of stations to explore if not done yet
            if neighbor['ID'] not in viriato_stations:
                nodes_to_explore.append(neighbor['ID'])
            # get the section between the actual node and the neighbor
            tracks = ap.section_tracks_between_original(actual_node, neighbor['ID'])
            for track in tracks:
                section_identity = track['DebugString']
                if section_identity not in added_sections:
                    # edge = [actual_node, neighbor['ID'], track['Weight']]
                    if not cache_section_track_id_distance.__contains__(track['ID']):
                        cache_section_track_id_distance[track['ID']] = track['Weight']

                    edge = (actual_node, neighbor['ID'], track['ID'])
                    edge_attribute = {'weight': track['Weight'], 'sectionTrackID': track['ID'],
                                      'debugString': track['DebugString']}
                    edges[edge] = edge_attribute

                    added_sections.append(section_identity)

    for node, attributes in viriato_stations_attributes.items():
        G_infra.add_node(node, **attributes)

    for edge, attributes in edges.items():
        G_infra.add_edge(edge[0], edge[1], edge[2], **attributes)

    # nx.single_source_dijkstra(infra_graph, 168, 191)
    # infra_graph.nodes[605]['NodeTracks']

    nx.write_gpickle(G_infra, p_graph_infra)
    return G_infra


def get_all_departing_and_arriving_nodes_at_stations(G):
    departing_trains_by_node = dict()  # dictionary with key : station & value : all departing nodes
    arriving_trains_by_node = dict()  # dictionary with key : station & value : all departing nodes

    for x, y in G.nodes(data=True):
        if y['type'] in ['arrivalNodePassing', 'departureNodePassing', 'origin', 'destination']:
            continue
        elif y['type'] == 'arrivalNode':
            if not arriving_trains_by_node.__contains__(x[0]):
                arriving_trains_by_node[x[0]] = list()
            arr_nodes_in_dict = arriving_trains_by_node[x[0]]
            arr_nodes_in_dict.append(x)
        elif y['type'] == 'departureNode':
            if not departing_trains_by_node.__contains__(x[0]):
                departing_trains_by_node[x[0]] = list()
            dep_nodes_in_dict = departing_trains_by_node[x[0]]
            dep_nodes_in_dict.append(x)

    return departing_trains_by_node, arriving_trains_by_node


def create_transit_edges_nodes_single_train(train, G_infra, idx_start_delay):

    s = 0  # nr of stop of a train
    total_train_length = len(train['TrainPathNodes'])
    train_id = train['ID']

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
        departure_time_last_node = datetime.datetime.strptime(train['TrainPathNodes'][idx_start_delay-1]['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
        departure_node_last_node = train['TrainPathNodes'][idx_start_delay-1]['NodeID']
        departure_node_last_node_name = (departure_node_last_node, train['TrainPathNodes'][idx_start_delay-1]['DepartureTime'],
                                         train['TrainPathNodes'][idx_start_delay-1]['ID'], 'd')
        attributes = {'train': train_id, 'type': 'departureNode', 'departureTime': departure_time_last_node,
                      'StopStatus': train['TrainPathNodes'][idx_start_delay-1]['StopStatus']}
        departure_nodes_attributes[departure_node_last_node_name] = attributes



    for tpn in train['TrainPathNodes'][idx_start_delay:]:  # loop trough all path nodes of a train
        s = s + 1
        if not G_infra.nodes[tpn['NodeID']]['in_area']:
            # train_left_area = True
            continue

        if tpn['StopStatus'] == 'commercialStop':  # consider only when train stops
            # update time and node
            arrival_time_this_node = datetime.datetime.strptime(tpn['ArrivalTime'], "%Y-%m-%dT%H:%M:%S")
            arrival_node_this_node = tpn['NodeID']
            departure_time_this_node = datetime.datetime.strptime(tpn['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
            departure_node_this_node = tpn['NodeID']

            if start_of_train:
                node_name_dep_this = (departure_node_this_node, tpn['DepartureTime'], tpn['ID'], 'd')
                departure_nodes.append(node_name_dep_this)
                attributes = {'train': train_id, 'type': 'departureNode', 'departureTime': departure_time_this_node, 'StopStatus': tpn['StopStatus']}
                departure_nodes_attributes[node_name_dep_this] = attributes
                departure_time_last_node = departure_time_this_node
                departure_node_last_node = departure_node_this_node
                departure_node_last_node_name = node_name_dep_this
                start_of_train = False
                continue

            elif s == total_train_length:  # end of a train
                node_name_arr_this = (arrival_node_this_node, tpn['ArrivalTime'], tpn['ID'], 'a')
                arrival_nodes.append(node_name_arr_this)
                attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node, 'StopStatus': tpn['StopStatus']}
                arrival_nodes_attributes[node_name_arr_this] = attributes

                # driving edge to this node
                run_time = arrival_time_this_node - departure_time_last_node
                driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds/60)])
                driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = {'flow': 0, 'type': 'driving', 'train_id': train_id}

                del departure_node_last_node
                del node_name_dep_this
                del node_name_arr_this
                del departure_time_last_node

            else:  # node in between two transit nodes

                # arrival nodes
                node_name_arr_this = (arrival_node_this_node, tpn['ArrivalTime'], tpn['ID'], 'a')
                arrival_nodes.append(node_name_arr_this)
                attributes = {'train': train_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node, 'StopStatus': tpn['StopStatus']}
                arrival_nodes_attributes[node_name_arr_this] = attributes
                # driving edge between last node and this node
                run_time = arrival_time_this_node - departure_time_last_node
                driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds/60)])
                driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = { 'flow': 0, 'type': 'driving', 'train_id': train_id}

                # departure nodes
                node_name_dep_this = (departure_node_this_node, tpn['DepartureTime'], tpn['ID'], 'd')
                departure_nodes.append(node_name_dep_this)
                attributes = {'train': train_id, 'type': 'departureNode', 'departureTime': departure_time_this_node, 'StopStatus': tpn['StopStatus']}
                # attributes = {'station': departure_node_this_node, 'departureTime': departure_time_this_node, 'train': train_id, 'type': 'departureNode'}
                departure_nodes_attributes[node_name_dep_this] = attributes
                # waiting edge between last node and this node
                wait_time = (departure_time_this_node - arrival_time_this_node)
                waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds/60)])
                waiting_edges_attributes[node_name_arr_this, node_name_dep_this] = {'flow': 0, 'type': 'waiting', 'train_id': train_id}

                # update the departure node for next iteration
                departure_node_last_node = departure_node_this_node
                departure_node_last_node_name = node_name_dep_this
                departure_time_last_node = departure_time_this_node
        # passing nodes
        elif tpn['StopStatus'] == 'passing':
            arrival_time_this_node = datetime.datetime.strptime(tpn['ArrivalTime'], "%Y-%m-%dT%H:%M:%S")
            arrival_node_this_node = tpn['NodeID']
            departure_time_this_node = datetime.datetime.strptime(tpn['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
            departure_node_this_node = tpn['NodeID']
            if start_of_train:
                node_name_dep_this = (departure_node_this_node, tpn['DepartureTime'], tpn['ID'], 'dp')
                departure_nodes.append(node_name_dep_this)
                attributes = {'train': train_id, 'type': 'departureNodePassing', 'departureTime': departure_time_this_node, 'StopStatus': tpn['StopStatus']}
                departure_nodes_attributes[node_name_dep_this] = attributes
                departure_time_last_node = departure_time_this_node
                departure_node_last_node = departure_node_this_node
                departure_node_last_node_name = node_name_dep_this
                start_of_train = False
                continue

            elif s == total_train_length:  # end of a train
                node_name_arr_this = (arrival_node_this_node, tpn['ArrivalTime'], tpn['ID'], 'ap')
                arrival_nodes.append(node_name_arr_this)
                attributes = {'train': train_id, 'type': 'arrivalNodePassing', 'arrivalTime': arrival_time_this_node, 'StopStatus': tpn['StopStatus']}
                arrival_nodes_attributes[node_name_arr_this] = attributes
                # driving edge to this node
                run_time = arrival_time_this_node - departure_time_last_node
                driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds/60)])
                driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = {'flow': 0, 'type': 'driving', 'train_id': train_id}

                del departure_node_last_node
                del node_name_dep_this
                del node_name_arr_this
                del departure_time_last_node

            else:  # node in between two transit nodes
                node_name_arr_this = (arrival_node_this_node, tpn['ArrivalTime'], tpn['ID'], 'ap')
                arrival_nodes.append(node_name_arr_this)
                attributes = {'train': train_id, 'type': 'arrivalNodePassing', 'arrivalTime': arrival_time_this_node, 'StopStatus': tpn['StopStatus']}

                arrival_nodes_attributes[node_name_arr_this] = attributes
                # driving edge between last node and this node
                run_time = arrival_time_this_node - departure_time_last_node
                driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds/60)])
                driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = {'flow': 0, 'type': 'driving', 'train_id': train_id}

                node_name_dep_this = (departure_node_this_node, tpn['DepartureTime'], tpn['ID'], 'dp')
                departure_nodes.append(node_name_dep_this)
                attributes = {'train': train_id, 'type': 'departureNodePassing', 'departureTime': departure_time_this_node, 'StopStatus': tpn['StopStatus']}
                departure_nodes_attributes[node_name_dep_this] = attributes
                # waiting edge between last node and this node
                wait_time = (departure_time_this_node - arrival_time_this_node)
                waiting_edges.append([node_name_arr_this, node_name_dep_this, float(wait_time.seconds/60)])
                waiting_edges_attributes[node_name_arr_this, node_name_dep_this] = {'flow': 0, 'type': 'waiting', 'train_id': train_id}

                # update the departure node for next iteration
                departure_node_last_node = departure_node_this_node
                departure_node_last_node_name = node_name_dep_this
                departure_time_last_node = departure_time_this_node

    nodes_edges_dict = {'arrival_nodes': arrival_nodes, 'departure_nodes': departure_nodes,
                        'arrival_nodes_attr': arrival_nodes_attributes, 'departure_nodes_attr': departure_nodes_attributes,
                        'driving_edges': driving_edges, 'waiting_edges': waiting_edges,
                        'driving_attr': driving_edges_attributes, 'waiting_attr': waiting_edges_attributes}

    return nodes_edges_dict


def create_transit_edges_nodes_emergency_bus(bus):

    bus_id = bus['ID']

    arrival_nodes = []
    arrival_nodes_attributes = dict()
    departure_nodes = []
    departure_nodes_attributes = dict()
    driving_edges = list()
    driving_edges_attributes = dict()

    n = 0
    for tpn in bus['TrainPathNodes']:  # loop trough all path nodes of a bus
        n += 1
        # update time and node
        arrival_time_this_node = datetime.datetime.strptime(tpn['ArrivalTime'], "%Y-%m-%dT%H:%M:%S")
        arrival_node_this_node = tpn['NodeID']
        departure_time_this_node = datetime.datetime.strptime(tpn['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
        departure_node_this_node = tpn['NodeID']

        if n == 1:
            node_name_dep_this = (departure_node_this_node, tpn['DepartureTime'], tpn['ID'], 'd')
            departure_nodes.append(node_name_dep_this)
            attributes = {'train': bus_id, 'type': 'departureNode', 'departureTime': departure_time_this_node,
                          'StopStatus': tpn['StopStatus'], 'bus': 'EmergencyBus'}
            departure_nodes_attributes[node_name_dep_this] = attributes
            departure_time_last_node = departure_time_this_node
            departure_node_last_node = departure_node_this_node
            departure_node_last_node_name = node_name_dep_this

        elif n == 2:  # end of a bus
            node_name_arr_this = (arrival_node_this_node, tpn['ArrivalTime'], tpn['ID'], 'a')
            arrival_nodes.append(node_name_arr_this)
            attributes = {'train': bus_id, 'type': 'arrivalNode', 'arrivalTime': arrival_time_this_node,
                          'StopStatus': tpn['StopStatus'], 'bus': 'EmergencyBus'}

            arrival_nodes_attributes[node_name_arr_this] = attributes

            # driving edge to this node
            run_time = arrival_time_this_node - departure_time_last_node
            driving_edges.append([departure_node_last_node_name, node_name_arr_this, float(run_time.seconds/60)])
            driving_edges_attributes[departure_node_last_node_name, node_name_arr_this] = {'flow': 0, 'type': 'driving',
                                                                                           'bus_id': bus_id, 'bus': True}


    nodes_edges_dict = {'arrival_nodes': arrival_nodes, 'departure_nodes': departure_nodes,
                        'arrival_nodes_attr': arrival_nodes_attributes, 'departure_nodes_attr': departure_nodes_attributes,
                        'driving_edges': driving_edges, 'driving_attr': driving_edges_attributes}

    return nodes_edges_dict
