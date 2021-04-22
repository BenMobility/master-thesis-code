import requests
import json
import sys
import pickle
import datetime
from collections import Counter
import operator
import numpy as np
import numpy.lib.recfunctions as rfn
import utils
import neighborhood_structures as nh
# control option ä $ --> {}
# control option Ü ¨ --> []
# control option <   --> \


def apiUrl():
    return sys.argv[1]


def post_method(url, message_body):
    response = requests.post(url, message_body)
    if response.status_code > 299:
        if response.json()['message'] is None:
            message = "No message provided."
        else:
            message = response.json()['message']
        error_message = "Code: " + str(response.status_code) + ". Message: " + message
        print_message("Error occurred", error_message)
        raise Exception(error_message)
    else:
        return response


def get_method(url):
    response = requests.get(url)
    if response.status_code > 299:
        if response.json()['message'] is None:
            message = "No message provided."
        else:
            message = response.json()['message']
        error_message = "Code: " + str(response.status_code) + ". Message: " + message
        print_message("Error occurred", error_message)
        return -1
        #raise Exception(error_message)
    else:
        return response


def get_parameter(key):
    result = get_method(apiUrl() + "/parameters/" + key)
    return result.json()['Value']


def print_message(caption, message):
    message_body = '{ "messageLevel1": "' + caption + '", "messageLevel2": "' + message + '"}'
    post_method(apiUrl() + "/notifications", message_body)


def nodes(node_id):
    # print(apiUrl + "/nodes/" + str(node_id))
    return get_method(apiUrl() + "/nodes/" + str(node_id)).json()


def neighbor_nodes(node):
    call = apiUrl() + "/neighbor-nodes/" + str(node)
    # print(call)
    return get_method(call).json()


def get_crossing_mesoscopic_routing_edges(node_id):
    body_message = '{\n' \
                   '  "nodeID": ' + str(node_id) +\
                   '\n}'
    return post_method(apiUrl() + "/get-crossing-mesoscopic-routing-edges", body_message).json()


def assign_station_track(train_path_node_id, node_track_id):
    message_body = '{\n' \
                   '"TrainPathNodeID": ' + str(train_path_node_id) + ',\n' +\
                   '"NodeTrackID": "' + str(node_track_id) + '"\n}'
    return post_method(apiUrl() + "/assign-station-track", message_body).json()


def select_section_track(from_node, to_node, identifier, track_info, G_infra):
    edges_between = G_infra.get_edge_data(from_node, to_node)
    all_track_id = list()
    all_frequency = list()

    # tuple_key_debug = (1065, 605, 592, 'arrival')
    for track_id, attributes in edges_between.items():
        tuple_key = (track_id, from_node, to_node, identifier)
        all_track_id.extend([track_id])
        # print(tuple_key)
        if tuple_key in track_info.track_sequences_of_TPN.keys():
            len_sequences = len(track_info.track_sequences_of_TPN[tuple_key])
            all_frequency.extend([len_sequences])
        else:
            len_sequences = 0
            all_frequency.extend([len_sequences])
        # print(len_sequences)
    # sort the list by the number of trains on that track from low to high
    max_value = max(all_frequency)
    max_index = all_frequency.index(max_value)
    section_track_selected = all_track_id[max_index]
    # select track with most trains travelling in same direction
    # section_track_selected = list_track_sequences[-1]

    # call = apiUrl() + "/section-tracks-between/" + str(from_node) + "/" + str(to_node)
    # viriato_return = get_method(call).json()

    return section_track_selected

def section_tracks_between_original(from_node, to_node):

    call = apiUrl() + "/section-tracks-between/" + str(from_node) + "/" + str(to_node)
    return get_method(call).json()


def section_tracks(track_id):
    call = apiUrl() + "/section-tracks/" + str(track_id)
    # print(call)
    return get_method(call).json()


def section_tracks_parallel_to(section_track):
    if isinstance(section_track, int):
        call = apiUrl() + "/section-tracks-parallel-to/" + str(section_track)
    else:
        call = apiUrl() + "/section-tracks-parallel-to/" + str(section_track['ID'])
    return get_method(call).json()


def active_trains_within_time_range(from_time, to_time):
   # message_body = '{ "fromTime": "' + from_time + '", "toTime": "' + to_time + '"}'
   # return get_method(apiUrl() + "/trains?FromTime=2005-05-01T14%3A05%3A00&ToTime=2005-05-01T15%3A10%3A00").json()
   return get_method(apiUrl()+'/trains?FromTime='+ from_time +'&ToTime='+to_time).json()

def headway_times_btwn_for_sectionTrack(preceedingTPN, suceedingTPN, sectionTrackID, fromNode, toNode):
    msg = '/headway-times/between-train-path-nodes/'
    msg += str(preceedingTPN) + '/' + str(suceedingTPN) + '/for-section-track/' + str(sectionTrackID)+'/in-direction/'
    msg += str(fromNode) + '/' + str(toNode)

    return get_method(apiUrl() + msg).json()


def headway_times_btwn(preceedingTPN, suceedingTPN):
    msg = '/headway-times/between-train-path-nodes/'
    msg += str(preceedingTPN) + '/' + str(suceedingTPN)

    return get_method(apiUrl() + msg).json()



def active_trains_cut_to_time_range_driving_any_node(from_time, to_time, driven_node_IDs):

        nodes_str = str(driven_node_IDs)
        return get_method(apiUrl() + '/trains?FromTime=' + from_time + '&ToTime=' + to_time+'&NodeFilter='+nodes_str[1:len(nodes_str)-1]).json()


def active_trains_cut_to_time_range_driving_any_node_one(from_time, to_time, driven_node_IDs):
    nodes_str = str(driven_node_IDs)
    if get_method(apiUrl() + '/trains?FromTime=' + from_time + '&ToTime=' + to_time + '&NodeFilter=' + nodes_str) != -1:
        return get_method(apiUrl() + '/trains?FromTime=' + from_time + '&ToTime=' + to_time + '&NodeFilter=' + nodes_str).json();


def clone_train(trainID):

    message_body = '{ "TrainID": ' + str(trainID) + '}'
    cloned_train = post_method(apiUrl() + "/clone-train", message_body).json()

    return cloned_train


def node_message_routing_edges(nodeID, start_section_track, end_section_track, nodeTrack=None, junction=False, last_node=False):
    if junction:
        node_message = '    {\n' \
                       '      "nodeId": ' + str(nodeID) + ',\n' + \
                       '      "startSectionTrack": ' + start_section_track + ',\n' + \
                       '      "endSectionTrack": ' + end_section_track + '\n' + \
                       '    },' + '\n'
    elif last_node:
        node_message = '    {\n'\
                       '      "nodeId": ' + str(nodeID) + ',\n' + \
                       '      "startSectionTrack": ' + start_section_track + ',\n' + \
                       '      "endNodeTrack": ' + str(nodeTrack) + '\n' + \
                       '    }\n  ]\n}'

    else:
        node_message = '    {\n' \
                       '      "nodeId": ' + str(nodeID) + ',\n' + \
                       '      "startSectionTrack": ' + start_section_track + ',\n' + \
                       '      "endNodeTrack": ' + str(nodeTrack) + '\n' + \
                       '    },' + '\n' + '    {' + '\n' + \
                       '      "nodeId": ' + str(nodeID) + ',\n' + \
                       '      "startNodeTrack": ' + str(nodeTrack) + ',\n' + \
                       '      "endSectionTrack": ' + end_section_track + '\n' + \
                       '    },' + '\n'

    return node_message


def reroute_train(train_to_reroute, path, idx_start_end, track_info, G_infra):
    '''
    section_tracks_between(node1, node2, tpn_track_sequences, infra_graph)
    :param train_to_reroute: the train which has to be rerouted
    :param idx_start_end: list of indeces of TrainPathNodes in the train to reroute
    :param path: the shortest path with closed track, first/last node are the start/end of rerouting
    :return: rerouted train on viriato platform
    '''
    idx_start_end_initial = idx_start_end.copy()
    nr_nodes_added_newPath = len(path[1]) - len(train_to_reroute['TrainPathNodes'][idx_start_end_initial[0]:idx_start_end_initial[1]+1])
    idx_1stNode_after_RR = idx_start_end_initial[1] + 1  # nr_nodes_added_newPath
    sequence_nr_start = train_to_reroute['TrainPathNodes'][0]['SequenceNumber']
    sequence_nr_end = train_to_reroute['TrainPathNodes'][-1]['SequenceNumber']

    # 1st node
    viriato_node = nodes(path[1][0])
    nodeTrack_startRR = viriato_node['NodeTracks'][0]['ID']
    pathNodeId_startRR = train_to_reroute['TrainPathNodes'][idx_start_end[0]]['ID']

    # end_section_track = section_tracks_between(path[1][0], path[1][1])[0]['ID']
    end_section_track = select_section_track(path[1][0], path[1][1], 'departure', track_info, G_infra)

    node_message = '    {\n' \
                   '      "nodeId": ' + str(path[1][0]) + ',\n' +\
                   '      "startNodeTrack": ' + str(nodeTrack_startRR) + ',\n' + \
                   '      "endSectionTrack": ' + str(end_section_track) + '\n    }, \n'
    message_body = node_message

    j = 1  # index in path list
    #  end section track of first node ist start_section_track of next node if a station
    track_ID_from_last_node = end_section_track
    track_ID_to_next_node = track_ID_from_last_node
    last_node_junction = False
    for node in path[1][1:-1]:
        viriato_node = nodes(node)
        track_ID_from_last_node = track_ID_to_next_node
        # track_ID_to_next_node = section_tracks_between(path[1][j], path[1][j + 1])[0]['ID']
        track_ID_to_next_node = select_section_track(path[1][j], path[1][j + 1], 'departure', track_info, G_infra)
        if len(viriato_node['NodeTracks']) == 0:
            # it is a junction
            message_body += node_message_routing_edges(viriato_node['ID'], str(track_ID_from_last_node), str(track_ID_to_next_node), junction=True)
        else:
            nodeTrack = viriato_node['NodeTracks'][0]['ID']  # assign the node track
            message_body += node_message_routing_edges(viriato_node['ID'], str(track_ID_from_last_node),  str(track_ID_to_next_node), nodeTrack)
        j += 1

    # last rerouted node, last node of the path to the next node in the train path
    viriato_node = nodes(path[1][-1])
    node_track = viriato_node['NodeTracks'][0]['ID']
    track_ID_from_last_node = track_ID_to_next_node
    try:
        first_node_after_RR = train_to_reroute['TrainPathNodes'][idx_1stNode_after_RR]
        train_RR_from_start_to_end = False
    except IndexError:
        print('Last rerouted node of the train is the same as end of RR path `?')
        train_RR_from_start_to_end = True

    take_1stnode_afterRR_on_originalPath = False
    # if train_RR_from_start_to_end:
    #    pathNodeId_endRR = train_to_reroute['TrainPathNodes'][idx_1stNode_after_RR-1]['ID']
    #    nodeTrack_end = viriato_node['NodeTracks'][0]['ID']
    if not take_1stnode_afterRR_on_originalPath:
        message_body += node_message_routing_edges(viriato_node['ID'], str(track_ID_from_last_node),None, node_track, last_node=True)
        pathNodeId_endRR = train_to_reroute['TrainPathNodes'][idx_start_end_initial[1]]['ID']
        nodeTrack_end = node_track
    elif take_1stnode_afterRR_on_originalPath:
        # track_ID_to_next_node = section_tracks_between(viriato_node['ID'], first_node_after_RR['NodeID'])[0]['ID']
        track_ID_to_next_node = select_section_track(viriato_node['ID'], first_node_after_RR['NodeID'], 'departure', track_info, G_infra)

        message_body += node_message_routing_edges(viriato_node['ID'], str(track_ID_from_last_node), str(track_ID_to_next_node), node_track)

        # remaining nodes of the initial route until node has station tracks
        viriato_node = nodes(first_node_after_RR['NodeID'])
        track_ID_from_last_node = track_ID_to_next_node
        track_ID_to_next_node = None
        add_nr_nodes = 1  #nr of nodes until end of routing edges found
        if len(viriato_node['NodeTracks']) != 0:
            nodeTrack_end = viriato_node['NodeTracks'][0]['ID']
            message_body += node_message_routing_edges(viriato_node['ID'], str(track_ID_from_last_node), str(track_ID_to_next_node), nodeTrack_end, last_node=True)
            pathNodeId_endRR = train_to_reroute['TrainPathNodes'][idx_1stNode_after_RR]['ID']
        else:
            last_node_junction = True
            while last_node_junction:
                # track_ID_to_next_node = section_tracks_between(viriato_node['ID'], train_to_reroute['TrainPathNodes'][idx_1stNode_after_RR + add_nr_nodes]['NodeID'])[0]['ID']
                track_ID_to_next_node = select_section_track(viriato_node['ID'], train_to_reroute['TrainPathNodes'][idx_1stNode_after_RR + add_nr_nodes]['NodeID'], 'departure', track_info, G_infra)
                message_body += node_message_routing_edges(viriato_node['ID'], str(track_ID_from_last_node), str(track_ID_to_next_node), junction=True)
                viriato_node = nodes(train_to_reroute['TrainPathNodes'][idx_1stNode_after_RR + add_nr_nodes]['NodeID'])
                track_ID_from_last_node = track_ID_to_next_node
                if len(viriato_node['NodeTracks']) > 0:
                    # last node reached
                    nodeTrack_end = viriato_node['NodeTracks'][0]['ID']
                    message_body += node_message_routing_edges(viriato_node['ID'], str(track_ID_from_last_node), None, nodeTrack_end, last_node=True)
                    pathNodeId_endRR = train_to_reroute['TrainPathNodes'][idx_1stNode_after_RR + add_nr_nodes]['ID']
                    last_node_junction = False
                    add_nr_nodes += 1
                else:
                    add_nr_nodes += 1

    message_head = '{\n  "TrainID": ' + str(train_to_reroute['ID']) + ',\n' +\
                   '  "StartTrainPathNodeId": ' + str(pathNodeId_startRR) + ',\n' +\
                   '  "EndTrainPathNodeId": ' + str(pathNodeId_endRR) + ',\n' +\
                   '  "RoutingEdges": [\n'
    message_body = message_head + message_body
    # print(message_body)
    # assign the node track of RR start / end node. Attention !!! returns a train not cut to time window
    # train_before_reroute = train_to_reroute.copy()
    length_train_before_RR = len(train_to_reroute['TrainPathNodes'])

    train_to_reroute = assign_station_track(pathNodeId_startRR, nodeTrack_startRR)

    # identify if the rerouted train has a node outside the time window, adapt the indices if so
    rerouted_train_not_cut_to_timeWindow = False
    if sequence_nr_start != train_to_reroute['TrainPathNodes'][0]['SequenceNumber'] \
            or sequence_nr_end != train_to_reroute['TrainPathNodes'][-1]['SequenceNumber']:
        # rerouted train has changed, not cut to time window, find the index of train path nodes up to where to cut
        nr_nodes_added_before = sequence_nr_start - train_to_reroute['TrainPathNodes'][0]['SequenceNumber']
        # new indices to cut train path nodes so that all trains before / afterwards are the same
        cut_before_idx = nr_nodes_added_before
        cut_after_idx = length_train_before_RR + nr_nodes_added_before + nr_nodes_added_newPath
        rerouted_train_not_cut_to_timeWindow = True
    else:
        cut_before_idx = 0

    # assign the node track of the first and last RR node
    # todo train node ids are changed if I assign a station track to cloned train, ask Matthias why ?
    train_to_reroute = assign_station_track(pathNodeId_endRR, nodeTrack_end)
    # reroute the train
    train_to_reroute = post_method(apiUrl() + "/reroute-train", message_body).json()

    if rerouted_train_not_cut_to_timeWindow:
        # train_to_reroute['TrainPathNodes'] = train_to_reroute['TrainPathNodes'][cut_before_idx:cut_after_idx]
        train_to_reroute['CutIndices'] = [cut_before_idx, cut_after_idx]
    else:
        train_to_reroute['CutIndices'] = None
    # todo, remove this once the rr works as wished
    if train_RR_from_start_to_end or not take_1stnode_afterRR_on_originalPath:
        idx_endRR = idx_1stNode_after_RR + nr_nodes_added_newPath - 1
    else:
        idx_endRR = idx_1stNode_after_RR + nr_nodes_added_newPath-1 + add_nr_nodes

    idx_startRR = idx_start_end_initial[0]
    # index representing all routing edges in rerouting method, used for update train time
    train_to_reroute['StartEndRR_idx'] = [idx_startRR + cut_before_idx, idx_endRR + cut_before_idx]

    return train_to_reroute


def calculate_run_times(trainID):
    call = apiUrl() + "/calculate-run-times/" + str(trainID)
    return get_method(call).json()


def cancel_train(trainID):
    message_body = '{ "trainID": ' + str(trainID) + ' }'

    return post_method(apiUrl() + "/cancel-train", message_body).json()


def cancel_train_from(trainPathNodeID):
    message_body = '{ "trainPathNodeID": ' + str(trainPathNodeID) + ' }'

    return post_method(apiUrl() + "/cancel-train-from", message_body).json()


def cancel_train_to(trainPathNodeID):
    message_body = '{ "trainPathNodeID": ' + str(trainPathNodeID) + ' }'

    return post_method(apiUrl() + "/cancel-train-to", message_body).json()



def update_train_times(train, times, train_to_reroute):
    train_path_node_startRR = train['TrainPathNodes'][train['StartEndRR_idx'][0]]
    dep_time_last_node = train_path_node_startRR['DepartureTime']
    train_path_firstRR_node = train['TrainPathNodes'][train['StartEndRR_idx'][0]+1]
    train_path_node_endRR = train['TrainPathNodes'][train['StartEndRR_idx'][1]]
    train_path_node_lastBefore_endRR = train['TrainPathNodes'][train['StartEndRR_idx'][1]-1]

    # train_path_node_afterRR = train['TrainPathNodes'][train['StartEndRR_idx'][1]+1]
    idx_endRR = train['StartEndRR_idx'][1]
    idx_afterRR = idx_endRR

    body_message = '{ ' \
                   '"TrainId": ' + str(train['ID']) + ', '\
                   '"Times":  ['

    start_node_time_update = False
    start_remaining_part_of_train = False  # used for the part after RR which is same as in original train
    break_iteration = False
    for time in times['Times']:

        if train_path_firstRR_node['ID'] == time['TrainPathNodeID']:
            # handle first node to update the train time, this is the first rerouted node
            start_node_time_update = True
            min_runtime = utils.duration_transform_to_timedelta(time['MinimumRunTime'])
            arr_time = datetime.datetime.strptime(dep_time_last_node, "%Y-%m-%dT%H:%M:%S") + min_runtime
            if time['MinimumStopTime'] == 'P0D':
                dep_time_last_node = arr_time
            else:
                dep_time_last_node = arr_time + utils.duration_transform_to_timedelta(time['MinimumStopTime'])
            #  datetime.datetime.strptime(train_path_nodes['DepartureTime'], "%Y-%m-%dT%H:%M:%S")

            time_msg = '{' \
                       '"TrainPathNodeID": ' + str(time["TrainPathNodeID"]) + ', ' \
                       '"ArrivalTime": "' + arr_time.strftime("%Y-%m-%dT%H:%M:%S") + '", '\
                       '"DepartureTime": "' + dep_time_last_node.strftime("%Y-%m-%dT%H:%M:%S") + '"' +\
                       '}'
            body_message = body_message + time_msg
        elif start_node_time_update:
            #if train_path_node_endRR['ID'] == time['TrainPathNodeID']:
            # todo When to use which minimum Runtime ?
            if train_path_node_lastBefore_endRR['ID'] == time['TrainPathNodeID']:
                # Todo, remove the minus -1 for idx after RR once the todo before is cleared
                idx_afterRR = idx_afterRR - 1
                # end of rerouting reached
                start_node_time_update = False
                start_remaining_part_of_train = True

            # take care of the other nodes until end of RR
            min_runtime = utils.duration_transform_to_timedelta(time['MinimumRunTime'])
            arr_time = dep_time_last_node + min_runtime
            if time['MinimumStopTime'] == 'P0D':
                dep_time_last_node = arr_time
            else:
                dep_time_last_node = arr_time + utils.duration_transform_to_timedelta(time['MinimumStopTime'])

            time_msg = ',' \
                       '{' \
                       '"TrainPathNodeID": ' + str(time["TrainPathNodeID"]) + ', '\
                       '"ArrivalTime": "' + arr_time.strftime("%Y-%m-%dT%H:%M:%S") + '", ' \
                       '"DepartureTime": "' + dep_time_last_node.strftime("%Y-%m-%dT%H:%M:%S") + '"' + \
                       '}'
            body_message = body_message + time_msg
        elif start_remaining_part_of_train:

            if time['TrainPathNodeID'] == train['TrainPathNodes'][-1]['ID']:
                print('lets skip the remaining Times')
                break_iteration = True

            # first node after RR of train, back on initial track
            idx_afterRR += 1
            min_runtime = utils.duration_transform_to_timedelta(train['TrainPathNodes'][idx_afterRR]['MinimumRunTime'])
            # min_runtime = utils.duration_transform_to_timedelta(time['MinimumRunTime'])
            arr_time = dep_time_last_node + min_runtime
            if train['TrainPathNodes'][idx_afterRR]['MinimumStopTime'] == 'P0D':
                dep_time_last_node = arr_time
            else:
                min_stop_time = train['TrainPathNodes'][idx_afterRR]['MinimumStopTime']
                dep_time_last_node = arr_time + utils.duration_transform_to_timedelta(min_stop_time)
            time_msg = ',{' \
                       '"TrainPathNodeID": ' + str(time["TrainPathNodeID"]) + ', ' \
                       '"ArrivalTime": "' + arr_time.strftime("%Y-%m-%dT%H:%M:%S") + '", ' \
                       '"DepartureTime": "' + dep_time_last_node.strftime("%Y-%m-%dT%H:%M:%S") + '"' + \
                       '}'
            body_message = body_message + time_msg
        else:
            # dep_time_last_node = datetime.datetime.strptime(time['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
            # print('not startet RR')
            continue
    body_message = body_message + ']}'

    # print(body_message)
    train_updated_times = post_method(apiUrl() + "/update-train-times", body_message).json()

    return train_updated_times


def set_section_track(train_path_node_id, section_track_id):
    message_body = '{"TrainPathNodeID": ' + str(train_path_node_id['ID']) + \
                   ', "SectionTrackID": ' + str(section_track_id) + '}'
    print(message_body)

    return post_method(apiUrl() + "/set-section-track", message_body).json()


def get_possessions_section_track_closure(fromTime, toTime):
    #Oliver last version
    #return get_method(apiUrl() + '/possessions-within-time-range/section-track-closures/from/' + fromTime + '/to/'
     #              + toTime).json()

    return get_method(apiUrl()+'/possessions/section-track-closures?FromTime=' + fromTime + '&ToTime=' + toTime).json();
    #return get_method(apiUrl()+'/possessions/node-track-closures?FromTime=2003-05-01T07%3A30%3A00&ToTime=2003-05-01T08%3A05%3A45').json();
#  my methods using ap
def reroute_train_via_set_section_track(train, closed_section_track):
    # identify parallel section tracks
    parallel_section_tracks = section_tracks_parallel_to(closed_section_track)
    candidates_section_track = []
    for n in range(0, len(parallel_section_tracks)):
        if parallel_section_tracks[n]['ID'] != closed_section_track['ID']:
            candidates_section_track.append(parallel_section_tracks[n]['ID'])

    if len(candidates_section_track) == 0:
        print('  WARNING no parallel section track for closed section track ', closed_section_track['ID'], ' available')
    elif len(candidates_section_track) == 1:
        print('  Only one parallel section track available, rerouted on this track')
        set_track = candidates_section_track[0]
    else:
        print('  multiple parallel section tracks available')
        set_track = candidates_section_track[0]

    # identify train path node traveling on closed_section_track
    train_path_nodes = utils.build_dict(train['TrainPathNodes'], key="SectionTrackID")
    train_path_node_on_closed_track = train_path_nodes[closed_section_track['ID']]

    # reroute train on a parallel track
    return set_section_track(train_path_node_on_closed_track, set_track)


def check_trains_occurence(cut_trains):
    '''
    small check if each train is occuring only once
    :param cut_trains: trains visiting any node
    :return:
    '''
    unique_trains = list()
    for train in cut_trains:
        unique_trains.append(train['ID'])
    counter = Counter(unique_trains)
    sorted_c = sorted(counter.items(), key=operator.itemgetter(1))
    # using list comprehension to get values only
    res = [lis[1] for lis in sorted_c]
    maxi = max(res)
    if maxi != 1:
        raise Exception('Some errors occured, not every train is unique !')


def get_all_visited_nodes_in_time_window():
    time_window = get_parameter('ReferenceTimeWindow')
    trains = active_trains_within_time_range(time_window['FromTime'], time_window['ToTime'])
    print_message('Count trains ', str(len(trains)))
    # all_node_ids = set()
    all_node_ids = list()
    for train in trains:
        for train_path_node in train['TrainPathNodes']:
            # all_node_ids.add(train_path_node['NodeID'])
            all_node_ids.append(train_path_node['NodeID'])
    return all_node_ids


def get_all_viriato_node_ids_to_code_dictionary(all_node_ids):
    code_id_dictionary = {}
    id_code_dictionary = {}
    all_node_ids = set(all_node_ids)  # set only stores unique values
    for node_id in all_node_ids:
        node = nodes(node_id)
        code = node['Code']
        code_id_dictionary[node_id] = code
        id_code_dictionary[code] = node_id

    return code_id_dictionary, id_code_dictionary


def add_field_com_stop(nodesSBB, nodes_commercial_stops):
    nodesSBB = rfn.append_fields(nodesSBB, 'commercial_stop', np.ones(nodesSBB.shape[0]), dtypes='i8', usemask=False,
                                 asrecarray=True)
    c = 0
    for station in nodesSBB:
        if station.Code not in nodes_commercial_stops:
            station.commercial_stop = 0
            c = c + 1
    print('\n', c, ' of ', nodesSBB.shape[0], ' stations do not have a commercial stops \n')
    return nodesSBB


def cut_trains_AreaOfInterest(cut_trains, stations_in_area):
    train_index = 0
    cut_trains_to_Area = list()
    for train in cut_trains:
        nodes_in_area = list()
        for j in range(0, len(train['TrainPathNodes'])):
            # print(train['TrainPathNodes'][j]['NodeID'])
            if train['TrainPathNodes'][j]['NodeID'] in stations_in_area:
                nodes_in_area.append(train['TrainPathNodes'][j].copy())
        if len(nodes_in_area) > 1:
            cut_trains_to_Area.append(train.copy())
            cut_trains_to_Area[train_index]['TrainPathNodes'] = nodes_in_area
            train_index += 1
        else:
            # all nodes are outside area --> delete train
            # print('hi')
            continue
    return cut_trains_to_Area
            # print(G.edges)


def cut_trains_AreaOfInterest_and_adapt_Viriato_DB(cut_trains, stations_in_area, parameters):
    train_index = 0
    cut_trains_to_Area = list()

    for train in cut_trains:
        # if in a former run a emergency train is called and the algorithm platform is not restarted, the train is still
        # active in Viriato, therefore cancel them. For debugging you might do not want to restart the platform always
        if train['DebugString'] == 'RVZH_9999_1_J05 tt_(S)':
            cancel_train(train['ID'])
            continue
        train_leaves_area = False
        train_enters_area = False
        train_outside_area = False
        train_inside_area = False

        nodes_in_area = list()
        nodes_outside_area = list()
        nodes_idx_entering_area = list()
        nodes_idx_leaving_area = list()
        comm_stops = 0
        for j in range(0, len(train['TrainPathNodes'])):
            if train['TrainPathNodes'][j]['NodeID'] in stations_in_area:
                if train['TrainPathNodes'][j]['StopStatus'] == 'commercialStop':
                    comm_stops += 1

                nodes_in_area.append(train['TrainPathNodes'][j].copy())
                if not train_inside_area:
                    train_enters_area = True
                    train_inside_area = True
                else:
                    train_enters_area = False

                if train_enters_area:
                    nodes_idx_entering_area.append(j)

            else:
                nodes_outside_area.append(train['TrainPathNodes'][j].copy())
                if not train_outside_area:
                    train_leaves_area = True
                    train_outside_area = True
                else:
                    train_leaves_area = False
                if train_leaves_area:
                    nodes_idx_leaving_area.append(j)


                train_outside_area = True
        if comm_stops <= 1:
            cancel_train(train['ID'])
            continue

        if len(nodes_idx_leaving_area) == 1 and len(nodes_idx_entering_area) == 0:
            print('How do we get here ?')
        elif len(nodes_idx_leaving_area) == 0 and len(nodes_idx_entering_area) == 0:
            print('How do we get here ?')
        elif len(nodes_idx_leaving_area) == 1 and len(nodes_idx_entering_area) == 1:
            if nodes_idx_entering_area[0] < nodes_idx_leaving_area[0]:
                # train starts in area and leaves area
                if train['TrainPathNodes'][nodes_idx_entering_area[0]]['SequenceNumber'] == 0:
                    # train start is inside the area
                    if nodes_idx_leaving_area[0] > 1:
                        cancel_train_from(train['TrainPathNodes'][nodes_idx_leaving_area[0]-1]['ID'])
                    else:
                        cancel_train(train['ID'])

                else:
                    # train start is cut to the time window
                    try:
                        tpn_cancel_to = train['TrainPathNodes'][nodes_idx_entering_area[0]]['ID']
                        tpn_cancel_from = train['TrainPathNodes'][nodes_idx_leaving_area[0]-1]['ID']
                        cancel_train_to(tpn_cancel_to)
                        cancel_train_from(tpn_cancel_from)
                    except Exception:
                        print('wait')
            else:
                # train starts outside area and enters the area
                cancel_train_to(train['TrainPathNodes'][nodes_idx_entering_area[0]]['ID'])

            # cancel to first node inside and cancel from first outside
        elif len(nodes_idx_leaving_area) == 2 and len(nodes_idx_entering_area) == 1:
            # this train starts outside the area, runs through the area and ends outside
            tpn_cancel_to = train['TrainPathNodes'][nodes_idx_entering_area[0]]['ID']
            tpn_cancel_from = train['TrainPathNodes'][nodes_idx_leaving_area[1]-1]['ID']
            cancel_train_to(tpn_cancel_to)
            cancel_train_from(tpn_cancel_from)
        elif len(nodes_idx_leaving_area) == 1 and len(nodes_idx_entering_area) == 2:
            # if train['TrainPathNodes'][nodes_idx_entering_area[1]]['NodeID'] == train['TrainPathNodes'][nodes_idx_leaving_area[0]-1]['NodeID']:
            tpn_cancel_from = train['TrainPathNodes'][nodes_idx_leaving_area[0]-1]
            tpn_cancel_to = train['TrainPathNodes'][nodes_idx_entering_area[1]]
            nh.short_turn_train_viriato_preselection(parameters, train, tpn_cancel_from, tpn_cancel_to)
        elif len(nodes_idx_leaving_area) == 2 and len(nodes_idx_entering_area) == 2:
                tpn_cancel_from = train['TrainPathNodes'][nodes_idx_leaving_area[0]-1]
                tpn_cancel_to = train['TrainPathNodes'][nodes_idx_entering_area[1]]
                nh.short_turn_train_viriato_preselection(parameters, train, tpn_cancel_from, tpn_cancel_to)
        elif len(nodes_idx_leaving_area) > 2 or len(nodes_idx_entering_area) > 2:
            print('TPN ')
        else:
            pass

    return cut_trains_to_Area


def cut_single_train_AreaOfInterest(cut_train, nodesSBB):

    '''
    :param cut_train: trains to cut
    :param nodesSBB: the nodes within the AoI
    :return: the train with nodes inside AoI, None if totally outside
    '''

    train_index = 0
    cut_train_to_Area = cut_train.copy()
    nodes_in_area = list()

    for j in range(0, len(cut_train['TrainPathNodes'])):
        # print(train['TrainPathNodes'][j]['NodeID'])
        if cut_train['TrainPathNodes'][j]['NodeID'] in nodesSBB.Code:
                nodes_in_area.append(cut_train['TrainPathNodes'][j].copy())

    if len(nodes_in_area) > 1:
        cut_train_to_Area['TrainPathNodes'] = nodes_in_area
        train_index += 1
    else:
        cut_train_to_Area = None
        # all nodes are outside area --> delete train
        # print('hi')

    return cut_train_to_Area


def cut_trains_AreaOfInterest_TimeWindow(cut_trains, stations_in_area, timeWindow):

    fromTime = datetime.datetime.strptime(timeWindow['FromTime'], "%Y-%m-%dT%H:%M:%S")
    toTime = datetime.datetime.strptime(timeWindow['ToTime'], "%Y-%m-%dT%H:%M:%S")
    cut_trains_to_Area_time = list()
    last_tpn = None
    first_node_before_set = False
    last_node_after_set = False
    start = True
    for train in cut_trains:
        tpn_in_area = list()
        for tpn in train['TrainPathNodes']:
            # print(train['TrainPathNodes'][j]['NodeID'])

            if tpn['NodeID'] in stations_in_area:
                tpn_departure_time = datetime.datetime.strptime(tpn['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
                tpn_arrival_time = datetime.datetime.strptime(tpn['ArrivalTime'], "%Y-%m-%dT%H:%M:%S")
                if fromTime <= tpn_arrival_time <= toTime:
                    if not first_node_before_set and not start:
                        first_node_before_set = True
                        tpn_in_area.append(last_tpn)
                    tpn_in_area.append(tpn)
                else:
                    if last_tpn in tpn_in_area and not start and not last_node_after_set:
                        if last_tpn_departure_time < toTime:
                            tpn_in_area.append(tpn)
                last_tpn = tpn
                last_tpn_departure_time = tpn_departure_time
            start = False
        if len(tpn_in_area) > 1:
            train['TrainPathNodes'] = tpn_in_area
        else:
            continue
        cut_trains_to_Area_time.append(train)
    return cut_trains_to_Area_time


