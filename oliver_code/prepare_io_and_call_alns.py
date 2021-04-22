import algorithm_platform_methods as ap
import networkx as nx
import heapq
import datetime
import utils
import alns as alns
from operator import itemgetter


def prepare_io_and_call_alns(G, G_infra, cut_trains_to_area, parameters):

    # receive the closed tracks of Viriato (defined in viriato)
    possessions = ap.get_possessions_section_track_closure(parameters.time_window['FromTime'],
                                                           parameters.time_window['ToTime'])


    # so that no trains passes the closed track (maybe they could be removed instead)
    G_infra, closed_SectionTrackIDs = increase_weight_of_closed_tracks(G_infra, possessions,
                                                                       parameters.weight_closed_tracks)


    # Get all information about track occupancy, defined in
    track_info = TrackInformation(cut_trains_to_area, closed_SectionTrackIDs, G_infra)
    high = heapq.nlargest(5, track_info.nr_usage_tracks.items(), key=lambda i: i[1])
    low = heapq.nsmallest(10, track_info.nr_usage_tracks.items(), key=lambda i: i[1])

    # db has duplicates
    get_duplicate_trains_on_track = False
    if get_duplicate_trains_on_track:
        duplicates = identify_sameTrains_sameTrack_sameTime(track_info)


    # check if graph is acyclic
    # todo, remove debug here
    debug = False
    if not debug:
        is_graph_acyclic = utils.check_acyclicity_of_Graph(G)
        if is_graph_acyclic:
            G_topological_sort = nx.topological_sort(G)

    set_solution = alns.alns(G, G_infra, cut_trains_to_area, track_info, parameters)

    return set_solution


def identify_sameTrains_sameTrack_sameTime(track_info):
    duplicates = []
    for track, values in track_info.track_sequences_of_TPN.items():
        seen = {}
        for entries in values:
            time = entries[1]
            if not time in seen.keys():
                seen[time] = [track, entries]
            else:
                seen[time].append([track, entries])
        for k, v in seen.items():
            if len(v) > 2:
                duplicates.append(v)
    return duplicates


def clone_train_to_reroute(nodesSBB, trainID, train_to_reroute):
    cloned_train = ap.clone_train(trainID)
    cloned_train = ap.cut_single_train_AreaOfInterest(cloned_train, nodesSBB)
    cloned_train['idx'] = train_to_reroute['idx']
    if cloned_train['TrainPathNodes'][cloned_train['idx'][0]]['SequenceNumber'] != \
            train_to_reroute['TrainPathNodes'][train_to_reroute['idx'][0]]['SequenceNumber']:
        cloned_train['idx'] = None


def clone_train_to_reroute_cut_to_length_initial_train(train_to_clone):
    cloned_train = ap.clone_train(train_to_clone['ID'])
    sequence_start = train_to_clone['TrainPathNodes'][0]['SequenceNumber']

    sequence = 0
    node_list = list()
    for node in cloned_train['TrainPathNodes'][sequence_start:]:
        if sequence > len(train_to_clone['TrainPathNodes']) - 1:
            break
        elif node['SequenceNumber'] == train_to_clone['TrainPathNodes'][sequence]['SequenceNumber']:
            node_list.append(node)
            sequence += 1
    cloned_train['TrainPathNodes'] = node_list
    cloned_train['idx'] = train_to_clone['idx']

    return cloned_train


def increase_weight_of_closed_tracks(G_infra, possessions, weight=10e8):
    '''
    :param G_infra:
    :param possessions: list of all closed tracks, return of viriato method closed section tracks
    :param weight: new weight for the closed tracks, has to be very very high
    :return: updated infrastructure graph : closed tracks have new weight, initial_weight stored as an attribute and
    closed_SectionTracksID is a list of all closed tracks
    '''
    closed_SectionTrackIDs = []
    for track in possessions:
        if track['SectionTrackID'] not in closed_SectionTrackIDs:
            closed_SectionTrackIDs.append(track['SectionTrackID'])
        else:
            continue

        # get the closed track in the infrastructure graph

        tracks_between_nodes = G_infra[track['FromNodeID']][track['ToNodeID']]
        for key, value in tracks_between_nodes.items():
            if value['sectionTrackID'] != track['SectionTrackID']:
                continue
            else:
                G_infra[track['FromNodeID']][track['ToNodeID']][key]['initial_weight'] = \
                    G_infra[track['FromNodeID']][track['ToNodeID']][key]['weight']
                G_infra[track['FromNodeID']][track['ToNodeID']][key]['weight'] = weight

    return G_infra, closed_SectionTrackIDs


def identify_trains_driving_closed_sections(closed_SectionTrackIDs, cut_trains):
    '''
    :param closed_SectionTrackIDs: List of the ID of closed Section tracks
    :param cut_trains: trains driving on network in timewindow
    :return: trains_on_closed_tracks: list containing dicts of all trains, driving on any closed track, train[key = 'idx'] returns the index in
            train path nodes where the closed track is used
            trains_closed_track_at_end: list of trains where track is closed between 1st and 2nd node
    '''

    # identify all trains driving on closed tracks
    i = 0
    trains_on_closed_tracks = []
    trains_closed_track_at_beginning = []
    for train in cut_trains:
        i += 1
        # if train['ID'] == 3798349:
        #    print('wait')
        for train_path_node in train['TrainPathNodes']:
            if train_path_node['SectionTrackID'] in closed_SectionTrackIDs:
                idx = train['TrainPathNodes'].index(train_path_node)
                if idx == 1:
                    trains_closed_track_at_beginning.append(train)
                    break

                train_on_closed_track = train.copy()
                train_on_closed_track['idx'] = idx
                # train found driving on closed track
                trains_on_closed_tracks.append(train_on_closed_track)
                break
    return trains_on_closed_tracks, trains_closed_track_at_beginning


class TrackInformation:
    def __init__(self, cut_trains_to_area, closed_SectionTrackIDs, G_infra):
        self.name = 'TrackInfo'
        nr_usage_tracks, trains_on_closed_tracks, track_sequences_of_TPN, tpn_info, trainID_debugString, \
            tuple_key_value_of_tpn_ID_arrival, tuple_key_value_of_tpn_ID_departure = \
                used_tracks_all_trains(cut_trains_to_area, closed_SectionTrackIDs, G_infra)

        self.nr_usage_tracks = nr_usage_tracks
        self.trains_on_closed_tracks = trains_on_closed_tracks
        self.track_sequences_of_TPN = track_sequences_of_TPN
        self.tpn_information = tpn_info
        self.trainID_debugString = trainID_debugString
        self.tuple_key_value_of_tpn_ID_arrival = tuple_key_value_of_tpn_ID_arrival
        self.tuple_key_value_of_tpn_ID_departure = tuple_key_value_of_tpn_ID_departure



class Track_info_to_none:
    def __init__(self):
        self.name = 'track_info_empty'

        self.nr_usage_tracks = None
        self.trains_on_closed_tracks = None
        self.track_sequences_of_TPN = None
        self.tpn_information = None
        self.trainID_debugString = None
        self.tuple_key_value_of_tpn_ID_arrival = None
        self.tuple_key_value_of_tpn_ID_departure = None

def used_tracks_all_trains(cut_trains, closed_SectionTrackIDs, G_infra):
    nr_usage_tracks = dict()  # dictionary of tracks used by train, key: train ID , value: number of usage
    trainID_debugString = dict()  # dictionary, key: trainID, value:debugstring
    tpn_information = dict()  # key : tpn_id value : {arrTime, depTime, fromNode, toNode, sectionTrackID,
    #                                                  sectionTrackID_toNextNode, nextNodeID, nextTPN_ID}
    tuple_key_value_of_tpn_ID_arrival = dict()
    tuple_key_value_of_tpn_ID_departure = dict()

    # dictionary of the sequences of train path nodes, key:(sectionTrackID, fromNode, toNode) value: [TPN1, TPN2...]
    track_sequences_of_TPN = dict()

    # identify all trains driving on closed tracks
    trains_on_closed_tracks = []

    for train in cut_trains:
        # train leaving and entering AoI at Node 300
        # if train['DebugString'] == 'FV IR 505 tt_(IR)':
        # print('wait')
        trainID_debugString[train['ID']] = train['DebugString']

        used_tracks_single_train(closed_SectionTrackIDs, nr_usage_tracks, tpn_information, track_sequences_of_TPN,
                                 train, trains_on_closed_tracks, tuple_key_value_of_tpn_ID_arrival,
                                 tuple_key_value_of_tpn_ID_departure, idx_start_delay=0)

    # sort the sequence tracks dict by arrival time from early to late
    for k, v in track_sequences_of_TPN.items():
        #        f.write(str(k))
        track_sequences_of_TPN[k] = sorted(v, key=itemgetter(1))

    return nr_usage_tracks, trains_on_closed_tracks, track_sequences_of_TPN, tpn_information, trainID_debugString,\
           tuple_key_value_of_tpn_ID_arrival, tuple_key_value_of_tpn_ID_departure


def used_tracks_single_train(closed_SectionTrackIDs, nr_usage_tracks, tpn_information, track_sequences_of_TPN, train,
                             trains_on_closed_tracks, tuple_key_value_of_tpn_ID_arrival,
                             tuple_key_value_of_tpn_ID_departure, idx_start_delay):
    start_train = True
    on_closed_track = False
    tpn_node_index = -1
    for node in train['TrainPathNodes'][idx_start_delay:]:
        tpn_node_index += 1
        if start_train:
            if node['SectionTrackID'] in closed_SectionTrackIDs:
                idx = train['TrainPathNodes'].index(node)
                train_on_closed_track['idx'] = [idx]
                # train found driving on closed track
                trains_on_closed_tracks.append(train_on_closed_track)

            tpn_information[node['ID']] = {
                'ArrivalTime': datetime.datetime.strptime(node['ArrivalTime'], "%Y-%m-%dT%H:%M:%S"),
                'DepartureTime': datetime.datetime.strptime(node['DepartureTime'], "%Y-%m-%dT%H:%M:%S"),
                'RunTime': None, 'fromNode': None, 'toNode': node['NodeID'], 'SectionTrack': None,
                'nextTPN_ID': train['TrainPathNodes'][tpn_node_index + 1]['ID'], 'TrainID': train['ID']}

            nextTPN_SectionTrack = train['TrainPathNodes'][tpn_node_index + 1]['SectionTrackID']
            nextTPN_NodeID = train['TrainPathNodes'][tpn_node_index + 1]['NodeID']
            tuple_key = (None, None, node['NodeID'], 'arrival')
            if not track_sequences_of_TPN.__contains__(tuple_key):
                track_sequences_of_TPN[tuple_key] = []
            track_sequences_of_TPN[tuple_key].append([node['ID'], node['ArrivalTime'], train['ID']])
            tuple_key_value_of_tpn_ID_arrival[node['ID']] = [tuple_key, [node['ID'], node['ArrivalTime'], train['ID']]]

            tuple_key = (nextTPN_SectionTrack, node['NodeID'], nextTPN_NodeID, 'departure')
            if not track_sequences_of_TPN.__contains__(tuple_key):
                track_sequences_of_TPN[tuple_key] = []
            track_sequences_of_TPN[tuple_key].append([node['ID'], node['DepartureTime'], train['ID']])
            tuple_key_value_of_tpn_ID_departure[node['ID']] = [tuple_key,
                                                               [node['ID'], node['DepartureTime'], train['ID']]]
            fromNode = node['NodeID']

            start_train = False
            continue

        if node['SectionTrackID'] in closed_SectionTrackIDs:
            if on_closed_track:
                # train has multiple paths on closed tracks
                train_on_closed_track['idx'].extend([train['TrainPathNodes'].index(node)])
            else:
                train_on_closed_track = train.copy()
                idx = train['TrainPathNodes'].index(node)
                train_on_closed_track['idx'] = [idx]
                # train found driving on closed track
                trains_on_closed_tracks.append(train_on_closed_track)
                on_closed_track = True

        # create the track sequence of TPN list/dict
        toNode = node['NodeID']

        if fromNode == toNode:
            # this train has probably been cut to time window & AoI, leaves area and enters again the same node later
            tpn_information[node['ID']] = {
                'ArrivalTime': datetime.datetime.strptime(node['ArrivalTime'], "%Y-%m-%dT%H:%M:%S"),
                'DepartureTime': datetime.datetime.strptime(node['DepartureTime'], "%Y-%m-%dT%H:%M:%S"),
                'RunTime': None, 'fromNode': None, 'toNode': node['NodeID'], 'SectionTrack': None,
                'nextTPN_ID': train['TrainPathNodes'][tpn_node_index + 1]['ID'], 'TrainID': train['ID']}

            nextTPN_SectionTrack = train['TrainPathNodes'][tpn_node_index + 1]['SectionTrackID']
            nextTPN_NodeID = train['TrainPathNodes'][tpn_node_index + 1]['NodeID']
            tuple_key = (None, None, node['NodeID'], 'arrival')
            if not track_sequences_of_TPN.__contains__(tuple_key):
                track_sequences_of_TPN[tuple_key] = []
            track_sequences_of_TPN[tuple_key].append([node['ID'], node['ArrivalTime'], train['ID']])
            tuple_key_value_of_tpn_ID_arrival[node['ID']] = [tuple_key,
                                                             [node['ID'], node['ArrivalTime'], train['ID']]]

            tuple_key = (nextTPN_SectionTrack, node['NodeID'], nextTPN_NodeID, 'departure')
            if not track_sequences_of_TPN.__contains__(tuple_key):
                track_sequences_of_TPN[tuple_key] = []
            track_sequences_of_TPN[tuple_key].append([node['ID'], node['DepartureTime'], train['ID']])
            tuple_key_value_of_tpn_ID_departure[node['ID']] = [tuple_key,
                                                               [node['ID'], node['ArrivalTime'], train['ID']]]
            fromNode = toNode
            continue
        else:
            tuple_key = (node['SectionTrackID'], fromNode, toNode, 'arrival')
            if not track_sequences_of_TPN.__contains__(tuple_key):
                track_sequences_of_TPN[tuple_key] = []
            track_sequences_of_TPN[tuple_key].append([node['ID'], node['ArrivalTime'], train['ID']])
            tuple_key_value_of_tpn_ID_arrival[node['ID']] = [tuple_key,
                                                             [node['ID'], node['ArrivalTime'], train['ID']]]

            train_ends = False
            if tpn_node_index + 1 < len(train['TrainPathNodes']):
                nextTPN_ID = train['TrainPathNodes'][tpn_node_index + 1]['ID']
                nextTPN_NodeID = train['TrainPathNodes'][tpn_node_index + 1]['NodeID']
                nextTPN_SectionTrack = train['TrainPathNodes'][tpn_node_index + 1]['SectionTrackID']
            else:
                nextTPN_ID = None
                nextTPN_NodeID = None
                nextTPN_SectionTrack = None
                train_ends = True

            if nextTPN_NodeID is not None and toNode != nextTPN_NodeID:
                tuple_key = (nextTPN_SectionTrack, toNode, nextTPN_NodeID, 'departure')
                if not track_sequences_of_TPN.__contains__(tuple_key):
                    track_sequences_of_TPN[tuple_key] = []
                track_sequences_of_TPN[tuple_key].append([node['ID'], node['DepartureTime'], train['ID']])
                tuple_key_value_of_tpn_ID_departure[node['ID']] = [tuple_key,
                                                                   [node['ID'], node['DepartureTime'], train['ID']]]

            train_leaves_area = False
            if toNode == nextTPN_NodeID:
                train_leaves_area = True
                nextTPN_ID = None
                nextTPN_NodeID = None
                nextTPN_SectionTrack = None

                tuple_key = (nextTPN_SectionTrack, toNode, None, 'departure')
                if not track_sequences_of_TPN.__contains__(tuple_key):
                    track_sequences_of_TPN[tuple_key] = []
                track_sequences_of_TPN[tuple_key].append([node['ID'], node['DepartureTime'], train['ID']])
                tuple_key_value_of_tpn_ID_departure[node['ID']] = [tuple_key, [node['ID'],
                                                                               node['DepartureTime'], train['ID']]]

            if not train_leaves_area and train_ends:  # end of a train
                tuple_key = (nextTPN_SectionTrack, toNode, nextTPN_NodeID, 'departure')
                if not track_sequences_of_TPN.__contains__(tuple_key):
                    track_sequences_of_TPN[tuple_key] = []
                track_sequences_of_TPN[tuple_key].append([node['ID'], node['DepartureTime'], train['ID']])
                tuple_key_value_of_tpn_ID_departure[node['ID']] = [tuple_key,
                                                                   [node['ID'], node['DepartureTime'], train['ID']]]

            tpn_information[node['ID']] = {
                'ArrivalTime': datetime.datetime.strptime(node['ArrivalTime'], "%Y-%m-%dT%H:%M:%S"),
                'DepartureTime': datetime.datetime.strptime(node['DepartureTime'], "%Y-%m-%dT%H:%M:%S"),
                'RunTime': utils.duration_transform_to_timedelta(node['MinimumRunTime']),
                'fromNode': fromNode, 'toNode': toNode, 'SectionTrack': node['SectionTrackID'],
                'nextTPN_ID': nextTPN_ID, 'TrainID': train['ID']}

            fromNode = toNode

        # get the amount of trains running on a track
        if node['SectionTrackID'] is None:
            # neglect the first node for the distance calculation
            continue
        if not nr_usage_tracks.__contains__(node['SectionTrackID']):
            nr_usage_tracks[node['SectionTrackID']] = 1
        else:
            nr_usage_tracks[node['SectionTrackID']] += 1
