import utils
import datetime
import algorithm_platform_methods as ap
import networkx as nx
from operator import itemgetter


def restore_disruption_feasibility(G_infra, track_info, parameters):
    # restore disruption feasibility
    # print('\n Start of the restore disruption feasibility')
    changed_trains = {}

    for train_disruption_infeasible in track_info.trains_on_closed_tracks:
        train_original = train_disruption_infeasible.copy()
        # Todo Remember: train_to_reroute['idx'] = idx_trainPathNode_not_reached
        trainID = train_disruption_infeasible['ID']
        # print('TrainID: ', trainID)
        # if train_disruption_infeasible['DebugString'] == 'RVZH S 20 tt_(S)':
        #    print('wait')

        # if a train is disrupted to the last node, cancel train from the last commercial stop
        if train_disruption_infeasible['TrainPathNodes'][train_disruption_infeasible['idx'][0]] == \
                train_disruption_infeasible['TrainPathNodes'][-1] or \
                train_disruption_infeasible['TrainPathNodes'][train_disruption_infeasible['idx'][0]] == \
                train_disruption_infeasible['TrainPathNodes'][-2]:

            # find the last commercial stop before the node which is reached by the disrupted track
            i = 1
            while not train_disruption_infeasible['TrainPathNodes'][train_disruption_infeasible['idx'][0] - i][
                      'StopStatus'] == 'commercialStop':
                i += 1
            # once the last node with comm stop reached, cancel train from here
            train_disruption_infeasible['CancelTrainFrom'] = train_disruption_infeasible['TrainPathNodes'][
                train_disruption_infeasible['idx'][0] - i]['ID']

            dbg_string = train_disruption_infeasible['DebugString']
            remove_tpn_info_canceled_train(track_info, train_disruption_infeasible, train_disruption_infeasible['CancelTrainFrom'])
            changed_trains[trainID] = {'train_id': trainID, 'DebugString': dbg_string, 'Action': 'CancelFrom',
                                       'tpn_cancel_from': train_disruption_infeasible['CancelTrainFrom']}

            train_disruption_infeasible = ap.cancel_train_from(train_disruption_infeasible['CancelTrainFrom'])
            # print('train canceled')
            continue

        # try to reroute the train, if it is not possible, apply cancel cancel from or short turn, depending on case
        train_disruption_infeasible = try_to_reroute_train_via_different_nodes(train_disruption_infeasible, G_infra,
                                                                               parameters)

        if train_disruption_infeasible['Reroute']:
            rerouted_train = ap.reroute_train(train_disruption_infeasible, train_disruption_infeasible['ReroutePath'],
                                              train_disruption_infeasible['indexReroute'], track_info, G_infra)
            # print(train_disruption_infeasible['ID'] == rerouted_train['ID']) ##
            cut_train_idx = None
            if rerouted_train is not None:
                cut_train_idx = rerouted_train['CutIndices']
            else:
                pass
                # print('what to do with this train, check ap reroute method...')
            run_time = ap.calculate_run_times(rerouted_train['ID'])
            # update time from start of rerouting
            train_disruption_infeasible = update_rr_train_times_feasible_path(rerouted_train, run_time, track_info, G_infra)
            if train_disruption_infeasible['CancelTrainID'] is not None:
                dbg_string = train_disruption_infeasible['DebugString']
                remove_tpn_info_canceled_train(track_info, train_disruption_infeasible)
                train_disruption_infeasible = ap.cancel_train(train_disruption_infeasible['CancelTrainID'])
                # print('train canceled')
                changed_trains[trainID] = {'train_id': trainID, 'DebugString': dbg_string, 'Action': 'Cancel'}
                continue
            elif train_disruption_infeasible['CancelTrainFrom'] is not None:
                dbg_string = train_disruption_infeasible['DebugString']
                remove_tpn_info_canceled_train(track_info, train_disruption_infeasible,
                                               tpnID_cancel_from=train_disruption_infeasible['CancelTrainFrom'])
                changed_trains[trainID] = {'train_id': trainID, 'DebugString': dbg_string, 'Action': 'CancelFrom',
                                           'tpn_cancel_from': train_disruption_infeasible['CancelTrainFrom']}
                train_disruption_infeasible = ap.cancel_train_from(train_disruption_infeasible['CancelTrainFrom'])


                continue
            update_tpn_information(0, track_info, train_disruption_infeasible, train_original)
            # edge_test = graphcreator.trains_for_transit_nodes_and_edges([train_disruption_infeasible], nodesSBB=None)
            changed_trains[trainID] = {'train_id': trainID, 'DebugString': train_disruption_infeasible['DebugString'], 'Action': 'Reroute',
                                       'StartEndRR_tpnID': train_disruption_infeasible['TrainPathNodes'][train_disruption_infeasible['StartEndRR_idx'][0]]['ID'],
                                       'add_stop_time': train_disruption_infeasible['add_stop_time']}

            if cut_train_idx is not None:
                train_disruption_infeasible['TrainPathNodes'] = train_disruption_infeasible['TrainPathNodes'][
                                                              cut_train_idx[0]:cut_train_idx[1]]
            # check feasibility
        elif train_disruption_infeasible['CancelTrainID'] is not None:
            # print(trainID)
            # print(train_disruption_infeasible['CancelTrainID'])
            dbg_string = train_disruption_infeasible['DebugString']
            remove_tpn_info_canceled_train(track_info, train_disruption_infeasible)
            train_disruption_infeasible = ap.cancel_train(train_disruption_infeasible['CancelTrainID'])
            changed_trains[trainID] = {'DebugString': dbg_string, 'Action': 'Cancel'}
            # print('train canceled')
        elif train_disruption_infeasible['CancelTrainFrom'] is not None:
            dbg_string = train_disruption_infeasible['DebugString']
            train_before_cancel = train_disruption_infeasible
            remove_tpn_info_canceled_train(track_info, train_disruption_infeasible,
                                           tpnID_cancel_from=train_disruption_infeasible['CancelTrainFrom'])
            train_disruption_infeasible = ap.cancel_train_from(train_disruption_infeasible['CancelTrainFrom'])
            changed_trains[trainID] = {'train_id': trainID, 'DebugString': dbg_string, 'Action': 'CancelFrom',
                                       'tpn_cancel_from': train_disruption_infeasible['CancelTrainFrom']}

           #  print('train canceled from node')
        elif train_disruption_infeasible['ShortTurn']:
            dbg_string = train_disruption_infeasible['DebugString']
            tpn_initial_train = train_disruption_infeasible['TrainPathNodes'].copy()
            cancel_idx = train_disruption_infeasible['idx']
            cancel_tpnIDS = [train_disruption_infeasible['TrainPathNodes'][cancel_idx[0]-1],
                             train_disruption_infeasible['TrainPathNodes'][cancel_idx[1]]]

            remove_tpn_info_short_turned_train(track_info, train_disruption_infeasible)
            #   copy original train twice, cancel original, cancel 1st train just before closed track, cancel 2nd copy until after closed track passed twice
            train_before_disruption, train_after_disruption = short_turn_train(parameters, train_disruption_infeasible)
            # print('ShortTurn this train, several time on closed track')
            changed_trains[trainID] = {'train_id': trainID, 'DebugString': dbg_string, 'Action': 'ShortTurn', 'initial_tpn': tpn_initial_train,
                                       'tpns_cancel_from_to': cancel_tpnIDS, 'train_before': train_before_disruption,
                                       'train_after': train_after_disruption}

    return changed_trains


def try_to_reroute_train_via_different_nodes(train, G_infra, parameters):
    '''
    :param train:
    :param closed_SectionTrackIDs:
    :param G_infra:
    :param weight_closed_track:
    :return: Returns the train as well the options which should be considered to apply
    '''

    train['CancelTrainID'] = None
    train['CancelTrainFrom'] = None
    train['Reroute'] = False
    train['ShortTurn'] = False

    if len(train['idx']) == 1:
        idx = train['idx'][0]
    else:

        # print('This train runs several time on a closed track, what should we do ?')
        train['ShortTurn'] = True
        return train

    # print('\n Start rerouting train with id :', train['ID'])
    #   identify not reached node because of closed section track
    sequence_nr_not_reached = train['TrainPathNodes'][idx]['SequenceNumber']

    # identify the last node which can be reached and has commercial stop
    i = 1
    while train['TrainPathNodes'][idx - i][
        'StopStatus'] != 'commercialStop':  # and len(nodes(train['TrainPathNodes'][idx-i]['Nodeid'])['NodeTracks']) > 0:
        if train['TrainPathNodes'][idx - i] == train['TrainPathNodes'][0]:
            # print(' start of train reached before commercial stop with node tracks')
            train['CancelTrainID'] = train['ID']
            return train
        i += 1
    reroute_from_node = train['TrainPathNodes'][idx - i]
    if len(ap.nodes(reroute_from_node['NodeID'])['NodeTracks']) == 0:
        # print('This train has its first comm stop with no Node Tracks available, not possible to reroute')
        # check if this node is the first node of the train
        if reroute_from_node['SequenceNumber'] == train['TrainPathNodes'][0]['SequenceNumber']:
            # todo Double check if this if is even necessary, seems like a second check of same condition
            train['CancelTrainID'] = train['ID']
            return train
        else:
            cancel_train_from_node = True
            train['CancelTrainFrom'] = reroute_from_node['ID']
            return train

    index_reroute = [idx - i, None]

    # identify the node which can not be reached and has commercial stop
    i = 0
    while train['TrainPathNodes'][idx + i]['StopStatus'] != 'commercialStop':
        if train['TrainPathNodes'][idx + i] == train['TrainPathNodes'][-1]:
            # print(' End of train reached before commercial stop')
            train['CancelTrainFrom'] = reroute_from_node['ID']
            return train
        i += 1
    reroute_to_node = train['TrainPathNodes'][idx + i]
    if len(ap.nodes(reroute_to_node['NodeID'])['NodeTracks']) == 0:
        # print('This train has its next comm stop with no Node Tracks available, not possible to reroute')
        # print(' next comm stop of train is a DST, cancel this train from reroute_from_node')
        if reroute_from_node['ID'] == train['TrainPathNodes'][0]['ID']:
            # if it is the first node of train path, cancel train
            train['CancelTrainID'] = train['ID']
        else:
            train['CancelTrainFrom'] = reroute_from_node['ID']
        return train

    index_reroute[1] = idx + i

    try:  # try to find a path to the node not reached
        # with the cutoff of the weight_closed_tracks we make sure the path via this edges is not considered
        path = nx.single_source_dijkstra(G_infra, reroute_from_node['NodeID'], reroute_to_node['NodeID'],
                                         cutoff=parameters.weight_closed_tracks)
        # index_reroute[1] = idx
        path_found = True
        train['Reroute'] = True
        train['ReroutePath'] = path
        train['indexReroute'] = index_reroute
        return train
    except nx.NetworkXNoPath:
        path_found = False
        # print('No path found to the node not reached, will try to reroute to node after the closed track')

    #   if no path found identify the node after the closed section track
    if sequence_nr_not_reached != train['TrainPathNodes'][-1]['SequenceNumber'] and not path_found:
        #   look in TrainPathNodes until the next commercial stop is found
        n = index_reroute[1] + 1
        while train['TrainPathNodes'][n]['StopStatus'] != 'commercialStop':
            if train['TrainPathNodes'][n] == train['TrainPathNodes'][-1]:
                # print('end of train reached before next commercial stop')
                train['CancelTrainFrom'] = reroute_from_node['ID']
                return train
            n += 1
        first_node_after_closure = train['TrainPathNodes'][n]['NodeID']
        try:
            path = nx.single_source_dijkstra(G_infra, reroute_from_node['NodeID'], first_node_after_closure,
                                             cutoff=parameters.weight_closed_tracks)
            index_reroute[1] = n
            viriato_node_rerouted_to = ap.nodes(first_node_after_closure)
            path_found = True
            train['Reroute'] = True
            train['ReroutePath'] = path
            train['indexReroute'] = index_reroute
            return train

        except nx.NetworkXNoPath:
            path_found = False
            # print('No path found to the next node of not reached node')
            train['CancelTrainFrom'] = reroute_from_node['ID']
            return train

    elif sequence_nr_not_reached == train['TrainPathNodes'][-1]['SequenceNumber'] and not path_found:
        # print('end of train reached before new node for rerouting could be found')
        # print('rerouting of train not possible, \n' , 'closed track is before last node of train')
        train['CancelTrainFrom'] = reroute_from_node['ID']
        return train


def update_rr_train_times_feasible_path(train_to_update, run_times, track_info, G_infra):
    train = train_to_update
    # print(train.copy())
    fmt = "%Y-%m-%dT%H:%M:%S"
    if train['DebugString'] == 'RVZH S 20 tt_(S)':
        print('wait')
    run_times_dict = utils.build_dict(run_times['Times'], 'TrainPathNodeID')
    node_idx = train['StartEndRR_idx'][0]  # first rerouted node
    train_path_node_startRR = train['TrainPathNodes'][node_idx]
    # check the arrival time of the start of RR, sometimes it can be changed due to new runtime calculation
    dep_time_last_node_before_RR = datetime.datetime.strptime(train['TrainPathNodes'][node_idx-1]['DepartureTime'], fmt)
    arrival_time_start_RR_initial = datetime.datetime.strptime(train_path_node_startRR['ArrivalTime'], fmt)
    # if a train is rr from the start of a train, it might be that the runtime is 0
    if run_times['Times'][node_idx]['MinimumRunTime'] is not None:
        runtime_to_start_RR = utils.duration_transform_to_timedelta(run_times['Times'][node_idx]['MinimumRunTime'])
    else:
        runtime_to_start_RR = None

    if runtime_to_start_RR is None:
        run_times_dict[train_path_node_startRR['ID']]['MinimumRunTime'] = None
        pass
    elif dep_time_last_node_before_RR + runtime_to_start_RR > arrival_time_start_RR_initial:
        arrival_time_start_RR_initial = dep_time_last_node_before_RR + runtime_to_start_RR
        # assign the correct arrival time to the train path node
        train['TrainPathNodes'][node_idx]['ArrivalTime'] = arrival_time_start_RR_initial.strftime("%Y-%m-%dT%H:%M:%S")
        departure_time_start_RR = arrival_time_start_RR_initial + utils.duration_transform_to_timedelta(run_times['Times'][node_idx]['MinimumStopTime'])
        train['TrainPathNodes'][node_idx]['DepartureTime'] = departure_time_start_RR.strftime("%Y-%m-%dT%H:%M:%S")
    # stop_time = utils.transform_timedelta_to_ISO8601(dep_time_last_node - train_path_node_startRR['ArrivalTime'])
    stop_time = utils.duration_transform_to_timedelta(train_path_node_startRR['MinimumStopTime'])
    tpn_train_added_sequences = []
    visited_tpns = []
    body_message = '{ "TrainId": ' + str(train['ID']) + ', "Times":  ['

    # index for the remaining RR nodes until end of train path
    node_idx_RR = node_idx
    start_TT_update = True
    runtime_RR_train_feasible = {}
    last_node_of_train = False

    add_stop_time = datetime.timedelta(minutes=0)

    for node in train['TrainPathNodes'][node_idx:]:
        # loop through all nodes beginning at first rerouted node
        if node['ID'] not in visited_tpns and not last_node_of_train:
            j = 0
            run_times_of_section = dict()  # key NodeID, value runtime from calculate runtime
            tpn_rr_train = dict()  # key, TPN ID, value, all entries in initial train tpn
            # tpn_section[runTimes RR Train tpn of Section, node IDs of Section, trainID, RR Train tpn of Section]
            tpn_section = [run_times_of_section, [node['ID']], train['ID'], tpn_rr_train]
            # first tpn node of section (should be a station to check dep. time feasibility)
            run_times_of_section[node['ID']] = run_times_dict[train['TrainPathNodes'][node_idx_RR + j]['ID']]
            tpn_rr_train[node['ID']] = train['TrainPathNodes'][node_idx_RR + j]
            j += 1
            if node_idx_RR + j == len(train['TrainPathNodes'])-1:
                last_node_of_train = True
            #    print('Last Node of Train reached')
            # find tpn of section until next stations, where a train could be parked potentially
            while G_infra.nodes[train['TrainPathNodes'][node_idx_RR + j]['NodeID']]['NodeTracks'] is None:
                tpn_section[1].extend([train['TrainPathNodes'][node_idx_RR + j]['ID']])
                tpn_rr_train[train['TrainPathNodes'][node_idx_RR + j]['ID']] = train['TrainPathNodes'][node_idx_RR + j]
                # find the run_time to next node
                run_times_of_section[train['TrainPathNodes'][node_idx_RR + j]['ID']] = run_times_dict[train['TrainPathNodes'][node_idx_RR + j]['ID']]
                j += 1
                # check if we reach the last node of the train, probably all end of trains have nodeTracks
                if node_idx_RR + j == len(train['TrainPathNodes'])-1:
                    if G_infra.nodes[train['TrainPathNodes'][node_idx_RR + j]['NodeID']]['NodeTracks'] is None:
                        print('this train ends at a station without station tracks')
                        break

            tpn_section[1].append(train['TrainPathNodes'][node_idx_RR + j]['ID'])
            run_times_of_section[train['TrainPathNodes'][node_idx_RR + j]['ID']] = run_times_dict[train['TrainPathNodes'][node_idx_RR + j]['ID']]
            tpn_rr_train[train['TrainPathNodes'][node_idx_RR + j]['ID']] = train['TrainPathNodes'][node_idx_RR + j]
            if node_idx_RR + j == len(train['TrainPathNodes'])-1:
                last_node_of_train = True
            node_idx_RR += j
            # Selection of the departure Times for the inputs into the greedy feasibility check
            if start_TT_update:
                # For this node I have to take the Departure Time of initial RR train and not from Runtime calculation
                dep_time_sectionStart = tpn_rr_train[node['ID']]['DepartureTime']
                dep_time_tpn = dep_time_sectionStart

            else:
                try:
                    arr_time_tpn_string = runtime_RR_train_feasible[node['ID']]['ArrivalTime']
                except KeyError:
                    print('how that ?')
                arr_time_tpn = datetime.datetime.strptime(arr_time_tpn_string, fmt)
                dep_time_tpn = arr_time_tpn + utils.duration_transform_to_timedelta(
                    run_times_of_section[node['ID']]['MinimumStopTime'])

            # find the free capacity for all nodes until next station in tracks_used_in_section
            section_clear = False
            nr_iterations = 0

            tpn_section_added_sequences = None
            while not section_clear and nr_iterations <= 15:
                nr_iterations += 1
                # last node is a tuple (dep_time_last_node, nodeID)
                if start_TT_update:
                    if isinstance(dep_time_tpn, str):
                        dep_time_tpn = datetime.datetime.strptime(dep_time_tpn, "%Y-%m-%dT%H:%M:%S")

                    dep_time_tpn_before = dep_time_tpn
                    section_clear, dep_time_tpn, runtime_section_feasible, tpn_section_added_sequences = \
                        check_train_section_runtime_feasibility(dep_time_tpn, track_info, tpn_section, G_infra)
                    if section_clear:
                        runtime_RR_train_feasible.update(runtime_section_feasible)
                        start_TT_update = False
                        # Update the arrival time of the last tpn of the section, is the input for the next section !!!
                        tpn_section[3][tpn_section[1][-1]]['ArrivalTime'] = runtime_RR_train_feasible[tpn_section[1][-1]]['ArrivalTime']
                        tpn_train_added_sequences.append(tpn_section_added_sequences)
                        visited_tpns.extend([x for x in tpn_section[1][0:-1]])
                else:
                    section_clear, dep_time_tpn, runtime_section_feasible, tpn_section_added_sequences = \
                        check_train_section_runtime_feasibility(dep_time_tpn, track_info, tpn_section, G_infra)
                    if section_clear:
                        runtime_RR_train_feasible.update(runtime_section_feasible)
                        # Update the arrival time of the last tpn of the section, is the input for the next section !!!
                        tpn_section[3][tpn_section[1][-1]]['ArrivalTime'] = runtime_RR_train_feasible[tpn_section[1][-1]]['ArrivalTime']
                        tpn_train_added_sequences.append(tpn_section_added_sequences)
                        visited_tpns.extend([x for x in tpn_section[1][0:-1]])
                # delay the departure time of the last node by one minute and try again
                if not section_clear:
                    add_stop_time += dep_time_tpn - dep_time_tpn_before
            # tpn_train_added_sequences.append(tpn_section_added_sequences)

            if not section_clear:
                train_updated_times = train_to_update
                if train_to_update['StartEndRR_idx'][0] > 10:
                    train_updated_times['CancelTrainFrom'] = train_to_update['TrainPathNodes'][train_to_update['StartEndRR_idx'][0]]['ID']
                    train_updated_times['CancelTrainID'] = None
                else:
                    train_updated_times['CancelTrainFrom'] = None
                    train_updated_times['CancelTrainID'] = train_updated_times['ID']
                return train_updated_times
        # if node has been visited, continue
        else:
            continue

    last_nodeID_of_train = node['ID']
    stop_time_last_node = utils.duration_transform_to_timedelta(runtime_RR_train_feasible[last_nodeID_of_train]['MinimumStopTime'])
    arrival_time_last_node = datetime.datetime.strptime(runtime_RR_train_feasible[last_nodeID_of_train]['ArrivalTime'], fmt)
    departure_time_last_node = datetime.datetime.strftime((arrival_time_last_node + stop_time_last_node), fmt)
    runtime_RR_train_feasible[last_nodeID_of_train]['DepartureTime'] = departure_time_last_node

    time_msg = str()
    start = True
    for tpn, value in runtime_RR_train_feasible.items():
        if start:
            start = False
            if value['RunTime'] is None:
                str_runtime = ' '
            else:
                str_runtime = ', "MinimumRunTime": "' + value['RunTime'] + '"'
            time_msg += '\n  {' \
                        '"TrainPathNodeID": ' + str(tpn) + ',' \
                        ' "ArrivalTime": "' + value['ArrivalTime'] + '",' + \
                        ' "DepartureTime": "' + value['DepartureTime'] + '"' + \
                        str_runtime + '}'
        else:
            time_msg += ',\n  {' \
                           '"TrainPathNodeID": ' + str(tpn) + ',' \
                           ' "ArrivalTime": "' + value['ArrivalTime'] + '",' + \
                           ' "DepartureTime": "' + value['DepartureTime'] + '",' + \
                           ' "MinimumRunTime": "' + value['RunTime'] + '"' + '}'
    body_message += time_msg

    # train_path_node_afterRR = train['TrainPathNodes'][train['StartEndRR_idx'][1]+1]
    # idx_endRR = train['StartEndRR_idx'][1]
    # idx_afterRR = idx_endRR

    body_message = body_message + ']}'

    debug = False
    if debug:
        file1 = open("update_trainTimes_debug.txt", "w")
        # \n is placed to indicate EOL (End of Line)
        file1.write("body message \n")
        file1.writelines(body_message)
        file1.write("\n\n calculated run times \n")
        file1.write(str(run_times['Times']))
        file1.write("\n\n updated feasible run times \n")
        file1.write(str(runtime_RR_train_feasible))
        file1.close()
    try:
        train_updated_times = ap.post_method(ap.apiUrl() + "/update-train-times", body_message).json()
    except Exception:
        print('wait what ?')
    #  Cut indices before cut to area !
    train_updated_times['CutIndices'] = train['CutIndices']
    train_updated_times['StartEndRR_idx'] = train['StartEndRR_idx']
    train_updated_times['CancelTrainID'] = None
    train_updated_times['CancelTrainFrom'] = None
    train_updated_times['add_stop_time'] = add_stop_time
    # print('update train time successful')
    return train_updated_times


def update_train_times_feasible_path_delay_operator(train_to_delay, time_to_delay, track_info, G_infra, tpn_index_start):
    train = train_to_delay
    # print(train.copy())
    # run_times_dict = utils.build_dict(run_times['Times'], 'TrainPathNodeID')
    node_idx = tpn_index_start  # train is completely delayed from start to end
    train_path_node_start = train['TrainPathNodes'][node_idx]
    # check the arrival time of the start of RR, sometimes it can be changed due to new runtime calculation
    dep_time_start = datetime.datetime.strptime(train['TrainPathNodes'][node_idx]['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
    dep_time_start += datetime.timedelta(minutes=time_to_delay)
    # if a train is rr from the start of a train, it might be that the runtime is 0
    # if run_times['Times'][node_idx]['MinimumRunTime'] is not None: Todo, remove comments for delay partially
    #    runtime_to_start_RR = utils.duration_transform_to_timedelta(run_times['Times'][node_idx]['MinimumRunTime'])
    # else:
    #     runtime_to_start_RR = None

    # if runtime_to_start_RR is None:
    #    run_times_dict[train_path_node_startRR['ID']]['MinimumRunTime'] = None
    #    pass
    # elif dep_time_last_node_before_RR + runtime_to_start_RR > arrival_time_start_RR_initial:
    #    arrival_time_start_RR_initial = dep_time_last_node_before_RR + runtime_to_start_RR
        # assign the correct arrival time to the train path node
    #    train['TrainPathNodes'][node_idx]['ArrivalTime'] = arrival_time_start_RR_initial.strftime("%Y-%m-%dT%H:%M:%S")
    #    departure_time_start_RR = arrival_time_start_RR_initial + utils.duration_transform_to_timedelta(run_times['Times'][node_idx]['MinimumStopTime'])
    #    train['TrainPathNodes'][node_idx]['DepartureTime'] = departure_time_start_RR.strftime("%Y-%m-%dT%H:%M:%S")
    # stop_time = utils.transform_timedelta_to_ISO8601(dep_time_last_node - train_path_node_startRR['ArrivalTime'])

    stop_time = utils.duration_transform_to_timedelta(train_path_node_start['MinimumStopTime'])
    tpn_train_added_sequences = []
    visited_tpns = []
    body_message = '{ "TrainId": ' + str(train['ID']) + ', "Times":  ['

    # index for the remaining RR nodes until end of train path
    node_idx_delay = node_idx
    start_TT_update = True
    runtime_delayed_train_feasible = {}
    last_node_of_train = False

    for node in train['TrainPathNodes'][node_idx:]:
        # loop through all nodes beginning at first rerouted node
        if node['ID'] not in visited_tpns and not last_node_of_train:
            j = 0
            # run_times_of_section = dict()  # key NodeID, value runtime from calculate runtime
            tpns_train = dict()  # key, TPN ID, value, all entries in initial train tpn
            # tpn_section[runTimes RR Train tpn of Section, node IDs of Section, trainID, RR Train tpn of Section]
            tpn_section = [[node['ID']], train['ID'], tpns_train]
            # first tpn node of section (should be a station to check dep. time feasibility)

            tpns_train[node['ID']] = train['TrainPathNodes'][node_idx_delay + j]
            j += 1
            if node_idx_delay + j == len(train['TrainPathNodes'])-1:
                last_node_of_train = True
                # print('Last Node of Train reached')
            # find tpn of section until next stations, where a train could be parked potentially
            try:
                while G_infra.nodes[train['TrainPathNodes'][node_idx_delay + j]['NodeID']]['NodeTracks'] is None and not last_node_of_train:
                    tpn_section[0].extend([train['TrainPathNodes'][node_idx_delay + j]['ID']])
                    tpns_train[train['TrainPathNodes'][node_idx_delay + j]['ID']] = train['TrainPathNodes'][node_idx_delay + j]
                    # find the run_time to next node
                    # run_times_of_section[train['TrainPathNodes'][node_idx_delay + j]['ID']] = run_times_dict[train['TrainPathNodes'][node_idx_delay + j]['ID']]
                    j += 1
                    # check if we reach the last node of the train, probably all end of trains have nodeTracks
                    if node_idx_delay + j == len(train['TrainPathNodes'])-1:
                        if G_infra.nodes[train['TrainPathNodes'][node_idx_delay + j]['NodeID']]['NodeTracks'] is None:
                            # print('this train ends at a station without station tracks')
                            break
            except IndexError:
                pass
                # print('whats wrong here ? ')
            tpn_section[0].append(train['TrainPathNodes'][node_idx_delay + j]['ID'])
            # run_times_of_section[train['TrainPathNodes'][node_idx_delay + j]['ID']] = run_times_dict[train['TrainPathNodes'][node_idx_delay + j]['ID']]
            tpns_train[train['TrainPathNodes'][node_idx_delay + j]['ID']] = train['TrainPathNodes'][node_idx_delay + j]
            if node_idx_delay + j == len(train['TrainPathNodes'])-1:
                last_node_of_train = True
            node_idx_delay += j
            # Selection of the departure Times for the inputs into the greedy feasibility check
            if start_TT_update:
                # For this node I have to take the Departure Time of initial RR train and not from Runtime calculation
                dep_time_sectionStart = dep_time_start
                dep_time_tpn = dep_time_sectionStart

            else:
                try:
                    arr_time_tpn_string = runtime_delayed_train_feasible[node['ID']]['ArrivalTime']
                except KeyError:
                    pass
                    # print('how that ?')
                arr_time_tpn = datetime.datetime.strptime(arr_time_tpn_string, "%Y-%m-%dT%H:%M:%S")
                dep_time_tpn = arr_time_tpn + utils.duration_transform_to_timedelta(tpns_train[node['ID']]['MinimumStopTime'])

            # find the free capacity for all nodes until next station in tracks_used_in_section
            section_clear = False
            nr_iterations = 0

            tpn_section_added_sequences = None
            while not section_clear and nr_iterations <= 15:
                nr_iterations += 1
                # last node is a tuple (dep_time_last_node, nodeID)
                if start_TT_update:
                    if isinstance(dep_time_tpn, str):
                        dep_time_tpn = datetime.datetime.strptime(dep_time_tpn, "%Y-%m-%dT%H:%M:%S")

                    dep_time_tpn_before = dep_time_tpn
                    section_clear, dep_time_tpn, runtime_section_feasible, tpn_section_added_sequences = \
                        check_train_section_runtime_feasibility_delay_operator(dep_time_tpn, track_info, tpn_section, G_infra)
                    if section_clear:
                        runtime_delayed_train_feasible.update(runtime_section_feasible)
                        start_TT_update = False
                        # Update the arrival time of the last tpn of the section, is the input for the next section !!!
                        tpn_section[2][tpn_section[0][-1]]['ArrivalTime'] = runtime_delayed_train_feasible[tpn_section[0][-1]]['ArrivalTime']
                        tpn_train_added_sequences.append(tpn_section_added_sequences)
                        visited_tpns.extend([x for x in tpn_section[0][0:-1]])
                else:
                    section_clear, dep_time_tpn, runtime_section_feasible, tpn_section_added_sequences = \
                        check_train_section_runtime_feasibility_delay_operator(dep_time_tpn, track_info, tpn_section, G_infra)
                    if section_clear:
                        runtime_delayed_train_feasible.update(runtime_section_feasible)
                        # Update the arrival time of the last tpn of the section, is the input for the next section !!!
                        tpn_section[2][tpn_section[0][-1]]['ArrivalTime'] = runtime_delayed_train_feasible[tpn_section[0][-1]]['ArrivalTime']  # tpn_section[1][-1]
                        tpn_train_added_sequences.append(tpn_section_added_sequences)
                        visited_tpns.extend([x for x in tpn_section[0][0:-1]])
                # delay the departure time of the last node by one minute and try again
                if not section_clear:
                    # arrival_time_node = node['ArrivalTime']
                    # arrival_time_node_datetime = datetime.datetime.strptime(arrival_time_node, "%Y-%m-%dT%H:%M:%S")
                    stop_time += dep_time_tpn - dep_time_tpn_before
            # tpn_train_added_sequences.append(tpn_section_added_sequences)

            if not section_clear:
                train_updated_times = train_to_delay
                # if train_to_update['StartEndRR_idx'][0] > 10:
                #    train_updated_times['CancelTrainFrom'] = train_to_update['TrainPathNodes'][train_to_update['StartEndRR_idx'][0]]['ID']
                #    train_updated_times['CancelTrainID'] = None
                #else:
                train_updated_times['Delay'] = 'infeasible'
                return train_updated_times
        # if node has been visited, continue
        else:
            continue

    last_nodeID_of_train = node['ID']
    runtime_delayed_train_feasible[last_nodeID_of_train]['DepartureTime'] = runtime_delayed_train_feasible[last_nodeID_of_train]['ArrivalTime']

    time_msg = str()
    start = True
    for tpn, value in runtime_delayed_train_feasible.items():
        if start:
            start = False
            if value['RunTime'] is None:
                str_runtime = ' '
            else:
                str_runtime = ', "MinimumRunTime": "' + value['RunTime'] + '"'
            time_msg += '\n  {' \
                        '"TrainPathNodeID": ' + str(tpn) + ',' \
                        ' "ArrivalTime": "' + value['ArrivalTime'] + '",' + \
                        ' "DepartureTime": "' + value['DepartureTime'] + '"' + \
                        str_runtime + '}'
        else:
            time_msg += ',\n  {' \
                           '"TrainPathNodeID": ' + str(tpn) + ',' \
                           ' "ArrivalTime": "' + value['ArrivalTime'] + '",' + \
                           ' "DepartureTime": "' + value['DepartureTime'] + '",' + \
                           ' "MinimumRunTime": "' + value['RunTime'] + '"' + '}'
    body_message += time_msg

    body_message = body_message + ']}'

    # print(body_message)
    # print(train['DebugString'])
    debug = False
    if debug:
        file1 = open("update_trainTimes_debug.txt", "w")
        # \n is placed to indicate EOL (End of Line)
        file1.write("body message \n")
        file1.writelines(body_message)
        file1.write("\n\n calculated run times \n")
        file1.write(str(train))
        file1.write("\n\n updated feasible run times \n")
        file1.write(str(runtime_delayed_train_feasible))
        file1.close()
    # try:
    #    train_updated_times = ap.post_method(ap.apiUrl() + "/update-train-times", body_message).json()
    # except Exception:
    #    print('wait what ?')
    #  Cut indices before cut to area !
    # train_updated_times['CutIndices'] = train['CutIndices']
    # train_updated_times['StartEndRR_idx'] = train['StartEndRR_idx']
    # train_updated_times['CancelTrainID'] = None
    # train_updated_times['CancelTrainFrom'] = None
    train['Delay'] = 'feasible'
    train['runtime_delay_feasible'] = runtime_delayed_train_feasible
    train['body_message'] = body_message
    # print('update train time successful')
    return train


def check_train_section_runtime_feasibility(dep_time_tpn_sectionStart, track_info, tpns_section, G_infra, section_start=True):
    dep_time_tpn_string = dep_time_tpn_sectionStart.strftime("%Y-%m-%dT%H:%M:%S")
    dep_time_tpn = dep_time_tpn_sectionStart

    # nodeID_last_tpn = last_node[1]
    runTime_section = tpns_section[0]
    tpn_ids_section = tpns_section[1]
    # tpns = [track_info.tpn_information[x] for x in tpn_ids_section]# tpns_section[2]
    trainID = tpns_section[2]
    tpn_rr_train = tpns_section[3]
    tpn_sequences_added_section = []

    runtime_section_feasible = dict()  # resulting runtime of the new path
    section_clear = False
    tpn_outside_AoI = False
    next_tpn_outside_AoI = False
    last_tpn_outside_AoI = False
    arrival_feasibility_not_needed = False
    no_tpn_sequences_added = False  # in the case this and next node are outside of AoI, no sequences are added
    i = -1

    for tpn_id in tpn_ids_section:
        i += 1
        tpn_clear = False
        tpn = tpn_rr_train[tpn_id]
        tpn_runtime = runTime_section[tpn_id]
        if not G_infra.nodes[tpn['NodeID']]['in_area']:
            tpn_outside_AoI = True
        if i + 1 < len(tpn_ids_section):
            next_tpn_id = tpn_rr_train[tpn_ids_section[i+1]]['NodeID']
            if not G_infra.nodes[next_tpn_id]['in_area']:
                next_tpn_outside_AoI = True
        if i != 0:
            last_tpn_id = tpn_rr_train[tpn_ids_section[i - 1]]['NodeID']
            if not G_infra.nodes[last_tpn_id]['in_area']:
                last_tpn_outside_AoI = True

        # if not infra_graph.edges[tpn['NodeID']]['in_area']
        if section_start:
            if not tpn_outside_AoI and not next_tpn_outside_AoI:
                dep_time_tpn, tpn_clear, tpn_sequences_added = check_tpn_departure_feasibility \
                         (dep_time_tpn_sectionStart, dep_time_tpn_string, i, tpn, tpn_clear,
                          tpn_id, tpn_ids_section, tpn_rr_train, track_info, trainID)
            else:
                tpn_clear = True
            if not tpn_clear:
                return section_clear, dep_time_tpn, runtime_section_feasible, tpn_sequences_added_section
            else:
                if not tpn_outside_AoI and not next_tpn_outside_AoI:
                    tpn_sequences_added_section.append(tpn_sequences_added)
                dep_time_tpn_string = dep_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
                runtime_section_feasible[tpn_id] = {'TrackID': tpn['SectionTrackID'], 'ArrivalTime': tpn['ArrivalTime'],
                                                    'DepartureTime': dep_time_tpn_string,
                                                    'RunTime': tpn_runtime['MinimumRunTime'],
                                                    'MinimumStopTime': tpn_runtime['MinimumStopTime']}
                section_start = False
                continue
        # initialize the travel time
        min_runtime_tpn = utils.duration_transform_to_timedelta(runTime_section[tpn_id]['MinimumRunTime'])
        arrival_time_tpn = dep_time_tpn + min_runtime_tpn

        if tpn_outside_AoI or last_tpn_outside_AoI:
            tpn_clear = True
            arrival_feasibility_not_needed = True
        iter_arrival_check = 0
        while not tpn_clear and iter_arrival_check <= 3:
            iter_arrival_check += 1
            # check arrival time feasibility of tpn, if not feasible increase runtime and check again
            arrival_time_tpn_before = arrival_time_tpn
            arrival_time_tpn, tpn_clear, tpn_sequences_added, deltaT_forDeparture = check_arrival_feasibility_tpn(arrival_time_tpn,  i, tpn,
                                                  tpn_clear, tpn_id, tpn_ids_section, tpn_rr_train, track_info, trainID)

            if not tpn_clear and deltaT_forDeparture is None:
                min_runtime_tpn += arrival_time_tpn - arrival_time_tpn_before
            elif not tpn_clear and deltaT_forDeparture is not None:
                dep_time_tpn_startSection = dep_time_tpn_sectionStart + deltaT_forDeparture
                remove_tpn_added_to_track_info_tpn_sequences(tpn_sequences_added_section, track_info)
                tpn_sequences_added_section = None
                return section_clear, dep_time_tpn_startSection, runtime_section_feasible, tpn_sequences_added_section
        if not arrival_feasibility_not_needed:
            tpn_sequences_added_section.append(tpn_sequences_added)
        if tpn_clear and i + 1 == len(tpn_ids_section):
            # last arrival node of the section
            section_clear = True
            min_runtime_tpn_duration = utils.transform_timedelta_to_ISO8601(min_runtime_tpn)
            arrival_time_tpn_string = arrival_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
            runtime_section_feasible[tpn_id] = {'TrackID': tpn['SectionTrackID'], 'ArrivalTime': arrival_time_tpn_string,
                                                'DepartureTime': None,
                                                'RunTime': min_runtime_tpn_duration,
                                                'MinimumStopTime': tpn_runtime['MinimumStopTime']}
        else:
            # departure feasibility of the node in section
            tpn_clear = False
            # if utils.duration_transform_to_timedelta(tpn_runtime['MinimumStopTime']).seconds != 0:
            #    print(' here we are')
            dep_time_tpn = arrival_time_tpn + utils.duration_transform_to_timedelta(tpn_runtime['MinimumStopTime'])
            if not tpn_outside_AoI and not next_tpn_outside_AoI:
                dep_time_tpn_before = dep_time_tpn
                dep_time_tpn_string = dep_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
                dep_time_tpn, tpn_clear, tpn_sequences_added = check_tpn_departure_feasibility \
                                         (dep_time_tpn, dep_time_tpn_string, i, tpn, tpn_clear,
                                         tpn_id, tpn_ids_section, tpn_rr_train, track_info, trainID)
            else:  # not needed to check the dep feasibility if one of both nodes is outside the area
                tpn_clear = True
                no_tpn_sequences_added = True

            if not tpn_clear:
                deltaT_forDeparture = dep_time_tpn - dep_time_tpn_before
                dep_time_tpn = dep_time_tpn_sectionStart + deltaT_forDeparture
                remove_tpn_added_to_track_info_tpn_sequences(tpn_sequences_added_section, track_info)
                tpn_sequences_added_section = None
                return section_clear, dep_time_tpn, runtime_section_feasible, tpn_sequences_added_section

            else:
                if not no_tpn_sequences_added:
                    tpn_sequences_added_section.append(tpn_sequences_added)
                min_runtime_tpn_duration = utils.transform_timedelta_to_ISO8601(min_runtime_tpn)
                arrival_time_tpn_string = arrival_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
                dep_time_tpn_string = dep_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
                runtime_section_feasible[tpn_id] = {'TrackID': tpn['SectionTrackID'],
                                                    'ArrivalTime': arrival_time_tpn_string,
                                                    'DepartureTime': dep_time_tpn_string,
                                                    'RunTime': min_runtime_tpn_duration,
                                                    'MinimumStopTime': tpn_runtime['MinimumStopTime']}

    return section_clear, dep_time_tpn, runtime_section_feasible, tpn_sequences_added_section


def check_train_section_runtime_feasibility_delay_operator(dep_time_tpn_sectionStart, track_info, tpns_section, G_infra, section_start=True):
    fmt = "%Y-%m-%dT%H:%M:%S"
    dep_time_tpn_string = dep_time_tpn_sectionStart.strftime("%Y-%m-%dT%H:%M:%S")
    dep_time_tpn = dep_time_tpn_sectionStart

    # nodeID_last_tpn = last_node[1]
    # runTime_section = tpns_section[0]
    tpn_ids_section = tpns_section[0]
    # tpns = [track_info.tpn_information[x] for x in tpn_ids_section]# tpns_section[2]
    trainID = tpns_section[1]
    tpns_train = tpns_section[2]
    tpn_sequences_added_section = []

    runtime_section_feasible = dict()  # resulting runtime of the new path
    section_clear = False
    tpn_outside_AoI = False
    next_tpn_outside_AoI = False
    last_tpn_outside_AoI = False
    arrival_feasibility_not_needed = False
    no_tpn_sequences_added = False  # in the case this and next node are outside of AoI, no sequences are added
    i = -1

    for tpn_id in tpn_ids_section:
        i += 1
        tpn_clear = False
        tpn = tpns_train[tpn_id]
        # tpn_runtime = runTime_section[tpn_id]
        if not G_infra.nodes[tpn['NodeID']]['in_area']:
            tpn_outside_AoI = True
        if i + 1 < len(tpn_ids_section):
            next_tpn_id = tpns_train[tpn_ids_section[i+1]]['NodeID']
            if not G_infra.nodes[next_tpn_id]['in_area']:
                next_tpn_outside_AoI = True
        if i != 0:
            last_tpn_NodeId = tpns_train[tpn_ids_section[i - 1]]['NodeID']
            if not G_infra.nodes[last_tpn_NodeId]['in_area']:
                last_tpn_outside_AoI = True

        # if not infra_graph.edges[tpn['NodeID']]['in_area']
        if section_start:
            if not tpn_outside_AoI and not next_tpn_outside_AoI:
                dep_time_tpn, tpn_clear, tpn_sequences_added = check_tpn_departure_feasibility_delay_operator \
                                                    (dep_time_tpn_sectionStart, dep_time_tpn_string, i, tpn, tpn_clear,
                                                     tpn_id, tpn_ids_section, tpns_train, track_info, trainID)
            else:
                tpn_clear = True
            if not tpn_clear:
                return section_clear, dep_time_tpn, runtime_section_feasible, tpn_sequences_added_section
            else:
                if not tpn_outside_AoI and not next_tpn_outside_AoI:
                    tpn_sequences_added_section.append(tpn_sequences_added)
                dep_time_tpn_string = dep_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
                runtime_section_feasible[tpn_id] = {'TrackID': tpn['SectionTrackID'], 'ArrivalTime': tpn['ArrivalTime'],
                                                    'DepartureTime': dep_time_tpn_string,
                                                    'RunTime': tpns_train[tpn_id]['MinimumRunTime'],
                                                    'MinimumStopTime': tpns_train[tpn_id]['MinimumStopTime']}
                section_start = False
                continue
        # initialize the travel time
        run_time = datetime.datetime.strptime(tpns_train[tpn_id]['ArrivalTime'], fmt) - \
                   datetime.datetime.strptime(tpns_train[tpn_ids_section[i - 1]]['DepartureTime'], fmt)
        arrival_time_tpn = dep_time_tpn + run_time

        if tpn_outside_AoI or last_tpn_outside_AoI:
            tpn_clear = True
            arrival_feasibility_not_needed = True
        iter_arrival_check = 0
        while not tpn_clear and iter_arrival_check <= 3:
            iter_arrival_check += 1
            # check arrival time feasibility of tpn, if not feasible increase runtime and check again
            arrival_time_tpn_before = arrival_time_tpn
            arrival_time_tpn, tpn_clear, tpn_sequences_added, deltaT_forDeparture = check_arrival_feasibility_tpn_delay_operator\
                        (arrival_time_tpn,  i, tpn, tpn_clear, tpn_id, tpn_ids_section, tpns_train, track_info, trainID)

            if not tpn_clear and deltaT_forDeparture is None:
                run_time += arrival_time_tpn - arrival_time_tpn_before
            elif not tpn_clear and deltaT_forDeparture is not None:
                dep_time_tpn_startSection = dep_time_tpn_sectionStart + deltaT_forDeparture
                remove_tpn_added_to_track_info_tpn_sequences(tpn_sequences_added_section, track_info)
                tpn_sequences_added_section = None
                return section_clear, dep_time_tpn_startSection, runtime_section_feasible, tpn_sequences_added_section
        if not arrival_feasibility_not_needed:
            tpn_sequences_added_section.append(tpn_sequences_added)
        if tpn_clear and i + 1 == len(tpn_ids_section):
            # last arrival node of the section
            section_clear = True
            min_runtime_tpn_duration = utils.transform_timedelta_to_ISO8601(run_time)
            arrival_time_tpn_string = arrival_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
            runtime_section_feasible[tpn_id] = {'TrackID': tpn['SectionTrackID'], 'ArrivalTime': arrival_time_tpn_string,
                                                'DepartureTime': None,
                                                'RunTime': tpns_train[tpn_id]['MinimumRunTime'],
                                                'MinimumStopTime': tpns_train[tpn_id]['MinimumStopTime']}
        else:
            # departure feasibility of the node in section
            tpn_clear = False
            # if utils.duration_transform_to_timedelta(tpn_runtime['MinimumStopTime']).seconds != 0:
            #    print(' here we are')
            dep_time_tpn = arrival_time_tpn + utils.duration_transform_to_timedelta(tpns_train[tpn_id]['MinimumStopTime'])
            if not tpn_outside_AoI and not next_tpn_outside_AoI:
                dep_time_tpn_before = dep_time_tpn
                dep_time_tpn_string = dep_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
                dep_time_tpn, tpn_clear, tpn_sequences_added = check_tpn_departure_feasibility_delay_operator \
                                         (dep_time_tpn, dep_time_tpn_string, i, tpn, tpn_clear,
                                         tpn_id, tpn_ids_section, tpns_train, track_info, trainID)
            else:  # not needed to check the dep feasibility if one of both nodes is outside the area
                tpn_clear = True
                no_tpn_sequences_added = True

            if not tpn_clear:
                deltaT_forDeparture = dep_time_tpn - dep_time_tpn_before
                dep_time_tpn = dep_time_tpn_sectionStart + deltaT_forDeparture
                remove_tpn_added_to_track_info_tpn_sequences(tpn_sequences_added_section, track_info)
                tpn_sequences_added_section = None
                return section_clear, dep_time_tpn, runtime_section_feasible, tpn_sequences_added_section

            else:
                if not no_tpn_sequences_added:
                    tpn_sequences_added_section.append(tpn_sequences_added)
                # min_runtime_tpn_duration = utils.transform_timedelta_to_ISO8601(min_runtime_tpn)
                arrival_time_tpn_string = arrival_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
                dep_time_tpn_string = dep_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
                runtime_section_feasible[tpn_id] = {'TrackID': tpn['SectionTrackID'],
                                                    'ArrivalTime': arrival_time_tpn_string,
                                                    'DepartureTime': dep_time_tpn_string,
                                                    'RunTime': tpns_train[tpn_id]['MinimumRunTime'],
                                                    'MinimumStopTime': tpns_train[tpn_id]['MinimumStopTime']}
        # check if any other parallel track is free
        check_parallel_track = False
        if check_parallel_track:
            parallel_tracks = ap.section_tracks_parallel_to(tpn['SectionTrack'])
            for other_track in parallel_tracks:
                if other_track['ID'] == tpn['SectionTrack']:
                    continue
                else:  # elif all(elem not in track_info.time_track_is_occupied[other_track['ID']] for elem in minutes_track_is_used):
                    track_to_node_used = other_track['ID']
                    print(' track %i : is free to use' % other_track['ID'])

                    break


    return section_clear, dep_time_tpn, runtime_section_feasible, tpn_sequences_added_section


def check_tpn_departure_feasibility(dep_time_tpn, dep_time_tpn_string, i, tpn, tpn_clear, tpn_id,
                                    tpn_ids_section, tpn_rr_train, track_info, trainID):
    # check only the departure node of this tp !
    # min_runtime_tpn = tpn_runtime['MinimumRunTime']
    # min_stop_time = tpn_runtime['MinStopTime']
    tpn_NodeID = tpn['NodeID']
    try:
        nextTPN_NodeID = tpn_rr_train[tpn_ids_section[i + 1]]['NodeID']
    except IndexError:
        print('wait')
    track_nextTPN = tpn_rr_train[tpn_ids_section[i + 1]]['SectionTrackID']
    nextTPN_ID = tpn_rr_train[tpn_ids_section[i + 1]]['ID']
    tuple_key = (track_nextTPN, tpn_NodeID, nextTPN_NodeID, 'departure')
    tpn_sequences_added = None
    try:
        if [tpn_id, dep_time_tpn_string, trainID] not in track_info.track_sequences_of_TPN[tuple_key]:
            add_tpn_to_track_sequences(tpn_id, dep_time_tpn_string, trainID, track_info, tuple_key)
    except KeyError:
        print('Tuple Key Error in check tpn dep feasibility ')
        print(tuple_key)
        print([tpn_id, dep_time_tpn_string, trainID])

    index_TPN_on_track = track_info.track_sequences_of_TPN[tuple_key].index([tpn_id, dep_time_tpn_string, trainID])

    if index_TPN_on_track == 0:
        condition_pre = True
    else:
        preceeding_TPN = track_info.track_sequences_of_TPN[tuple_key][index_TPN_on_track - 1]
        try:
            min_headway_preceeding = ap.headway_times_btwn_for_sectionTrack(preceeding_TPN[0], nextTPN_ID, track_nextTPN,
                                                                        tpn_NodeID, nextTPN_NodeID)
        except Exception:
            min_headway_preceeding = {'headwayTime': 'PT2M'}

        delta_hw_pre = utils.duration_transform_to_timedelta(min_headway_preceeding['headwayTime'])
        try:
            pre_tpn_info = track_info.tpn_information[preceeding_TPN[0]]
            condition_pre = pre_tpn_info['DepartureTime'] + delta_hw_pre <= dep_time_tpn
        except KeyError:
            condition_pre = True
            # this train has probably been cancelled and therefore the tpn is removed


    if index_TPN_on_track + 1 == len(track_info.track_sequences_of_TPN[tuple_key]):
        condition_suc = True
    else:
        succeeding_TPN = track_info.track_sequences_of_TPN[tuple_key][index_TPN_on_track + 1]
        try:
            min_headway_succeeding = ap.headway_times_btwn_for_sectionTrack(nextTPN_ID, succeeding_TPN[0], track_nextTPN,
                                                                            tpn_NodeID, nextTPN_NodeID)
        except Exception:
            min_headway_succeeding = {'headwayTime': 'PT2M'}
        delta_hw_suc = utils.duration_transform_to_timedelta(min_headway_succeeding['headwayTime'])
        try:
            suc_tpn_info = track_info.tpn_information[succeeding_TPN[0]]
            condition_suc = dep_time_tpn <= suc_tpn_info['DepartureTime'] - delta_hw_suc
        except KeyError:
            # this train has probably been cancelled and therefore the tpn is removed
            condition_suc = True


    if condition_pre and condition_suc:
        # departure feasible
        # check opposite track direction departure
        dep_time_tpn, tpn_clear = check_departure_feasibility_tpn_opposite_track_direction\
            (dep_time_tpn, dep_time_tpn_string, nextTPN_NodeID, tpn_NodeID, tpn_clear, tpn_id, track_info, track_nextTPN,
             trainID)

        tpn_sequences_added = {tuple_key: [tpn_id, dep_time_tpn_string, trainID] }

    elif condition_pre and not condition_suc or not condition_pre and not condition_suc:
        # not feasible
        dep_time_tpn = suc_tpn_info['DepartureTime'] + delta_hw_suc
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, dep_time_tpn_string, trainID])
    elif not condition_pre and condition_suc:
        # not feasible
        dep_time_tpn = pre_tpn_info['DepartureTime'] + delta_hw_pre
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, dep_time_tpn_string, trainID])

    return dep_time_tpn, tpn_clear, tpn_sequences_added


def check_departure_feasibility_tpn_opposite_track_direction(dep_time_tpn, dep_time_tpn_string, nextTPN_NodeID,
                                                             tpn_NodeID, tpn_clear, tpn_id, track_info, track_nextTPN,
                                                             trainID):
    tuple_key_opposite_direction = (track_nextTPN, nextTPN_NodeID, tpn_NodeID, 'arrival')
    if not tuple_key_opposite_direction in track_info.track_sequences_of_TPN.keys():
        # this track is not driven in opposite direction
        tpn_clear = True
    else:
        add_tpn_to_track_sequences(tpn_id, dep_time_tpn_string, trainID, track_info, tuple_key_opposite_direction)
        index_TPN_on_track_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction].index(
            [tpn_id, dep_time_tpn_string, trainID])
        # preceeding train
        if index_TPN_on_track_op == 0:
            condition_pre_op = True
        else:
            min_headway_preceeding = {'headwayTime': 'PT2M'}
            delta_hw_pre_op = utils.duration_transform_to_timedelta(min_headway_preceeding['headwayTime'])
            preceeding_TPN_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction][
                index_TPN_on_track_op - 1]
            try:
                pre_tpn_info_op = track_info.tpn_information[preceeding_TPN_op[0]]
                condition_pre_op = pre_tpn_info_op['ArrivalTime'] + delta_hw_pre_op <= dep_time_tpn
            except KeyError:
                condition_pre_op = True

        # suceeding train
        if index_TPN_on_track_op + 1 == len(track_info.track_sequences_of_TPN[tuple_key_opposite_direction]):
            condition_suc_op = True
        else:
            min_headway_succeeding = {'headwayTime': 'PT2M'}
            delta_hw_suc_op = utils.duration_transform_to_timedelta(min_headway_succeeding['headwayTime'])
            succeeding_TPN_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction][
                index_TPN_on_track_op + 1]
            try:
                suc_tpn_info_op = track_info.tpn_information[succeeding_TPN_op[0]]
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

        # remove it from opposite direction track sequences
        track_info.track_sequences_of_TPN[tuple_key_opposite_direction].remove([tpn_id, dep_time_tpn_string, trainID])
    return dep_time_tpn, tpn_clear


def check_tpn_departure_feasibility_delay_operator(dep_time_tpn, dep_time_tpn_string, i, tpn, tpn_clear, tpn_id,
                                                   tpn_ids_section, tpns_train, track_info, trainID):
    # check only the departure node of this tp !
    # min_runtime_tpn = tpn_runtime['MinimumRunTime']
    # min_stop_time = tpn_runtime['MinStopTime']
    tpn_NodeID = tpn['NodeID']
    try:
        nextTPN_NodeID = tpns_train[tpn_ids_section[i + 1]]['NodeID']
    except IndexError:
        pass
        # print('wait')
    track_nextTPN = tpns_train[tpn_ids_section[i + 1]]['SectionTrackID']
    nextTPN_ID = tpns_train[tpn_ids_section[i + 1]]['ID']
    tuple_key = (track_nextTPN, tpn_NodeID, nextTPN_NodeID, 'departure')

    tpn_sequences_added = None

    condition_pre, condition_suc, no_tains_in_opposite_direction = False, False, False

    try:
        if [tpn_id, dep_time_tpn_string, trainID] not in track_info.track_sequences_of_TPN[tuple_key]:
            add_tpn_to_track_sequences(tpn_id, dep_time_tpn_string, trainID, track_info, tuple_key)
    except KeyError:
        # no other trains on this track in this direction
        condition_pre, condition_suc, no_tains_in_opposite_direction = True, True, True
        # print(tuple_key)
        # print([tpn_id, dep_time_tpn_string, trainID])

    if not condition_pre and not condition_suc:
        index_TPN_on_track = track_info.track_sequences_of_TPN[tuple_key].index([tpn_id, dep_time_tpn_string, trainID])
    # except KeyError:
    #    print('something went wrong')

        if index_TPN_on_track == 0:
            condition_pre = True
        else:
            preceeding_TPN = track_info.track_sequences_of_TPN[tuple_key][index_TPN_on_track - 1]
            # try:
            #    min_headway_preceeding = ap.headway_times_btwn_for_sectionTrack(preceeding_TPN[0], nextTPN_ID, track_nextTPN,
            #                                                                tpn_NodeID, nextTPN_NodeID)
            # except Exception:

            min_headway_preceeding = {'headwayTime': 'PT2M'}

            delta_hw_pre = utils.duration_transform_to_timedelta(min_headway_preceeding['headwayTime'])
            try:
                pre_tpn_info = track_info.tpn_information[preceeding_TPN[0]]
                condition_pre = pre_tpn_info['DepartureTime'] + delta_hw_pre <= dep_time_tpn
            except KeyError:
                condition_pre = True
                # this train has probably been cancelled and therefore the tpn is removed

        if index_TPN_on_track + 1 == len(track_info.track_sequences_of_TPN[tuple_key]):
            condition_suc = True
        else:
            succeeding_TPN = track_info.track_sequences_of_TPN[tuple_key][index_TPN_on_track + 1]
            # try:
            #    min_headway_succeeding = ap.headway_times_btwn_for_sectionTrack(nextTPN_ID, succeeding_TPN[0], track_nextTPN,
            #                                                                    tpn_NodeID, nextTPN_NodeID)
            # except Exception:
            min_headway_succeeding = {'headwayTime': 'PT2M'}

            delta_hw_suc = utils.duration_transform_to_timedelta(min_headway_succeeding['headwayTime'])
            try:
                suc_tpn_info = track_info.tpn_information[succeeding_TPN[0]]
                condition_suc = dep_time_tpn <= suc_tpn_info['DepartureTime'] - delta_hw_suc
            except KeyError:
                # this train has probably been cancelled and therefore the tpn is removed
                condition_suc = True

    if condition_pre and condition_suc:
        # departure feasible
        dep_time_tpn, tpn_clear = check_departure_feasibility_tpn_opposite_track_direction \
            (dep_time_tpn, dep_time_tpn_string, nextTPN_NodeID, tpn_NodeID, tpn_clear, tpn_id, track_info,
             track_nextTPN, trainID)
        if not no_tains_in_opposite_direction:
            tpn_sequences_added = {tuple_key: [tpn_id, dep_time_tpn_string, trainID]}

    elif condition_pre and not condition_suc or not condition_pre and not condition_suc:
        # not feasible
        dep_time_tpn = suc_tpn_info['DepartureTime'] + delta_hw_suc
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, dep_time_tpn_string, trainID])
    elif not condition_pre and condition_suc:
        # not feasible
        dep_time_tpn = pre_tpn_info['DepartureTime'] + delta_hw_pre
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, dep_time_tpn_string, trainID])

    return dep_time_tpn, tpn_clear, tpn_sequences_added


def check_arrival_feasibility_tpn(arr_time_tpn, i, tpn, tpn_clear, tpn_id, tpn_ids_section, tpn_rr_train, track_info, trainID):
    arr_time_tpn_string = arr_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
    # check the arrival node of this tp !
    # min_runtime_tpn = tpn_runtime['MinimumRunTime']
    # min_stop_time = tpn_runtime['MinStopTime']
    tpn_NodeID = tpn['NodeID']
    lastTPN_NodeID = tpn_rr_train[tpn_ids_section[i-1]]['NodeID']
    track_TPN = tpn_rr_train[tpn_ids_section[i]]['SectionTrackID']
    tuple_key = (track_TPN, lastTPN_NodeID, tpn_NodeID, 'arrival')
    tpn_sequences_added = None
    deltaT_forDeparture = datetime.timedelta(seconds=0)
    try:
        if [tpn_id, arr_time_tpn_string, trainID] not in track_info.track_sequences_of_TPN[tuple_key]:
            add_tpn_to_track_sequences(tpn_id, arr_time_tpn_string, trainID, track_info, tuple_key)
    except KeyError:
        pass
        # print('Wait')
    index_TPN_on_track = track_info.track_sequences_of_TPN[tuple_key].index([tpn_id, arr_time_tpn_string, trainID])

    if index_TPN_on_track == 0:
        condition_pre = True
    else:
        preceeding_TPN = track_info.track_sequences_of_TPN[tuple_key][index_TPN_on_track - 1]
        try:
            min_headway_preceeding = ap.headway_times_btwn_for_sectionTrack(preceeding_TPN[0], tpn_id, track_TPN, lastTPN_NodeID,
                                                                            tpn_NodeID)
        except Exception:
            min_headway_preceeding = {'headwayTime': 'PT2M'}

        delta_hw_pre = utils.duration_transform_to_timedelta(min_headway_preceeding['headwayTime'])
        try:
            pre_tpn_info = track_info.tpn_information[preceeding_TPN[0]]
            condition_pre = pre_tpn_info['ArrivalTime'] + delta_hw_pre <= arr_time_tpn
        except KeyError:
            # this train has probably been cancelled and therefore the tpn is removed
            condition_pre = True

    if index_TPN_on_track + 1 == len(track_info.track_sequences_of_TPN[tuple_key]):
        condition_suc = True
    else:
        succeeding_TPN = track_info.track_sequences_of_TPN[tuple_key][index_TPN_on_track + 1]
        try:
            min_headway_succeeding = ap.headway_times_btwn_for_sectionTrack(tpn_id, succeeding_TPN[0], track_TPN, lastTPN_NodeID,
                                                                            tpn_NodeID)
        except Exception:
            min_headway_succeeding = {'headwayTime': 'PT2M'}

        delta_hw_suc = utils.duration_transform_to_timedelta(min_headway_succeeding['headwayTime'])
        try:
            suc_tpn_info = track_info.tpn_information[succeeding_TPN[0]]
            condition_suc = arr_time_tpn <= suc_tpn_info['ArrivalTime'] - delta_hw_suc
        except KeyError:
            condition_suc = True
            # this train has probably been cancelled and therefore the tpn is removed

    if condition_pre and condition_suc:
        # arrival feasible
        # check arrival feasibility opposite track direction
        arr_time_tpn, deltaT_forDeparture, tpn_clear = check_tpn_arrival_feasibility_opposite_direction(arr_time_tpn,
                    arr_time_tpn_string, deltaT_forDeparture, lastTPN_NodeID, tpn_NodeID, tpn_clear, tpn_id, track_TPN,
                    track_info, trainID)

        tpn_sequences_added = {tuple_key: [tpn_id, arr_time_tpn_string, trainID]}

    elif condition_pre and not condition_suc or not condition_pre and not condition_suc:
        # not feasible
        deltaT_forDeparture = suc_tpn_info['ArrivalTime'] + delta_hw_pre - arr_time_tpn
        if deltaT_forDeparture < datetime.timedelta(seconds=0):
            deltaT_forDeparture = datetime.datetime.strptime(succeeding_TPN[1], "%Y-%m-%dT%H:%M:%S") + delta_hw_suc - arr_time_tpn

        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, arr_time_tpn_string, trainID])

    elif not condition_pre and condition_suc:
        # not feasible
        deltaT_forDeparture = None  # try do increase runtime at next iteration
        arr_time_tpn = pre_tpn_info['ArrivalTime'] + delta_hw_pre
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, arr_time_tpn_string, trainID])

    return arr_time_tpn, tpn_clear, tpn_sequences_added, deltaT_forDeparture


def check_tpn_arrival_feasibility_opposite_direction(arr_time_tpn, arr_time_tpn_string, deltaT_forDeparture,
                                                     lastTPN_NodeID, tpn_NodeID, tpn_clear, tpn_id, track_TPN,
                                                     track_info, trainID):
    tuple_key_opposite_direction = (track_TPN, tpn_NodeID, lastTPN_NodeID, 'departure')
    if not tuple_key_opposite_direction in track_info.track_sequences_of_TPN.keys():
        # this track is not driven in opposite direction
        tpn_clear = True
    else:
        add_tpn_to_track_sequences(tpn_id, arr_time_tpn_string, trainID, track_info, tuple_key_opposite_direction)
        index_TPN_on_track_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction].index(
            [tpn_id, arr_time_tpn_string, trainID])
        # preceeding train
        min_headway_preceeding = {'headwayTime': 'PT2M'}
        delta_hw_pre_op = utils.duration_transform_to_timedelta(min_headway_preceeding['headwayTime'])

        if index_TPN_on_track_op == 0:
            condition_pre_op = True
        else:
            preceeding_TPN_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction][
                index_TPN_on_track_op - 1]
            try:
                pre_tpn_info_op = track_info.tpn_information[preceeding_TPN_op[0]]
                condition_pre_op = pre_tpn_info_op['DepartureTime'] + delta_hw_pre_op <= arr_time_tpn
            except KeyError:
                condition_pre_op = True
        # succeeding train
        min_headway_succeeding = {'headwayTime': 'PT2M'}
        delta_hw_suc_op = utils.duration_transform_to_timedelta(min_headway_succeeding['headwayTime'])
        if index_TPN_on_track_op + 1 == len(track_info.track_sequences_of_TPN[tuple_key_opposite_direction]):
            condition_suc_op = True
        else:
            succeeding_TPN_op = track_info.track_sequences_of_TPN[tuple_key_opposite_direction][
                index_TPN_on_track_op + 1]
            try:
                suc_tpn_info_op = track_info.tpn_information[succeeding_TPN_op[0]]
                condition_suc_op = arr_time_tpn <= suc_tpn_info_op['DepartureTime'] - delta_hw_suc_op
            except KeyError:
                # this train has probably been cancelled and therefore the tpn is removed
                condition_suc_op = True

        if condition_suc_op and condition_pre_op:
            tpn_clear = True

        elif condition_pre_op and not condition_suc_op or not condition_pre_op and not condition_suc_op:
            deltaT_forDeparture = suc_tpn_info_op['DepartureTime'] + delta_hw_pre_op - arr_time_tpn
            if deltaT_forDeparture < datetime.timedelta(seconds=0):
                deltaT_forDeparture = datetime.datetime.strptime(succeeding_TPN_op[1], "%Y-%m-%dT%H:%M:%S") \
                                      + delta_hw_suc_op - arr_time_tpn

        elif not condition_pre_op and condition_suc_op:
            # not feasible
            deltaT_forDeparture = None  # try do increase runtime at next iteration
            arr_time_tpn = pre_tpn_info_op['DepartureTime'] + delta_hw_pre_op
            tpn_clear = False

        # remove it from opposite direction track sequences
        track_info.track_sequences_of_TPN[tuple_key_opposite_direction].remove([tpn_id, arr_time_tpn_string, trainID])
    return arr_time_tpn, deltaT_forDeparture, tpn_clear


def check_arrival_feasibility_tpn_delay_operator(arr_time_tpn, i, tpn, tpn_clear, tpn_id, tpn_ids_section, tpns_train, track_info, trainID):
    arr_time_tpn_string = arr_time_tpn.strftime("%Y-%m-%dT%H:%M:%S")
    # check the arrival node of this tp !
    # min_runtime_tpn = tpn_runtime['MinimumRunTime']
    # min_stop_time = tpn_runtime['MinStopTime']
    tpn_NodeID = tpn['NodeID']
    lastTPN_NodeID = tpns_train[tpn_ids_section[i-1]]['NodeID']
    track_TPN = tpns_train[tpn_id]['SectionTrackID']
    tuple_key = (track_TPN, lastTPN_NodeID, tpn_NodeID, 'arrival')
    tpn_sequences_added = None
    deltaT_forDeparture = datetime.timedelta(seconds=0)
    condition_suc, condition_pre = False, False
    no_tains_in_opposite_direction = False
    try:
        if [tpn_id, arr_time_tpn_string, trainID] not in track_info.track_sequences_of_TPN[tuple_key]:
            add_tpn_to_track_sequences(tpn_id, arr_time_tpn_string, trainID, track_info, tuple_key)
    except KeyError:
        # no other trains on this track in this direction
        condition_pre, condition_suc, no_tains_in_opposite_direction = True, True, True
    if not condition_pre and not condition_suc:
        index_TPN_on_track = track_info.track_sequences_of_TPN[tuple_key].index([tpn_id, arr_time_tpn_string, trainID])

        if index_TPN_on_track == 0:
            condition_pre = True
        else:
            preceeding_TPN = track_info.track_sequences_of_TPN[tuple_key][index_TPN_on_track - 1]
            # try:
            #    min_headway_preceeding = ap.headway_times_btwn_for_sectionTrack(preceeding_TPN[0], tpn_id, track_TPN, lastTPN_NodeID,
            #                                                                    tpn_NodeID)
            #except Exception:
            min_headway_preceeding = {'headwayTime': 'PT2M'}

            delta_hw_pre = utils.duration_transform_to_timedelta(min_headway_preceeding['headwayTime'])
            try:
                pre_tpn_info = track_info.tpn_information[preceeding_TPN[0]]
                condition_pre = pre_tpn_info['ArrivalTime'] + delta_hw_pre <= arr_time_tpn
            except KeyError:
                # this train has probably been cancelled and therefore the tpn is removed
                condition_pre = True

        if index_TPN_on_track + 1 == len(track_info.track_sequences_of_TPN[tuple_key]):
            condition_suc = True
        else:
            succeeding_TPN = track_info.track_sequences_of_TPN[tuple_key][index_TPN_on_track + 1]
            # try:
            #    min_headway_succeeding = ap.headway_times_btwn_for_sectionTrack(tpn_id, succeeding_TPN[0], track_TPN, lastTPN_NodeID,
            #                                                                    tpn_NodeID)
            # except Exception:
            min_headway_succeeding = {'headwayTime': 'PT2M'}

            delta_hw_suc = utils.duration_transform_to_timedelta(min_headway_succeeding['headwayTime'])
            try:
                suc_tpn_info = track_info.tpn_information[succeeding_TPN[0]]
                condition_suc = arr_time_tpn <= suc_tpn_info['ArrivalTime'] - delta_hw_suc
            except KeyError:
                condition_suc = True
                # this train has probably been cancelled and therefore the tpn is removed

    if condition_pre and condition_suc:
        # departure feasible
        arr_time_tpn, deltaT_forDeparture, tpn_clear = check_tpn_arrival_feasibility_opposite_direction(arr_time_tpn, arr_time_tpn_string, deltaT_forDeparture, lastTPN_NodeID,
                                                         tpn_NodeID, tpn_clear, tpn_id, track_TPN, track_info, trainID)
        if not no_tains_in_opposite_direction:
            tpn_sequences_added = {tuple_key: [tpn_id, arr_time_tpn_string, trainID]}

    elif condition_pre and not condition_suc or not condition_pre and not condition_suc:
        # not feasible
        try:
            deltaT_forDeparture = suc_tpn_info['ArrivalTime'] + delta_hw_pre - arr_time_tpn
        except UnboundLocalError:
            deltaT_forDeparture = datetime.timedelta(minutes=5)
        if deltaT_forDeparture < datetime.timedelta(seconds=0):
            deltaT_forDeparture = datetime.datetime.strptime(succeeding_TPN[1], "%Y-%m-%dT%H:%M:%S") + delta_hw_suc - arr_time_tpn
        # todo, double check this condition ! why is the suceeding tpn arrival so much different to tpn info ?
        #         if suc_tpn_info['ArrivalTime'] > arr_time_tpn:
        # 			deltaT_forDeparture = suc_tpn_info['ArrivalTime'] + delta_hw_pre - arr_time_tpn
        #         else:
        # 			deltaT_forDeparture = delta_hw_pre + arr_time_tpn - suc_tpn_info['ArrivalTime']
        # arr_time_tpn = suc_tpn_info['ArrivalTime'] + delta_hw_suc
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, arr_time_tpn_string, trainID])

    elif not condition_pre and condition_suc:
        # not feasible
        deltaT_forDeparture = None  # try do increase runtime at next iteration
        arr_time_tpn = pre_tpn_info['ArrivalTime'] + delta_hw_pre
        track_info.track_sequences_of_TPN[tuple_key].remove([tpn_id, arr_time_tpn_string, trainID])

    return arr_time_tpn, tpn_clear, tpn_sequences_added, deltaT_forDeparture


def short_turn_train(parameters, train_disruption_infeasible):
    cloned_train1 = ap.clone_train(train_disruption_infeasible['ID'])
    cloned_train2 = ap.clone_train(train_disruption_infeasible['ID'])
    cloned_trains = ap.cut_trains_AreaOfInterest_TimeWindow([cloned_train1, cloned_train2], parameters.stations_in_area,
                                                            parameters.time_window)
    cloned_train1 = cloned_trains[0]
    cloned_train2 = cloned_trains[1]
    idx_tpn_closed_track = train_disruption_infeasible['idx']
    # cancel train 1 from
    cloned_train1 = ap.cancel_train_from(cloned_train1['TrainPathNodes'][idx_tpn_closed_track[0] - 1]['ID'])
    # cancel train 2 to
    cloned_train2 = ap.cancel_train_to(cloned_train2['TrainPathNodes'][idx_tpn_closed_track[1]]['ID'])
    cloned_trains = ap.cut_trains_AreaOfInterest_TimeWindow([cloned_train1, cloned_train2],
                                                            parameters.stations_in_area,
                                                            parameters.time_window)
    cloned_train1 = cloned_trains[0]
    cloned_train2 = cloned_trains[1]
    train_disruption_infeasible = ap.cancel_train(train_disruption_infeasible['ID'])
    return cloned_train1, cloned_train2


def short_turn_train_viriato_preselection(parameters, train_to_shortTurn, tpn_cancel_from, tpn_cancel_to):

    cloned_train1 = ap.clone_train(train_to_shortTurn['ID'])
    cloned_train2 = ap.clone_train(train_to_shortTurn['ID'])

    i = 0
    while cloned_train1['TrainPathNodes'][i]['SequenceNumber'] != tpn_cancel_from['SequenceNumber']:
        i += 1
    tpn_cancel_from = cloned_train1['TrainPathNodes'][i]['ID']

    i = 0
    while cloned_train1['TrainPathNodes'][i]['SequenceNumber'] != tpn_cancel_to['SequenceNumber']:
        i += 1
    tpn_cancel_to = cloned_train1['TrainPathNodes'][i]['ID']

    # cancel train 1 from
    cloned_train1 = ap.cancel_train_from(tpn_cancel_from)
    # cancel train 2 to
    cloned_train2 = ap.cancel_train_to(tpn_cancel_to)
    cloned_trains = ap.cut_trains_AreaOfInterest_TimeWindow([cloned_train1, cloned_train2],
                                                            parameters.stations_in_area,
                                                            parameters.time_window)
    cloned_train1 = cloned_trains[0]
    cloned_train2 = cloned_trains[1]
    train_disruption_infeasible = ap.cancel_train(train_to_shortTurn['ID'])

    return cloned_train1, cloned_train2


def call_emergency_train_scen_low():
    emergency_train = ap.get_parameter('emergencyTrain_Scen_Low')
    emergency_train = ap.clone_train(emergency_train['ID'])

    return emergency_train


def update_tpn_information(node_idx, track_info, train_updated_times, train_original):
    # remove all tpn of original train
    for tpn in train_original['TrainPathNodes']:
        try:
            del track_info.tpn_information[tpn['ID']]
        except KeyError:
            continue

    # Update the tpn information
    from_tpn_index = node_idx
    if train_updated_times['TrainPathNodes'][from_tpn_index - 1]['SequenceNumber'] >= 0:
        fromNode = train_updated_times['TrainPathNodes'][from_tpn_index - 1]['NodeID']
    else:
        fromNode = None
    tpn_node_index = from_tpn_index

    for node in train_updated_times['TrainPathNodes'][from_tpn_index:]:
        toNode = node['NodeID']
        if tpn_node_index + 1 < len(train_updated_times['TrainPathNodes']):
            nextTPN_ID = train_updated_times['TrainPathNodes'][tpn_node_index + 1]['ID']
        else:
            nextTPN_ID = None
        if node['MinimumRunTime'] is not None:
            min_run_time = utils.duration_transform_to_timedelta(node['MinimumRunTime'])
        else:
            min_run_time = None

        track_info.tpn_information[node['ID']] = {
            'ArrivalTime': datetime.datetime.strptime(node['ArrivalTime'], "%Y-%m-%dT%H:%M:%S"),
            'DepartureTime': datetime.datetime.strptime(node['DepartureTime'], "%Y-%m-%dT%H:%M:%S"),
            'RunTime': min_run_time, 'fromNode': fromNode, 'toNode': toNode, 'SectionTrack': node['SectionTrackID'],
            'nextTPN_ID': nextTPN_ID, 'TrainID': train_updated_times['ID']}

        fromNode = toNode
        tpn_node_index += 1


def remove_tpn_info_canceled_train(track_info, train_disruption_infeasible, tpnID_cancel_from=None):
    if tpnID_cancel_from is None:
        for node in train_disruption_infeasible['TrainPathNodes']:
            try:
                del track_info.tpn_information[node['ID']]
            except KeyError:
                pass
    else:
        for node in train_disruption_infeasible['TrainPathNodes']:
            if node['ID'] != tpnID_cancel_from:
                continue
            else:
                try:
                    del track_info.tpn_information[node['ID']]
                except KeyError:
                    pass


def remove_tpn_info_short_turned_train(track_info, train_disruption_infeasible):
    idx_to_cancel = train_disruption_infeasible['idx']

    for node in train_disruption_infeasible['TrainPathNodes'][idx_to_cancel[0]:idx_to_cancel[1]+1]:
            try:
                del track_info.tpn_information[node['ID']]
            except KeyError:
                pass


def remove_tpn_added_to_track_info_tpn_sequences(tpn_sequences_added_section, track_info):
    for sequ in tpn_sequences_added_section:
        if sequ is None:
            continue
        for key, value in sequ.items():
            try:
                track_info.track_sequences_of_TPN[key].remove(value)
            except ValueError:
                pass
                # print('\n  Value Error, value not in in track info track sequ')
                # print(key, value)

def add_tpn_to_track_sequences(tpn_ID, tpn_arr_or_dep_Time, trainID, track_info, tuple_key):
    '''tpn_id, dep_time_tpn, trainID
    :param tpns: train path node to add into the sequence on track
    :param track_info: object with the sequence on track in it
    :param trainID: of the tpn train
    :param tuple_key: key for the sequence list (sectionTrack, fromNode, toNode)
    :return: None
    '''
    track_info.track_sequences_of_TPN[tuple_key].append([tpn_ID, tpn_arr_or_dep_Time, trainID])
    track_info.track_sequences_of_TPN[tuple_key] = sorted(track_info.track_sequences_of_TPN[tuple_key],
                                                          key=itemgetter(1))


def sort_track_sequences_of_TPN_by_time(track_info, tuple_key):
    '''
    :param tpns: train path node to add into the sequence on track
    :param track_info: object with the sequence on track in it
    :param trainID: of the tpn train
    :param tuple_key: key for the sequence list (sectionTrack, fromNode, toNode)
    :return: None
    '''
    track_info.track_sequences_of_TPN[tuple_key] = sorted(track_info.track_sequences_of_TPN[tuple_key],
                                                          key=itemgetter(1))