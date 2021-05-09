"""
Created on Mon Mar 01 2021

@author: BenMobility

To easily test code lines with Viriato

"""
from py_client import algorithm_interface
import viriato_interface
import networkx as nx
import random
import numpy as np
import shortest_path
import timetable_graph
import infrastructure_graph
import alns_platform
import helpers
import pickle
import datetime
import copy
import numpy.lib.recfunctions as rfn

# Viriato
base_url = 'http://localhost:8080'  # Viriato localhost
np.random.seed(42)  # Random seed of the whole program

# %% Read pickle files for alns
timetable_initial_graph = np.load('output/pickle/timetable_initial_graph_for_alns.pkl', allow_pickle=True)
infra_graph = np.load('output/pickle/infra_graph_for_alns.pkl', allow_pickle=True)
trains_timetable = np.load('output/pickle/trains_timetable_for_alns.pkl', allow_pickle=True)
parameters = np.load('output/pickle/parameters_for_alns.pkl', allow_pickle=True)
parameters.number_iteration = 100

# %% Passenger assigment

# Get the priority list with all the odt pairs
odt_priority_list_original = parameters.odt_as_list

# Set the count to zero for the loop
i = 0

# Assign the passengers based on the priority list
for odt in odt_priority_list_original:
    # Break the loop if reached the last odt to avoid index error
    if i == len(odt_priority_list_original):
        print('End of the passenger assignment')
        break

    # Compute the shortest path with dijkstra
    try:
        l, p = shortest_path.single_source_dijkstra(timetable_initial_graph, odt[0], odt[1])
        # Save the path
        odt_priority_list_original[i][4] = p
        # Save the length of the path
        try:
            odt_priority_list_original[i][5] = round(l, 1)
        except IndexError:
            odt_priority_list_original[i].append(round(l, 1))

        # Assign the flow on the timetable graph's edges
        for j in range(len(p) - 1):
            try:
                if sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) + odt[3] > parameters.train_capacity:
                    try:
                        # Check if the current passenger is already seated in the train
                        if p[j - 1][2] == p[j][2]:
                            # Initialize the parameters for the capacity constraint checks
                            k = 1
                            odt_with_lower_priority_name = []
                            odt_with_lower_priority_flow = []
                            odt_with_lower_priority_index = []

                            # Remove assigned odt with lower priority on the edge
                            while sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) + odt[3] > \
                                    parameters.train_capacity:
                                try:
                                    # Check if the assigned odt is already seated in the train, if so, go to next
                                    # assigned odt
                                    if timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k] in \
                                            timetable_initial_graph[p[j - 1]][p[j]]['odt_assigned']:
                                        k += 1
                                    # If not assigned in the previous edge, hence the assigned passenger must be from
                                    # another train
                                    else:
                                        # Save the assigned
                                        odt_with_lower_priority_name.append(
                                            timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k])
                                        odt_with_lower_priority_flow.append(
                                            timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k][3])
                                        odt_with_lower_priority_index.append(-k)

                                        # Check if removing the assigned odt from the train is enough, if not, need to
                                        # add another assigned from the list
                                        if sum(odt_with_lower_priority_flow) >= odt[3]:
                                            for odt_with_lower_priority in odt_with_lower_priority_name:
                                                # Extract the odt to get the recorded path from the original priority
                                                # list
                                                extract_odt = [item for item in odt_priority_list_original
                                                               if item[0:2] == odt_with_lower_priority[0:2]]

                                                # Find the index on the original list
                                                index_in_original_list = odt_priority_list_original.index(
                                                    extract_odt[0])

                                                # Get the path from the original list
                                                extract_odt_path = extract_odt[0][4]

                                                # Get the index of the last node before the full capacity train
                                                index_last_node_on_path_before_capacity = extract_odt_path.index(p[j])

                                                # Split the path into delete and keep path
                                                odt_path_to_delete = extract_odt_path[
                                                                     index_last_node_on_path_before_capacity:]
                                                odt_path_to_keep = extract_odt_path[
                                                                   :index_last_node_on_path_before_capacity]

                                                # Modify the original path and erase the length, needs to be recomputed
                                                try:
                                                    odt_priority_list_original[
                                                        index_in_original_list][4] = odt_path_to_keep
                                                    odt_priority_list_original[index_in_original_list][5] = None
                                                except ValueError:
                                                    print(f'{odt_priority_list_original[index_in_original_list][0:2]}'
                                                          f' at index {index_in_original_list} has already a changed '
                                                          f'value but it was not recorded properly. Please check '
                                                          f'passenger assignment')

                                                # Delete the flow and the odt_assigned
                                                for n in range(len(odt_path_to_delete) - 1):
                                                    try:
                                                        index_to_delete = timetable_initial_graph[
                                                            odt_path_to_delete[n]][
                                                            odt_path_to_delete[n + 1]]['odt_assigned'].index(
                                                            odt_with_lower_priority)
                                                        del timetable_initial_graph[odt_path_to_delete[n]][
                                                            odt_path_to_delete[n + 1]]['flow'][index_to_delete]
                                                        del timetable_initial_graph[odt_path_to_delete[n]][
                                                            odt_path_to_delete[n + 1]]['odt_assigned'][index_to_delete]
                                                    except (KeyError, ValueError):
                                                        continue

                                                if 'odt_facing_capacity_constrain' in locals():
                                                    # Record the odt with the last node before capacity constraint.
                                                    # [odt, last node, index, edge, new path, number of trial]
                                                    odt_info = [odt_with_lower_priority, odt_path_to_keep[-1],
                                                                odt_path_to_delete[0:2], [], 1]
                                                    odt_facing_capacity_constrain.append(odt_info)
                                                else:
                                                    odt_facing_capacity_constrain = [odt_with_lower_priority,
                                                                                     odt_path_to_keep[-1],
                                                                                     odt_path_to_delete[0:2], [], 1]
                                            # Done with the recording of oft facing capacity constraint
                                            break
                                        # Not enough seats released, need at least one more group to leave
                                        else:
                                            k += 1
                                # Not suppose to happen, but it might if there an assignment mistake
                                except IndexError:
                                    print(
                                        f'Train is at full capacity and the current odt {odt} is already seated, '
                                        f'but the algorithm cannot find the assigned odt that is assigned but not '
                                        f'seated in the train.')
                                    break
                        else:
                            if 'odt_facing_capacity_constrain' in locals():
                                # Record the odt with the last node before capacity constraint.
                                # [odt, last node, index, edge, new path, number of trial]
                                odt_info = [odt[0:4], p[j-1], [p[j], p[j + 1]], [], 1]
                                odt_facing_capacity_constrain.append(odt_info)
                            else:
                                odt_facing_capacity_constrain = [[odt[0:4], p[j-1], [p[j], p[j + 1]], [], 1]]

                            # Done for this odt, do not need to continue to assign further. go to the next one
                            break

                    # It means that the previous edge is home to the first station,
                    # hence the passenger is not seated in the train
                    except IndexError:
                        if 'odt_facing_capacity_constrain' in locals():
                            # Record the odt with the last node before capacity constraint.
                            # [odt, last node, index, edge, new path, number of trial]
                            odt_info = [odt[0:4], p[j-1], [p[j], p[j + 1]], [], 1]
                            odt_facing_capacity_constrain.append(odt_info)
                        else:
                            odt_facing_capacity_constrain = [[odt[0:4], p[j-1], [p[j], p[j + 1]], [], 1]]

                        # Done for this odt, do not need to continue to assign further. go to the next one
                        break
                else:
                    timetable_initial_graph[p[j]][p[j + 1]]['flow'].append(odt[3])
                    timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'].append(odt[0:4])

            # If there is a key error, it means it is either a home-station edge, station-destination edge or a transfer
            # hence we go check the next node
            except KeyError:
                pass
    # If there is no path, it raises an error. Record the none path and add the penalty
    except nx.exception.NetworkXNoPath:
        odt_priority_list_original[i][4] = None
        try:
            odt_priority_list_original[i][5] = parameters.penalty_no_path
        except IndexError:
            odt_priority_list_original[i].append(parameters.penalty_no_path)
    i += 1

# %% Sort the list in descending manner with the level of importance

odt_facing_capacity_constrain.sort(key=lambda x: x[0][2], reverse=True)

# %% Second loop passenger assignment for odt facing capacity constraint

# %% Duplicates in odt_capacity_constraint

# keep = []
# index = []
# i = 0
# for odt in odt_facing_capacity_constrain:
#     if odt[0][0:2] in keep:
#         datetime.datetime.strptime(odt[1][1], "%Y-%m-%dT%H:%M:%S")
#         i += 1
#         pass
#     else:
#         keep.append(odt[0][0:2])
#         index.append(i)
#         i += 1
# %% Flow assignment
# for j in range(len(p) - 1):
#     try:
#         if sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) + odt[3] > parameters.train_capacity:
#             try:
#                 # Check if the current passenger is already seated in the train
#                 if p[j - 1][2] == p[j][2]:
#                     # Initialize the parameters for the capacity constraint checks
#                     k = 1
#                     odt_with_lower_priority_name = []
#                     odt_with_lower_priority_flow = []
#                     odt_with_lower_priority_index = []
#
#                     # Remove assigned odt with lower priority on the edge
#                     while sum(timetable_initial_graph[p[i]][p[i + 1]]['flow']) + odt[3] > parameters.train_capacity:
#                         try:
#                             # Check if the assigned odt is already seated in the train, if so, go to next assigned odt
#                             if timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k] in \
#                                     timetable_initial_graph[p[j - 1]][p[j]]['odt_assigned']:
#                                 k += 1
#                             # If not assigned in the previous edge, hence the assigned passenger must be from another
#                             # train
#                             else:
#                                 # Save the assigned
#                                 odt_with_lower_priority_name.append(
#                                     timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k])
#                                 odt_with_lower_priority_flow.append(
#                                     timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k][3])
#                                 odt_with_lower_priority_index.append(k)
#
#                                 # Check if removing the assigned odt from the train is enough, if not, need to add
#                                 # another assigned
#                                 # from the list
#                                 if sum(odt_with_lower_priority_flow) > odt[3]:
#                                     for odt_with_lower_priority in odt_with_lower_priority_name:
#                                         if 'odt_facing_capacity_constrain' in locals():
#                                             # Record the odt with the last node before capacity constraint.
#                                             # [odt, last node, index, edge, new path, number of trial]
#                                             odt_info = [odt, p[j], j, [p[j], p[j + 1]], [], 1]
#                                             odt_facing_capacity_constrain.append(odt_info)
#                                         else:
#                                             odt_facing_capacity_constrain = [[odt, p[j], j, [p[j], p[j + 1]], [], 1]]
#
#                         # Not suppose to happen, but it might if there an assignment mistake
#                         except IndexError:
#                             print(
#                                 f'Train is at full capacity and the current odt {odt} is already seated, '
#                                 f'but the algorithm cannot find the assigned odt that is assigned but not seated in '
#                                 f'the train.')
#                             break
#                 else:
#                     if 'odt_facing_capacity_constrain' in locals():
#                         # Record the odt with the last node before capacity constraint.
#                         # [odt, last node, index, edge, new path, number of trial]
#                         odt_info = [odt, p[j], j, [p[j], p[j + 1]], [], 1]
#                         odt_facing_capacity_constrain.append(odt_info)
#                     else:
#                         odt_facing_capacity_constrain = [[odt, p[j], j, [p[j], p[j + 1]], [], 1]]
#
#                     # Done for this odt, do not need to continue to assign further. go to the next one
#                     break
#
#             # It means that the previous edge is home to the first station,
#             # hence the passenger is not seated in the train
#             except IndexError:
#                 if 'odt_facing_capacity_constrain' in locals():
#                     # Record the odt with the last node before capacity constraint.
#                     # [odt, last node, index, edge, new path, number of trial]
#                     odt_info = [odt, p[j], j, [p[j], p[j + 1]], [], 1]
#                     odt_facing_capacity_constrain.append(odt_info)
#                 else:
#                     odt_facing_capacity_constrain = [[odt, p[j], j, [p[j], p[j + 1]], [], 1]]
#
#                 # Done for this odt, do not need to continue to assign further. go to the next one
#                 break
#         else:
#             timetable_initial_graph[p[j]][p[j + 1]]['flow'].append(odt[3])
#             timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'].append(odt[0:4])
#
#     # If there is a key error, it means it is either a home-station edge, station-destination edge or a transfer,
#     # hence we go check the next node
#     except KeyError:
#         pass
#
# # %% full capacity constraint checks
# if p[i - 1][2] == p[i][2]:
#     # Initialize the parameters for the capacity constraint checks
#     j = 1
#     odt_with_lower_priority_name = []
#     odt_with_lower_priority_flow = []
#     odt_with_lower_priority_index = []
#
#     # Remove assigned odt with lower priority on the edge
#     while sum(timetable_initial_graph[p[i]][p[i + 1]]['flow']) + odt[3] > parameters.train_capacity:
#         try:
#             # Check if the assigned odt is already seated in the train, if so, go to next assigned odt
#             if timetable_initial_graph[p[i]][p[i + 1]]['odt_assigned'][-j] in \
#                     timetable_initial_graph[p[i-1]][p[i]]['odt_assigned']:
#                 j += 1
#             # If not assigned in the previous edge, hence the assigned passenger must be from another train
#             else:
#                 # Save the assigned
#                 odt_with_lower_priority_name.append(timetable_initial_graph[p[i]][p[i + 1]]['odt_assigned'][-j])
#                 odt_with_lower_priority_flow.append(
#                     timetable_initial_graph[p[i]][p[i + 1]]['odt_assigned'][-j][3])
#                 odt_with_lower_priority_index.append(j)
#
#                 # Check if removing the assigned odt from the train is enough, if not, need to add another assigned
#                 # from the list
#                 if sum(odt_with_lower_priority_flow) > odt[3]:
#                     for odt_with_lower_priority in odt_with_lower_priority_name:
#                         if 'odt_facing_capacity_constrain' in locals():
#                             # Record the odt with the last node before capacity constraint.
#                             # [odt, last node, index, edge, new path, number of trial]
#                             odt_info = [odt, p[i], i, [p[i], p[i + 1]], [], 1]
#                             odt_facing_capacity_constrain.append(odt_info)
#                         else:
#                             odt_facing_capacity_constrain = [[odt, p[i], i, [p[i], p[i + 1]], [], 1]]
#
#         # Not suppose to happen, but it might if there an assignment mistake
#         except IndexError:
#             print(f'Train is at full capacity and the current odt {odt} is already seated, but the algorithm cannot find'
#                   f' the assigned odt that is assigned but not seated in the train.')
#             break
# %% Start ALNS
# set_solutions = alns_platform.start(timetable_initial_graph, infra_graph, trains_timetable, parameters)


# %%

# # %% Time window from Viriato and close tracks ids from disruption scenario
# time_window = viriato_interface.get_time_window()
# close_track_ids = viriato_interface.get_section_track_closure_ids(time_window)


# Since it is a bidirectional graph, to and from are the same nodes
# neighbor_nodes_from = viriato_interface.get_neighboring_nodes_from_node(168)
# neighbor_nodes_to = viriato_interface.get_neighboring_nodes_to_node(168)
# neighbor_nodes_between = viriato_interface.get_neighboring_nodes_between(168, 168)
# section_tracks_from = viriato_interface.get_section_track_from(168)
# section_tracks_to = viriato_interface.get_section_track_to(168)

# # %% Infrastructure graph and load parameters
# infra_graph, sbb_nodes = infrastructure_graph.build_infrastructure_graph(time_window)
# parameters = helpers.Parameters(infra_graph, time_window, close_track_ids, list_parameters)
#
# # %% Original timetable graph
# trains_timetable = timetable_graph.get_trains_timetable(time_window, sbb_nodes, parameters)

# # Timetable with waiting edges and transfer edges
# timetable_waiting_transfer, stations_with_commercial_stop = \
#     timetable_graph.get_timetable_with_waiting_transfer_edges(trains_timetable, parameters)
#
# # %% test
# zone_candidates = create_zone_candidates_of_stations(station_candidates)

# %% to compare graphs
# infra_graph = nx.read_gpickle('output/pickle/infra_graph.pickle')
# graph_infrastructure = nx.read_gpickle('oliver_code/oliver_pickles/graph_infrastructure.pickle')
#
# infra_graph_edges = infra_graph.edges
# graph_infrastructure_edges = graph_infrastructure.edges
#
# np.savetxt('output/checkpoint/infra_graph_edges.csv', infra_graph_edges)  # My code output
# np.savetxt('output/checkpoint/graph_infrastructure_edges.csv', graph_infrastructure_edges)  # Oliver's output
#
# timetable_graph_with_waiting_transfer_edges = nx.read_gpickle('output/pickle/timetable_graph_only_trains.pickle')
# timetable_graph_oliver_pickle =
# nx.read_gpickle('oliver_code/oliver_pickles/graph_transfer_m4_M15_threshold8000.pickle')

# timetable_graph_with_waiting_transfer_edges_edges = timetable_graph_with_waiting_transfer_edges.edges
# timetable_graph_oliver_pickle_edges = timetable_graph_oliver_pickle.edges
#
# np.savetxt('output/checkpoint/timetable_waiting_transfer_benoit.csv',
# timetable_graph_with_waiting_transfer_edges_edges)
# np.savetxt('output/checkpoint/timetable_waiting_transfer_oliver.csv', timetable_graph_oliver_pickle_edges)

# to check the timetable graph nodes
# a = nx.get_node_attributes(timetable_initial_graph, 'type')
# b = [a.keys(), a.values()]
# b_0 = [str(i) for i in b[0]]
# np.savetxt('output/checkpoint/nodes_timetable_graph_b1.csv', b_1, fmt='%s')

# to check the timetable graph edges
# a = nx.get_edge_attributes(timetable_initial_graph, 'type')
# a_0 = [str(i) for i in a.keys()]
# a_1 = [str(i) for i in a.values()]
# desti = [1, 2, 3, 5, 13, 14, 21, 23, 24, 26, 28, 30,
# 31, 32, 33, 37, 39, 43, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 81,
# 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 111, 112, 113, 115, 116, 117,
# 119, 121, 131, 133, 135, 136, 137, 139, 141, 151, 152, 153, 154, 156, 157, 160, 161, 171, 172, 173, 174, 175, 176,
# 177, 178, 180, 181, 182, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 211, 212, 213, 214, 215, 216, 217, 218,
# 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 231, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
# 2933, 2938, 4021, 4022, 4023, 4024, 4026, 4027, 4028, 4029, 4030, 4031, 4033, 4034, 4035, 4036, 4037, 4038, 4039,
# 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4048, 4061, 4062, 4066, 4067, 4072, 4074, 4075, 4077, 4081, 4083, 4084,
# 4093, 4107, 4123, 4305, 4308, 4312, 4314, 4315, 4317, 4318, 4319, 4321, 4322, 4571, 4601, 23001, 23002, 23003,
# 23004, 26101, 26102, 26103, 26104, 26105, 26106, 26107, 26108, 26109, 26110, 26111, 26112] 2 3 5 13 14 21 23 24 26
# 28 30 31 32 33 37 39 43 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 81 82 83 84 85 86 87 88
# 89 90 91 92 93 94 95 96 97 99 100 101 102 111 112 113 115 116 117 119 121 131 133 135 136 137 139 141 151 152 153
# 154 156 157 160 161 171 172 173 174 175 176 177 178 180 181 182 191 192 193 194 195 196 197 198 199 200 211 212 213
# 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 231 241 242 243 244 245 246 247 248 249 250 251
# 2933 2938 4021 4022 4023 4024 4026 4027 4028 4029 4030 4031 4033 4034 4035 4036 4037 4038 4039 4040 4041 4042 4043
# 4044 4045 4046 4048 4061 4062 4066 4067 4072 4074 4075 4077 4081 4083 4084 4093 4107 4123 4305 4308 4312 4314 4315
# 4317 4318 4319 4321 4322 4571 4601 23001 23002 23003 23004 26101 26102 26103 26104 26105 26106 26107 26108 26109
# 26110 26111
# 26112

# get the weight of infra_graph
# infra_graph[199][538][1092]['weight'] # see if it is set very high since it is closed


# to check if there is a alone node...

# import networkx as nx
# a = nx.get_node_attributes(timetable_initial_graph, 'type')
# all_nodes = list(a.keys())
# isolate_nodes = []
# for i in all_nodes:
#     if nx.is_isolate(timetable_initial_graph, i):
#         isolate_nodes.append(i)

# to check how many passenger are not assigned because they are on a isolated node
# passenger_isolated = []
# for iso in isolate_nodes:
#     for od in odt_list:
#         if od[0] == iso:
#             passenger_isolated.append(od[3])
#             print(od[3])
# import numpy as np
# np.sum(passenger_isolated)


# # how to pickle write
# with open('output/pickle/mypickle.pickle', 'wb') as f:
#     pickle.dump(some_obj, f)

# how to pickle read
# with open('mypickle.pickle') as f:
#     loaded_obj = pickle.load(f)

# Pickle for recarray numpy
# filename
# filename_demand = 'output/pickle/demand_' + str(parameters.min_nb_passenger) + 'pass_' + \
#                   str(parameters.th_zone_selection) + 'm_' + '_zones_' + str(parameters.nb_zones_to_connect) + \
# '_stations_' + str(parameters.nb_stations_to_connect) + '.pickle'
# to load
# np.load(filename_demand, allow_pickle = True)
# to write
# demand_selected_zones.dump(filename_demand)

# to pickle networkx graph
# import networkx as nx
# nx.write_gpickle(timetable_initial_graph, 'output/pickle/initial_timetable_m' +
# str(int(parameters.transfer_m.total_seconds() / 60)) \
#                              + '_M' + str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
#                              str(parameters.th_zone_selection) + '.pickle')
#
# nx.read_gpickle('output/pickle/initial_timetable_m' + str(int(parameters.transfer_m.total_seconds() / 60)) \
#                              + '_M' + str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
#                              str(parameters.th_zone_selection) + '.pickle')

# how to pickle list and dict too
# import pickle
#
# with open('output/pickle/odt_list.pickle', 'wb') as f:
#     pickle.dump(odt_list, f)
# with open('output/pickle/odt_list.pickle', 'rb') as f:
#     mynewlist = pickle.load(f)

# # to check if a class object has the attribute you are looking for
# if hasattr(train_to_delay, 'emergency_train') and train_to_delay.emergency_train is True:
#     emergency_train = True


# to check the headway

# for train in trains_timetable:
#     for node in train.train_path_nodes:
#         if node.section_track_id == 1117:
#             print(node.node_id)
# algorithm_interface.get_headway_time(1117, 297, 401, 13635090, 13635452)
# datetime.timedelta(seconds=420)

# to check timetable nodes but not all
# i = 0
# for x, y in timetable_initial_graph.nodes(data=True):
#     print(x, y['type'])
#     i += 1
#     if i == 5:
#         break


# # UPDATE TIMES
# import copy
# import viriato_interface
# train_copy = copy.deepcopy(train)
# arrival_time_copy = train.train_path_nodes[17].arrival_time
# departure_time_copy = train.train_path_nodes[17].departure_time
# minimum_stop_time_copy = train.train_path_nodes[17].minimum_stop_time
# minimum_run_time_copy = train.train_path_nodes[17].minimum_run_time
# arrival_time_copy
# datetime.datetime(2005, 5, 10, 5, 46, 12)
# minimum_stop_time_copy
# datetime.timedelta(seconds=228)
# minimum_run_time_copy
# datetime.timedelta(seconds=114)
# train_path_node_section = [train_copy]
# import datetime
# new_arrival_time_copy = datetime.datetime(2005, 5, 10, 5, 46, 17)
# new_minimum_stop_time_copy = datetime.timedelta(seconds=223)
# new_minimum_run_time_copy = datetime.timedelta(seconds=119)
# train_copy_updated = viriato_interface.update_train_times(train_copy.id, train_copy.train_path_nodes[17].id,
# new_arrival_time_copy, departure_time_copy, new_minimum_run_time_copy, new_minimum_stop_time_copy)
# train_copy_updated.train_path_nodes[17].arrival_time
# datetime.datetime(2005, 5, 10, 5, 46, 17)
# train_path_node_section[0].train_path_nodes[17].arrival_time
# datetime.datetime(2005, 5, 10, 5, 46, 12)

# Get the index of the trains that are on the closed tracks [90, 95, 267, 268, 269, 270, 271, 272, 273, 283, 284,
# 285, 286, 287, 288, 289]
# import alns_platform
# track_info = alns_platform.TrackInformation(trains_timetable, closed_section_track_ids)
# import viriato_interface
# possessions = viriato_interface.get_section_track_closures(parameters.time_window)
# infra_graph, closed_section_track_ids = alns_platform.increase_weight_on_closed_tracks(infra_graph, possessions,
# parameters)
# track_info = alns_platform.TrackInformation(trains_timetable, closed_section_track_ids)
# i = 0
# index = []
# for train in trains_timetable:
#     for train_on_closed_track in track_info.trains_on_closed_tracks:
#         if train.id == train_on_closed_track:
#             index.append(i)
#     i +=1
