"""
Created on Mon Mar 01 2021

@author: BenMobility

To easily test code lines with Viriato

"""
import numpy as np
import shortest_path
import copy
import passenger_assignment
import alns_platform
import networkx as nx


# %% Benchmark
# timetable_initial_graph = np.load('output/pickle/timetable_benchmark.pkl', allow_pickle=True)
# parameters = np.load('output/pickle/parameters_benchmark.pkl', allow_pickle=True)
# odt_list = np.load('output/pickle/odt_list_benchmark.pkl', allow_pickle=True)
# origin_name_desired_dep_time = np.load('output/pickle/odt_desired_benchmark.pkl', allow_pickle=True)
# origin_name_zone_dict = np.load('output/pickle/origin_name_benchmark.pkl', allow_pickle=True)
#
# shortest_path.find_path_for_all_passengers_and_remove_unserved_demand(timetable_initial_graph,
#                                                                       odt_list,
#                                                                       origin_name_desired_dep_time,
#                                                                       origin_name_zone_dict,
#                                                                       parameters)


# %% 1st passenger assignment

# timetable_initial_graph = np.load('output/pickle/timetable_initial_graph_for_alns.pkl', allow_pickle=True)
# parameters = np.load('output/pickle/parameters_for_alns.pkl', allow_pickle=True)
# parameters.train_capacity = 500
#
# # Assign the passenger on the timetable graph
# odt_facing_capacity_constraint, parameters, timetable_initial_graph = passenger_assignment.capacity_constraint_1st_loop(
#     parameters, timetable_initial_graph)
#
# # And save the output of the first list of odt facing capacity constraint
# alns_platform.pickle_results(odt_facing_capacity_constraint, 'output/pickle/odt_facing_capacity_constraint.pkl')
# alns_platform.pickle_results(parameters, 'output/pickle/parameters_with_first_assignment_done.pkl')
# alns_platform.pickle_results(timetable_initial_graph, 'output/pickle/timetable_with_first_assignment_done.pkl')

# %% Passenger assignment with capacity constraint 2nd and more iteration

# # Loading data
# timetable_initial_graph = np.load('output/pickle/timetable_with_first_assignment_done.pkl', allow_pickle=True)
# parameters = np.load('output/pickle/parameters_with_first_assignment_done.pkl', allow_pickle=True)
# odt_facing_capacity_constraint = np.load('output/pickle/odt_facing_capacity_constraint.pkl', allow_pickle=True)
#
# timetable_initial_graph, assigned, unassigned, odt_facing_capacity_dict_for_iteration, odt_priority_list_original = \
#     passenger_assignment.capacity_constraint_2nd_loop(parameters, odt_facing_capacity_constraint,
#                                                       timetable_initial_graph)
#
# # And save the output of the first list of odt facing capacity constraint
# alns_platform.pickle_results(odt_facing_capacity_dict_for_iteration,
#                              'output/pickle/odt_dict_after_passengers_assignment.pkl')
# alns_platform.pickle_results(odt_priority_list_original, 'output/pickle/odt_after_passengers_assignment.pkl')
# alns_platform.pickle_results(parameters, 'output/pickle/parameters_after_passengers_assignment.pkl')
# alns_platform.pickle_results(timetable_initial_graph, 'output/pickle/timetable_after_passengers_assignment.pkl')

# %% Passenger assignment with closed tracks
timetable_initial_graph = np.load('output/pickle/timetable_after_passengers_assignment.pkl', allow_pickle=True)
odt_facing_capacity_dict_for_iteration = np.load('output/pickle/odt_dict_after_passengers_assignment.pkl',
                                                 allow_pickle=True)
parameters = np.load('output/pickle/parameters_after_passengers_assignment.pkl', allow_pickle=True)
odt_priority_list_original = np.load('output/pickle/odt_after_passengers_assignment.pkl', allow_pickle=True)

# Get the edges on the closed tracks
edges_on_closed_tracks = passenger_assignment.get_edges_on_closed_tracks(parameters, timetable_initial_graph)

# Create the list of odt facing disruption
odt_facing_disruption = passenger_assignment.create_list_odt_facing_disruption(edges_on_closed_tracks,
                                                                               timetable_initial_graph,
                                                                               odt_priority_list_original)

# Assign the passengers facing disruption
timetable_initial_graph, assigned, unassigned, odt_facing_disruption, odt_priority_list_original = \
    passenger_assignment.assignment_with_disruption(odt_priority_list_original,
                                                    odt_facing_disruption,
                                                    timetable_initial_graph,
                                                    parameters)
# %% Start ALNS
# set_solutions = alns_platform.start(timetable_initial_graph, infra_graph, trains_timetable, parameters)


# # %% to compare graphs
# # infra_graph = nx.read_gpickle('output/pickle/infra_graph.pickle')
# # graph_infrastructure = nx.read_gpickle('oliver_code/oliver_pickles/graph_infrastructure.pickle')
# #
# # infra_graph_edges = infra_graph.edges
# # graph_infrastructure_edges = graph_infrastructure.edges
# #
# # np.savetxt('output/checkpoint/infra_graph_edges.csv', infra_graph_edges)  # My code output
# # np.savetxt('output/checkpoint/graph_infrastructure_edges.csv', graph_infrastructure_edges)  # Oliver's output
# #
# # timetable_graph_with_waiting_transfer_edges = nx.read_gpickle('output/pickle/timetable_graph_only_trains.pickle')
# # timetable_graph_oliver_pickle =
# # nx.read_gpickle('oliver_code/oliver_pickles/graph_transfer_m4_M15_threshold8000.pickle')
#
# # timetable_graph_with_waiting_transfer_edges_edges = timetable_graph_with_waiting_transfer_edges.edges
# # timetable_graph_oliver_pickle_edges = timetable_graph_oliver_pickle.edges
# #
# # np.savetxt('output/checkpoint/timetable_waiting_transfer_benoit.csv',
# # timetable_graph_with_waiting_transfer_edges_edges)
# # np.savetxt('output/checkpoint/timetable_waiting_transfer_oliver.csv', timetable_graph_oliver_pickle_edges)
#
# # to check the timetable graph nodes
# # a = nx.get_node_attributes(timetable_initial_graph, 'type')
# # b = [a.keys(), a.values()]
# # b_0 = [str(i) for i in b[0]]
# # np.savetxt('output/checkpoint/nodes_timetable_graph_b1.csv', b_1, fmt='%s')
#
# # to check the timetable graph edges
# # a = nx.get_edge_attributes(timetable_initial_graph, 'type')
# # a_0 = [str(i) for i in a.keys()]
# # a_1 = [str(i) for i in a.values()]
# # desti = [1, 2, 3, 5, 13, 14, 21, 23, 24, 26, 28, 30,
# # 31, 32, 33, 37, 39, 43, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 81,
# # 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 111, 112, 113, 115, 116, 117,
# # 119, 121, 131, 133, 135, 136, 137, 139, 141, 151, 152, 153, 154, 156, 157, 160, 161, 171, 172, 173, 174, 175, 176,
# # 177, 178, 180, 181, 182, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 211, 212, 213, 214, 215, 216, 217, 218,
# # 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 231, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
# # 2933, 2938, 4021, 4022, 4023, 4024, 4026, 4027, 4028, 4029, 4030, 4031, 4033, 4034, 4035, 4036, 4037, 4038, 4039,
# # 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4048, 4061, 4062, 4066, 4067, 4072, 4074, 4075, 4077, 4081, 4083, 4084,
# # 4093, 4107, 4123, 4305, 4308, 4312, 4314, 4315, 4317, 4318, 4319, 4321, 4322, 4571, 4601, 23001, 23002, 23003,
# # 23004, 26101, 26102, 26103, 26104, 26105, 26106, 26107, 26108, 26109, 26110, 26111, 26112] 2 3 5 13 14 21 23 24 26
# # 28 30 31 32 33 37 39 43 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 81 82 83 84 85 86 87 88
# # 89 90 91 92 93 94 95 96 97 99 100 101 102 111 112 113 115 116 117 119 121 131 133 135 136 137 139 141 151 152 153
# # 154 156 157 160 161 171 172 173 174 175 176 177 178 180 181 182 191 192 193 194 195 196 197 198 199 200 211 212 213
# # 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 231 241 242 243 244 245 246 247 248 249 250 251
# # 2933 2938 4021 4022 4023 4024 4026 4027 4028 4029 4030 4031 4033 4034 4035 4036 4037 4038 4039 4040 4041 4042 4043
# # 4044 4045 4046 4048 4061 4062 4066 4067 4072 4074 4075 4077 4081 4083 4084 4093 4107 4123 4305 4308 4312 4314 4315
# # 4317 4318 4319 4321 4322 4571 4601 23001 23002 23003 23004 26101 26102 26103 26104 26105 26106 26107 26108 26109
# # 26110 26111
# # 26112
#
# # get the weight of infra_graph
# # infra_graph[199][538][1092]['weight'] # see if it is set very high since it is closed
#
#
# # to check if there is a alone node...
#
# # import networkx as nx
# # a = nx.get_node_attributes(timetable_initial_graph, 'type')
# # all_nodes = list(a.keys())
# # isolate_nodes = []
# # for i in all_nodes:
# #     if nx.is_isolate(timetable_initial_graph, i):
# #         isolate_nodes.append(i)
#
# # to check how many passenger are not assigned because they are on a isolated node
# # passenger_isolated = []
# # for iso in isolate_nodes:
# #     for od in odt_list:
# #         if od[0] == iso:
# #             passenger_isolated.append(od[3])
# #             print(od[3])
# # import numpy as np
# # np.sum(passenger_isolated)
#
#
# # # how to pickle write
# # with open('output/pickle/mypickle.pickle', 'wb') as f:
# #     pickle.dump(some_obj, f)
#
# # how to pickle read
# # with open('mypickle.pickle') as f:
# #     loaded_obj = pickle.load(f)
#
# # Pickle for recarray numpy
# # filename
# # filename_demand = 'output/pickle/demand_' + str(parameters.min_nb_passenger) + 'pass_' + \
# #                   str(parameters.th_zone_selection) + 'm_' + '_zones_' + str(parameters.nb_zones_to_connect) + \
# # '_stations_' + str(parameters.nb_stations_to_connect) + '.pickle'
# # to load
# # np.load(filename_demand, allow_pickle = True)
# # to write
# # demand_selected_zones.dump(filename_demand)
#
# # to pickle networkx graph
# # import networkx as nx
# # nx.write_gpickle(timetable_initial_graph, 'output/pickle/initial_timetable_m' +
# # str(int(parameters.transfer_m.total_seconds() / 60)) \
# #                              + '_M' + str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
# #                              str(parameters.th_zone_selection) + '.pickle')
# #
# # nx.read_gpickle('output/pickle/initial_timetable_m' + str(int(parameters.transfer_m.total_seconds() / 60)) \
# #                              + '_M' + str(int(parameters.transfer_M.total_seconds() / 60)) + '_threshold_' + \
# #                              str(parameters.th_zone_selection) + '.pickle')
#
# # how to pickle list and dict too
# # import pickle
# #
# # with open('output/pickle/odt_list.pickle', 'wb') as f:
# #     pickle.dump(odt_list, f)
# # with open('output/pickle/odt_list.pickle', 'rb') as f:
# #     mynewlist = pickle.load(f)
#
# # # to check if a class object has the attribute you are looking for
# # if hasattr(train_to_delay, 'emergency_train') and train_to_delay.emergency_train is True:
# #     emergency_train = True
#
#
# # to check the headway
#
# # for train in trains_timetable:
# #     for node in train.train_path_nodes:
# #         if node.section_track_id == 1117:
# #             print(node.node_id)
# # algorithm_interface.get_headway_time(1117, 297, 401, 13635090, 13635452)
# # datetime.timedelta(seconds=420)
#
# # to check timetable nodes but not all
# # i = 0
# # for x, y in timetable_initial_graph.nodes(data=True):
# #     print(x, y['type'])
# #     i += 1
# #     if i == 5:
# #         break
#
#
# # # UPDATE TIMES
# # import copy
# # import viriato_interface
# # train_copy = copy.deepcopy(train)
# # arrival_time_copy = train.train_path_nodes[17].arrival_time
# # departure_time_copy = train.train_path_nodes[17].departure_time
# # minimum_stop_time_copy = train.train_path_nodes[17].minimum_stop_time
# # minimum_run_time_copy = train.train_path_nodes[17].minimum_run_time
# # arrival_time_copy
# # datetime.datetime(2005, 5, 10, 5, 46, 12)
# # minimum_stop_time_copy
# # datetime.timedelta(seconds=228)
# # minimum_run_time_copy
# # datetime.timedelta(seconds=114)
# # train_path_node_section = [train_copy]
# # import datetime
# # new_arrival_time_copy = datetime.datetime(2005, 5, 10, 5, 46, 17)
# # new_minimum_stop_time_copy = datetime.timedelta(seconds=223)
# # new_minimum_run_time_copy = datetime.timedelta(seconds=119)
# # train_copy_updated = viriato_interface.update_train_times(train_copy.id, train_copy.train_path_nodes[17].id,
# # new_arrival_time_copy, departure_time_copy, new_minimum_run_time_copy, new_minimum_stop_time_copy)
# # train_copy_updated.train_path_nodes[17].arrival_time
# # datetime.datetime(2005, 5, 10, 5, 46, 17)
# # train_path_node_section[0].train_path_nodes[17].arrival_time
# # datetime.datetime(2005, 5, 10, 5, 46, 12)
#
# # Get the index of the trains that are on the closed tracks [90, 95, 267, 268, 269, 270, 271, 272, 273, 283, 284,
# # 285, 286, 287, 288, 289]
# # import alns_platform
# # track_info = alns_platform.TrackInformation(trains_timetable, closed_section_track_ids)
# # import viriato_interface
# # possessions = viriato_interface.get_section_track_closures(parameters.time_window)
# # infra_graph, closed_section_track_ids = alns_platform.increase_weight_on_closed_tracks(infra_graph, possessions,
# # parameters)
# # track_info = alns_platform.TrackInformation(trains_timetable, closed_section_track_ids)
# # i = 0
# # index = []
# # for train in trains_timetable:
# #     for train_on_closed_track in track_info.trains_on_closed_tracks:
# #         if train.id == train_on_closed_track:
# #             index.append(i)
# #     i +=1
