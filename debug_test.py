import numpy as np
import alns_platform
import copy
import helpers
import neighbourhood_operators
import timetable_graph

np.random.seed(42)
n_iteration = 3

infra_graph = np.load('output/pickle/debug/infra_graph_'+str(n_iteration)+'.pkl', allow_pickle=True)
operator = np.load('output/pickle/debug/operator_'+str(n_iteration)+'.pkl', allow_pickle=True)
changed_trains = np.load('output/pickle/debug/changed_trains_'+str(n_iteration)+'.pkl', allow_pickle=True)
trains_timetable = np.load('output/pickle/debug/trains_timetable_'+str(n_iteration)+'.pkl', allow_pickle=True)
edges_o_stations_d = np.load('output/pickle/debug/edges_o_stations_d_'+str(n_iteration)+'.pkl', allow_pickle=True)
track_info = np.load('output/pickle/debug/track_info_'+str(n_iteration)+'.pkl', allow_pickle=True)
timetable_prime_graph = np.load('output/pickle/debug/timetable_prime_graph_'+str(n_iteration)+'.pkl', allow_pickle=True)
parameters = np.load('output/pickle/debug/parameters_'+str(n_iteration)+'.pkl', allow_pickle=True)
odt_priority_list_original = np.load('output/pickle/debug/odt_priority_list_original_'+str(n_iteration)+'.pkl', allow_pickle=True)
timetable_solution_prime_graph = np.load('output/pickle/debug/timetable_solution_prime_graph_'+str(n_iteration)+'.pkl', allow_pickle=True)
initial_timetable = np.load('output/pickle/debug/initial_timetable_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_op_current = np.load('output/pickle/debug/z_op_current_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_de_reroute_current = np.load('output/pickle/debug/z_de_reroute_current_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_de_cancel_current = np.load('output/pickle/debug/z_de_cancel_current_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_tt_current = np.load('output/pickle/debug/z_tt_current_'+str(n_iteration)+'.pkl', allow_pickle=True)
timetable_solution_graph = np.load('output/pickle/debug/timetable_solution_graph_'+str(n_iteration)+'.pkl', allow_pickle=True)
scores = np.load('output/pickle/debug/scores_'+str(n_iteration)+'.pkl', allow_pickle=True)
temp_i = np.load('output/pickle/debug/temp_i_'+str(n_iteration)+'.pkl', allow_pickle=True)
solution_archive = np.load('output/pickle/debug/solution_archive_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_cur_accepted = np.load('output/pickle/debug/z_cur_accepted_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_cur_archived = np.load('output/pickle/debug/z_cur_archived_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_op_accepted = np.load('output/pickle/debug/z_op_accepted_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_de_reroute_accepted = np.load('output/pickle/debug/z_de_reroute_accepted_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_de_cancel_accepted = np.load('output/pickle/debug/z_de_cancel_accepted_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_tt_accepted = np.load('output/pickle/debug/z_tt_accepted_'+str(n_iteration)+'.pkl', allow_pickle=True)
z_for_pickle = np.load('output/pickle/debug/z_for_pickle_'+str(n_iteration)+'.pkl', allow_pickle=True)
n_iteration = np.load('output/pickle/debug/n_iteration_'+str(n_iteration)+'.pkl', allow_pickle=True)
number_temperature_changes = np.load('output/pickle/debug/number_temperature_changes_'+str(n_iteration)+'.pkl', allow_pickle=True)
all_accepted_solutions = np.load('output/pickle/debug/all_accepted_solutions_'+str(n_iteration)+'.pkl', allow_pickle=True)
return_to_archive_at_iteration = np.load('output/pickle/debug/return_to_archive_at_iteration_'+str(n_iteration)+'.pkl', allow_pickle=True)
iterations_until_return_archives = np.load('output/pickle/debug/iterations_until_return_archives_'+str(n_iteration)+'.pkl', allow_pickle=True)
number_usage = np.load('output/pickle/debug/number_usage_'+str(n_iteration)+'.pkl', allow_pickle=True)
probabilities = np.load('output/pickle/debug/probabilities_'+str(n_iteration)+'.pkl', allow_pickle=True)
# weights = np.load('output/pickle/debug/weights_'+str(n_iteration)+'.pkl', allow_pickle=True)
temperature_it = np.load('output/pickle/debug/temperature_it_'+str(n_iteration)+'.pkl', allow_pickle=True)

timetable_prime_graph, track_info, edges_o_stations_d, changed_trains, operator, \
            odt_facing_neighbourhood_operator, odt_priority_list_original = \
                alns_platform.apply_operator_to_timetable(operator, timetable_prime_graph, changed_trains, trains_timetable,
                                            track_info, infra_graph, edges_o_stations_d, parameters,
                                            odt_priority_list_original)

# Set the timetable_solution_graph parameters
timetable_solution_prime_graph.edges_o_stations_d = edges_o_stations_d
timetable_solution_prime_graph.timetable = trains_timetable
timetable_solution_prime_graph.graph = timetable_prime_graph
timetable_solution_prime_graph, timetable_prime_graph, odt_priority_list_original = \
    alns_platform.find_path_and_assign_pass_neighbourhood_operator(timetable_prime_graph,
                                                     parameters,
                                                     timetable_solution_prime_graph,
                                                     edges_o_stations_d,
                                                     odt_priority_list_original,
                                                     odt_facing_neighbourhood_operator)































# alns_platform.pickle_results(changed_trains, 'output/pickle/debug/changed_trains.pkl')
# alns_platform.pickle_results(trains_timetable,'output/pickle/debug/trains_timetable.pkl')
# alns_platform.pickle_results(edges_o_stations_d ,'output/pickle/debug/edges_o_stations_d.pkl')
# alns_platform.pickle_results(track_info,'output/pickle/debug/track_info.pkl')
# alns_platform.pickle_results(timetable_prime_graph,'output/pickle/debug/timetable_prime_graph.pkl')
# alns_platform.pickle_results(parameters,'output/pickle/debug/parameters.pkl')
# alns_platform.pickle_results(odt_priority_list_original,'output/pickle/debug/odt_priority_list_original.pkl')
# alns_platform.pickle_results(infra_graph, 'output/pickle/debug/infra_graph.pkl')
# alns_platform.pickle_results(operator, 'output/pickle/debug/operator.pkl')
# alns_platform.pickle_results(changed_trains, 'output/pickle/debug/changed_trains.pkl')
# alns_platform.pickle_results(trains_timetable,'output/pickle/debug/trains_timetable.pkl')
# alns_platform.pickle_results(edges_o_stations_d ,'output/pickle/debug/edges_o_stations_d.pkl')
# alns_platform.pickle_results(track_info,'output/pickle/debug/track_info.pkl')
# alns_platform.pickle_results(timetable_prime_graph,'output/pickle/debug/timetable_prime_graph.pkl')
# alns_platform.pickle_results(parameters,'output/pickle/debug/parameters.pkl')
# alns_platform.pickle_results(odt_priority_list_original,'output/pickle/debug/odt_priority_list_original.pkl')
# alns_platform.pickle_results(timetable_solution_prime_graph,'output/pickle/debug/timetable_solution_prime_graph.pkl')
# alns_platform.pickle_results(initial_timetable,'output/pickle/debug/initial_timetable.pkl')
# alns_platform.pickle_results(z_op_current,'output/pickle/debug/z_op_current.pkl')
# alns_platform.pickle_results(z_de_reroute_current,'output/pickle/debug/z_de_reroute_current.pkl')
# alns_platform.pickle_results(z_de_cancel_current,'output/pickle/debug/z_de_cancel_current.pkl')
# alns_platform.pickle_results(z_tt_current,'output/pickle/debug/z_tt_current.pkl')
# alns_platform.pickle_results(timetable_solution_graph,'output/pickle/debug/timetable_solution_graph.pkl')
# alns_platform.pickle_results(scores,'output/pickle/debug/scores.pkl')
# alns_platform.pickle_results(temp_i,'output/pickle/debug/temp_i.pkl')
# alns_platform.pickle_results(solution_archive,'output/pickle/debug/solution_archive.pkl')
# alns_platform.pickle_results(z_cur_accepted,'output/pickle/debug/z_cur_accepted.pkl')
# alns_platform.pickle_results(z_cur_archived,'output/pickle/debug/z_cur_archived.pkl')
# alns_platform.pickle_results(z_op_accepted,'output/pickle/debug/z_op_accepted.pkl')
# alns_platform.pickle_results(z_de_reroute_accepted,'output/pickle/debug/z_de_reroute_accepted.pkl')
# alns_platform.pickle_results(z_de_cancel_accepted,'output/pickle/debug/z_de_cancel_accepted.pkl')
# alns_platform.pickle_results(z_tt_accepted,'output/pickle/debug/z_tt_accepted.pkl')
# alns_platform.pickle_results(z_for_pickle,'output/pickle/debug/z_for_pickle.pkl')
# alns_platform.pickle_results(n_iteration,'output/pickle/debug/n_iteration.pkl')
# alns_platform.pickle_results(number_temperature_changes,'output/pickle/debug/number_temperature_changes.pkl')
# alns_platform.pickle_results(all_accepted_solutions,'output/pickle/debug/all_accepted_solutions.pkl')
# alns_platform.pickle_results(return_to_archive_at_iteration,'output/pickle/debug/return_to_archive_at_iteration.pkl')
# alns_platform.pickle_results(iterations_until_return_archives,'output/pickle/debug/iterations_until_return_archives.pkl')
# alns_platform.pickle_results(number_usage,'output/pickle/debug/number_usage.pkl')
# alns_platform.pickle_results(probabilities,'output/pickle/debug/probabilities.pkl')
# alns_platform.pickle_results(weights,'output/pickle/debug/weights.pkl')
# alns_platform.pickle_results(temperature_it,'output/pickle/debug/temperature_it.pkl')

# alns_platform.pickle_results(parameters,'output/pickle/debug/parameters_08_06.pkl')
# alns_platform.pickle_results(odt_facing_capacity_constraint,'output/pickle/debug/odt_facing_capacity_constraint_08_06.pkl')
# alns_platform.pickle_results(timetable_initial_graph,'output/pickle/debug/timetable_initial_graph_08_06.pkl')

