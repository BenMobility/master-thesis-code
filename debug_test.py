import numpy as np
import alns_platform
import copy
import helpers
import neighbourhood_operators
import timetable_graph

feasible_timetable_graph = True
prime_feasible_timetable_graph = True
iterations_until_return_archives = 100
return_to_archive_at_iteration = 100
n_iteration = 2
number_temperature_changes = 0
all_accepted_solutions = np.load('output/pickle/debug/all_accepted_solutions.pkl', allow_pickle=True)
changed_trains = np.load('output/pickle/debug/changed_trains.pkl', allow_pickle=True)
edges_o_stations_d = np.load('output/pickle/debug/edges_o_stations_d.pkl', allow_pickle=True)
infra_graph = np.load('output/pickle/debug/infra_graph.pkl', allow_pickle=True)
initial_timetable = np.load('output/pickle/debug/initial_timetable.pkl', allow_pickle=True)
number_usage = np.load('output/pickle/debug/number_usage.pkl', allow_pickle=True)
odt_priority_list_original = np.load('output/pickle/debug/odt_priority_list_original.pkl', allow_pickle=True)
parameters = np.load('output/pickle/debug/parameters.pkl', allow_pickle=True)
probabilities = np.load('output/pickle/debug/probabilities.pkl', allow_pickle=True)
scores = np.load('output/pickle/debug/scores.pkl', allow_pickle=True)
solution_archive = np.load('output/pickle/debug/solution_archive.pkl', allow_pickle=True)
solutions = np.load('output/pickle/debug/solutions.pkl', allow_pickle=True)
temp_i = np.load('output/pickle/debug/temp_i.pkl', allow_pickle=True)
temperature_it = np.load('output/pickle/debug/temperature_it.pkl', allow_pickle=True)
timetable_initial_graph = np.load('output/pickle/debug/timetable_initial_graph.pkl', allow_pickle=True)
timetable_prime_graph = np.load('output/pickle/debug/timetable_prime_graph.pkl', allow_pickle=True)
timetable_solution_graph = np.load('output/pickle/debug/timetable_solution_graph.pkl', allow_pickle=True)
timetable_solution_prime_graph = np.load('output/pickle/debug/timetable_solution_prime_graph.pkl', allow_pickle=True)
track_info = np.load('output/pickle/debug/track_info.pkl', allow_pickle=True)
trains_timetable = np.load('output/pickle/debug/trains_timetable.pkl', allow_pickle=True)
weights = np.load('output/pickle/debug/weights.pkl', allow_pickle=True)
z_for_pickle = np.load('output/pickle/debug/z_for_pickle.pkl', allow_pickle=True)

z_cur_accepted = z_for_pickle['z_cur_accepted']
z_cur_archived = z_for_pickle['z_cur_archived']
z_de_cancel_accepted = z_for_pickle['z_de_cancel_acc']
z_de_cancel_current = z_for_pickle['z_de_cancel_cur']
z_de_reroute_accepted = z_for_pickle['z_de_reroute_acc']
z_de_reroute_current = z_for_pickle['z_de_reroute_cur']
z_op_accepted = z_for_pickle['z_op_acc']
z_op_current = z_for_pickle['z_op_cur']
z_tt_accepted = z_for_pickle['z_tt_acc']
z_tt_current = z_for_pickle['z_tt_cur']

while any(t > 0 for t in temp_i) and n_iteration < parameters.number_iteration:
        print(f'\nIteration number: {n_iteration}')
        # Add an iteration in the count
        n_iteration += 1
        if feasible_timetable_graph:
                # Capture the current temperature
                temperature_it.append(temp_i.copy())
                # Get the trains timetable in a deepcopy
                trains_timetable = copy.deepcopy(timetable_solution_graph.time_table)
                # Get the track info
                track_info = alns_platform.TrackInformation(trains_timetable, parameters.closed_tracks)
                # Keep track of the changed trains in the current solution
                changed_trains = copy.deepcopy(timetable_solution_graph.changed_trains)
                # Identify the candidates for operators
                alns_platform.identify_candidates_for_operators(trains_timetable, parameters, timetable_solution_graph,
                                                  changed_trains)
                # Get the number of usage and the selected operator
                number_usage, operator = alns_platform.select_operator(probabilities, number_usage, changed_trains, track_info,
                                                         parameters)

                # Print the selected operator
                print(f'Selected operator: {operator}')

                # Create a new timetable with the current solution but without flow
                timetable_prime_graph = timetable_solution_graph.graph

                # Create a dict of edges that combines origin to stations and stations to destination edges
                edges_o_stations_d = helpers.CopyEdgesOriginStationDestination(
                        timetable_solution_graph.edges_o_stations_d)

                # Create a new empty solution
                timetable_solution_prime_graph = helpers.Solution()

                # Apply the operator on the current solution
                print('Apply the operator on the current solution.')
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
                timetable_solution_prime_graph.total_dist_train = alns_platform.distance_travelled_all_trains(trains_timetable,
                                                                                                infra_graph)
                timetable_solution_prime_graph.deviation_reroute_timetable = alns_platform.deviation_reroute_timetable(
                        trains_timetable,
                        initial_timetable,
                        changed_trains,
                        parameters)
                timetable_solution_prime_graph.deviation_cancel_timetable = alns_platform.deviation_cancel_timetable(trains_timetable,
                                                                                                       initial_timetable,
                                                                                                       changed_trains,
                                                                                                       parameters)
                timetable_solution_prime_graph.changed_trains = changed_trains
                timetable_solution_prime_graph.set_of_trains_for_operator = parameters.set_of_trains_for_operator

                # Archive the current solution
                z_op_current.append(timetable_solution_prime_graph.total_dist_train)
                z_de_reroute_current.append(timetable_solution_prime_graph.deviation_reroute_timetable)
                z_de_cancel_current.append(timetable_solution_prime_graph.deviation_cancel_timetable)
                z_tt_current.append(timetable_solution_prime_graph.total_traveltime)

                # Printout the current results
                print('z_o_current : ', timetable_solution_prime_graph.total_dist_train,
                      '\n z_d_reroute_current : ', timetable_solution_prime_graph.deviation_reroute_timetable,
                      '\n z_d_cancel_current : ', timetable_solution_prime_graph.deviation_cancel_timetable,
                      '\n z_p_current : ', timetable_solution_prime_graph.total_traveltime)

                # Archive limited to 80 solutions (memory issue, could be increased)
                timetable_solution_graph, scores, accepted_solution, archived_solution = \
                        alns_platform.archiving_acceptance_rejection(timetable_solution_graph, timetable_solution_prime_graph,
                                                       operator,
                                                       parameters, scores, temp_i, solution_archive)

                # Save the current solution, the accepted solution
                z_cur_accepted.append(accepted_solution)
                z_cur_archived.append(archived_solution)
                z_op_accepted.append(timetable_solution_graph.total_dist_train)
                z_de_reroute_accepted.append(timetable_solution_graph.deviation_reroute_timetable)
                z_de_cancel_accepted.append(timetable_solution_graph.deviation_cancel_timetable)
                z_tt_accepted.append(timetable_solution_graph.total_traveltime)

                # Save the multi objective results into a pickle
                alns_platform.pickle_results(z_for_pickle, 'output/pickle/z_pickle.pkl')

                # Printout the accepted solution
                print('z_o_accepted : ', timetable_solution_graph.total_dist_train,
                      '\n z_d_reroute_accepted : ', timetable_solution_graph.deviation_reroute_timetable,
                      '\n z_d_cancel_accepted : ', timetable_solution_graph.deviation_cancel_timetable,
                      '\n z_p_accepted : ', timetable_solution_graph.total_traveltime)

                # Print out the temperature and the number of iteration
                print('temperature : ', temp_i, ' iteration : ', n_iteration)

                # Update the temperature
                temp_i, number_temperature_changes = alns_platform.update_temperature(temp_i, n_iteration, number_temperature_changes,
                                                                        all_accepted_solutions, parameters)

                # Select a solution from the archive periodically
                timetable_solution_graph, iterations_until_return_archives, return_to_archive_at_iteration = \
                        alns_platform.periodically_select_solution(solution_archive, timetable_solution_graph, n_iteration,
                                                     iterations_until_return_archives, parameters,
                                                     return_to_archive_at_iteration)

                # Update the weights
                weights, scores, probabilities = alns_platform.update_weights(weights, scores, number_usage,
                                                                number_temperature_changes,
                                                                probabilities, parameters)
        if not prime_feasible_timetable_graph:
                # Add the temperature for the iteration
                temperature_it.append(temp_i)

                # Restore the disruption infeasibility's at the first iteration
                changed_trains = neighbourhood_operators.restore_disruption_feasibility(infra_graph, track_info,
                                                                                        parameters)

                # Get trains in the time window and area
                trains_timetable = alns_platform.get_all_trains_cut_to_time_window_and_area(parameters)

                # Save the initial timetable feasible graph
                parameters.initial_timetable_feasible = copy.deepcopy(trains_timetable)

                # Get all the information in one place for the tracks
                track_info = alns_platform.TrackInformation(trains_timetable, parameters.closed_tracks)

                # Create the restored feasibility graph
                timetable_prime_graph, edges_o_stations_d = \
                        timetable_graph.create_restored_feasibility_graph(trains_timetable, parameters)

                # Record the current solution
                timetable_solution_prime_graph = helpers.Solution()
                timetable_solution_prime_graph.timetable = copy.deepcopy(trains_timetable)

                # Passenger assignment and compute the objective for passenger
                solutions, timetable_prime_graph, odt_priority_list_original = \
                        alns_platform.find_path_and_assign_pass(timetable_prime_graph, parameters, timetable_solution_prime_graph,
                                                  edges_o_stations_d)

                # Record the results of the current solution timetable
                timetable_solution_prime_graph.total_dist_train = alns_platform.distance_travelled_all_trains(trains_timetable,
                                                                                                infra_graph)
                timetable_solution_prime_graph.deviation_reroute_timetable = alns_platform.deviation_reroute_timetable(
                        trains_timetable,
                        initial_timetable,
                        changed_trains,
                        parameters)
                timetable_solution_prime_graph.deviation_cancel_timetable = alns_platform.deviation_cancel_timetable(trains_timetable,
                                                                                                       initial_timetable,
                                                                                                       changed_trains,
                                                                                                       parameters)

                # Check if the current solution has a value for deviation, if not, it means that the algorithm platform
                # has not been restarted for the original timetable
                if timetable_solution_prime_graph.deviation_reroute_timetable == 0:
                        raise Exception(
                                'The deviation restored feasibility is 0, we need to restart the Viriato algorithm'
                                'platform')

                # Record all the results for the current solution
                timetable_solution_prime_graph.set_of_trains_for_operator['Cancel'] = \
                        copy.deepcopy(parameters.set_of_trains_for_operator['Cancel'])
                timetable_solution_prime_graph.set_of_trains_for_operator['Delay'] = \
                        copy.deepcopy(parameters.set_of_trains_for_operator['Delay'])
                timetable_solution_prime_graph.graph = timetable_graph.copy_graph_with_flow(timetable_prime_graph)
                timetable_solution_prime_graph.changed_trains = copy.deepcopy(changed_trains)
                timetable_solution_prime_graph.edges_o_stations_d = \
                        helpers.CopyEdgesOriginStationDestination(edges_o_stations_d)

                # Save the current and the accepted solution
                z_op_current.append(timetable_solution_prime_graph.total_dist_train)
                z_de_reroute_current.append(timetable_solution_prime_graph.deviation_reroute_timetable)
                z_de_cancel_current.append(timetable_solution_prime_graph.deviation_cancel_timetable)
                z_tt_current.append(timetable_solution_prime_graph.total_traveltime)
                z_op_accepted.append(timetable_solution_prime_graph.total_dist_train)
                z_de_reroute_accepted.append(timetable_solution_prime_graph.deviation_reroute_timetable)
                z_de_cancel_accepted.append(timetable_solution_prime_graph.deviation_cancel_timetable)
                z_tt_accepted.append(timetable_solution_prime_graph.total_traveltime)
                z_cur_accepted.append(True)
                z_cur_archived.append(True)

                # Pickle the results
                alns_platform.pickle_results(z_for_pickle, 'output/pickle/z_pickle.pkl')

                # Printout the results
                print(f'z_o: {timetable_solution_prime_graph.total_dist_train} \n'
                      f'z_d_reroute: {timetable_solution_prime_graph.deviation_reroute_timetable}\n'
                      f'z_d_cancel: {timetable_solution_prime_graph.deviation_cancel_timetable}\n'
                      f'z_p: {timetable_solution_prime_graph.total_traveltime}')

                # Save the solution in the archives
                solution_archive.append(timetable_solution_prime_graph)

                # Copy the solution for the next iteration
                timetable_solution_graph = alns_platform.get_solution_graph_with_copy_solution_prime(timetable_solution_prime_graph,
                                                                                       parameters)

                # Set the feasible graph and prime graph to true for next iteration
                feasible_timetable_graph = True
                prime_feasible_timetable_graph = True

