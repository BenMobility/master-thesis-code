import algorithm_platform_methods as ap
import numpy as np
import math
import neighborhood_structures as nh
import Classes
import prepare_io_and_call_alns as pio
import graphcreator
import shortestpath as sp
import networkx as nx
import copy
import utils
import datetime
import time
import pickle
import cProfile
import io
import pstats
import sp_topological_sort as sp_topo
# import resources


def alns(G_0, G_infra, cut_trains_to_area, track_info, parameters):
    initial_timetable = Classes.Timetables()
    initial_timetable.initial_timetable_infeasible = copy.deepcopy(cut_trains_to_area)
    weights = Classes.Weights()
    scores = Classes.Scores()
    nr_usage = Classes.Nr_usage()
    probabilities = Classes.Probabilities(weights)
    parameters.trains_on_closed_track_initial_timetable_infeasible = [train['ID'] for train in track_info.trains_on_closed_tracks]
    parameters.train_ids_initial_timetable_infeasible = [train['ID'] for train in initial_timetable.initial_timetable_infeasible]

    n_iteration = 0

    # could be optimized
    nr_temperature_changes = 0

    iterations_until_return_archives = 100  # Binder 2000
    return_to_archive_at_iteration = n_iteration + iterations_until_return_archives

    t_i = parameters.t_start
    solution_archive = []

    # operation, deviation and travel time cost
    z_op_current, z_de_current, z_tt_current = [], [], []
    z_op_accepted, z_de_accepted, z_tt_accepted = [], [], []
    z_cur_accepted, z_cur_archived, temperature_it = [], [], []

    # combine three different costs
    all_accepted_solutions = [z_op_accepted, z_de_accepted, z_tt_accepted]

    # save values - current, accepted etc.
    z_for_pickle = {'z_op_cur': z_op_current, 'z_de_cur': z_de_current, 'z_tt_cur': z_tt_current,
                    'z_op_acc': z_op_accepted, 'z_de_acc': z_de_accepted, 'z_tt_acc': z_tt_accepted,
                    'z_cur_acceppted': z_cur_accepted, 'z_cur_archived': z_cur_archived, 't_it': temperature_it}

    G_feasible, G_prime_feasible = False, False

    # profiling the iterations
    pr = start_end_profiler(start=True)

    while any(t > 0 for t in t_i) and n_iteration < 4000:
        n_iteration += 1
        if G_feasible:
            temperature_it.append(t_i.copy())
            cut_trains_to_area = copy.deepcopy(solution_G.time_table)
            track_info = pio.TrackInformation(cut_trains_to_area, parameters.closed_tracks, G_infra)
            changed_trains = copy.deepcopy(solution_G.changed_trains) #each train that was changed in the current solution
            identify_candidates_for_operators(cut_trains_to_area, parameters, solution_G, changed_trains)
            nr_usage, operator = select_operator(probabilities, nr_usage, changed_trains, track_info, parameters)
            # if n_iteration > 20:
            #    operator = 'Return'
            #    list_candids = candids_for_return_operator(changed_trains, parameters)
            #    if len(list_candids) <= 3:
            #        nr_usage, operator = select_operator(probabilities, nr_usage, changed_trains, track_info,
            #                                            parameters)

            print(operator)
            G_prime = copy_graph_and_remove_flow(solution_G.graph, parameters)
            # track_info = copy_track_info_from_solution_G(solution_G)
            edges_o_stations_d = Classes.Copy_edges_origin_station_destionation(solution_G.edges_o_stations_d)
            # create new empty solution obj
            solution_prime = Classes.Solutions()
            # check_Graph_number_edges_nodes(G_prime)
            G_prime, track_info, edges_o_stations_d, changed_trains, operator = \
                apply_operator_to_G(operator, G_prime, changed_trains, cut_trains_to_area, track_info, G_infra,
                                    edges_o_stations_d, parameters)

            solution_prime.edges_o_stations_d = edges_o_stations_d
            solution_prime.time_table = cut_trains_to_area
            solution_prime.graph = G_prime
            solution_prime, G_prime = find_path_and_assign_pass(G_prime, G_0, parameters, solution_prime, edges_o_stations_d)
            solution_prime.total_dist_train = distance_travelled_all_trains(cut_trains_to_area, G_infra)
            solution_prime.deviation_timetable = deviation_time_table(cut_trains_to_area, initial_timetable,
                                                                      changed_trains, parameters)
            solution_prime.changed_trains = changed_trains
            # solution_prime.track_info = copy_track_info_from_track_info(track_info)

            solution_prime.set_of_trains_for_operator = parameters.set_of_trains_for_operator

            z_op_current.append(solution_prime.total_dist_train)
            z_de_current.append(solution_prime.deviation_timetable)
            z_tt_current.append(solution_prime.total_traveltime)

            print('z_o_current : ', solution_prime.total_dist_train, ' z_d_current : ', solution_prime.deviation_timetable,
                  ' z_p_current : ', solution_prime.total_traveltime)

            # archive limited to 80 solutions (memory issue, could be increased)
            solution_G, scores, accepted_solution, archived_solution = archiving_acceptance_rejection(solution_G,
                                                    solution_prime, operator, parameters, scores, t_i, solution_archive)
            #
            z_cur_accepted.append(accepted_solution)
            z_cur_archived.append(archived_solution)
            z_op_accepted.append(solution_G.total_dist_train)
            z_de_accepted.append(solution_G.deviation_timetable)
            z_tt_accepted.append(solution_G.total_traveltime)
            pickle_results(z_for_pickle, 'z_pickle.pkl')

            print('z_o_accepted : ', solution_G.total_dist_train, ' z_d_accepted : ', solution_G.deviation_timetable,
                  ' z_p_accepted : ', solution_G.total_traveltime)

            print('temperature : ', t_i, ' iteration : ', n_iteration)
            t_i, nr_temperature_changes = update_temperature(t_i, n_iteration, nr_temperature_changes, all_accepted_solutions)
            solution_G, iterations_until_return_archives, return_to_archive_at_iteration = \
                periodically_select_solution(solution_archive, solution_G, n_iteration, iterations_until_return_archives
                                             , parameters, return_to_archive_at_iteration)
            weights, scores, probabilities = update_weights(weights, scores, nr_usage, nr_temperature_changes, probabilities)

        if not G_prime_feasible:
            temperature_it.append(t_i)
            # restore the disruption infeasibility's at the first iterations....
            changed_trains = nh.restore_disruption_feasibility(G_infra, track_info, parameters)
            cut_trains_to_area = get_all_trains_cut_to_time_window_and_area(parameters)
            parameters.initial_timetable_feasible = copy.deepcopy(cut_trains_to_area)
            # Get all information about track occupancy, defined in, uptade because of the changes
            track_info = pio.TrackInformation(cut_trains_to_area, parameters.closed_tracks, G_infra)
            G_prime, edges_o_stations_d = graphcreator.create_restored_feasibility_graph(cut_trains_to_area, parameters)

            solution_prime = Classes.Solutions()
            solution_prime.time_table = cut_trains_to_area.copy()

            # find paths and assign passengers and calculate z_p
            use_topo = False
            use_dijkstra = True
            if use_dijkstra:
                solutions, G_prime = find_path_and_assign_pass(G_prime, G_0, parameters, solution_prime, edges_o_stations_d)
            
            # could be optimized probably
            if use_topo:
                # find_path_and_assign_pass_topo(G, parameters)
                find_path_and_assign_pass_topo_sparse_graph(G_prime, parameters, edges_o_stations_d)

            solution_prime.total_dist_train = distance_travelled_all_trains(cut_trains_to_area, G_infra)
            solution_prime.deviation_timetable = deviation_time_table(cut_trains_to_area, initial_timetable,
                                                                      changed_trains, parameters)
            if solution_prime.deviation_timetable == 0:
                raise Exception('deviation restored feasibility is 0, restart Algorithm Platform ')

            solution_prime.set_of_trains_for_operator['Cancel'] = parameters.set_of_trains_for_operator['Cancel'].copy()
            solution_prime.set_of_trains_for_operator['Delay'] = parameters.set_of_trains_for_operator['Delay'].copy()
            solution_prime.graph = copy_graph_with_flow(G_prime, parameters)
            solution_prime.changed_trains = copy.deepcopy(changed_trains)
            solution_prime.edges_o_stations_d = Classes.Copy_edges_origin_station_destionation(edges_o_stations_d)
            # solution_prime.track_info = copy_track_info_from_track_info(track_info)
            z_op_current.append(solution_prime.total_dist_train)
            z_de_current.append(solution_prime.deviation_timetable)
            z_tt_current.append(solution_prime.total_traveltime)
            z_op_accepted.append(solution_prime.total_dist_train)
            z_de_accepted.append(solution_prime.deviation_timetable)
            z_tt_accepted.append(solution_prime.total_traveltime)
            z_cur_accepted.append(True)
            z_cur_archived.append(True)

            pickle_results(z_for_pickle, 'z_pickle.pkl')

            print('z_o : ', solution_prime.total_dist_train, ' z_d : ', solution_prime.deviation_timetable,
                  ' z_p : ', solution_prime.total_traveltime)

            solution_archive.append(solution_prime)
            solution_G = get_solution_G_with_copy_solution_prime(solution_prime, parameters)

            # accepted_solutions.append(solution_prime)
            G_feasible = True
            G_prime_feasible = True

    start_end_profiler('ALNS', end=True, pr=pr)

    pickle_archive_op_tt_de(solution_archive)
    pickle_archive_w_changed_Trains(solution_archive)
    pickle_results(weights, 'weights.pkl')
    pickle_results(probabilities, 'probabilities.pkl')
    return solution_archive




def archiving_acceptance_rejection(solution_G, solution_prime, operator, parameters, scores, t_i, solution_archive):
    archived_solution = False
    accepted_solution = False
    solution_prime_not_dominated_by_any_solution = True

    if True:
        for solution in solution_archive:
            cond_tt = solution_prime.total_traveltime < solution.total_traveltime
            cond_de = solution_prime.deviation_timetable < solution.deviation_timetable
            cond_op = solution_prime.total_dist_train < solution.total_dist_train
            if any([cond_tt, cond_op, cond_de]):
                continue
            else:
                solution_prime_not_dominated_by_any_solution = False
                break
        if solution_prime_not_dominated_by_any_solution:
            archived_solution = True
            accepted_solution = True
            print('solution added to archive')
            solution_archive.append(solution_prime)
            solution_G = get_solution_G_with_copy_solution_prime(solution_prime, parameters)
            scores = update_scores(operator, parameters.score_1, scores)

    if archived_solution:
        # Check if there are any solutions in the archive which are dominated by current solution
        index_dominated_solutions = []
        i = -1
        for solution in solution_archive[:-1]:
            i += 1
            cond_tt = solution_prime.total_traveltime <= solution.total_traveltime
            cond_de = solution_prime.deviation_timetable <= solution.deviation_timetable
            cond_op = solution_prime.total_dist_train <= solution.total_dist_train
            if all([cond_tt, cond_op, cond_de]):
                cond_tt = solution_prime.total_traveltime < solution.total_traveltime
                cond_de = solution_prime.deviation_timetable < solution.deviation_timetable
                cond_op = solution_prime.total_dist_train < solution.total_dist_train
                if any([cond_tt, cond_op, cond_de]):
                    index_dominated_solutions.append(i)
        if len(index_dominated_solutions) >= 1:
            index_dominated_solutions.reverse()
            for i in index_dominated_solutions:
                del solution_archive[i]
        pickle_full_archive(solution_archive)

    # t_i <- [t_op, t_de, t_tt]
    else:
        reaction_factor_op = 1
        reaction_factor_dev = 1

        z_o_prime = solution_prime.total_dist_train
        z_o = solution_G.total_dist_train
        try:
            acceptance_prob_operation = min(math.exp((-(z_o_prime - z_o) / t_i[0]))*reaction_factor_op, 1)
        except ZeroDivisionError:
            acceptance_prob_operation = 0.001
            print('division by zero, t_i', t_i)


        z_d_prime = solution_prime.deviation_timetable
        z_d = solution_G.deviation_timetable
        try:
            acceptance_prob_deviation = min(math.exp((-(z_d_prime - z_d) / t_i[1]))*reaction_factor_dev, 1)
        except ZeroDivisionError:
            acceptance_prob_deviation = 0.001
            print('division by zero, t_i', t_i)


        z_tt_prime = solution_prime.total_traveltime
        z_tt = solution_G.total_traveltime
        try:
            acceptance_prob_passenger = min(math.exp((-(z_tt_prime - z_tt) / t_i[2])), 1)
        except ZeroDivisionError:
            acceptance_prob_passenger = 0.001
            print('division by zero, t_i', t_i)


        accteptance_prob = acceptance_prob_passenger * acceptance_prob_deviation * acceptance_prob_operation

        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
        rd = np.random.uniform(0.0, 1.0)

        if rd < accteptance_prob:
            accepted_solution = True
            solution_G = get_solution_G_with_copy_solution_prime(solution_prime, parameters)
            scores = update_scores(operator, parameters.score_2, scores)
        else:
            scores = update_scores(operator, parameters.score_3, scores)

    if len(solution_archive) >= 80:
        # throw away worst solution if archive gets to big
        idx_throw_away = 0
        current_worst = solution_archive[idx_throw_away]

        print('max archive size reached, worst is deleted')
        for i in range(1, len(solution_archive)-1):
            if solution_archive[i].total_traveltime >= current_worst.total_traveltime:
                if solution_archive[i].total_traveltime > current_worst.total_traveltime:
                    idx_throw_away = i
                    continue
                else:
                    if solution_archive[i].deviation_timetable >= current_worst.deviation_timetable:
                        if solution_archive[i].deviation_timetable > current_worst.deviation_timetable:
                            idx_throw_away = i
                            continue
                        else:
                            if solution_archive[i].total_dist_train > current_worst.total_dist_train:
                                idx_throw_away = i

        print('solution archive reached max lenght, worst solution is deleted')
        del solution_archive[idx_throw_away]

    return solution_G, scores, accepted_solution, archived_solution  # , best_solution


def periodically_select_solution(solution_archive, solution_G, n_iteration, iterations_until_return_archives, parameters,
                                 return_to_archive_at_iteration):
    reaction_factor_return_archive = 0.9

    if n_iteration == return_to_archive_at_iteration:
        iterations_until_return_archives = reaction_factor_return_archive * iterations_until_return_archives
        if iterations_until_return_archives <= 20:
            iterations_until_return_archives = 20

        return_to_archive_at_iteration = int(n_iteration + round(iterations_until_return_archives, 0))

        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
        n = 8
        solutions_sorted_tt = []

        for solution in solution_archive:
            solutions_sorted_tt.append(solution.total_traveltime)
        if n > len(solutions_sorted_tt):
            n = len(solutions_sorted_tt) - 1

            # sort by travel time
        preselect_n_best_solutions_traveltime = list(np.argsort(np.array(solutions_sorted_tt))[-n:])
        rd_index = np.random.randint(0, len(preselect_n_best_solutions_traveltime))

        solution_G = get_solution_G_with_copy_solution_prime(solution_archive[rd_index], parameters)
        print('return to archive: selected solution :')
        print('z_o_accepted : ', solution_G.total_dist_train, ' z_d_accepted : ', solution_G.deviation_timetable,
              ' z_p_accepted : ', solution_G.total_traveltime)

    return solution_G, iterations_until_return_archives, return_to_archive_at_iteration



def update_scores(operator, score, scores):
    if operator == 'Cancel':
        scores.cc += score

    elif operator == 'CancelFrom':
        scores.pc += score

    elif operator == 'Delay':
        scores.cd += score

    elif operator == 'DelayFrom':
        scores.pd += score

    elif operator == 'EmergencyTrain':
        scores.et += score

    elif operator == 'EmergencyBus':
        scores.eb += score

    elif operator == 'Return':
        scores.ret += score

    return scores


def update_temperature(t_i, n_iteration, m, all_accepted_solutions):
    # all_accepted_solutions <-- [z_op_accepted, z_de_accepted, z_tt_accepted]
    warm_up_phase = 100  # Binder 300
    iterations_at_T_level = 20  # Binder 50
    number_of_temperature_change = 60  #100  # Binder 100
    p_0 = 0.999
    p_f = 0.001

    if n_iteration >= warm_up_phase and m != number_of_temperature_change:
        if n_iteration % iterations_at_T_level == 0:
            m += 1
            sigma_z_op = np.std(all_accepted_solutions[0])
            sigma_z_de = np.std(all_accepted_solutions[1])
            sigma_z_tt = np.std(all_accepted_solutions[2])
            t_i[0] = - sigma_z_op / (np.log(p_0 + ((p_f - p_0)/number_of_temperature_change) * m))
            t_i[1] = - sigma_z_de / (np.log(p_0 + ((p_f - p_0) / number_of_temperature_change) * m))
            t_i[2] = - sigma_z_tt / (np.log(p_0 + ((p_f - p_0) / number_of_temperature_change) * m))

    elif m == number_of_temperature_change:
        t_i = [0, 0, 0]

    return t_i, m


def update_weights(weights, scores, nr_usage, nr_temperature_updates, probabilities):
    if nr_temperature_updates % 2 == 0 and nr_temperature_updates != 0:
        reaction_factor = 0.5
        if nr_usage.cc != 0:
            weights.cc = (1 - 0.5)*weights.cc + reaction_factor * scores.cc / nr_usage.cc
            nr_usage.cc = 0
            scores.cc = 1

        if nr_usage.pc != 0:
            weights.pc = (1 - 0.5) * weights.pc + reaction_factor * scores.pc / nr_usage.pc
            nr_usage.pc = 0
            scores.pc = 1

        if nr_usage.cd != 0:
            weights.cd = (1 - 0.5) * weights.cd + reaction_factor * scores.cd / nr_usage.cd
            nr_usage.cd = 0
            scores.cd = 1

        if nr_usage.pd != 0:
            weights.pd = (1 - 0.5) * weights.pd + reaction_factor * scores.pd / nr_usage.pd
            nr_usage.pd = 0
            scores.pd = 1

        if nr_usage.et != 0:
            weights.et = (1 - 0.5) * weights.et + reaction_factor * scores.et / nr_usage.et
            nr_usage.et = 0
            scores.et = 1

        if nr_usage.eb != 0:
            weights.eb = (1 - 0.5) * weights.eb + reaction_factor * scores.eb / nr_usage.eb
            nr_usage.eb = 0
            scores.eb = 1

        if nr_usage.ret != 0:
            weights.ret = (1 - 0.5) * weights.ret + reaction_factor * scores.ret / nr_usage.ret
            nr_usage.ret = 0
            scores.ret = 1
        weights.sum = weights.cc + weights.cd + weights.eb + weights.et + weights.pc + weights.pd + weights.ret
        
        probabilities = Classes.Probabilities(weights) # changed when the weights are adapted
    return weights, scores, probabilities


def add_accepted_solution_for_Viriato(accepted_changes_current_solution, all_accepted_solutions, changes_per_accepted_solution,
                          accepted_solutions, solution_prime, solution_archive=None):

    accepted_solutions.append(solution_prime)
    all_accepted_solutions.append([solution_prime.total_traveltime, solution_prime.deviation_timetable,
                                   solution_prime.total_dist_train])
    index_current_solution = len(all_accepted_solutions)
    changes = [{key: value} for key, value in solution_prime.changed_trains.items()]
    accepted_changes_current_solution = accepted_changes_current_solution.append(changes)
    changes_per_accepted_solution[index_current_solution] = accepted_changes_current_solution

    if solution_archive is not None:
        solution_archive.append(solution_prime)


def copy_track_info_from_solution_G(solution_G):

    # track_info_copy = copy.deepcopy(solution_G.track_info)
    track_info = pio.Track_info_to_none()
    track_info.nr_usage_tracks = solution_G.track_info.nr_usage_tracks.copy()
    track_info.trains_on_closed_tracks = solution_G.track_info.trains_on_closed_tracks.copy()
    track_info.tpn_information = copy.deepcopy(solution_G.track_info.tpn_information)
    track_info.track_sequences_of_TPN = copy.deepcopy(solution_G.track_info.track_sequences_of_TPN)
    track_info.trainID_debugString = solution_G.track_info.trainID_debugString.copy()
    track_info.tuple_key_value_of_tpn_ID_arrival = copy.deepcopy(solution_G.track_info.tuple_key_value_of_tpn_ID_arrival)
    track_info.tuple_key_value_of_tpn_ID_departure = copy.deepcopy(solution_G.track_info.tuple_key_value_of_tpn_ID_departure)

    return track_info


def copy_track_info_from_track_info(track_info):

    track_info_return = pio.Track_info_to_none()
    track_info_return.nr_usage_tracks = track_info.nr_usage_tracks.copy()
    track_info_return.trains_on_closed_tracks = track_info.trains_on_closed_tracks.copy()
    track_info_return.tpn_information = copy.deepcopy(track_info.tpn_information)
    track_info_return.track_sequences_of_TPN = copy.deepcopy(track_info.track_sequences_of_TPN)
    track_info_return.trainID_debugString = track_info.trainID_debugString.copy()
    track_info_return.tuple_key_value_of_tpn_ID_arrival = copy.deepcopy(track_info.tuple_key_value_of_tpn_ID_arrival)
    track_info_return.tuple_key_value_of_tpn_ID_departure = copy.deepcopy(track_info.tuple_key_value_of_tpn_ID_departure)

    return track_info_return


def start_end_profiler(filename=None, start=False, end=False, pr=None):
    if start:
        pr = cProfile.Profile()
        pr.enable()
        return pr

    if end:
        pr.disable()
        s = io.StringIO()

        sortby_selected = 'cumulative'  # 'totalTime'
        # ps = pstats.Stats(pr, stream=s).sort_stats('totalTime')
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby_selected)
        # ps = pstats.Stats(pr).sort_stats(sortby)
        ps.print_stats()
        with open('Profile_' + sortby_selected + filename + '.txt', 'w+') as f:
            f.write(s.getvalue())


def pickle_results(z_for_pickle, filename):
    with open(filename, 'wb') as f:
        pickle.dump(z_for_pickle, f)


def apply_operator_to_G(operator, G, changed_trains, cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters):
    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)
    # operator = 'EmergencyTrain'
    # operator = 'CancelFrom'
    # operator = 'Cancel'
    if operator == 'Cancel':
        # cancel random train
        changed_trains, G, track_info, edges_o_stations_d = operator_cancel(G, changed_trains, cut_trains_to_area,
                                                                track_info, edges_o_stations_d, parameters)

    elif operator == 'CancelFrom':
        changed_trains, G, train_id_operated, track_info, edges_o_stations_d = operator_cancel_from(G, changed_trains,
                                                    cut_trains_to_area, track_info, G_infra,  edges_o_stations_d, parameters)

    elif operator == 'Delay':
        # delay random train
        changed_trains, G, train_id_to_delay, track_info, edges_o_stations_d = operator_complete_delay(G, changed_trains,
                                                    cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters)

    elif operator == 'DelayFrom':
        changed_trains, G, train_id_to_delay, track_info, edges_o_stations_d = operator_part_delay(G, changed_trains,
                                                   cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters)

    elif operator == 'EmergencyTrain':
        emergency_train = nh.call_emergency_train_scen_low()
        changed_trains, G, train_id_to_delay, track_info, edges_o_stations_d = operator_emergency_train(G, changed_trains,
                                        emergency_train, cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters)

    elif operator == 'EmergencyBus':

        changed_trains, G, train_id_to_delay, track_info, edges_o_stations_d = operator_emergency_bus(G, changed_trains,
                                        cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters)

    elif operator == 'Return':
        changed_trains, G, train_id_to_delay, track_info, edges_o_stations_d = \
            operator_return_train_to_initial_timetable(G, changed_trains, cut_trains_to_area, track_info, G_infra,
                                                       edges_o_stations_d, parameters)

    return G, track_info, edges_o_stations_d, changed_trains, operator


def select_operator(probabilities, nr_usages, changed_trains, track_info, parameters):
    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)
    if len(track_info.trains_on_closed_tracks) > 0:
        print('trains running on closed track !!!')
    operator = None
    while operator is None:
        rd = np.random.uniform(0.0, 1.0)
        if rd <= probabilities.cc:  # complete cancel
            nr_usages.cc += 1
            operator = 'Cancel'

        elif probabilities.cc < rd <= probabilities.pc:  # partial cancel
            nr_usages.pc += 1
            operator = 'CancelFrom'

        elif probabilities.cc < rd <= probabilities.cd:  # delay
            nr_usages.cd += 1
            operator = 'Delay'

        elif probabilities.cd < rd <= probabilities.pd:  # partial delay
            nr_usages.pd += 1
            operator = 'DelayFrom'

        elif probabilities.pd < rd <= probabilities.et:  # emergency train
            nr_usages.et += 1
            operator = 'EmergencyTrain'

        elif probabilities.et < rd <= probabilities.eb:  # emergency bus
            nr_usages.eb += 1
            operator = 'EmergencyBus'

        elif rd > probabilities.eb:
            list_return_candids = candids_for_return_operator(changed_trains, parameters)
            if len(list_return_candids) <= 3:
                continue
            nr_usages.ret += 1
            operator = 'Return'

    return nr_usages, operator



def candids_for_return_operator(changed_trains, parameters):
    no_action = ['EmergencyTrain', 'EmergencyBus', 'Return', 'Reroute']
    list_return_candids = [trainID for trainID, attr in changed_trains.items() if attr['Action'] not in
                           no_action and trainID not in parameters.trains_on_closed_track_initial_timetable_infeasible
                           and trainID in parameters.train_ids_initial_timetable_infeasible]
    return list_return_candids



def operator_return_train_to_initial_timetable(G, changed_trains, cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters):
    list_return_candids = candids_for_return_operator(changed_trains, parameters)
    if len(list_return_candids) <= 3:
        train_id_to_return = None
        print('empty_changed_trains')
        return changed_trains, G, train_id_to_return, track_info, edges_o_stations_d
    try:
        train_id_to_return = list_return_candids[np.random.randint(0, len(list_return_candids)-1)]
    except ValueError:
        print('what went wrong?')

    action_to_revert = changed_trains[train_id_to_return]['Action']
    print(' trainID', train_id_to_return, changed_trains[train_id_to_return]['DebugString'], ' action to revert :', action_to_revert)
    if action_to_revert is not 'Cancel':
        # if not cancel, remove the informations of train still in cut trains
        idx_train_in_timetable_prime = 0
        while cut_trains_to_area[idx_train_in_timetable_prime]['ID'] != train_id_to_return:
            idx_train_in_timetable_prime += 1

        train_timetable_prime = cut_trains_to_area[idx_train_in_timetable_prime]  # ['TrainPathNodes']

        edges_o_stations_d = remove_edges_of_train_from_o_stations_d(edges_o_stations_d, train_timetable_prime, G)
        nodes_attr_train, edges_of_train = remove_nodes_edges_of_train(G, train_id_to_return)

        remove_entries_from_track_sequences(track_info, train_timetable_prime)
    else:
        train_timetable_prime = None

    # get the train from initial timetable
    try:
        i = 0
        while parameters.initial_timetable_feasible[i]['ID'] != train_id_to_return:
            i += 1
        train_initial_updated = copy.deepcopy(parameters.initial_timetable_feasible[i])
    except IndexError:
        print('Train not found')

    # update the train times
    train_before_update = copy.deepcopy(train_initial_updated)
    time_to_delay = int(0)
    # try:
    train_initial_updated = nh.update_train_times_feasible_path_delay_operator(train_initial_updated, time_to_delay, track_info,
                                                                                    G_infra, tpn_index_start=0)
    # except UnboundLocalError:
    #    print(changed_trains[train_id_to_return])
    #    print(' changed train (trainid to return) what happened here ?')
        # solve it with: train_initial_updated['Delay'] = 'infeasible'
        #  train_initial_updated['Delay'] = None

    train_update_feasible = False
    if train_initial_updated['Delay'] == 'feasible':
        train_update_feasible = True
        # remove the entries in the tpn_information and update the time of the delayed train
        remove_entries_in_tpn_informations_and_update_tpns_of_return_train(track_info, train_initial_updated,
                                                                           train_timetable_prime, action_to_revert)
    else:
        train_initial_updated = train_before_update

    if action_to_revert == 'Cancel':
        cut_trains_to_area.append(train_initial_updated)
    else:
        cut_trains_to_area[idx_train_in_timetable_prime] = train_initial_updated

    # create and add driving and waiting edges and nodes to the Graph
    nodes_edges_dict = graphcreator.create_transit_edges_nodes_single_train(train_initial_updated, G_infra, idx_start_delay=0)

    add_transit_nodes_edges_single_train_to_Graph(G, nodes_edges_dict, bus=False)
    # create and add transfer edges to the Graph
    tpns = [tpn_id['ID'] for tpn_id in train_initial_updated['TrainPathNodes']]
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train = graphcreator.transfer_edges_single_train\
                                               (G, train_initial_updated, parameters.transfer_M, parameters.transfer_m,
                                                0, tpns)
    G.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(G, transfer_edges_attribute)
    # update the list of edges from origin to destination

    edges_o_stations_d = add_edges_of_train_from_o_stations_d(edges_o_stations_d, train_initial_updated, G, parameters, 0, tpns)

    # update the changed trains method
    if train_update_feasible:
        changed_trains[train_id_to_return] = {'train_id': train_id_to_return, 'DebugString': train_initial_updated['DebugString'],
                                             'Action': 'Return', 'body_message': train_initial_updated['body_message']}

    print(action_to_revert)
    return changed_trains, G, train_id_to_return, track_info, edges_o_stations_d


def operator_cancel(G, changed_trains, cut_trains_to_area, track_info, edges_o_stations_d, parameters):
    rd_train_idx = np.random.randint(0, len(cut_trains_to_area))
    train_to_cancel = cut_trains_to_area[rd_train_idx]
    train_id_to_cancel = train_to_cancel['ID']
    emergency_train = False
    bus = False
    if 'EmergencyTrain' in train_to_cancel.keys():
        emergency_train = True
    if 'EmergencyBus' in train_to_cancel.keys():
         bus = True

    edges_o_stations_d = remove_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_cancel, G)
    nodes_attr_train, edges_of_train = remove_nodes_edges_of_train(G, train_id_to_cancel)
    if not bus:
        # remove the entries in the tpn_informatioN
        remove_entries_from_track_sequences(track_info, train_to_cancel)
        remove_entries_in_tpn_informations_canceled_train(track_info, train_to_cancel, idx_start_cancel=0)

    if bus:
        # print('debug cancel Bus !!')
        changed_trains[train_id_to_cancel] = {'train_id': train_id_to_cancel,
                                           'DebugString': cut_trains_to_area[rd_train_idx]['DebugString'],
                                           'Action': 'Cancel', 'EmergencyBus': True}


    elif not emergency_train and not bus:
        changed_trains[train_id_to_cancel] = {'train_id': train_id_to_cancel,
                                              'DebugString': cut_trains_to_area[rd_train_idx]['DebugString'],
                                              'Action': 'Cancel'}
    elif emergency_train:
        changed_trains[train_id_to_cancel] = {'train_id': train_id_to_cancel,
                                              'DebugString': cut_trains_to_area[rd_train_idx]['DebugString'],
                                              'Action': 'Cancel', 'EmergencyTrain': True}

    del cut_trains_to_area[rd_train_idx]

    return changed_trains, G, track_info, edges_o_stations_d


def operator_cancel_from(G, changed_trains, cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters):
    train_found = False
    list_cancel_candids = list(set(parameters.set_of_trains_for_operator['Cancel']))
    if len(list_cancel_candids) >= 3:
        n_it = 0
        while not train_found and n_it <= 5:
            n_it += 1
            train_id_to_cancelFrom = list_cancel_candids[np.random.randint(0, len(list_cancel_candids))]
            i = 0
            try:
                while cut_trains_to_area[i]['ID'] != train_id_to_cancelFrom:
                    i += 1
                train_to_cancelFrom = cut_trains_to_area[i]
            except IndexError:
                continue
            comm_stops = 0
            for tpn in train_to_cancelFrom['TrainPathNodes']:
                if tpn['StopStatus'] == 'commercialStop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True
    if not train_found:
        # select randomly a train from all trains
        while not train_found:
            train_id_to_cancelFrom = cut_trains_to_area[np.random.randint(0, len(cut_trains_to_area))]['ID']
            i = 0
            while cut_trains_to_area[i]['ID'] != train_id_to_cancelFrom:
                i += 1
            train_to_cancelFrom = cut_trains_to_area[i]
            comm_stops = 0
            for tpn in train_to_cancelFrom['TrainPathNodes']:
                if tpn['StopStatus'] == 'commercialStop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True

    emergency_train = False
    if 'EmergencyTrain' in train_to_cancelFrom.keys():
        emergency_train = True

    # cancel from last comm stop
    idx_tpn_cancel_from = identify_last_departure_tpnID_of_train_to_cancelFrom(train_to_cancelFrom)
    tpn_cancel_from = train_to_cancelFrom['TrainPathNodes'][idx_tpn_cancel_from]

    edges_o_stations_d = remove_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_cancelFrom, G)
    nodes_attr_train, edges_of_train = remove_nodes_edges_of_train(G, train_id_to_cancelFrom)

    remove_entries_from_track_sequences(track_info, train_to_cancelFrom)
    remove_entries_in_tpn_informations_canceled_train(track_info, train_to_cancelFrom, idx_tpn_cancel_from)

    tpns_cancel = [tpn_id['ID'] for tpn_id in train_to_cancelFrom['TrainPathNodes']]

    # create and add driving and waiting edges and nodes to the Graph
    nodes_edges_dict = graphcreator.create_transit_edges_nodes_single_train(train_to_cancelFrom, G_infra, 0)
    add_transit_nodes_edges_single_train_to_Graph(G, nodes_edges_dict, bus=False)
    # create and add transfer edges to the Graph
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train = graphcreator.transfer_edges_single_train \
        (G, train_to_cancelFrom, parameters.transfer_M, parameters.transfer_m, 0, tpns_cancel)
    G.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(G, transfer_edges_attribute)

    # update the list of edges from origin to destination
    edges_o_stations_d = add_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_cancelFrom, G, parameters, 0,
                                                              tpns_cancel)

    # remove the entries in the tpn_information and update the time of the delayed train
    if not emergency_train:
        changed_trains[train_id_to_cancelFrom] = {'train_id': train_id_to_cancelFrom, 'DebugString': train_to_cancelFrom['DebugString'],
                                              'Action': 'CancelFrom', 'CancelFrom': train_to_cancelFrom['ID'],
                                              'tpn_cancel_from': tpn_cancel_from['ID'],
                                              'body_message': train_id_to_cancelFrom}
    else:
        changed_trains[train_id_to_cancelFrom] = {'train_id': train_id_to_cancelFrom, 'DebugString': train_to_cancelFrom['DebugString'],
                                                  'Action': 'CancelFrom', 'CancelFrom': train_to_cancelFrom['ID'],
                                                  'tpn_cancel_from': tpn_cancel_from['ID'], 'body_message': train_id_to_cancelFrom,
                                                  'EmergencyTrain': True}

    return changed_trains, G, train_id_to_cancelFrom, track_info, edges_o_stations_d


def operator_complete_delay(G, changed_trains, cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters):
    list_delay_candids = list(set(parameters.set_of_trains_for_operator['Delay']))
    train_id_to_delay = list_delay_candids[np.random.randint(0, len(list_delay_candids)-1)]
    i = 0
    while cut_trains_to_area[i]['ID'] != train_id_to_delay:
        i += 1

    train_to_delay = cut_trains_to_area[i]  # ['TrainPathNodes']
    tpns = [tpn_id['ID'] for tpn_id in train_to_delay['TrainPathNodes']]

    bus = False
    emergency_train = False
    if 'EmergencyBus' in train_to_delay.keys():
        bus = True
    elif 'EmergencyTrain' in train_to_delay.keys():
        emergency_train = True

    edges_o_stations_d = remove_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_delay, G)

    nodes_attr_train, edges_of_train = remove_nodes_edges_of_train(G, train_id_to_delay)

    delay_options = [5, 10, 15, 20]
    time_to_delay = delay_options[np.random.randint(0, len(delay_options))]

    # Remove entries of train in track_sequences
    if not bus:
        remove_entries_from_track_sequences(track_info, train_to_delay)

        # update the train times
        train_before_update = train_to_delay.copy()
        train_to_delay = nh.update_train_times_feasible_path_delay_operator(train_to_delay, time_to_delay, track_info,
                                                                        G_infra, tpn_index_start=0)
        train_update_feasible = False
        if train_to_delay['Delay'] == 'feasible':
            train_update_feasible = True
            # remove the entries in the tpn_information and update the time of the delayed train
            remove_entries_in_tpn_informations_and_update_tpns_of_delayed_train(track_info, train_to_delay, idx_start_delay=0)
        else:
            train_to_delay = train_before_update

        # create and add driving and waiting edges and nodes to the Graph
        nodes_edges_dict = graphcreator.create_transit_edges_nodes_single_train(train_to_delay, G_infra, idx_start_delay=0)
    else:
        # its a bus
        train_update_feasible = True
        train_to_delay = delay_tpns_of_bus(train_to_delay, time_to_delay)
        nodes_edges_dict = graphcreator.create_transit_edges_nodes_emergency_bus(train_to_delay)


    add_transit_nodes_edges_single_train_to_Graph(G, nodes_edges_dict, bus)
    # create and add transfer edges to the Graph
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train = graphcreator.transfer_edges_single_train\
                                               (G, train_to_delay, parameters.transfer_M, parameters.transfer_m,
                                                0, tpns)
    G.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(G, transfer_edges_attribute)
    # update the list of edges from origin to destination
    if not bus:
        edges_o_stations_d = add_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_delay, G, parameters, 0, tpns)
    else:
        edges_o_stations_d = add_edges_of_bus_from_o_stations_d(edges_o_stations_d, train_to_delay, G, parameters, 0, tpns)

    # update the changed trains method
    if not bus and not emergency_train and train_update_feasible:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay, 'DebugString': train_to_delay['DebugString'],
                                             'Action': 'Delay', 'body_message': train_to_delay['body_message']}
    elif bus:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay, 'DebugString': train_to_delay['DebugString'],
                                             'Action': 'Delay', 'EmergencyBus': True}
    elif emergency_train:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay, 'DebugString': train_to_delay['DebugString'],
                                             'Action': 'Delay', 'body_message': train_to_delay['body_message'],
                                             'EmergencyTrain': True}
    elif not train_update_feasible:
        pass

    return changed_trains, G, train_id_to_delay, track_info, edges_o_stations_d


def operator_part_delay(G, changed_trains, cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters):
    train_found = False
    # rd_train_idx = np.random.randint(0, len(cut_trains_to_area))
    # train_to_delay = cut_trains_to_area[rd_train_idx]  # ['TrainPathNodes']
    # train_id_to_delay = train_to_delay['ID']
    # identify the node to delay after...
    # tpn_idx_start_delay = np.random.randint(0, len(train_to_delay['TrainPathNodes'])-1)

    list_delay_candids = list(set(parameters.set_of_trains_for_operator['Delay']))
    if len(list_delay_candids) >= 3:
        n_it = 0
        while not train_found and n_it <= 3:
            n_it += 1
            train_id_to_delay = list_delay_candids[np.random.randint(0, len(list_delay_candids))]
            i = 0
            try:
                while cut_trains_to_area[i]['ID'] != train_id_to_delay:
                    i += 1
                train_to_delay = cut_trains_to_area[i]
            except IndexError:
                continue
            if 'EmergencyBus' in train_to_delay.keys():
                continue
            comm_stops = 0
            for tpn in train_to_delay['TrainPathNodes']:
                if tpn['StopStatus'] == 'commercialStop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True

    if not train_found:
        while not train_found:
            idx = np.random.randint(0, len(cut_trains_to_area))
            train_to_delay = cut_trains_to_area[idx]
            train_id_to_delay = train_to_delay['ID']
            if 'EmergencyBus' in train_to_delay.keys():
                continue
            comm_stops = 0
            for tpn in train_to_delay['TrainPathNodes']:
                if tpn['StopStatus'] == 'commercialStop':
                    comm_stops += 1
                if comm_stops > 3:
                    train_found = True

    # cancel from last comm stop
    idx_tpn_delay_from = identify_departure_tpnID_of_train_to_delayFrom(train_to_delay)
    tpn_cancel_from = train_to_delay['TrainPathNodes'][idx_tpn_delay_from]

    emergency_train = False
    if 'EmergencyTrain' in train_to_delay.keys():
        emergency_train = True

    tpns_delay = [tpn_id['ID'] for tpn_id in train_to_delay['TrainPathNodes'][idx_tpn_delay_from:]]

    edges_o_stations_d = remove_edges_of_part_delayed_train_from_o_stations_d(edges_o_stations_d, train_to_delay, G, tpns_delay)

    nodes_attr_train, edges_of_train = remove_nodes_edges_of_part_of_train(G, train_id_to_delay, tpns_delay)

    delay_options = [5, 10, 15, 20]
    time_to_delay = delay_options[np.random.randint(0, len(delay_options))]
    # Remove entries of train in track_sequences
    try:
        remove_entries_part_of_train_from_track_sequences(track_info, train_to_delay['TrainPathNodes'][idx_tpn_delay_from:])
    except ValueError:
        print('something went wrong')
        pass
    # update the train times
    train_before_update = train_to_delay.copy()
    train_to_delay = nh.update_train_times_feasible_path_delay_operator(train_to_delay, time_to_delay, track_info,
                                                                        G_infra, idx_tpn_delay_from)
    train_update_feasible = False
    if train_to_delay['Delay'] == 'feasible':
        train_update_feasible = True
        # remove the entries in the tpn_information and update the time of the delayed train
        remove_entries_in_tpn_informations_and_update_tpns_of_delayed_train(track_info, train_to_delay, idx_tpn_delay_from)
    else:
        train_to_delay = train_before_update

    # create and add driving and waiting edges and nodes to the Graph
    nodes_edges_dict = graphcreator.create_transit_edges_nodes_single_train(train_to_delay, G_infra, idx_tpn_delay_from)
    add_transit_nodes_edges_single_train_to_Graph(G, nodes_edges_dict, bus=False)
    # create and add transfer edges to the Graph
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train = graphcreator.transfer_edges_single_train\
                                               (G, train_to_delay, parameters.transfer_M, parameters.transfer_m,
                                                idx_tpn_delay_from, tpns_delay)
    G.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(G, transfer_edges_attribute)
    # update the list of edges from origin to destination
    edges_o_stations_d = add_edges_of_train_from_o_stations_d(edges_o_stations_d, train_to_delay, G, parameters,
                                                              idx_tpn_delay_from, tpns_delay)
    # update the changed trains method
    if not emergency_train and train_update_feasible:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay, 'DebugString':train_to_delay['DebugString'],
                                         'Action': 'Delay', 'body_message': train_to_delay['body_message']}
    elif train_update_feasible:
        changed_trains[train_id_to_delay] = {'train_id': train_id_to_delay, 'DebugString': train_to_delay['DebugString'],
                                             'Action': 'Delay', 'body_message': train_to_delay['body_message'],
                                           'EmergencyTrain': True}
    elif not train_update_feasible:
        pass

    return changed_trains, G, train_id_to_delay, track_info, edges_o_stations_d


def operator_emergency_train(G, changed_trains, emergency_train, cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters):
    train_id_et = emergency_train['ID']
    emergency_train['EmergencyTrain'] = True

    timeWindow_fromTime = parameters.time_window['datetime']['FromTime']
    timeWindow_toTime = parameters.time_window['datetime']['ToTime']
    timeWindow_duration_min = round(parameters.time_window['duration'].seconds / 60, 0) - 30
    # template train starts at 06:00, so just delay the start of the train by random time delay
    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)
    dep_time_delay = np.random.randint(0, timeWindow_duration_min)

    tpns = [tpn_id['ID'] for tpn_id in emergency_train['TrainPathNodes']]

    # update the train times
    emergency_train = nh.update_train_times_feasible_path_delay_operator(emergency_train, dep_time_delay, track_info,
                                                                        G_infra, tpn_index_start=0)
    if emergency_train['Delay'] == 'infeasible':
        return changed_trains, G, train_id_et, track_info, edges_o_stations_d

    # add the entries to the tpn_information and update the time of the delayed train
    add_entries_to_tpn_informations_and_update_tpns_of_demergency_train(track_info, emergency_train, idx_start_delay=0)

    # create and add driving and waiting edges and nodes to the Graph
    nodes_edges_dict = graphcreator.create_transit_edges_nodes_single_train(emergency_train, G_infra, idx_start_delay=0)
    add_transit_nodes_edges_single_train_to_Graph(G, nodes_edges_dict, bus=False)
    # create and add transfer edges to the Graph
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train = graphcreator.transfer_edges_single_train\
                                               (G, emergency_train, parameters.transfer_M, parameters.transfer_m,
                                                0, tpns)
    G.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(G, transfer_edges_attribute)
    # update the list of edges from origin to destination
    edges_o_stations_d = add_edges_of_train_from_o_stations_d(edges_o_stations_d, emergency_train, G, parameters, 0,
                                                              tpns)
    # update the changed trains method
    changed_trains[train_id_et] = {'train_id': train_id_et, 'DebugString': emergency_train['DebugString'], 'Action': 'EmergencyTrain',
                                   'body_message': emergency_train['body_message'], 'EmergencyTrain': True}
    # l, p = nx.multi_source_dijkstra(G, [(207, '2005-05-10T06:39:12', 7981154, 'a')], nodes_edges_dict['arrival_nodes'][1])
    cut_trains_to_area.append(emergency_train)
    # if False:
    #    for train in cut_trains_to_area:
    #        if train['DebugString'] == 'RVZH_9999_1_J05 tt_(S)':
    #            print(train['ID'])

    return changed_trains, G, train_id_et, track_info, edges_o_stations_d


def operator_emergency_bus(G, changed_trains, cut_trains_to_area, track_info, G_infra, edges_o_stations_d, parameters):
    bus_id_nr = 90000
    bus_id = 'Bus' + str(bus_id_nr)
    while bus_id in changed_trains.keys():
        bus_id_nr += 10
        bus_id = 'Bus' + str(bus_id_nr)

    timeWindow_fromTime = parameters.time_window['datetime']['FromTime']
    timeWindow_toTime = parameters.time_window['datetime']['ToTime']
    timeWindow_duration_min = round(parameters.time_window['duration'].seconds / 60, 0)
    # template train starts at 06:00, so just delay the start of the train by random time delay
    add_time_bus = np.random.randint(0, timeWindow_duration_min - 10)
    departure_time_bus = timeWindow_fromTime + datetime.timedelta(minutes=add_time_bus)

    emergency_bus = bus_add_bus_path_nodes_scenLow(bus_id, departure_time_bus)

    tpns_bus = [tpn_id['ID'] for tpn_id in emergency_bus['TrainPathNodes']]

    # create and add driving and waiting edges and nodes to the Graph
    nodes_edges_dict = graphcreator.create_transit_edges_nodes_emergency_bus(emergency_bus)
    add_transit_nodes_edges_single_train_to_Graph(G, nodes_edges_dict, bus=True)
    # create and add transfer edges to the Graph
    transfer_edges, transfer_edges_attribute, arrival_departure_nodes_train = graphcreator.transfer_edges_single_bus\
                                               (G, emergency_bus, parameters.transfer_MBus, parameters.transfer_mBus,
                                                0, tpns_bus)
    G.add_weighted_edges_from(transfer_edges)
    nx.set_edge_attributes(G, transfer_edges_attribute)
    # update the list of edges from origin to destination
    edges_o_stations_d = add_edges_of_bus_from_o_stations_d(edges_o_stations_d, emergency_bus, G, parameters, 0,
                                                              tpns_bus)

    # update the changed trains method
    changed_trains[bus_id] = {'train_id': bus_id, 'DebugString': emergency_bus['DebugString'],
                              'Action': 'EmergencyBus', 'body_message': None, 'EmergencyTrain': True}
    # l, p = nx.multi_source_dijkstra(G, [(199, '2005-05-10T06:20:00', 'Bus900001', 'd')], (543, '2005-05-10T06:30:00', 'Bus900002', 'a'))
    cut_trains_to_area.append(emergency_bus)

    return changed_trains, G, bus_id, track_info, edges_o_stations_d


def bus_add_bus_path_nodes_scenLow(busID, departure_time):
    bus = {}
    departure_time_str = datetime.datetime.strftime(departure_time, "%Y-%m-%dT%H:%M:%S")
    arrival_time = departure_time + datetime.timedelta(minutes=10)
    arrival_time_str = datetime.datetime.strftime(arrival_time, "%Y-%m-%dT%H:%M:%S")
    # Walisellen = 543
    # Dietlikon = 199
    bus['ID'] = busID
    bus['EmergencyBus'] = True
    if np.random.uniform(0.0, 1.0) < 0.5:
        start = 543
        end = 199
    else:
        start = 199
        end = 543
    bus['TrainPathNodes'] = [{
            "ID": busID + str(1),
            "SectionTrackID": None,
            "IsSectionTrackAscending": None,
            "NodeID": start,
            "NodeTrackID": None,
            "ArrivalTime": departure_time_str,
            "DepartureTime": departure_time_str,
            "MinimumRunTime": "P0D",
            "MinimumStopTime": "P0D",
            "StopStatus": "commercialStop",
            "SequenceNumber": 0
        },
        {
            "ID": busID + str(2),
            "SectionTrackID": None,
            "IsSectionTrackAscending": None,
            "NodeID": end,
            "NodeTrackID": None,
            "ArrivalTime": arrival_time_str,
            "DepartureTime": arrival_time_str,
            "MinimumRunTime": "PT10M",
            "MinimumStopTime": "P0D",
            "StopStatus": "commercialStop",
            "SequenceNumber": 1
        }]
    bus['DebugString'] = "EmergencyBus"
    return bus


def delay_tpns_of_bus(train_to_delay, time_to_delay):
    dep_time_start = datetime.datetime.strptime(train_to_delay['TrainPathNodes'][0]['DepartureTime'], "%Y-%m-%dT%H:%M:%S")
    dep_time_start += datetime.timedelta(minutes=time_to_delay)

    dep_time_str = datetime.datetime.strftime(dep_time_start, "%Y-%m-%dT%H:%M:%S")
    arrival_time = dep_time_start + datetime.timedelta(minutes=10)
    arrival_time_str = datetime.datetime.strftime(arrival_time, "%Y-%m-%dT%H:%M:%S")

    train_to_delay['TrainPathNodes'][0]['ArrivalTime'] = dep_time_str
    train_to_delay['TrainPathNodes'][0]['DepartureTime'] = dep_time_str

    train_to_delay['TrainPathNodes'][1]['ArrivalTime'] = arrival_time_str
    train_to_delay['TrainPathNodes'][1]['DepartureTime'] = arrival_time_str

    return train_to_delay


def create_graph_with_edges_o_stations_d(edges_o_stations_d, od=None, G=None):

    if G is None:
        G = nx.DiGraph()

    edges_o_stations = edges_o_stations_d.edges_o_stations
    edges_stations_d = edges_o_stations_d.edges_stations_d
    edges_o_stations_dict = edges_o_stations_d.edges_o_stations_dict  # key origin, value edges connecting to train nodes
    edges_stations_d_dict = edges_o_stations_d.edges_stations_d_dict

    if od is None:
        G.add_weighted_edges_from(edges_o_stations)
        G.add_weighted_edges_from(edges_stations_d)
    #    for origin, edges_attr in edges_o_stations_dict.items():
    #        G.add_weighted_edges_from(edges_attr)
    else:
        origin = od[0]
        dest = od[1]
        G.add_weighted_edges_from(edges_o_stations_dict[origin])
        G.add_weighted_edges_from(edges_stations_d_dict[dest])

    return G


def find_path_and_assign_pass_topo(G, parameters):
    tic = time.time()
    adj_dict, adj_list_topo_order, vertices_topo_order = sp_topo.generate_topological_order_adj_list(G)
    source_targets_dict = utils.transform_odt_into_dict_key_source(parameters.odt_as_list)
    dist = {node: float('Inf') for node in vertices_topo_order}
    path = {target: list() for target in vertices_topo_order}
    for source, targets in source_targets_dict.items():
        # target = source_target[1]
        len_topo, path_topo = sp_topo.sp_topological_order(G, source, adj_list_topo_order, dist, path,
                                                           vertices_topo_order)
    print('topo uses : [min]', str((time.time() - tic) / 60))


def find_path_and_assign_pass_topo_sparse_graph(G_prime, parameters, edges_o_stations_d):
    tic = time.time()

    # G_edges = create_graph_with_edges_o_stations_d(edges_o_stations_d)
    # G_fullGraph = create_graph_with_edges_o_stations_d(edges_o_stations_d, G=G_prime.copy())
    # G_ods = create_graph_with_edges_o_stations_d(edges_o_stations_d, G=None)

    # adj_dict_ods, adj_list_topo_order_ods, vertices_topo_order_ods = sp_topo.generate_topological_order_adj_list(G_ods)
    # adj_dict_ods, adj_list_topo_order_ods, vertices_topo_order_ods = sp_topo.generate_topological_order_adj_list(
    #     G_fullGraph)

    adj_dict, adj_list_topo_order, vertices_topo_order = sp_topo.generate_topological_order_adj_list(G_prime)

    # todo Explanation of procedure: destination to the end and sources to the beginning of vert topo order
    #   add the successors of the sources to the adj list in the same order in the beginning
    #   get the index of the successor of target in the adj list and append [target, {'weight': 2.0}]

    dist = {node: float('Inf') for node in vertices_topo_order}
    path = {target: list() for target in vertices_topo_order}
    edges_o_s = edges_o_stations_d.edges_o_stations_dict
    edges_s_d = edges_o_stations_d.edges_stations_d_dict
    served_unserved_passenger = [0, 0]

    for od in parameters.odt_as_list:
        # vertices_topo_order_od = vertices_topo_order.copy()
        adj_list_topo_order_od = adj_list_topo_order.copy()

        source = od[0]
        target = od[1]
        groupsize = od[3]
        dist[source] = float('inf')
        dist[target] = float('inf')
        path[source] = []
        path[target] = []

        # add source to top
        vertices_topo_order.insert(0, source)
        adj_sources = []
        for i in range(0, len(edges_o_s[source])):
            successor = edges_o_s[source][i][1]
            weight = {'weight': edges_o_s[source][i][2]}
            adj_sources.append([successor, weight])
        # add adj of source to top
        adj_list_topo_order_od.insert(0, adj_sources)

        # add target as ajd to list of predecessor
        idx_adj_list_successor = {}  # key index predecessor, value adj_list_to_target
        for edge in edges_s_d[target]:
            predecessor = edge[0]
            idx_predecessor_vertices = vertices_topo_order.index(predecessor)
            weight = {'weight': edge[2]}
            adj_pred = [target, weight]
            idx_adj_list_successor[idx_predecessor_vertices] = adj_pred
            adj_list_topo_order_od[idx_predecessor_vertices].append(adj_pred)

        # add target to the end
        vertices_topo_order.append(target)
        adj_list_topo_order_od.append([])


        len_topo, path_topo = sp_topo.sp_topological_order(G_prime, source, adj_list_topo_order_od, dist, path,
                                                           vertices_topo_order, target)

        # l, p = nx.single_source_dijkstra(G_fullGraph, source, target)

        if len_topo is not None:
            path_topo.insert(0, source)
            served_unserved_passenger[0] += groupsize
        else:
            served_unserved_passenger[1] += groupsize
        # remove target
        del vertices_topo_order[-1]
        # del adj_list_topo_order[-1]

        # remove adj from predecessor to target
        # for idx, adj_pred in idx_adj_list_successor.items():
        #    adj_list_topo_order[idx].remove(adj_pred)


        # remove source and its adj list
        del vertices_topo_order[0]
        # del adj_list_topo_order[0]
        del dist[source]
        del path[source]

    print('topo uses : [min]', str((time.time() - tic) / 60))
    print('served passengers : ', served_unserved_passenger[0], '  unserved passengers : ', served_unserved_passenger[1])


def find_path_and_assign_pass(G_prime, G, parameters, solutions, edges_o_stations_d):
    cutoff = parameters.time_window['duration'].seconds/60
    sp_by_source = False
    sp_by_odt = False
    sp_by_scipy = True
    if sp_by_source:
        G_prime, length, path, served_unserved_passengers, total_traveltime = sp.find_sp_for_all_sources_sparse_graph\
                                    (G_prime, G, parameters, edges_o_stations_d, cutoff=cutoff, assign_passenger=True)
    if sp_by_odt:
        G_prime, served_unserved_passengers, total_traveltime = sp.find_sp_for_sources_targets_graph(
            G_prime, G, parameters, edges_o_stations_d, cutoff=cutoff, assign_passenger=True)

    # this one was used
    if sp_by_scipy:
        G_prime, served_unserved_passengers, total_traveltime = sp.find_sp_for_all_ods_full_graph_scipy(G_prime, G,
                                                    parameters, edges_o_stations_d, cutoff=None, assign_passenger=True)

    solutions.total_traveltime = round(total_traveltime, 1)

    print_out = True
    if print_out:
        print(' passengers with path : ', served_unserved_passengers[0], ', passengers without path : ',
              served_unserved_passengers[1])

    return solutions, G_prime


def get_solution_G_with_copy_solution_prime(solution_prime, parameters):
    solution_G = Classes.Solutions()
    solution_G.set_of_trains_for_operator = solution_prime.set_of_trains_for_operator.copy()
    solution_G.time_table = copy.deepcopy(solution_prime.time_table)
    solution_G.graph = copy_graph_with_flow(solution_prime.graph, parameters)
    solution_G.total_dist_train = solution_prime.total_dist_train
    solution_G.total_traveltime = solution_prime.total_traveltime
    solution_G.deviation_timetable = solution_prime.deviation_timetable
    solution_G.changed_trains = copy.deepcopy(solution_prime.changed_trains)
    solution_G.edges_o_stations_d = Classes.Copy_edges_origin_station_destionation(solution_prime.edges_o_stations_d)
    solution_G.set_of_trains_for_operator['Cancel'] = parameters.set_of_trains_for_operator['Cancel'].copy()
    solution_G.set_of_trains_for_operator['Delay'] = parameters.set_of_trains_for_operator['Delay'].copy()

    # solution_G.track_info = copy_track_info_from_track_info(solution_prime.track_info)
    return solution_G


def copy_graph_with_flow(G, parameters = None):
    G_prime = nx.DiGraph()
    edges_G = []
    attr_edges_G = {}

    nodes = {n: v for n, v in G.node(data=True)}
    for u, v, attr in G.edges(data=True):
        edges_G.append((u, v, {'weight': attr['weight']}))
        attr_edges_G[(u, v)] = attr

    # edges_G = [(u, v, attr) for u, v, attr in G.edges(data=True)]

    del G

    G_prime.add_nodes_from(nodes.keys())
    nx.set_node_attributes(G_prime, nodes)
    G_prime.add_weighted_edges_from(edges_G)
    nx.set_edge_attributes(G_prime, attr_edges_G)
    # edges_G = [(u, v, attr) for u, v, attr in G_prime.edges(data=True)]

    # G_prime = remove_flow_on_graph(G_prime, parameters)

    debug = False
    if debug:
        for u, v, attr in edges_G:
            if 'weight' not in attr.keys():
                print(u, v, attr)

    return G_prime


def copy_graph_and_remove_flow(G, parameters):

    G_prime = nx.DiGraph()
    edges_G = []
    attr_edges_G = {}
    nodes = {n: v for n, v in G.node(data=True)}
    for u, v, attr in G.edges(data=True):
        edges_G.append((u, v, {'weight': attr['weight']}))
        attr_copied = attr.copy()
        if 'flow' in attr_copied.keys():
            attr_copied['flow'] = 0
        attr_edges_G[(u, v)] = attr_copied

    # edges_G = [(u, v, attr) for u, v, attr in G.edges(data=True)]

    del G

    G_prime.add_nodes_from(nodes.keys())
    nx.set_node_attributes(G_prime, nodes)
    G_prime.add_weighted_edges_from(edges_G)
    nx.set_edge_attributes(G_prime, attr_edges_G)
    # edges_G = [(u, v, attr) for u, v, attr in G_prime.edges(data=True)]

    # G_prime = remove_flow_on_graph(G_prime, parameters)

    debug = False
    if debug:
        for u, v, attr in edges_G:
            if 'weight' not in attr.keys():
                print(u, v, attr)

    return G_prime


def remove_flow_on_graph(G, parameters):
    # remove flow of p on graph ['type'] in ['driving', 'waiting']
    for (u, v, attr) in G.edges.data():
        if 'flow' in attr.keys():
            if attr['flow'] != 0:
                attr['flow'] = 0
                if 'initial_weight' in attr.keys():
                    attr['weight'] = attr['initial_weight']
    return G


def add_entries_to_tpn_informations_and_update_tpns_of_demergency_train(track_info, train_to_delay, idx_start_delay):
    for tpn in train_to_delay['TrainPathNodes'][idx_start_delay:]:
        # tuple_key_value_arr = track_info.tuple_key_value_of_tpn_ID_arrival.pop(tpn['ID'])
        # tuple_key_value_dep = track_info.tuple_key_value_of_tpn_ID_departure.pop(tpn['ID'])
        # tpn_info = track_info.tpn_information.pop(tpn['ID'])
        tpn = update_delayed_tpn(train_to_delay['runtime_delay_feasible'][tpn['ID']], tpn)

    # Update track info
    pio.used_tracks_single_train(track_info.trains_on_closed_tracks, track_info.nr_usage_tracks,
                                 track_info.tpn_information,
                                 track_info.track_sequences_of_TPN, train_to_delay, track_info.trains_on_closed_tracks,
                                 track_info.tuple_key_value_of_tpn_ID_arrival,
                                 track_info.tuple_key_value_of_tpn_ID_departure, idx_start_delay)


def remove_entries_in_tpn_informations_and_update_tpns_of_return_train(track_info, train_initial_updated, train_timetable_prime, action_to_revert):
    if action_to_revert == 'Cancel':
        for tpn in train_initial_updated['TrainPathNodes']:
            tpn = update_delayed_tpn(train_initial_updated['runtime_delay_feasible'][tpn['ID']], tpn)

    else:
        train_timetable_prime_dict = utils.build_dict(train_timetable_prime['TrainPathNodes'], 'ID')
        for tpn in train_initial_updated['TrainPathNodes']:
            if tpn['ID'] in train_timetable_prime_dict.keys():
                tuple_key_value_arr = track_info.tuple_key_value_of_tpn_ID_arrival.pop(tpn['ID'])
                tuple_key_value_dep = track_info.tuple_key_value_of_tpn_ID_departure.pop(tpn['ID'])
                tpn_info = track_info.tpn_information.pop(tpn['ID'])

            tpn = update_delayed_tpn(train_initial_updated['runtime_delay_feasible'][tpn['ID']], tpn)

    # Update track info
    pio.used_tracks_single_train(track_info.trains_on_closed_tracks, track_info.nr_usage_tracks,
                                 track_info.tpn_information,
                                 track_info.track_sequences_of_TPN, train_initial_updated, track_info.trains_on_closed_tracks,
                                 track_info.tuple_key_value_of_tpn_ID_arrival,
                                 track_info.tuple_key_value_of_tpn_ID_departure, 0)



def remove_entries_in_tpn_informations_and_update_tpns_of_delayed_train(track_info, train_to_delay, idx_start_delay):
    for tpn in train_to_delay['TrainPathNodes'][idx_start_delay:]:
        tuple_key_value_arr = track_info.tuple_key_value_of_tpn_ID_arrival.pop(tpn['ID'])
        tuple_key_value_dep = track_info.tuple_key_value_of_tpn_ID_departure.pop(tpn['ID'])
        tpn_info = track_info.tpn_information.pop(tpn['ID'])
        tpn = update_delayed_tpn(train_to_delay['runtime_delay_feasible'][tpn['ID']], tpn)

    # Update track info
    pio.used_tracks_single_train(track_info.trains_on_closed_tracks, track_info.nr_usage_tracks,
                                 track_info.tpn_information,
                                 track_info.track_sequences_of_TPN, train_to_delay, track_info.trains_on_closed_tracks,
                                 track_info.tuple_key_value_of_tpn_ID_arrival,
                                 track_info.tuple_key_value_of_tpn_ID_departure, idx_start_delay)


def remove_entries_in_tpn_informations_canceled_train(track_info, train_to_cancel, idx_start_cancel):
    for tpn in train_to_cancel['TrainPathNodes']:
        tuple_key_value_arr = track_info.tuple_key_value_of_tpn_ID_arrival.pop(tpn['ID'])
        tuple_key_value_dep = track_info.tuple_key_value_of_tpn_ID_departure.pop(tpn['ID'])
        tpn_info = track_info.tpn_information.pop(tpn['ID'])

    # Update track info
    if idx_start_cancel == 0:
        train_to_cancel['TrainPathNodes'] = []
    else:
        train_to_cancel['TrainPathNodes'] = train_to_cancel['TrainPathNodes'][:idx_start_cancel+1]

        pio.used_tracks_single_train(track_info.trains_on_closed_tracks, track_info.nr_usage_tracks,
                                     track_info.tpn_information, track_info.track_sequences_of_TPN,
                                     train_to_cancel, track_info.trains_on_closed_tracks,
                                     track_info.tuple_key_value_of_tpn_ID_arrival,
                                     track_info.tuple_key_value_of_tpn_ID_departure, 0)


def remove_entries_from_track_sequences(track_info, train_to_delay):
    for tpn in train_to_delay['TrainPathNodes']:
        try:
            tuple_key_arr, value_arr = track_info.tuple_key_value_of_tpn_ID_arrival[tpn['ID']]
            tuple_key_dep, value_dep = track_info.tuple_key_value_of_tpn_ID_departure[tpn['ID']]
            track_info.track_sequences_of_TPN[tuple_key_arr].remove(value_arr)
            track_info.track_sequences_of_TPN[tuple_key_dep].remove(value_dep)
        except ValueError:
            # whole train is not in list....
            print(train_to_delay)
            print(tuple_key_arr)
            print(tuple_key_dep)
            continue


def remove_entries_part_of_train_from_track_sequences(track_info, tpns_train_part):
    for tpn in tpns_train_part:
        tuple_key_arr, value_arr = track_info.tuple_key_value_of_tpn_ID_arrival[tpn['ID']]
        tuple_key_dep, value_dep = track_info.tuple_key_value_of_tpn_ID_departure[tpn['ID']]
        track_info.track_sequences_of_TPN[tuple_key_arr].remove(value_arr)
        track_info.track_sequences_of_TPN[tuple_key_dep].remove(value_dep)


def add_transit_nodes_edges_single_train_to_Graph(G, nodes_edges_dict, bus=False):

    G.add_nodes_from(nodes_edges_dict['arrival_nodes_attr'])
    nx.set_node_attributes(G, nodes_edges_dict['arrival_nodes_attr'])
    G.add_nodes_from(nodes_edges_dict['departure_nodes_attr'])
    nx.set_node_attributes(G, nodes_edges_dict['departure_nodes_attr'])
    G.add_weighted_edges_from(nodes_edges_dict['driving_edges'])
    nx.set_edge_attributes(G, nodes_edges_dict['driving_attr'])
    if not bus:
        G.add_weighted_edges_from(nodes_edges_dict['waiting_edges'])
        nx.set_edge_attributes(G, nodes_edges_dict['waiting_attr'])


def remove_nodes_edges_of_train(G, train_id):
    nodes_attr_train = {x: y for x, y in G.nodes(data=True) if 'train' in y.keys() and y['train'] == train_id}
    # edges_of_train = {(u, v): attr for u, v, attr in G.edges(nodes_attr_train.keys(), data=True)}
    edges_of_train = {}
    for u, v, attr in G.edges(data=True):
        if u in nodes_attr_train.keys() or v in nodes_attr_train.keys():
            edges_of_train[(u, v)] = attr

    G.remove_edges_from(edges_of_train)
    G.remove_nodes_from(nodes_attr_train.keys())

    return nodes_attr_train, edges_of_train


def remove_nodes_edges_of_part_of_train(G, train_id, tpns_part):
    nodes_attr_train = {x: y for x, y in G.nodes(data=True) if 'train' in y.keys() and y['train'] == train_id and
                        x[2] in tpns_part}
    edges_of_train = {(u, v): attr for u, v, attr in G.edges(nodes_attr_train.keys(), data=True)}
    G.remove_edges_from(edges_of_train)
    G.remove_nodes_from(nodes_attr_train.keys())

    return nodes_attr_train, edges_of_train


def add_edges_of_train_from_o_stations_d(edges_o_stations_d, train, G, parameters, tpn_idx_start_delay, tpns_delay):

    arr_dep_nodes_train = [(n, v) for n, v in G.nodes(data=True) if v['train'] == train['ID']
                           and v['type'] in ['arrivalNode', 'departureNode'] and n[2] in tpns_delay]

    # odt_by_destination = parameters.odt_by_destination
    odt_by_origin = parameters.odt_by_origin
    zone_candidates = parameters.zone_candidates

    first_arrival_of_train = identify_first_arrival_tpnID_of_train(train, tpn_idx_start_delay)
    last_departure_of_train = identify_last_departure_tpnID_of_train(train, tpn_idx_start_delay)
    fmt = "%Y-%m-%dT%H:%M"
    min_wait = parameters.min_wait
    max_wait = parameters.max_wait

    for transit_node, attr in arr_dep_nodes_train:
        station = transit_node[0]
        if attr['type'] == 'arrivalNode':
            if not transit_node[2] == first_arrival_of_train and tpn_idx_start_delay == 0:
                zones_to_connect = zone_candidates[station]
                for zone, tt_to_zone in zones_to_connect.items():
                    edge = [transit_node, zone, tt_to_zone]
                    if zone in edges_o_stations_d.edges_stations_d_dict.keys():
                        edges_o_stations_d.edges_stations_d_dict[zone].append(edge)
                    else:
                        edges_o_stations_d.edges_stations_d_dict[zone] = [edge]
                    edges_o_stations_d.edges_stations_d.append(edge)

        elif attr['type'] == 'departureNode':
            if not transit_node[2] == last_departure_of_train:
                zones_to_connect = zone_candidates[station]
                for zone, tt_to_zone in zones_to_connect.items():
                    if zone not in odt_by_origin.keys():
                        continue
                    for trip in odt_by_origin[zone]:
                        desired_dep_time = parameters.origin_name_desired_dep_time[(trip[0], trip[1])]
                        desired_dep_time = datetime.datetime.strptime(desired_dep_time, fmt)
                        time_delta = attr['departureTime'] - desired_dep_time
                        if min_wait < time_delta < max_wait:
                            edge = [trip[0], transit_node, tt_to_zone]
                            if trip[0] in edges_o_stations_d.edges_o_stations_dict.keys():
                                edges_o_stations_d.edges_o_stations_dict[trip[0]].append(edge)
                            else:
                                edges_o_stations_d.edges_o_stations_dict[trip[0]] = [edge]
                            edges_o_stations_d.edges_o_stations.append(edge)

    return edges_o_stations_d


def add_edges_of_bus_from_o_stations_d(edges_o_stations_d, train, G, parameters, tpn_idx_start_delay, tpns_delay):

    arr_dep_nodes_bus = [(n, v) for n, v in G.nodes(data=True) if v['train'] == train['ID']
                           and v['type'] in ['arrivalNode', 'departureNode'] and n[2] in tpns_delay]


    # odt_by_destination = parameters.odt_by_destination
    odt_by_origin = parameters.odt_by_origin
    zone_candidates = parameters.zone_candidates

    fmt = "%Y-%m-%dT%H:%M"
    min_wait = parameters.min_wait
    max_wait = parameters.max_wait

    for transit_node, attr in arr_dep_nodes_bus:
        station = transit_node[0]
        if attr['type'] == 'arrivalNode':
            zones_to_connect = zone_candidates[station]
            for zone, tt_to_zone in zones_to_connect.items():
                edge = [transit_node, zone, tt_to_zone]
                if zone in edges_o_stations_d.edges_stations_d_dict.keys():
                    edges_o_stations_d.edges_stations_d_dict[zone].append(edge)
                else:
                    edges_o_stations_d.edges_stations_d_dict[zone] = [edge]
                edges_o_stations_d.edges_stations_d.append(edge)

        elif attr['type'] == 'departureNode':
            zones_to_connect = zone_candidates[station]
            for zone, tt_to_zone in zones_to_connect.items():
                if zone not in odt_by_origin.keys():
                    continue
                for trip in odt_by_origin[zone]:
                    desired_dep_time = parameters.origin_name_desired_dep_time[(trip[0], trip[1])]
                    desired_dep_time = datetime.datetime.strptime(desired_dep_time, fmt)
                    time_delta = attr['departureTime'] - desired_dep_time
                    if min_wait < time_delta < max_wait:
                        edge = [trip[0], transit_node, tt_to_zone]
                        if trip[0] in edges_o_stations_d.edges_o_stations_dict.keys():
                            edges_o_stations_d.edges_o_stations_dict[trip[0]].append(edge)
                        else:
                            edges_o_stations_d.edges_o_stations_dict[trip[0]] = [edge]
                        edges_o_stations_d.edges_o_stations.append(edge)

    return edges_o_stations_d

# if flow is very low or if it is delayed, trains can be cancelled
# train could be selected for operator
def identify_candidates_for_operators(cut_trains_to_area, parameters, solution_G, changed_trains):

    G_prime = solution_G.graph
    parameters.set_of_trains_for_operator['Cancel'] = solution_G.set_of_trains_for_operator['Cancel'].copy()
    parameters.set_of_trains_for_operator['Delay'] = solution_G.set_of_trains_for_operator['Delay'].copy()
    for train in cut_trains_to_area:
        comm_stops = 0
        if isinstance(train['ID'], str):
            # it is a bus
            continue
        for tpn in train['TrainPathNodes']:
            if tpn['StopStatus'] == 'commercialStop':
                comm_stops += 1
        if comm_stops <= 2:
            parameters.set_of_trains_for_operator['Cancel'].append(train['ID'])

    all_train_flows = {}
    for (u, v, attr) in G_prime.edges.data():
        if 'flow' in attr.keys():
            if attr['type'] != 'driving':
                continue
            if 'bus_id' in attr.keys():
                if not all_train_flows.__contains__(attr['bus_id']):
                    all_train_flows[attr['bus_id']] = {'total_flow': attr['flow'], 'nr_of_edgeds_with_flow': 1,
                                                       'avg_flow': attr['flow']}
                else:
                    train_flow = all_train_flows[attr['train_id']]
                    train_flow['total_flow'] += attr['flow']
                    train_flow['nr_of_edgeds_with_flow'] += 1
                    train_flow['avg_flow'] = train_flow['total_flow'] / train_flow['nr_of_edgeds_with_flow']

            else:
                if not all_train_flows.__contains__(attr['train_id']):
                    all_train_flows[attr['train_id']] = {'total_flow': attr['flow'], 'nr_of_edgeds_with_flow': 1,
                                                     'avg_flow': attr['flow']}
                else:
                    train_flow = all_train_flows[attr['train_id']]
                    train_flow['total_flow'] += attr['flow']
                    train_flow['nr_of_edgeds_with_flow'] += 1
                    train_flow['avg_flow'] = train_flow['total_flow'] / train_flow['nr_of_edgeds_with_flow']

    overall_flow_trains = 0
    overall_flow_bus = 0
    for train_id, attr in all_train_flows.items():
        bus = False
        if isinstance(train_id, str):  # 'EmergencyBus':
            bus = True
            max_cap = parameters.bus_capacity
            overall_flow_bus += attr['total_flow']
        else:
            max_cap = parameters.train_capacity
            overall_flow_trains += attr['total_flow']
        if attr['avg_flow'] <= 0.5 * max_cap:
            parameters.set_of_trains_for_operator['Cancel'].append(train_id)
            parameters.set_of_trains_for_operator['Delay'].append(train_id)
    # l, p = nx.multi_source_dijkstra(G, [(199, '2005-05-10T06:14:18', 7974602, 'a')], (543, '2005-05-10T06:30:00', 'Bus900002', 'a'))
    for train_id, attr in changed_trains.items():
        if attr['Action'] in ['Reroute', 'ShortTurn', 'CancelFrom']:
            if train_id in parameters.set_of_trains_for_operator['Cancel']:
                parameters.set_of_trains_for_operator['Cancel'].append(train_id)
            if train_id in parameters.set_of_trains_for_operator['Delay']:
                parameters.set_of_trains_for_operator['Delay'].append(train_id)

    print('Total flow trains', overall_flow_trains, 'Total flow bus', overall_flow_bus)


    # for (u, v, attr) in G_prime.edges.data():
    #    if 'flow' in attr.keys():
    #        if attr['flow'] != 0:
    #           print(attr)


def identify_first_arrival_tpnID_of_train(train, tpn_idx_start_delay):
    i = tpn_idx_start_delay
    while train['TrainPathNodes'][i]['StopStatus'] != 'commercialStop':
        i += 1
        # if i == len(train['TrainPathNodes']) -1:
    return train['TrainPathNodes'][i]['ID']


def identify_last_departure_tpnID_of_train(train, tpn_idx_start_delay):
    i = len(train['TrainPathNodes']) - 1

    while train['TrainPathNodes'][i]['StopStatus'] != 'commercialStop':
        i -= 1
        # if i == len(train['TrainPathNodes']) -1:
    return train['TrainPathNodes'][i]['ID']


def identify_last_departure_tpnID_of_train_to_cancelFrom(train):
    i = len(train['TrainPathNodes']) - 2

    while train['TrainPathNodes'][i]['StopStatus'] != 'commercialStop':
        i -= 1
        # if i == len(train['TrainPathNodes']) -1:
    return i


def identify_departure_tpnID_of_train_to_delayFrom(train):
    idx_candids = []
    i = 0
    for tpn in train['TrainPathNodes']:
        i += 1
        if tpn['StopStatus'] == 'commercialStop':
            idx_candids.append(i)

    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)
    i = idx_candids[np.random.randint(1, len(idx_candids)-2)]
    return i


def remove_edges_of_train_from_o_stations_d(edges_o_stations_d, train, G):
    arr_dep_nodes_train = [(n, v) for n, v in G.nodes(data=True) if v['train'] == train['ID']
                           and v['type'] in ['arrivalNode', 'departureNode']]

    for station, attr in arr_dep_nodes_train:
        if attr['type'] == 'arrivalNode':
            idx_station_d = sorted([i for i, x in enumerate(edges_o_stations_d.edges_stations_d) if x[0] == station], reverse=True)
            for index in idx_station_d:
                edge = edges_o_stations_d.edges_stations_d[index]
                # remove edge from dict keyed by destination
                edges_o_stations_d.edges_stations_d_dict[edge[1]].remove(edge)
                if len(edges_o_stations_d.edges_stations_d_dict[edge[1]]) == 0:
                    del edges_o_stations_d.edges_stations_d_dict[edge[1]]
                    # print('no edges from o to station anymore')
                del edges_o_stations_d.edges_stations_d[index]

        elif attr['type'] == 'departureNode':
            idx_o_station = sorted([i for i, x in enumerate(edges_o_stations_d.edges_o_stations) if x[1] == station], reverse=True)
            for index in idx_o_station:
                edge = edges_o_stations_d.edges_o_stations[index]
                # remove edge from dictionary keyed by origin
                edges_o_stations_d.edges_o_stations_dict[edge[0]].remove(edge)
                if len(edges_o_stations_d.edges_o_stations_dict[edge[0]]) == 0:
                    # print('no edges from station to d anymore')
                    del edges_o_stations_d.edges_o_stations_dict[edge[0]]
                del edges_o_stations_d.edges_o_stations[index]

    return edges_o_stations_d


def remove_edges_of_part_delayed_train_from_o_stations_d(edges_o_stations_d, train, G, tpns_delay):
    '''
    :param edges_o_stations_d:
    :param train:
    :param G:
    :param tpns_delay: list of tpns to delay, starting from start delay idx
    :return:
    '''
    arr_dep_nodes_train = [(n, v) for n, v in G.nodes(data=True) if v['train'] == train['ID']
                           and v['type'] in ['arrivalNode', 'departureNode'] and n[2] in tpns_delay]

    for station, attr in arr_dep_nodes_train:
        if attr['type'] == 'arrivalNode':
            idx_station_d = sorted([i for i, x in enumerate(edges_o_stations_d.edges_stations_d) if x[0] == station], reverse=True)
            for index in idx_station_d:
                edge = edges_o_stations_d.edges_stations_d[index]
                # remove edge from dict keyed by destination
                try:
                    edges_o_stations_d.edges_stations_d_dict[edge[1]].remove(edge)
                except ValueError:
                    print('why not in the list ?')
                if len(edges_o_stations_d.edges_stations_d_dict[edge[1]]) == 0:
                    del edges_o_stations_d.edges_stations_d_dict[edge[1]]
                    # print('no edges from o to station anymore')
                del edges_o_stations_d.edges_stations_d[index]

        elif attr['type'] == 'departureNode':
            idx_o_station = sorted([i for i, x in enumerate(edges_o_stations_d.edges_o_stations) if x[1] == station], reverse=True)
            for index in idx_o_station:
                edge = edges_o_stations_d.edges_o_stations[index]
                # remove edge from dictionary keyed by origin
                edges_o_stations_d.edges_o_stations_dict[edge[0]].remove(edge)
                if len(edges_o_stations_d.edges_o_stations_dict[edge[0]]) == 0:
                    # print('no edges from station to d anymore')
                    del edges_o_stations_d.edges_o_stations_dict[edge[0]]
                del edges_o_stations_d.edges_o_stations[index]

    return edges_o_stations_d


def update_delayed_tpn(runtime_delay_feasible, tpn):
    tpn['DepartureTime'] = runtime_delay_feasible['DepartureTime']
    tpn['ArrivalTime'] = runtime_delay_feasible['ArrivalTime']
    return tpn


def get_all_trains_cut_to_time_window_and_area(parameters):

    for station_in_area in parameters.stations_in_area:
        cut_trains_to_area = ap.active_trains_cut_to_time_range_driving_any_node_one(parameters.time_window['FromTime'],
                                                                            parameters.time_window['ToTime'],
                                                                            station_in_area)
    #cut_trains_to_area = ap.active_trains_within_time_range(parameters.time_window['FromTime'],parameters.time_window['ToTime'])
    cut_trains_to_area = ap.cut_trains_AreaOfInterest(cut_trains_to_area, parameters.stations_in_area)

    return cut_trains_to_area


def distance_travelled_all_trains(cut_trains_to_area, G_infra):
    total_distance = 0
    for train in cut_trains_to_area:
        if isinstance(train, int):
            print('train canceled ?')
            continue
        for tpn in train['TrainPathNodes']:
            if tpn['SectionTrackID'] is not None:
                    if tpn['SectionTrackID'] in train.keys():
                        total_distance += G_infra.graph['cache_trackID_dist'][tpn['SectionTrackID']]
    # the distance is in decimeter --> divide by 10 * 1000 to have it in km
    total_distance = round(total_distance / (10*1000), 1)
    return total_distance


def deviation_time_table(cut_trains_to_area, initial_timetable, changed_trains, parameters):
    # parameters to penalize the operation, c - cancel, d - delay, e - emergency train, r - rerouting
    parameters.set_of_trains_for_operator = {'Cancel': [], 'CancelFrom': [], 'Delay': [], 'DelayFrom': []}
    fmt = "%Y-%m-%dT%H:%M:%S"
    d_c = 50 # cancelation
    d_d = 1 #delay
    d_e = 1000 # emergency
    d_b = d_e / 10 # bus
    d_r = 10 #rerouted train

    total_deviation = 0  # in minutes
    timetable_0 = utils.build_dict(initial_timetable.initial_timetable_infeasible, 'ID')

    # timetable of the current solution
    timetable_prime = utils.build_dict(cut_trains_to_area, 'ID')

    for train_id, value in changed_trains.items():
        action = value['Action']
        if action == 'Cancel':
            if 'EmergencyTrain' in value.keys():
                total_deviation += 0
                continue
            elif 'EmergencyBus' in value.keys():
                total_deviation += 0
                continue

            total_deviation += d_c
            departure_first_tpn = timetable_0[train_id]['TrainPathNodes'][0]['DepartureTime']
            departure_first_tpn = datetime.datetime.strptime(departure_first_tpn, "%Y-%m-%dT%H:%M:%S")
            arrival_last_tpn = timetable_0[train_id]['TrainPathNodes'][-1]['ArrivalTime']
            arrival_last_tpn = datetime.datetime.strptime(arrival_last_tpn, "%Y-%m-%dT%H:%M:%S")
            total_deviation += d_c * ((arrival_last_tpn - departure_first_tpn).seconds / 60)

        elif action == 'CancelFrom':
            if 'EmergencyTrain' in value.keys():
                total_deviation += deviation_emergency_train(timetable_prime, d_e, total_deviation, train_id, parameters)
                continue

            for tpn in timetable_0[train_id]['TrainPathNodes']:
                if tpn['ID'] == value['tpn_cancel_from']:
                    dep_time_canceledFrom = tpn['DepartureTime']
                    dep_time_canceledFrom = datetime.datetime.strptime(dep_time_canceledFrom, "%Y-%m-%dT%H:%M:%S")
                    arr_time_end_train = timetable_0[train_id]['TrainPathNodes'][-1]['ArrivalTime']
                    arr_time_end_train = datetime.datetime.strptime(arr_time_end_train, "%Y-%m-%dT%H:%M:%S")
                    total_deviation += d_c * (arr_time_end_train - dep_time_canceledFrom).seconds / 60

        elif action == 'ShortTurn':
            # Like cancelled in between
            for tpn in timetable_0[train_id]['TrainPathNodes']:
                if tpn['ID'] == value['tpns_cancel_from_to'][0]:
                    arrival_tpn_beforeTurn = tpn['ArrivalTime']
                    arrival_tpn_beforeTurn = datetime.datetime.strptime(arrival_tpn_beforeTurn, "%Y-%m-%dT%H:%M:%S")

                elif tpn['ID'] == value['tpns_cancel_from_to'][1]:
                    depart_tpn_afterTurn = tpn['DepartureTime']
                    depart_tpn_afterTurn = datetime.datetime.strptime(depart_tpn_afterTurn, "%Y-%m-%dT%H:%M:%S")

                    total_deviation += d_c * (depart_tpn_afterTurn - arrival_tpn_beforeTurn)

                # changed_trains[trainID] = {'DebugString': dbg_string, 'Action': 'ShortTurn ', 'initial_tpn': tpn_initial_train,
                #                          'tpns_cancel_from_to': cancel_tpnIDS, 'train_before': train_before_disruption,
                #                           'train_after': train_after_disruption}

        elif action == 'Reroute':
            if 'add_stop_time' in value.keys():
                if value['add_stop_time'] > parameters.delayTime_to_consider_cancel:
                    parameters.set_of_trains_for_operator['Cancel'].append(train_id)

            if train_id in timetable_prime.keys():
                for tpn in timetable_prime[train_id]['TrainPathNodes']:
                    if tpn['ID'] == value['StartEndRR_tpnID']:
                        dep_time_start_RR = tpn['DepartureTime']
                        dep_time_start_RR = datetime.datetime.strptime(dep_time_start_RR, "%Y-%m-%dT%H:%M:%S")
                        dep_time_end_train = timetable_prime[train_id]['TrainPathNodes'][-1]['ArrivalTime']
                        dep_time_end_train = datetime.datetime.strptime(dep_time_end_train, "%Y-%m-%dT%H:%M:%S")
                        total_deviation += d_r * (dep_time_end_train - dep_time_start_RR).seconds/60
                        break


        elif action == 'Delay' or action == 'Return':
            try:
                if 'EmergencyTrain' in timetable_prime[train_id].keys():
                    total_deviation = deviation_emergency_train(timetable_prime, d_e, total_deviation, train_id, parameters)
                    continue
                elif 'EmergencyBus' in timetable_prime[train_id].keys():
                    total_deviation = deviation_emergency_bus(timetable_prime, d_b, total_deviation, train_id, parameters)
                    continue
            except KeyError:
                print('something went wrong')
            max_delay_tpn, total_delay_train = deviation_delay_train(fmt, timetable_0, timetable_prime, train_id)
            total_deviation += total_delay_train.seconds / 60 * d_d

            changed_trains[train_id]['total_delay'] = total_delay_train
            changed_trains[train_id]['tpn_max_delay'] = max_delay_tpn

            if total_delay_train > parameters.delayTime_to_consider_cancel:
                if train_id not in parameters.set_of_trains_for_operator['Cancel']:
                    parameters.set_of_trains_for_operator['Cancel'].append(train_id)
            # if max_delay_tpn[0] > parameters.delayTime_to_consider_partCancel:
            #    if train_id not in parameters.set_of_trains_for_operator['CancelFrom']:
            #        parameters.set_of_trains_for_operator['CancelFrom'].append([train_id, max_delay_tpn[1]])

        elif action == 'DelayFrom':
            if 'EmergencyTrain' in timetable_prime[train_id].keys():
                deviation_emergency_train(timetable_prime, d_e, total_deviation, train_id, parameters)
                continue

            max_delay_tpn, total_delay_train = deviation_delay_train(fmt, timetable_0, timetable_prime, train_id)

            total_deviation += total_delay_train.seconds / 60 * d_d

            changed_trains[train_id]['total_delay'] = total_delay_train
            changed_trains[train_id]['tpn_max_delay'] = max_delay_tpn

            if total_delay_train > parameters.delayTime_to_consider_cancel:
                if train_id not in parameters.set_of_trains_for_operator['Cancel']:
                    parameters.set_of_trains_for_operator['Cancel'].append(train_id)
            # if max_delay_tpn[0] > parameters.delayTime_to_consider_partCancel:
            #    if train_id not in parameters.set_of_trains_for_operator['CancelFrom']:
            #        parameters.set_of_trains_for_operator['CancelFrom'].append([train_id, max_delay_tpn[1]])

        elif action == 'EmergencyTrain':
            total_deviation = deviation_emergency_train(timetable_prime, d_e, total_deviation, train_id, parameters)

        elif action == 'EmergencyBus':
            total_deviation = deviation_emergency_bus(timetable_prime, d_b, total_deviation, train_id, parameters)

    return round(total_deviation, 1)


def deviation_delay_train(fmt, timetable_0, timetable_prime, train_id):
    total_delay_train = datetime.timedelta(minutes=0)
    max_delay_tpn = [datetime.timedelta(minutes=0), 999]
    try:
        tpn_prime = utils.build_dict(timetable_prime[train_id]['TrainPathNodes'], 'ID')
    except KeyError:
        print('whi is train not in timetable prime')
    for tpn_0 in timetable_0[train_id]['TrainPathNodes']:
        try:
            if not tpn_0['ID'] in tpn_prime.keys():
                # this train has been canceled from a point, therefore this tpn is not in tpn prime anymore
                continue
            if tpn_0['DepartureTime'] == tpn_prime[tpn_0['ID']]['DepartureTime']:
                continue
            else:
                dep_tpn_0 = datetime.datetime.strptime(tpn_0['DepartureTime'], fmt)
                dep_tpn_prime = datetime.datetime.strptime(tpn_prime[tpn_0['ID']]['DepartureTime'], fmt)
                deviation_tpn = dep_tpn_prime - dep_tpn_0
                if max_delay_tpn[0] < deviation_tpn:
                    max_delay_tpn = [deviation_tpn, tpn_0['ID']]

            total_delay_train += deviation_tpn
        except KeyError:
            print('somethin went wrond in deviation calculation')

    return max_delay_tpn, total_delay_train


def deviation_emergency_train(timetable_prime, d_e, total_deviation, train_id, parameters):

    fmt = "%Y-%m-%dT%H:%M:%S"

    dep_time_start = timetable_prime[train_id]['TrainPathNodes'][0]['DepartureTime']
    dep_time_start = datetime.datetime.strptime(dep_time_start, fmt)
    arr_time_end_train = timetable_prime[train_id]['TrainPathNodes'][-1]['ArrivalTime']
    arr_time_end_train = datetime.datetime.strptime(arr_time_end_train, fmt)
    total_deviation += d_e + (arr_time_end_train - dep_time_start).seconds / 60

    parameters.set_of_trains_for_operator['Cancel'].append(train_id)
    # parameters.set_of_trains_for_operator['CancelFrom'].append(train_id)
    parameters.set_of_trains_for_operator['Delay'].append(train_id)
    # parameters.set_of_trains_for_operator['DelayFrom'].append(train_id)

    return total_deviation


def deviation_emergency_bus(timetable_prime, d_b, total_deviation, train_id, parameters):

    fmt = "%Y-%m-%dT%H:%M:%S"

    dep_time_start = timetable_prime[train_id]['TrainPathNodes'][0]['DepartureTime']
    dep_time_start = datetime.datetime.strptime(dep_time_start, fmt)
    arr_time_end_train = timetable_prime[train_id]['TrainPathNodes'][-1]['ArrivalTime']
    arr_time_end_train = datetime.datetime.strptime(arr_time_end_train, fmt)
    total_deviation += d_b + (arr_time_end_train - dep_time_start).seconds / 60

    parameters.set_of_trains_for_operator['Cancel'].append(train_id)
    parameters.set_of_trains_for_operator['Delay'].append(train_id)

    return total_deviation


def check_Graph_number_edges_nodes(G):
    # nodes_origins = [x for x, y in G.nodes(data=True) if y['type'] == 'origin']
    # nodes_destination = [x for x, y in G.nodes(data=True) if y['type'] == 'destination']
    train_node_types = ['arrivalNode', 'departureNode', 'arrivalNodePassing', 'departureNodePassing']
    nodes_trains = [x for x, y in G.nodes(data=True) if y['type'] in train_node_types]
    od, d, w, t = 0, 0, 0, 0
    for edge in G.edges.data():
        value = edge[2]
        if 'type' in value.keys():
            if value['type'] == 'driving':
                d += 1
            elif value['type'] == 'waiting':
                w += 1
            elif value['type'] == 'transfer':
                t += 1
        else:
            od += 1
    print('end')


def pickle_archive_w_changed_Trains(solution_archive):
    z_op, z_de, z_tt, changed_trains = [], [], [], []

    archive_for_pickle = {'z_op': z_op, 'z_de': z_de, 'z_tt': z_tt, 'changed_trains': changed_trains}
    for solution in solution_archive:
        z_op.append(solution.total_dist_train)
        z_de.append(solution.deviation_timetable)
        z_tt.append(solution.total_traveltime)
        changed_trains.append(solution.changed_trains)
    pickle_results(archive_for_pickle, 'z_archive.pkl')


def pickle_archive_op_tt_de(solution_archive):
    z_op, z_de, z_tt = [], [], []

    archive_for_pickle = {'z_op': z_op, 'z_de': z_de, 'z_tt': z_tt}
    for solution in solution_archive:
        z_op.append(solution.total_dist_train)
        z_de.append(solution.deviation_timetable)
        z_tt.append(solution.total_traveltime)
    pickle_results(archive_for_pickle, 'z_archive.pkl')


def pickle_full_archive(solution_archive):

    pickle_results(solution_archive, 'solution_archive.pkl')

