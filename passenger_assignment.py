"""
Created on Sun May  09 2021

@author: BenMobility

Passenger assignment with capacity constraint
"""
import networkx as nx
import shortest_path
import copy


# %% Passenger assignment
def capacity_constraint_1st_loop(parameters, timetable_initial_graph):
    """
    Method that assign all the passengers from the odt priority list. It will capture all the odt that are facing
    capacity constraint on their trip. Will return the passenger assignment of the origin odt plus a list of all the odt
    facing the capacity constraint
    :param parameters: Needs to have as an input the odt priority list
    :param timetable_initial_graph: Digraph with trains timetable edges and home-destination-transfers edges as well.
    :return: Odt_facing_capacity_constraint, a list of all the odt facing capacity constraint. parameters and timetable
    initial graph with updated values from passenger assignment
    """
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
            _, p = shortest_path.single_source_dijkstra(timetable_initial_graph, odt[0], odt[1])
            # Save the path
            odt_priority_list_original[i][4] = p
            # Save the length of the path
            try:
                odt_priority_list_original[i][5] = 0
            except IndexError:
                odt_priority_list_original[i].append(0)

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
                                        # If not assigned in the previous edge, hence the assigned passenger must be
                                        # from another train
                                        else:
                                            # Save the assigned
                                            odt_with_lower_priority_name.append(
                                                timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k])
                                            odt_with_lower_priority_flow.append(
                                                timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k][3])
                                            odt_with_lower_priority_index.append(-k)

                                            # Check if removing the assigned odt from the train is enough, if not, need\
                                            # to add another assigned from the list
                                            if parameters.train_capacity >= \
                                                    sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) - \
                                                    sum(odt_with_lower_priority_flow) + odt[3]:
                                                for odt_with_lower_priority in odt_with_lower_priority_name:
                                                    # Extract the odt to get the recorded path from the original
                                                    # priority list
                                                    extract_odt = [item for item in odt_priority_list_original
                                                                   if item[0:4] == odt_with_lower_priority[0:4]]

                                                    # Find the index on the original list
                                                    index_in_original_list = odt_priority_list_original.index(
                                                        extract_odt[0])

                                                    # Get the path from the original list
                                                    extract_odt_path = extract_odt[0][4]

                                                    # Get the index of the last node before the full capacity train
                                                    index_last_node_on_path_before_capacity = extract_odt_path.index(
                                                        p[j])

                                                    # Split the path into delete and keep path
                                                    odt_path_to_delete = extract_odt_path[
                                                                         index_last_node_on_path_before_capacity:]
                                                    odt_path_to_keep = extract_odt_path[
                                                                       :index_last_node_on_path_before_capacity]

                                                    # Modify the original path and erase the length, needs to be
                                                    # recomputed
                                                    try:
                                                        odt_priority_list_original[
                                                            index_in_original_list][4] = odt_path_to_keep
                                                        odt_priority_list_original[index_in_original_list][5] = 0
                                                    except ValueError:
                                                        print(f'{odt_priority_list_original[index_in_original_list][0]}'
                                                              f' at index {index_in_original_list} has already a '
                                                              f'changed value but it was not recorded properly. Please'
                                                              f'check passenger assignment')

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
                                                                odt_path_to_delete[n + 1]]['odt_assigned'][
                                                                index_to_delete]
                                                        except (KeyError, ValueError):
                                                            continue

                                                    if 'odt_facing_capacity_constrain' in locals():
                                                        # Record the odt with the last node before capacity constraint.
                                                        # [odt, odt path to keep, edge full, number of trial]
                                                        odt_info = [odt_with_lower_priority,
                                                                    odt_path_to_keep,
                                                                    odt_path_to_delete[0:2],
                                                                    1]
                                                        odt_facing_capacity_constrain.append(odt_info)
                                                    else:
                                                        odt_facing_capacity_constrain = [[odt_with_lower_priority,
                                                                                         odt_path_to_keep,
                                                                                         odt_path_to_delete[0:2],
                                                                                         1]]

                                                # Finally, add the current odt on the clean edge
                                                timetable_initial_graph[p[j]][p[j + 1]]['flow'].append(odt[3])
                                                timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'].append(odt[0:4])
                                                # Done with the recording of odt facing capacity constraint
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
                                    odt_info = [odt[0:4], p[:j], [p[j], p[j + 1]], 1]
                                    odt_facing_capacity_constrain.append(odt_info)
                                else:
                                    odt_facing_capacity_constrain = [[odt[0:4], p[:j], [p[j], p[j + 1]], 1]]

                                # Done for this odt, update the path on the original list
                                # Find the index on the original list
                                index_in_original_list = odt_priority_list_original.index(odt)

                                # Update the new path
                                odt_priority_list_original[index_in_original_list][4] = p[:j]
                                # do not need to continue to assign further. go to the next one
                                break

                        # It means that the previous edge is home to the first station,
                        # hence the passenger is not seated in the train
                        except IndexError:
                            if 'odt_facing_capacity_constrain' in locals():
                                # Record the odt with the last node before capacity constraint.
                                # [odt, last node, index, edge, new path, number of trial]
                                odt_info = [odt[0:4], p[:j], [p[j], p[j + 1]], 1]
                                odt_facing_capacity_constrain.append(odt_info)
                            else:
                                odt_facing_capacity_constrain = [[odt[0:4], p[:j], [p[j], p[j + 1]], 1]]

                            # Done for this odt, do not need to continue to assign further. go to the next one
                            # But before need to assign the path to the original list
                            # Find the index on the original list
                            index_in_original_list = odt_priority_list_original.index(odt)

                            # Update the new path
                            odt_priority_list_original[index_in_original_list][4] = p[:j]
                            break
                    else:
                        timetable_initial_graph[p[j]][p[j + 1]]['flow'].append(odt[3])
                        timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'].append(odt[0:4])

                # If there is a key error, it means it is either a home-station edge, station-destination edge or a
                # transfer, hence we go check the next node
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

    # Sort in a descending order for the new list of odt facing capacity constraint
    odt_facing_capacity_constrain.sort(key=lambda x: x[0][2], reverse=True)
    return odt_facing_capacity_constrain, parameters, timetable_initial_graph
# %% Assigning flow for the 2nd and more iteration


def capacity_constraint_2nd_loop(parameters, odt_facing_capacity_constraint, timetable_initial_graph):
    """
    method that iterates over the list of odt facing the trains with full capacity. It reassigns the passengers on the
    the timetable_initial_graph
    :param parameters: list of parameters from the main file
    :param odt_facing_capacity_constraint: list of odt facing train with full capacity
    :param timetable_initial_graph: multigraph with trains flow and odt assigned on the edges
    :return: timetable with assigned passengers, number of passenger assigned, number of passenger unassigned and
    dictionary of all the odt lists facing the capacity constraint
    """
    # Make a copy of the first priority list
    odt_priority_list_original = copy.deepcopy(parameters.odt_as_list)

    # Create a dictionary for the iterations
    odt_facing_capacity_dict_for_iteration = {0: copy.deepcopy(odt_facing_capacity_constraint)}

    # Start the loop for the passenger assignment with capacity constraint
    m = 0
    while True:
        try:
            # Select the list based on the iteration of m
            odt_list = odt_facing_capacity_dict_for_iteration[m]

            # remove the duplicates from the list
            odt_list = list(remove_the_duplicates(odt_list))

            # Set the count to zero for the loop
            i = 0

            # Assign the passengers based on the priority list
            for odt in odt_list:

                # Break the loop if reached the last odt to avoid index error

                if i == len(odt_list):
                    print('End of the passenger assignment')
                    break

                # Compute the shortest path with dijkstra
                try:
                    # First make sure that they won't use the path with the full capacity constraint, do not forget to
                    # save and restore the initial weight on the edge
                    initial_weight = timetable_initial_graph[odt[2][0]][odt[2][1]]['weight']
                    timetable_initial_graph[odt[2][0]][odt[2][1]]['weight'] = parameters.weight_closed_tracks
                    _, p = shortest_path.single_source_dijkstra(timetable_initial_graph,
                                                                odt[1][-1],
                                                                odt[0][1],
                                                                cutoff=parameters.weight_closed_tracks - 1)
                    timetable_initial_graph[odt[2][0]][odt[2][1]]['weight'] = initial_weight
                    # Save the path, origin to destination with the new path
                    odt_list[i][1] = odt_list[i][1] + p[1:]

                    # Assign the flow on the timetable graph's edges starting from the new path only
                    for j in range(len(p) - 1):
                        try:
                            # Check the train capacity on the next edge
                            if sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) + odt[0][3] > \
                                    parameters.train_capacity:
                                try:
                                    # Check if the current passenger is already seated in the train
                                    if p[j - 1][2] == p[j][2]:
                                        # Initialize the parameters for the capacity constraint checks
                                        k = 1
                                        # todo: change with lower priority to boarding at current node
                                        odt_with_lower_priority_name = []
                                        odt_with_lower_priority_flow = []
                                        odt_with_lower_priority_index = []

                                        # Remove assigned odt with lower priority on the edge
                                        while sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) + odt[0][3] > \
                                                parameters.train_capacity:
                                            try:
                                                # Check if the assigned odt is already seated in the train, if so, go to
                                                # next assigned odt
                                                if timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k] in \
                                                        timetable_initial_graph[p[j - 1]][p[j]]['odt_assigned']:
                                                    k += 1
                                                # If not assigned in the previous edge, hence the assigned passenger
                                                # must be from another train
                                                else:
                                                    # Save the assigned
                                                    odt_with_lower_priority_name.append(
                                                        timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k])
                                                    odt_with_lower_priority_flow.append(
                                                        timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k][3])
                                                    odt_with_lower_priority_index.append(-k)

                                                    # Check if removing the assigned odt from the train is enough, if
                                                    # not, need to add another assigned from the list
                                                    if parameters.train_capacity >= \
                                                        sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) - \
                                                            sum(odt_with_lower_priority_flow) + odt[0][3]:
                                                        for odt_with_lower_priority in odt_with_lower_priority_name:

                                                            # Extract the odt to get the recorded path from the original
                                                            # priority list
                                                            extract_odt = [item for item in odt_priority_list_original
                                                                           if item[0:4] == odt_with_lower_priority[0:4]]

                                                            # Find the index on the original list
                                                            index_in_original_list = odt_priority_list_original.index(
                                                                extract_odt[0])

                                                            # Verify the type of extract_odt between odt original format
                                                            # or
                                                            # Get the path from the original list
                                                            extract_odt_path = extract_odt[0][4]

                                                            # Get the index of the last node before the full capacity
                                                            # train
                                                            index_last_node_on_path_before_capacity = \
                                                                extract_odt_path.index(p[j])

                                                            # Split the path into delete and keep path
                                                            odt_path_to_delete = \
                                                                extract_odt_path[
                                                                index_last_node_on_path_before_capacity:]

                                                            odt_path_to_keep = extract_odt_path[
                                                                               :index_last_node_on_path_before_capacity]

                                                            # Modify the original path and erase the length, needs to be
                                                            # recomputed
                                                            try:
                                                                odt_priority_list_original[
                                                                    index_in_original_list][4] = list(odt_path_to_keep)
                                                                odt_priority_list_original[
                                                                    index_in_original_list][5] = 0
                                                            except ValueError:
                                                                # in order to stay inside the lines for code writing
                                                                message = \
                                                                    odt_priority_list_original[
                                                                        index_in_original_list][0:2]
                                                                print(f'{message} at index {index_in_original_list} '
                                                                      f'has already a changed value but it was not'
                                                                      f' recorded properly. Please check passenger '
                                                                      f'assignment')

                                                            # Delete the flow and the odt_assigned
                                                            for n in range(len(odt_path_to_delete) - 1):
                                                                try:
                                                                    index_to_delete = timetable_initial_graph[
                                                                        odt_path_to_delete[n]][
                                                                        odt_path_to_delete[n + 1]][
                                                                        'odt_assigned'].index(odt_with_lower_priority)
                                                                    del timetable_initial_graph[odt_path_to_delete[n]][
                                                                        odt_path_to_delete[n + 1]]['flow'][
                                                                        index_to_delete]
                                                                    del timetable_initial_graph[odt_path_to_delete[n]][
                                                                        odt_path_to_delete[n + 1]]['odt_assigned'][
                                                                        index_to_delete]
                                                                except (KeyError, ValueError):
                                                                    # KeyError means it is a transfer edge where there
                                                                    # is no flow or odt_assigned. ValueError can be
                                                                    # already removed from the edge. How? good question.
                                                                    continue

                                                            # Need to check if the boarding odt is reaching its maximum
                                                            # iteration before assigning it to the next list. To do so
                                                            # We first need to check if it has been already on the
                                                            # capacity list to check the number of iteration
                                                            if any(item[0] == odt_with_lower_priority
                                                                   for item in odt_list):
                                                                number_of_iteration = \
                                                                    [item[-1] for item in odt_list
                                                                     if item[0] == odt_with_lower_priority]
                                                            else:
                                                                number_of_iteration = [0]

                                                            # If the maximum number is reached, the passenger leave the
                                                            # system from the last point on their trip
                                                            if number_of_iteration[0] + 1 > \
                                                                    parameters.max_iteration_recompute_path:
                                                                odt_priority_list_original[
                                                                    index_in_original_list][4] = list(odt_path_to_keep)
                                                                odt_priority_list_original[
                                                                    index_in_original_list][5] = \
                                                                    parameters.penalty_no_path
                                                            else:
                                                                try:
                                                                    # If they have not reached their maximum number of
                                                                    # recalculation of their trip, put the current odt
                                                                    # in the next list
                                                                    odt_new_list = \
                                                                        odt_facing_capacity_dict_for_iteration[m+1]
                                                                    odt_info = [odt_with_lower_priority,  # ODT name
                                                                                odt_path_to_keep,  # ODT path keep
                                                                                odt_path_to_delete[0:2],    # Edge full
                                                                                number_of_iteration[0]+1]
                                                                    odt_facing_capacity_dict_for_iteration[
                                                                        m + 1].append(odt_info)
                                                                except KeyError:
                                                                    odt_facing_capacity_dict_for_iteration[m+1] =\
                                                                        list([[odt_with_lower_priority,
                                                                               odt_path_to_keep,
                                                                               odt_path_to_delete[0:2],
                                                                               number_of_iteration[0]+1]])

                                                            # Check if the odt_with_lower priority is in the odt facing
                                                            # capacity constraint list. If it does, need to delete it
                                                            try:
                                                                extract_odt_facing_capacity_constraint = \
                                                                    [item for item in odt_list[i:]
                                                                     if item[0][0:4] == odt_with_lower_priority[0:4]]

                                                                index_in_odt_list = odt_list.index(
                                                                    extract_odt_facing_capacity_constraint[0])

                                                                del odt_list[index_in_odt_list]

                                                            # If Value Error, it means it is not in the list and we can
                                                            # continue
                                                            except (ValueError, IndexError):
                                                                pass

                                                        # Finally, add the current odt on the clean edge
                                                        timetable_initial_graph[p[j]][p[j + 1]][
                                                            'flow'].append(odt[0][3])
                                                        timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'].append(
                                                            odt[0])
                                                        # Done with the recording of oft facing capacity constraint
                                                        break
                                                    # Not enough seats released, need at least one more group to leave
                                                    else:
                                                        k += 1
                                            # Not suppose to happen, but it might if there an assignment mistake
                                            except IndexError:
                                                print(
                                                    f'Train is at full capacity and the current odt {odt[0]} is already'
                                                    f' seated, but the algorithm cannot find the assigned odt that is'
                                                    f' assigned but not seated in the train.')
                                                break

                                    # It means, that the next train is at full capacity. Hence, the current odt journey
                                    # needs to be computed from here in the next list
                                    else:

                                        # First, we need to check if the odt has reached the limit of recomputing path
                                        if odt[3] + 1 > parameters.max_iteration_recompute_path:
                                            # Extract the odt to get the recorded path from the original
                                            # priority list
                                            extract_odt = [item for item in odt_priority_list_original
                                                           if item[0:4] == odt[0][0:4]]

                                            # Find the index on the original list
                                            index_in_original_list = odt_priority_list_original.index(
                                                extract_odt[0])

                                            # Update the original with the penalty
                                            odt_priority_list_original[index_in_original_list][4] = \
                                                list(odt_list[i][1][:-(len(p)-j)])  # Keep the assigned path
                                            odt_priority_list_original[index_in_original_list][5] = \
                                                parameters.penalty_no_path

                                            # Do not need to go further, next odt please.
                                            break

                                        else:
                                            try:
                                                # If they have not reached their maximum number of
                                                # recalculation of their trip, put the current odt
                                                # in the next list
                                                odt_new_list = \
                                                    odt_facing_capacity_dict_for_iteration[m + 1]
                                                odt_info = [odt[0],                         # ODT name
                                                            odt_list[i][1][:-(len(p)-j)],  # ODT path to keep
                                                            [p[j], p[j + 1]],               # Edge full
                                                            odt[3]+1]                       # Number of iteration
                                                odt_facing_capacity_dict_for_iteration[m + 1].append(odt_info)
                                            except KeyError:
                                                odt_facing_capacity_dict_for_iteration[m + 1] = \
                                                    list([[odt[0],
                                                           odt_list[i][1][:-(len(p)-j)],
                                                           [p[j], p[j + 1]],
                                                           odt[3]+1]])

                                        # Done for this odt, do not need to continue to assign further. go to the next
                                        # one but before need to assign the path to the original list
                                        # Find the index on the original list
                                        # Get the original length
                                        length_on_original_path = len(odt[1]) - len(p[1:])

                                        # Transform the odt to have the same format
                                        odt_with_original_format = [odt[0][0],
                                                                    odt[0][1],
                                                                    odt[0][2],
                                                                    odt[0][3],
                                                                    list(odt[1][:length_on_original_path]),
                                                                    0]

                                        index_in_original_list = \
                                            odt_priority_list_original.index(odt_with_original_format)

                                        # Update the new path and set the value to 0. In case the previous iteration,
                                        # the odt was consider with a penalty but now finds a way
                                        odt_priority_list_original[index_in_original_list][4] = \
                                            list(odt_list[i][1][:-(len(p)-j)])
                                        odt_priority_list_original[index_in_original_list][5] = 0

                                        # Do not need to go further. Next odt please.
                                        break

                                # It means that the previous edge is home to the first station,
                                # hence the passenger is not seated in the train
                                except IndexError:
                                    # First, we need to check if the odt has reached the limit of recomputing path
                                    if odt[3] + 1 > parameters.max_iteration_recompute_path:
                                        # Extract the odt to get the recorded path from the original
                                        # priority list
                                        extract_odt = [item for item in odt_priority_list_original
                                                       if item[0:4] == odt[0][0:4]]

                                        # Find the index on the original list
                                        index_in_original_list = odt_priority_list_original.index(
                                            extract_odt[0])

                                        # Update the original with the penalty
                                        odt_priority_list_original[index_in_original_list][4] = \
                                            list(odt_list[i][1][:-(len(p) - j)])  # Keep the assigned path
                                        odt_priority_list_original[index_in_original_list][5] = \
                                            parameters.penalty_no_path

                                        # Do not need to go further. Next odt please.
                                        break

                                    else:
                                        # Need to check if the boarding odt is reaching its maximum
                                        # iteration before assigning it to the next list. To do so
                                        # We first need to check if it has been already on the
                                        # capacity list to check the number of iteration
                                        if any(item[0] == odt_with_lower_priority
                                               for item in odt_list):
                                            number_of_iteration = \
                                                [item[-1] for item in odt_list
                                                 if item[0] == odt_with_lower_priority]
                                        else:
                                            number_of_iteration = [0]

                                        try:
                                            # If they have not reached their maximum number of
                                            # recalculation of their trip, put the current odt
                                            # in the next list
                                            odt_new_list = \
                                                odt_facing_capacity_dict_for_iteration[m + 1]
                                            odt_info = [odt[0],  # ODT name
                                                        odt_list[i][1][:-(len(p) - j)],  # ODT path to keep
                                                        [p[j], p[j + 1]],  # Edge full
                                                        number_of_iteration[0]+1]  # Number of iteration
                                            odt_facing_capacity_dict_for_iteration[m + 1].append(odt_info)
                                        except KeyError:
                                            odt_facing_capacity_dict_for_iteration[m + 1] = \
                                                [[odt[0],
                                                 odt_list[i][1][:-(len(p) - j)],
                                                 [p[j], p[j + 1]],
                                                 number_of_iteration[0]+1]]

                                    # Done for this odt, do not need to continue to assign further. go to the next one
                                    # But before need to assign the path to the original list
                                    # Find the index on the original list
                                    index_in_original_list = odt_priority_list_original.index(odt)

                                    # Update the new path
                                    odt_priority_list_original[index_in_original_list][4] = list(p[:j])
                                    odt_priority_list_original[index_in_original_list][5] = 0

                                    # Do not need to go further. Next odt please.
                                    break
                            else:
                                # Assign the current odt to the edge for the flow (group size) and odt_assigned with
                                # the name.
                                timetable_initial_graph[p[j]][p[j + 1]]['flow'].append(odt[0][3])
                                timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'].append(odt[0])

                        # If there is a key error, it means it is either a home-station edge, station-destination edge
                        # or a transfer, hence we go check the next node
                        except KeyError:
                            pass

                    # Once all the path is assigned with the current odt, update the original list with the new path
                    if j == (len(p) - 2):
                        # Update the odt info on the original list
                        extract_odt = [item for item in odt_priority_list_original
                                       if item[0:4] == odt[0][0:4]]

                        # Find the index on the original list
                        index_in_original_list = odt_priority_list_original.index(
                            extract_odt[0])

                        # Update the original odt with the new path
                        odt_priority_list_original[index_in_original_list][4] = list(odt_list[i][1])

                        # Keep to zero if no penalty
                        odt_priority_list_original[index_in_original_list][5] = 0

                # If there is no path, it raises an error. Record the none path and add the penalty
                except nx.exception.NetworkXNoPath:
                    # Extract the odt to get the recorded path from the original
                    # priority list
                    extract_odt = [item for item in odt_priority_list_original
                                   if item[0:4] == odt[0][0:4]]

                    # Find the index on the original list
                    index_in_original_list = odt_priority_list_original.index(extract_odt[0])

                    odt_priority_list_original[index_in_original_list][4] = list(odt[1])
                    try:
                        odt_priority_list_original[index_in_original_list][5] = parameters.penalty_no_path
                    except IndexError:
                        odt_priority_list_original[index_in_original_list].append(parameters.penalty_no_path)
                # Next node
                i += 1
            # Next list
            m += 1
            print(f'Moving to next iteration: {m}')

        # Once it has no more list, compute the number of assigned and not assigned passengers based on the original
        except KeyError:
            assigned = 0
            unassigned = 0
            count_assigned = 0
            count_unassigned = 0
            for odt in odt_priority_list_original:
                if odt[5] == 0:
                    assigned += odt[3]
                    count_assigned += 1
                else:
                    unassigned += odt[3]
                    count_unassigned += 1
            print('End of passenger assignment')
            print(f'\nThe number of assigned passengers: {assigned} for a total of group: {count_assigned}')
            print(f'The number of unassigned passengers: {unassigned} for a total of group: {count_unassigned}')
            break

    return timetable_initial_graph, assigned, unassigned, odt_facing_capacity_dict_for_iteration, \
           odt_priority_list_original


def assignment_with_disruption(odt_priority_list_original, odt_facing_disruption, timetable_initial_graph, parameters):
    """
    method that iterates over the list of odt facing the trains with full capacity. It reassigns the passengers on the
    the timetable_initial_graph
    :param parameters:
    :param odt_priority_list_original:
    :param parameters: list of parameters from the main file
    :param odt_facing_disruption: list of odt facing train with full capacity
    :param timetable_initial_graph: multigraph with trains flow and odt assigned on the edges
    :return: timetable with assigned passengers, number of passenger assigned, number of passenger unassigned and
    dictionary of all the odt lists facing the capacity constraint
    """
    # Create a dictionary for the iterations
    odt_facing_capacity_dict_for_iteration = {0: copy.deepcopy(odt_facing_disruption)}

    # Start the loop for the passenger assignment with capacity constraint
    m = 0
    while True:
        try:
            # Select the list based on the iteration of m
            odt_list = odt_facing_capacity_dict_for_iteration[m]

            # remove the duplicates from the list
            odt_list = list(remove_the_duplicates(odt_list))

            # Set the count to zero for the loop
            i = 0

            # Assign the passengers based on the priority list
            for odt in odt_list:

                # Break the loop if reached the last odt to avoid index error
                if i == len(odt_list):
                    print('End of the passenger assignment')
                    break

                # Compute the shortest path with dijkstra
                try:
                    # First make sure that they won't use the path with closed tracks
                    timetable_initial_graph[odt[2][0]][odt[2][1]]['weight'] = parameters.weight_closed_tracks
                    _, p = shortest_path.single_source_dijkstra(timetable_initial_graph,
                                                                odt[1][-1],
                                                                odt[0][1],
                                                                cutoff=parameters.weight_closed_tracks - 1)
                    # Save the path, origin to destination with the new path
                    odt_list[i][1] = odt_list[i][1] + p[1:]

                    # Assign the flow on the timetable graph's edges starting from the new path only
                    for j in range(len(p) - 1):
                        try:
                            # Check the train capacity on the next edge
                            if sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) + odt[0][3] > \
                                    parameters.train_capacity:
                                try:
                                    # Check if the current passenger is already seated in the train
                                    if p[j - 1][2] == p[j][2]:
                                        # Initialize the parameters for the capacity constraint checks
                                        k = 1
                                        # todo: change with lower priority to boarding at current node
                                        odt_with_lower_priority_name = []
                                        odt_with_lower_priority_flow = []
                                        odt_with_lower_priority_index = []

                                        # Remove assigned odt with lower priority on the edge
                                        while sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) + odt[0][3] > \
                                                parameters.train_capacity:
                                            try:
                                                # Check if the assigned odt is already seated in the train, if so, go to
                                                # next assigned odt
                                                if timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k] in \
                                                        timetable_initial_graph[p[j - 1]][p[j]]['odt_assigned']:
                                                    k += 1
                                                # If not assigned in the previous edge, hence the assigned passenger
                                                # must be from another train
                                                else:
                                                    # Save the assigned
                                                    odt_with_lower_priority_name.append(
                                                        timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k])
                                                    odt_with_lower_priority_flow.append(
                                                        timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'][-k][3])
                                                    odt_with_lower_priority_index.append(-k)

                                                    # Check if removing the assigned odt from the train is enough, if
                                                    # not, need to add another assigned from the list
                                                    if parameters.train_capacity >= \
                                                        sum(timetable_initial_graph[p[j]][p[j + 1]]['flow']) - \
                                                            sum(odt_with_lower_priority_flow) + odt[0][3]:
                                                        for odt_with_lower_priority in odt_with_lower_priority_name:

                                                            # Extract the odt to get the recorded path from the original
                                                            # priority list
                                                            extract_odt = [item for item in odt_priority_list_original
                                                                           if item[0:4] == odt_with_lower_priority[0:4]]

                                                            # Find the index on the original list
                                                            index_in_original_list = odt_priority_list_original.index(
                                                                extract_odt[0])

                                                            # Verify the type of extract_odt between odt original format
                                                            # or
                                                            # Get the path from the original list
                                                            extract_odt_path = extract_odt[0][4]

                                                            # Get the index of the last node before the full capacity
                                                            # train
                                                            index_last_node_on_path_before_capacity = \
                                                                extract_odt_path.index(p[j])

                                                            # Split the path into delete and keep path
                                                            odt_path_to_delete = \
                                                                extract_odt_path[
                                                                index_last_node_on_path_before_capacity:]

                                                            odt_path_to_keep = extract_odt_path[
                                                                               :index_last_node_on_path_before_capacity]

                                                            # Modify the original path and erase the length, needs to be
                                                            # recomputed
                                                            try:
                                                                odt_priority_list_original[
                                                                    index_in_original_list][4] = list(odt_path_to_keep)
                                                                odt_priority_list_original[
                                                                    index_in_original_list][5] = 0
                                                            except ValueError:
                                                                # in order to stay inside the lines for code writing
                                                                message = \
                                                                    odt_priority_list_original[
                                                                        index_in_original_list][0:2]
                                                                print(f'{message} at index {index_in_original_list} '
                                                                      f'has already a changed value but it was not'
                                                                      f' recorded properly. Please check passenger '
                                                                      f'assignment')

                                                            # Delete the flow and the odt_assigned
                                                            for n in range(len(odt_path_to_delete) - 1):
                                                                try:
                                                                    index_to_delete = timetable_initial_graph[
                                                                        odt_path_to_delete[n]][
                                                                        odt_path_to_delete[n + 1]][
                                                                        'odt_assigned'].index(odt_with_lower_priority)
                                                                    del timetable_initial_graph[odt_path_to_delete[n]][
                                                                        odt_path_to_delete[n + 1]]['flow'][
                                                                        index_to_delete]
                                                                    del timetable_initial_graph[odt_path_to_delete[n]][
                                                                        odt_path_to_delete[n + 1]]['odt_assigned'][
                                                                        index_to_delete]
                                                                except (KeyError, ValueError):
                                                                    # KeyError means it is a transfer edge where there
                                                                    # is no flow or odt_assigned. ValueError can be
                                                                    # already removed from the edge. How? good question.
                                                                    continue

                                                            # Need to check if the boarding odt is reaching its maximum
                                                            # iteration before assigning it to the next list. To do so
                                                            # We first need to check if it has been already on the
                                                            # capacity list to check the number of iteration
                                                            if any(item[0] == odt_with_lower_priority
                                                                   for item in odt_list):
                                                                number_of_iteration = \
                                                                    [item[-1] for item in odt_list
                                                                     if item[0] == odt_with_lower_priority]
                                                            else:
                                                                number_of_iteration = [0]

                                                            # If the maximum number is reached, the passenger leave the
                                                            # system from the last point on their trip
                                                            if number_of_iteration[0] + 1 > \
                                                                    parameters.max_iteration_recompute_path:
                                                                odt_priority_list_original[
                                                                    index_in_original_list][4] = list(odt_path_to_keep)
                                                                odt_priority_list_original[
                                                                    index_in_original_list][5] = \
                                                                    parameters.penalty_no_path
                                                            else:
                                                                try:
                                                                    # If they have not reached their maximum number of
                                                                    # recalculation of their trip, put the current odt
                                                                    # in the next list
                                                                    odt_new_list = \
                                                                        odt_facing_capacity_dict_for_iteration[m+1]
                                                                    odt_info = [odt_with_lower_priority,  # ODT name
                                                                                odt_path_to_keep,  # ODT path keep
                                                                                odt_path_to_delete[0:2],    # Edge full
                                                                                number_of_iteration[0]+1]
                                                                    odt_facing_capacity_dict_for_iteration[
                                                                        m + 1].append(odt_info)
                                                                except KeyError:
                                                                    odt_facing_capacity_dict_for_iteration[m+1] =\
                                                                        list([[odt_with_lower_priority,
                                                                               odt_path_to_keep,
                                                                               odt_path_to_delete[0:2],
                                                                               number_of_iteration[0]+1]])

                                                            # Check if the odt_with_lower priority is in the odt facing
                                                            # capacity constraint list. If it does, need to delete it
                                                            try:
                                                                extract_odt_facing_capacity_constraint = \
                                                                    [item for item in odt_list[i:]
                                                                     if item[0][0:4] == odt_with_lower_priority[0:4]]

                                                                index_in_odt_list = odt_list.index(
                                                                    extract_odt_facing_capacity_constraint[0])

                                                                del odt_list[index_in_odt_list]

                                                            # If Value Error, it means it is not in the list and we can
                                                            # continue
                                                            except (ValueError, IndexError):
                                                                pass

                                                        # Finally, add the current odt on the clean edge
                                                        timetable_initial_graph[p[j]][p[j + 1]][
                                                            'flow'].append(odt[0][3])
                                                        timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'].append(
                                                            odt[0])
                                                        # Done with the recording of oft facing capacity constraint
                                                        break
                                                    # Not enough seats released, need at least one more group to leave
                                                    else:
                                                        k += 1
                                            # Not suppose to happen, but it might if there an assignment mistake
                                            except IndexError:
                                                print(
                                                    f'Train is at full capacity and the current odt {odt[0]} is already'
                                                    f' seated, but the algorithm cannot find the assigned odt that is'
                                                    f' assigned but not seated in the train.')
                                                break

                                    # It means, that the next train is at full capacity. Hence, the current odt journey
                                    # needs to be computed from here in the next list
                                    else:

                                        # First, we need to check if the odt has reached the limit of recomputing path
                                        if odt[3] + 1 > parameters.max_iteration_recompute_path:
                                            # Extract the odt to get the recorded path from the original
                                            # priority list
                                            extract_odt = [item for item in odt_priority_list_original
                                                           if item[0:4] == odt[0][0:4]]

                                            # Find the index on the original list
                                            index_in_original_list = odt_priority_list_original.index(
                                                extract_odt[0])

                                            # Update the original with the penalty
                                            odt_priority_list_original[index_in_original_list][4] = \
                                                list(odt_list[i][1][:-(len(p)-j)])  # Keep the assigned path
                                            odt_priority_list_original[index_in_original_list][5] = \
                                                parameters.penalty_no_path

                                            # Do not need to go further, next odt please.
                                            break

                                        else:
                                            try:
                                                # If they have not reached their maximum number of
                                                # recalculation of their trip, put the current odt
                                                # in the next list
                                                odt_new_list = \
                                                    odt_facing_capacity_dict_for_iteration[m + 1]
                                                odt_info = [odt[0],                         # ODT name
                                                            odt_list[i][1][:-(len(p)-j)],  # ODT path to keep
                                                            [p[j], p[j + 1]],               # Edge full
                                                            odt[3]+1]                       # Number of iteration
                                                odt_facing_capacity_dict_for_iteration[m + 1].append(odt_info)
                                            except KeyError:
                                                odt_facing_capacity_dict_for_iteration[m + 1] = \
                                                    list([[odt[0],
                                                           odt_list[i][1][:-(len(p)-j)],
                                                           [p[j], p[j + 1]],
                                                           odt[3]+1]])

                                        # Done for this odt, do not need to continue to assign further. go to the next
                                        # one but before need to assign the path to the original list
                                        # Find the index on the original list
                                        # Get the original length
                                        length_on_original_path = len(odt[1]) - len(p[1:])

                                        # Transform the odt to have the same format
                                        odt_with_original_format = [odt[0][0],
                                                                    odt[0][1],
                                                                    odt[0][2],
                                                                    odt[0][3],
                                                                    list(odt[1][:length_on_original_path]),
                                                                    0]

                                        index_in_original_list = \
                                            odt_priority_list_original.index(odt_with_original_format)

                                        # Update the new path and set the value to 0. In case the previous iteration,
                                        # the odt was consider with a penalty but now finds a way
                                        odt_priority_list_original[index_in_original_list][4] = \
                                            list(odt_list[i][1][:-(len(p)-j)])
                                        odt_priority_list_original[index_in_original_list][5] = 0

                                        # Do not need to go further. Next odt please.
                                        break

                                # It means that the previous edge is home to the first station,
                                # hence the passenger is not seated in the train
                                except IndexError:
                                    # First, we need to check if the odt has reached the limit of recomputing path
                                    if odt[3] + 1 > parameters.max_iteration_recompute_path:
                                        # Extract the odt to get the recorded path from the original
                                        # priority list
                                        extract_odt = [item for item in odt_priority_list_original
                                                       if item[0:4] == odt[0][0:4]]

                                        # Find the index on the original list
                                        index_in_original_list = odt_priority_list_original.index(
                                            extract_odt[0])

                                        # Update the original with the penalty
                                        odt_priority_list_original[index_in_original_list][4] = \
                                            list(odt_list[i][1][:-(len(p) - j)])  # Keep the assigned path
                                        odt_priority_list_original[index_in_original_list][5] = \
                                            parameters.penalty_no_path

                                        # Do not need to go further. Next odt please.
                                        break

                                    else:
                                        # Need to check if the boarding odt is reaching its maximum
                                        # iteration before assigning it to the next list. To do so
                                        # We first need to check if it has been already on the
                                        # capacity list to check the number of iteration
                                        if any(item[0] == odt_with_lower_priority
                                               for item in odt_list):
                                            number_of_iteration = \
                                                [item[-1] for item in odt_list
                                                 if item[0] == odt_with_lower_priority]
                                        else:
                                            number_of_iteration = [0]

                                        try:
                                            # If they have not reached their maximum number of
                                            # recalculation of their trip, put the current odt
                                            # in the next list
                                            odt_new_list = \
                                                odt_facing_capacity_dict_for_iteration[m + 1]
                                            odt_info = [odt[0],  # ODT name
                                                        odt_list[i][1][:-(len(p) - j)],  # ODT path to keep
                                                        [p[j], p[j + 1]],  # Edge full
                                                        number_of_iteration[0]+1]  # Number of iteration
                                            odt_facing_capacity_dict_for_iteration[m + 1].append(odt_info)
                                        except KeyError:
                                            odt_facing_capacity_dict_for_iteration[m + 1] = \
                                                [[odt[0],
                                                 odt_list[i][1][:-(len(p) - j)],
                                                 [p[j], p[j + 1]],
                                                 number_of_iteration[0]+1]]

                                    # Done for this odt, do not need to continue to assign further. go to the next one
                                    # But before need to assign the path to the original list
                                    # Find the index on the original list
                                    index_in_original_list = odt_priority_list_original.index(odt)

                                    # Update the new path
                                    odt_priority_list_original[index_in_original_list][4] = list(p[:j])
                                    odt_priority_list_original[index_in_original_list][5] = 0

                                    # Do not need to go further. Next odt please.
                                    break
                            else:
                                # Assign the current odt to the edge for the flow (group size) and odt_assigned with
                                # the name.
                                timetable_initial_graph[p[j]][p[j + 1]]['flow'].append(odt[0][3])
                                timetable_initial_graph[p[j]][p[j + 1]]['odt_assigned'].append(odt[0])

                        # If there is a key error, it means it is either a home-station edge, station-destination edge
                        # or a transfer, hence we go check the next node
                        except KeyError:
                            pass

                    # Once all the path is assigned with the current odt, update the original list with the new path
                    if j == (len(p) - 2):
                        # Update the odt info on the original list
                        extract_odt = [item for item in odt_priority_list_original if item[0:4] == odt[0][0:4]]

                        # Find the index on the original list
                        index_in_original_list = odt_priority_list_original.index(
                            extract_odt[0])

                        # Update the original odt with the new path
                        odt_priority_list_original[index_in_original_list][4] = list(odt_list[i][1])

                        # Keep to zero if no penalty
                        odt_priority_list_original[index_in_original_list][5] = 0

                # If there is no path, it raises an error. Record the none path and add the penalty
                except nx.exception.NetworkXNoPath:
                    # Extract the odt to get the recorded path from the original
                    # priority list
                    extract_odt = [item for item in odt_priority_list_original
                                   if item[0:4] == odt[0][0:4]]

                    # Find the index on the original list
                    index_in_original_list = odt_priority_list_original.index(extract_odt[0])

                    odt_priority_list_original[index_in_original_list][4] = list(odt[1])
                    try:
                        odt_priority_list_original[index_in_original_list][5] = parameters.penalty_no_path
                    except IndexError:
                        odt_priority_list_original[index_in_original_list].append(parameters.penalty_no_path)
                # Next node
                i += 1
            # Next list
            m += 1
            print(f'Moving to next iteration: {m}')

        # Once it has no more list, compute the number of assigned and not assigned passengers based on the original
        except KeyError:
            assigned = 0
            unassigned = 0
            count_assigned = 0
            count_unassigned = 0
            for odt in odt_priority_list_original:
                if odt[5] == 0:
                    assigned += odt[3]
                    count_assigned += 1
                else:
                    unassigned += odt[3]
                    count_unassigned += 1
            print('End of passenger assignment')
            print(f'\nThe number of assigned passengers: {assigned} for a total of group: {count_assigned}')
            print(f'The number of unassigned passengers: {unassigned} for a total of group: {count_unassigned}')
            break

    return timetable_initial_graph, assigned, unassigned, odt_facing_capacity_dict_for_iteration, \
           odt_priority_list_original

# %% Remove the duplicates


def remove_the_duplicates(odt_facing_capacity_constraint):
    """
    method that removes the duplicates on the list while making sure to keep only the odt with the shortest trip due to
    the fact it has faced the capacity constraint.
    :param odt_facing_capacity_constraint: list of odt that face a train with full capacity.
    :return: odt_facing_capacity_constraint: new list without duplicates.
    """
    # Set a dictionary that keeps the index from the previous list and the odt name. Set a list of all duplicates
    # indexes and set i to for the iteration and keeping track of the index from the previous list
    seen = {}  # Dict in order to keep the index from the list and the odt name for checking
    duplicates_to_delete = []
    i = 0

    # Loop through all the odt in the list
    for item in odt_facing_capacity_constraint:

        # If the item has been already seen in the list. One of them has faced a train with full capacity earlier. Need
        # to keep that one only
        if item[0][0:4] in seen.values():
            index_odt_list_in_seen = list(seen.keys())[list(seen.values()).index(item[0][0:4])]
            length_in_seen = len(odt_facing_capacity_constraint[index_odt_list_in_seen][1])
            length_current_item = len(item[1])

            # Check which one is the earliest one based on the length of their path. If the earliest is the item, we
            # Need to remove the other one from the seen list and replace it by the item. If not, save the index of the
            # other
            if length_in_seen > length_current_item:
                duplicates_to_delete.append(index_odt_list_in_seen)
                seen.pop(index_odt_list_in_seen)
                seen[i] = item[0][0:4]
            else:
                duplicates_to_delete.append(i)
        # It has not been seen, so add the item the seen dict.
        else:
            seen[i] = item[0][0:4]
        i += 1

    for index_to_delete in sorted(duplicates_to_delete, reverse=True):
        del odt_facing_capacity_constraint[index_to_delete]

    # Put in order of priority
    odt_facing_capacity_constraint.sort(key=lambda x: x[0][2], reverse=True)
    return odt_facing_capacity_constraint


def get_edges_on_closed_tracks(parameters, timetable_initial_graph):
    """
    Method that create a dictionary of nodes. Departure node as key and Arrival node as value.
    :param parameters: To get the path arrival nodes on closed track computed before in the main file
    :param timetable_initial_graph: directed graph that contains all the edges
    :return: Dictionary of departure and arrival nodes for every edge on closed tracks.
    """
    # Get the edges on the closed tracks
    edges_on_closed_tracks = {}

    # Loop through all the nodes on closed tracks and get the departure nodes accordingly with the arrival node
    for arrival in parameters.path_nodes_on_closed_track:
        departure_node = [pred for pred in timetable_initial_graph.predecessors(arrival)]
        edges_on_closed_tracks[departure_node[0]] = arrival

    return edges_on_closed_tracks


def create_list_odt_facing_disruption(edges_on_closed_tracks, timetable_initial_graph, odt_priority_list_original):
    """
    Method that creates a list of all the odt that are facing the disruption.
    :param edges_on_closed_tracks: list of all the edges on the closed tracks
    :param timetable_initial_graph: directed graph with all the edges including flow and odt assigned
    :param odt_priority_list_original: the first list of all the odt ordered by the priority rules
    :return: list of odt facing disruption
    """
    # Create the empty list
    odt_facing_disruption = []

    # Loop through all the edges on closed tracks
    for departure_node, arrival_node in edges_on_closed_tracks.items():

        # Loop through all the odt assigned on the closed track
        for current_odt in timetable_initial_graph[departure_node][arrival_node]['odt_assigned']:

            # Get the information from the first list
            extract_odt = [item for item in odt_priority_list_original if item[0:4] == current_odt]
            extract_odt_path = extract_odt[0][4]
            index_last_node_on_path_before_disruption = extract_odt_path.index(departure_node)
            odt_path_to_keep = extract_odt_path[:index_last_node_on_path_before_disruption]

            # Get the index from original list for future update
            index_in_original_list = odt_priority_list_original.index(extract_odt[0])

            # Delete the flow and the odt_assigned
            odt_path_to_delete = extract_odt_path[index_last_node_on_path_before_disruption:]
            for n in range(len(odt_path_to_delete) - 1):
                try:
                    index_to_delete = timetable_initial_graph[
                        odt_path_to_delete[n]][odt_path_to_delete[n + 1]]['odt_assigned'].index(current_odt)
                    del timetable_initial_graph[odt_path_to_delete[n]][odt_path_to_delete[n + 1]]['flow'][
                        index_to_delete]
                    del timetable_initial_graph[odt_path_to_delete[n]][odt_path_to_delete[n + 1]]['odt_assigned'][
                        index_to_delete]
                except (KeyError, ValueError):
                    # KeyError means it is a transfer edge where there
                    # is no flow or odt_assigned. ValueError can be
                    # already removed from the edge. How? good question.
                    continue

            # Check number of iteration from the previous odt_facing_capacity
            number_iteration = 0
            # todo: takes too much time for little results...
            # for _, odt_list_capacity in sorted(odt_facing_capacity_dict_for_iteration.items(), reverse=True):
            #     if any(item[0][0] == extract_odt[0][0:4] for item in odt_list_capacity):
            #         number_iteration = [item[0][3] for item in odt_list_capacity if item[0][0] == extract_odt[0][0:4]]
            #         break

            # Transform the odt on the odt facing disruption format
            odt_facing_format = [extract_odt[0][0:4],
                                 list(odt_path_to_keep),
                                 [departure_node, arrival_node],
                                 number_iteration + 1]

            odt_facing_disruption.append(odt_facing_format)

            # Update the original list with the new path and set to 0 if it was the penalty value
            odt_priority_list_original[index_in_original_list][4] = list(odt_path_to_keep)
            odt_priority_list_original[index_in_original_list][5] = 0

    return odt_facing_disruption
