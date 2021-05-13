"""
Created on Sun May  09 2021

@author: BenMobility

Passenger assignment with capacity constraint
"""
import networkx as nx
import shortest_path


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
                                                                   if item[0:2] == odt_with_lower_priority[0:2]]

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
                                                        odt_facing_capacity_constrain = [odt_with_lower_priority,
                                                                                         odt_path_to_keep,
                                                                                         odt_path_to_delete[0:2],
                                                                                         1]

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
# %% Sort the list in descending manner with the level of importance

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

    return odt_facing_capacity_constraint
