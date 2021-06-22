"""
Created on Mon Mar 01 2021

@author: BenMobility

To easily test code lines with Viriato

"""
import numpy as np
import shortest_path
import copy
import passenger_assignment
import helpers
import alns_platform
import networkx as nx
import matplotlib.pyplot as plt

solution_archive = np.load('output/pickle/solution_archive.pkl', allow_pickle=True)
total_distance_overall = [item.total_dist_train for item in solution_archive]
most_distance_index = np.argmax(total_distance_overall)
least_distance_index = np.argmin(total_distance_overall)
total_travel_time_overall = [item.total_traveltime for item in solution_archive]
most_travel_time_index = np.argmax(total_travel_time_overall)
least_travel_time_index = np.argmin(total_travel_time_overall)

solution_worst_operation = solution_archive[most_distance_index]
solution_worst_travel_time = solution_archive[most_travel_time_index]
solution_least_operation = solution_archive[least_distance_index]
solution_least_travel_time = solution_archive[least_travel_time_index]

# assigned, unassigned = helpers.compute_assigned_not_assigned(odt_priority_list_original)
