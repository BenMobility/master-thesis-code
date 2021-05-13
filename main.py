"""
Created on Mon Jan  25 2021

@author: BenMobility

Main code for repair scheduling in railways
"""
# %% Imports
import infrastructure_graph
import viriato_interface
import helpers
import timetable_graph
import shortest_path
import alns_platform
import passenger_assignment
import numpy as np

# Viriato
base_url = 'http://localhost:8080'  # Viriato localhost
random_seed = 42
np.random.seed(random_seed)  # Random seed for the main code

# %% filter, alns
filter_passengers = False
start_alns = False
debug_mode_passenger = False
debug_mode_train = False

if debug_mode_passenger:
    print('\nDebug mode for passenger is on.')
if debug_mode_train:
    print('\nDebug mode for train is on.')

# %% Parameters initialization

# Parameters
th_zone_selection = 8000  # Threshold of the zone selection in meters [0]
nb_zones_to_connect = 1  # Number of zones to connect in total [1]
nb_stations_to_connect = 2  # Number of stations to connect (in euclidean) [2]
min_nb_passenger = 0  # Minimum number of passenger [3]
min_transfer_time = 4  # Minimum transfer time (m) in minutes [4]
max_transfer_time = 15  # Maximum transfer time (M) in minutes [5]
min_transfer_time_bus = 1  # Minimum transfer time in minutes for emergency buses [6]
max_transfer_time_bus = 20  # Maximum transfer time in minutes for emergency buses [7]
origin_train_departure_min_time = 2  # Minimum waiting time in minutes for the origin train departure [8]
origin_train_departure_max_time = 20  # Maximum waiting time in minutes for the origin train departure [9]
beta_transfer = 10  # Beta for transfer edges minutes [10]
beta_waiting = 2.5  # Beta for waiting edges min/min [11]
penalty_edge = 1000  # Penalty edge for not assigned passengers in the system [12] todo: change for something else
score_1 = 10  # Score for the ALNS algorithm, accepted solution. [13]
score_2 = 2  # Score for the ALNS algorithm, accepted solution, yet worsened solution. [14]
score_3 = 0  # Score for the ALNS algorithm, rejected solution. [15]
t_start_op = 10 ** 6  # Starting temperature for Simulated Annealing on the operation objective [16]
t_start_de = 10 ** 6  # Starting temperature for Simulated Annealing on the deviation objective [17]
t_start_tt = 10 ** 6  # Starting temperature for Simulated Annealing on the travel time objective [18]
weight_closed_tracks = 10e9  # Weight for closed tracks edges [19]
train_capacity = 500  # Number of passenger per train [20]
bus_capacity = 100  # Number of passenger per bus in Zurich [21]
penalty_no_path = 9000  # Penalty of no assignment equals duration in minutes [22] todo: define better
delayTime_to_consider_cancel = 30  # Delay time to consider for full cancel in minutes [23]
delayTime_to_consider_partCancel = 10  # Delay time to consider for part cancel in minutes [24]
commercial_stops = 1  # Threshold for a train to be considered in the timetable [25]
time_discretization = 10  # Time steps for the passenger grouping [26]
group_size_passenger = 80  # Size of each passenger group (number of passengers) [27]

# Save/Read pickle
save_pickle = True  # Save the output in a pickle file
read_pickle = True  # Read the input pickle files

# Create home connection from scratch
create_timetable_home_connections_from_scratch = False  # True, if you want to connect stations to homes

# Passenger assignment
full_od_file = False  # True, if you want to read to full od file ~287,000 KB [28]
read_od_departure_time_file = True  # False, if you want to make it from scratch [29]
create_group_passengers = True  # True, if you want to group the passengers for the assignment [30]
max_iteration_recompute_path = 3  # The limit of recomputing the passenger path [31]
read_selected_zones_demand_and_travel_time = True  # True, if you want to read csv files of demand and travel time [32]
capacity_constraint = False  # False, if capacity constraint is included or not in the passenger assign. [53]
assign_passenger = True  # True, makes assign passenger during the shortest path algorithm in ALNS [38]

#  ALNS Neighbourhood search
number_iteration = 10  # Number of iteration for the ALNS [33]
number_iteration_archive = 100  # Number of iteration before archiving the solution, binder use 2000 [34]
delay_options = [5, 10, 15, 20]  # The options for how long delaying the train for delay operator [35]
time_delta_delayed_bus = 10  # Time added from the delayed departure time of the bus to compute its arrival time [36]
min_headway = 120  # Minimum headway between two consecutive trains (in seconds) for succeeding and preceding head [37]
deviation_penalty_cancel = 50  # Penalty for deviated the initial timetable with a cancellation [39]
deviation_penalty_delay = 1  # Penalty for deviated the initial timetable with a delay [40]
deviation_penalty_emergency = 1000  # Penalty for deviated the initial timetable with an emergency [41]
deviation_penalty_bus = deviation_penalty_emergency / 10  # Penalty for deviated the initial timetable with a bus [42]
deviation_penalty_rerouted = 10  # Penalty for deviated the initial timetable with a rerouted train [43]

# Acceptance or rejection solutions
reaction_factor_operation = 1  # Factor for acceptance probability of operation objective [44]
reaction_factor_deviation = 1  # Factor for acceptance probability of deviation objective [45]
warm_up_phase = 100  # Initial phase for the update temperature in the simulated annealing [46]
iterations_temperature_level = 20  # Iterations for changing temperature level [47]
number_of_temperature_change = 60  # Max iterations for the temperature change [48]
reaction_factor_return_archive = 0.9  # Factor to return into the archive to select a solution [49]
reaction_factor_weights = 0.5  # Factor to update the weights for each operator [50]

# Greedy feasibility check train insertion
max_iteration_feasibility_check = 15  # Maximum number of iterations for the greedy feasibility check [51]
max_iteration_section_check = 3  # Maximum number of iterations for the section feasibility check [52]

# Save all the parameters in a list to store after in a class
list_parameters = [th_zone_selection, nb_zones_to_connect, nb_stations_to_connect, min_nb_passenger,
                   min_transfer_time, max_transfer_time, min_transfer_time_bus, max_transfer_time_bus,
                   origin_train_departure_min_time, origin_train_departure_max_time, beta_transfer, beta_waiting,
                   penalty_edge, score_1, score_2, score_3, t_start_op, t_start_de, t_start_tt,
                   weight_closed_tracks, train_capacity, bus_capacity, penalty_no_path,
                   delayTime_to_consider_cancel, delayTime_to_consider_partCancel, commercial_stops,
                   time_discretization, group_size_passenger, full_od_file, read_od_departure_time_file,
                   create_group_passengers, max_iteration_recompute_path, read_selected_zones_demand_and_travel_time,
                   number_iteration, number_iteration_archive, delay_options, time_delta_delayed_bus, min_headway,
                   assign_passenger, deviation_penalty_cancel, deviation_penalty_delay, deviation_penalty_emergency,
                   deviation_penalty_bus, deviation_penalty_rerouted, reaction_factor_operation,
                   reaction_factor_deviation, warm_up_phase, iterations_temperature_level,
                   number_of_temperature_change, reaction_factor_return_archive, reaction_factor_weights,
                   max_iteration_feasibility_check, max_iteration_section_check, capacity_constraint]

# %% Time window from Viriato and close tracks ids from disruption scenario
print('\nMain code is running.')
print(f'Random seed used is: {random_seed}')
time_window = viriato_interface.get_time_window()
print(f'The time window for this experiment is: {time_window.from_time} to {time_window.to_time}.')
closed_track_ids = viriato_interface.get_section_track_closure_ids(time_window)

# %% Infrastructure graph and load parameters
print('\nBuilding the infrastructure graph.')
infra_graph, sbb_nodes, nodes_code, id_nodes = infrastructure_graph.build_infrastructure_graph(time_window, save_pickle)
print('Infrastructure graph done!')
parameters = helpers.Parameters(infra_graph, time_window, closed_track_ids, list_parameters)

# %% Original timetable graph
print('\nCreate the trains timetable.')
trains_timetable = timetable_graph.get_trains_timetable(time_window, sbb_nodes, parameters, debug_mode_train)
print('Trains timetable done!')

# Get all the path nodes on the closed track for the passenger assignments
print('keep track of the path nodes on the closed tracks.')
track_info = alns_platform.TrackInformation(trains_timetable, closed_track_ids)
path_nodes_on_closed_track = []
for train_on_closed_track in track_info.trains_on_closed_tracks:
    for node in train_on_closed_track.train_path_nodes:
        if any([node.section_track_id == c for c in closed_track_ids]):
            path_nodes_on_closed_track.append((node.node_id, node.arrival_time.strftime("%Y-%m-%dT%H:%M:%S"), node.id,
                                               'a'))

# Timetable with waiting edges and transfer edges
print('\nCreate timetable with waiting edges and transfer edges.')
timetable_waiting_transfer, stations_with_commercial_stop = \
    timetable_graph.get_timetable_with_waiting_transfer_edges(trains_timetable, parameters)
print('Done!')

# Update the sbb nodes with the attribute of all the station with commercial stops
print('\nAdd commercial stops attributes.')
sbb_nodes = helpers.add_field_com_stop(sbb_nodes, stations_with_commercial_stop)
print('Done.')
parameters.stations_comm_stop = stations_with_commercial_stop

# Upload the data from selected zones, demand and travel time zone to zone
print('\nReading files with demand zones and travel time.')
selected_zones, demand_selected_zones, travel_time_selected_zones = helpers.reading_demand_zones_travel_time(sbb_nodes,
                                                                                                             parameters)

# Create the OD matrix with the desired departure time for the passengers
print('\nGet the desired departure time for all the passengers.')
od_departure_time = helpers.get_od_departure_time(parameters, demand_selected_zones)

# Connect the train stations with all the passenger homes
print('\nConnect the train stations nodes with all the passengers homes.')

# Connect home of the passenger to the stations (similar to the transfer connection)
timetable_initial_graph, odt_list, odt_by_origin, odt_by_dest, station_candidates, origin_name_desired_dep_time, \
origin_name_zone_dict = timetable_graph.connect_home_stations(timetable_waiting_transfer, sbb_nodes,
                                                              travel_time_selected_zones, selected_zones, id_nodes,
                                                              od_departure_time, parameters,
                                                              create_timetable_home_connections_from_scratch)
print('The timetable initial graph is done!')

# Zone candidates
print('\n Define the zone candidates.')
zone_candidates = helpers.create_zone_candidates_of_stations(station_candidates)
print('Done.')

# Store in parameters, because of the high computational time
parameters.station_candidates = station_candidates
parameters.zone_candidates = zone_candidates
parameters.odt_by_origin = odt_by_origin
parameters.odt_by_destination = odt_by_dest
parameters.origin_name_desired_dep_time = origin_name_desired_dep_time
parameters.path_nodes_on_closed_track = path_nodes_on_closed_track

# In order to have a little computational time
if debug_mode_passenger:
    odt_list = odt_list[0:1000]

# Add the path column in the odt_list
[x.append([]) for x in odt_list]
parameters.odt_as_list = odt_list

# Assign the passenger on the timetable graph
odt_facing_capacity_constraint, parameters, timetable_initial_graph = passenger_assignment.capacity_constraint_1st_loop(
    parameters, timetable_initial_graph)

timetable_initial_graph, assigned, unassigned, odt_facing_capacity_dict_for_iteration = \
    passenger_assignment.capacity_constraint_2nd_loop(parameters, odt_facing_capacity_constraint,
                                                      timetable_initial_graph)

# And save the output of the first list of odt facing capacity constraint
alns_platform.pickle_results(odt_facing_capacity_constraint, 'output/pickle/odt_facing_capacity_constraint.pkl')
alns_platform.pickle_results(odt_facing_capacity_dict_for_iteration,
                             'output/pickle/odt_facing_capacity_dict_facing_capacity_constraint.pkl')
alns_platform.pickle_results(parameters, 'output/pickle/parameters_with_first_assignment_done.pkl')
alns_platform.pickle_results(timetable_initial_graph, 'output/pickle/timetable_with_first_assignment_done.pkl')

# todo: remove filter the passengers
# Filter the passengers
if filter_passengers:
    shortest_path.find_path_for_all_passengers_and_remove_unserved_demand(timetable_initial_graph, odt_list,
                                                                          origin_name_desired_dep_time,
                                                                          origin_name_zone_dict,
                                                                          parameters)

# Record the variables needed for ALNS into pickle files
alns_platform.pickle_results(timetable_initial_graph, 'output/pickle/timetable_initial_graph_for_alns.pkl')
alns_platform.pickle_results(infra_graph, 'output/pickle/infra_graph_for_alns.pkl')
alns_platform.pickle_results(trains_timetable, 'output/pickle/trains_timetable_for_alns.pkl')
alns_platform.pickle_results(parameters, 'output/pickle/parameters_for_alns.pkl')

# %% Run the ALNS
if start_alns:
    set_solutions = alns_platform.start(timetable_initial_graph, infra_graph, trains_timetable, parameters)

# %% Print the results
if start_alns:
    picked_solution = helpers.pick_best_solution(set_solutions)
    # todo: Store the solution to viriato
    print('end of algorithm  \n total running time in [sec] : see profiler')
    # todo: I want to make sure the code follow each steps. and to compute time effort for each one of them
