import algorithm_platform_methods as ap
import datetime
import graphcreator


class Edges_origin_station_destionation:
    def __init__(self, G, max_wait, min_wait, parameters):
        self.name = 'edges_o_stations_d'

        return_edges_nodes = True

        origin_nodes, origin_nodes_attributes, destination_nodes, destination_nodes_attributes, edges_o_stations, \
            edges_o_stations_attr, edges_stations_d, edges_stations_d_attr, edges_o_stations_dict, edges_stations_d_dict = \
            graphcreator.generate_edges_origin_station_destination(G, max_wait, min_wait, parameters, True)

        # self.origin_nodes = origin_nodes
        # self.origin_nodes_attributes = origin_nodes_attributes
        # self.destination_nodes = destination_nodes
        # self.destination_nodes_attributes = destination_nodes_attributes
        self.edges_o_stations = edges_o_stations
        # self.edges_o_stations_attr = edges_o_stations_attr
        self.edges_stations_d = edges_stations_d
        # self.edges_stations_d_attr = edges_stations_d_attr
        self.edges_o_stations_dict = edges_o_stations_dict  # key origin, value edges connecting to train nodes
        self.edges_stations_d_dict = edges_stations_d_dict  # key destination, value edges connecting to


class Copy_edges_origin_station_destionation:
    def __init__(self, edges_o_stations_d):
        self.name = 'edges_o_stations_d'

        # self.origin_nodes = origin_nodes
        # self.origin_nodes_attributes = origin_nodes_attributes
        # self.destination_nodes = destination_nodes
        # self.destination_nodes_attributes = destination_nodes_attributes
        copy_edges = []
        for edge in edges_o_stations_d.edges_o_stations:
            copy_edges.append(edge.copy())
        self.edges_o_stations = copy_edges

        copy_edges = []
        for edge in edges_o_stations_d.edges_stations_d:
            copy_edges.append(edge.copy())
        self.edges_stations_d = copy_edges

        copy_dict = {}
        for origin, transfers in edges_o_stations_d.edges_o_stations_dict.items():
            copy_dict[origin] = transfers.copy()
        self.edges_o_stations_dict = copy_dict  # key origin, value edges connecting to train nodes

        copy_dict = {}
        for destination, transfers in edges_o_stations_d.edges_stations_d_dict.items():
            copy_dict[destination] = transfers.copy()
        self.edges_stations_d_dict = copy_dict  # key destination, value edges connecting to


# store the value of the parameters
class Parameters:
    def __init__(self, G_infra):
        self.name = 'parameters'
        time_window = ap.get_parameter('ReferenceTimeWindow')
        # Format = "%Y-%m-%dT%H:%M"
        time_window['fmt'] = "%Y-%m-%dT%H:%M:%S"
        time_window['datetime'] = {'FromTime': datetime.datetime.strptime(time_window['FromTime'], time_window['fmt']),
                                   'ToTime': datetime.datetime.strptime(time_window['ToTime'], time_window['fmt'])}
        time_window['duration'] = time_window['datetime']['ToTime'] - time_window['datetime']['FromTime']
        self.time_window = time_window

        self.closed_tracks = get_all_closed_tracks(possessions=ap.get_possessions_section_track_closure(time_window['FromTime'], time_window['ToTime']))
        self.trains_on_closed_track_initial_timetable_infeasible = None

        self.initial_timetable_infeasible = None
        self.initial_timetable_feasible = None
        self.train_ids_initial_timetable_infeasible = None

        self.score_1 = 10
        self.score_2 = 2
        self.score_3 = 0

        t_i = 10**6
        self.t_start = [t_i, t_i, t_i]  # start temperature [z_op, z_de, z_tt]

        self.weight_closed_tracks = 10e9
        self.train_capacity = 500
        self.bus_capacity = 100  # bus zurich
        self.penalty_no_path = int(time_window['duration'].seconds/60)  # the time window of three hours in minutes

        # parameters for graph creation
        # transfer edges Binder 17 efficient m = 4, M = 15
        self.transfer_m = datetime.timedelta(minutes=4)
        self.transfer_M = datetime.timedelta(minutes=15)
        # transfer edges emergency bus
        self.transfer_mBus = datetime.timedelta(minutes=1)
        self.transfer_MBus = datetime.timedelta(minutes=20)

        # origin train departure waiting times
        self.min_wait = datetime.timedelta(minutes=2)
        self.max_wait = datetime.timedelta(minutes=20)

        # home connections
        self.station_candidates = None
        self.zone_candidates = None
        self.stations_in_area = [n for n in G_infra.nodes if G_infra.nodes[n]['in_area']]
        self.stations_comm_stop = None
        self.odt_by_origin = None  # list which is the input to sp algorithm
        self.odt_by_destination = None  # dictionary with odt and number of passengers, key 1st level origin zone, 2nd level origin departure node name (e.g. '1_06:15')
        self.odt_as_list = None
        self.origin_name_desired_dep_time = None

        self.set_of_trains_for_operator = {'Cancel': [], 'CancelFrom': [], 'Delay': [], 'DelayFrom': []}
        self.delayTime_to_consider_cancel = datetime.timedelta(minutes=30)
        self.delayTime_to_consider_partCancel = datetime.timedelta(minutes=10)


def get_all_closed_tracks(possessions):
    closed_SectionTrackIDs = []
    for track in possessions:
        if track['SectionTrackID'] not in closed_SectionTrackIDs:
            closed_SectionTrackIDs.append(track['SectionTrackID'])
        else:
            continue
    return closed_SectionTrackIDs


class Timetables:
    def __init__(self):
        self.name = 'initial_timetables'

        self.initial_timetable_infeasible = None
        self.initial_timetable_feasible = None


class Solutions:
    def __init__(self):
        self.name = 'solutions'

        self.set_of_trains_for_operator = None
        self.time_table = None
        self.total_traveltime = None
        self.total_dist_train = None
        self.deviation_timetable = None
        self.graph = None
        self.changed_trains = None
        self.track_info = None
        self.edges_o_stations_d = None
        self.set_of_trains_for_operator = {}


class Weights:
    def __init__(self):
        self.name = 'weight'

        # self.rr = 1  # Rerouting
        self.cc = 1  # complete cancel
        self.pc = 1  # partial cancel
        self.cd = 1  # complete delay
        self.pd = 1  # partial delay
        self.et = 1  # emergency train
        self.eb = 1  # emergency bus
        self.ret = 1  # return to initial train

        self.sum = self.cc + self.pc + self.cd + self.pd + self.et + self.eb + self.ret  # self.rr


class Scores:
    def __init__(self):
        self.name = 'score'
        # self.rr = 0  # Rerouting
        self.cc = 0  # complete cancel
        self.pc = 0  # partial cancel
        self.cd = 0  # complete delay
        self.pd = 0  # partial delay
        self.et = 0  # emergency train
        self.eb = 0  # emergency bus
        self.ret = 0  # return to initial train


class Nr_usage:
    def __init__(self):
        self.name = 'nr_usage'
        # self.rr = 0  # Rerouting
        self.cc = 0  # complete cancel
        self.pc = 0  # partial cancel
        self.cd = 0  # complete delay
        self.pd = 0  # partial delay
        self.et = 0  # emergency train
        self.eb = 0  # emergency bus
        self.ret = 0  # return to initial train


class Probabilities:
    def __init__(self, weights):
        self.name = 'probabilities'
        # self.rr = weights.rr / weights.sum  # Rerouting
        min_prob = 0.05
        remaining_prob = 1 - 7 * min_prob
        self.cc = min_prob + (weights.cc / weights.sum) * remaining_prob   # complete cancel
        self.pc = min_prob + self.cc + (weights.pc / weights.sum) * remaining_prob  # partial cancel
        self.cd = min_prob + self.pc + (weights.cd / weights.sum) * remaining_prob  # complete delay
        self.pd = min_prob + self.cd + (weights.pd / weights.sum) * remaining_prob  # partial delay
        self.et = min_prob + self.pd + (weights.et / weights.sum) * remaining_prob  # emergency train
        self.eb = min_prob + self.et + (weights.eb / weights.sum) * remaining_prob  # emergency bus
        self.ret = min_prob + self.eb + (weights.ret / weights.sum) * remaining_prob   # return to initial train

