import numpy as np
import networkx as nx
import passenger_assignment

od = None
odt_priority_list_original = np.load('output/pickle/odt_priority_list_original_alns.pkl', allow_pickle=True)
timetable_initial_graph = np.load('output/pickle/timetable_alns.pkl', allow_pickle=True)
parameters = np.load('output/pickle/parameters_for_alns.pkl', allow_pickle=True)
timetable_full_graph = np.load('output/pickle/timetable_full_graph.pkl', allow_pickle=True)

timetable_graph = np.load('output/pickle/timetable_edges_o_stations_d.pkl', allow_pickle=True)
edges_o_stations_d = np.load('output/pickle/edges_o_stations_d.pkl', allow_pickle=True)

if timetable_graph is None:
        timetable_graph = nx.DiGraph()

edges_o_stations = edges_o_stations_d.edges_o_stations
edges_stations_d = edges_o_stations_d.edges_stations_d

# Edges_o_stations_dict: key origin, value edges connecting to train nodes
edges_o_stations_dict = edges_o_stations_d.edges_o_stations_dict
edges_stations_d_dict = edges_o_stations_d.edges_stations_d_dict


if od is None:
        timetable_graph.add_weighted_edges_from(edges_o_stations)
        timetable_graph.add_weighted_edges_from(edges_stations_d)
        # Get the attribute for origin and destination nodes (for dijkstra algorithm)
        nodes_type = {}
        for node in timetable_graph.nodes:
                if isinstance(node, str):
                        nodes_type[node] = {'type': 'origin'}
                elif not isinstance(node, tuple):
                        nodes_type[node] = {'type': 'destination'}
        nx.set_node_attributes(timetable_graph, nodes_type)

else:
        origin = od[0]
        dest = od[1]
        timetable_graph.add_weighted_edges_from(edges_o_stations_dict[origin])
        timetable_graph.add_weighted_edges_from(edges_stations_d_dict[dest])
        # Get the attribute for origin and destination nodes (for dijkstra algorithm)
        nodes_type = {}
        for node in timetable_graph.nodes:
                if isinstance(node, str):
                        nodes_type[node] = {'type': 'origin'}
                elif not isinstance(node, tuple):
                        nodes_type[node] = {'type': 'destination'}
        nx.set_node_attributes(timetable_graph, nodes_type)

