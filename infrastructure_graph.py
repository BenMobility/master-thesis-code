"""
Created on Thu Feb 25 2021

@author: BenMobility

Build infrastructure graph for the master thesis main codes.
"""
from networkx import MultiGraph
import networkx as nx
import viriato_interface
import numpy as np
import pickle


def build_infrastructure_graph(time_window, save_pickle):
    """
    function that build an infrastructure graph from Viriato and SBB nodes.
    :param save_pickle: boolean parameter to check if we want to save the output in a pickle file
    :param time_window: Viriato object called previously on the main code.
    :return: infra_graph: Multigraph from networkx that contains multiple characteristics
    :return: sbb_nodes: all the sbb nodes with zone, sbb ID, Viriato ID, node name and coordinates (type: Rec array)
    :return: node_codes:nodes_code: all visited node codes (type : list)
    :return: id_nodes: all visited node ids (type: list)
    """
    # Nodes in Viriato
    node_codes, id_nodes = viriato_interface.get_all_visited_nodes_in_time_window(time_window)

    # Nodes in SBB network with Viriato nodes
    sbb_nodes = viriato_interface.get_sbb_nodes(node_codes)

    # Initialize lists of dictionaries
    viriato_stations = []  # List of stations for infrastructure graph
    viriato_stations_attributes = {}  # Dictionary of attributes for Viriato stations
    edges = {}  # List of links between stations for infrastructure graph
    added_sections = []  # List of sections already added to the graph

    # Initialize the infrastructure graph
    cache_section_track_id_distance = {}
    infra_graph = MultiGraph(cache_trackID_dist=cache_section_track_id_distance)
    nodes_to_explore = [sbb_nodes.Code[0]]  # List of nodes to explore

    while len(nodes_to_explore) != 0:
        actual_node = nodes_to_explore.pop(0)
        # Check if node is already explored
        if actual_node not in viriato_stations:
            viriato_stations.append(int(actual_node))
            node_info = viriato_interface.get_node_info(actual_node)
            # Create attributes, depending on if its in the area or not
            attributes = dict()
            attributes['Code'] = node_info.code
            attributes['DebugString'] = node_info.debug_string
            attributes['in_area'] = False
            if actual_node in sbb_nodes.Code:
                attributes['in_area'] = True
            if len(node_info.node_tracks) > 0:
                attributes['NodeTracks'] = node_info.node_tracks
            else:
                attributes['NodeTracks'] = None
            viriato_stations_attributes[actual_node] = attributes

        #  Find all neighbors of actual node (used from node, since it is bidirectional, have the same neighbors with to
        neighbors = viriato_interface.get_neighboring_nodes_from_node(actual_node)

        # Loop through neighbors to identify the sections between the stations
        for neighbor in neighbors:
            # Add the neighbor to the list of stations to explore if not done yet
            if neighbor.id not in viriato_stations:
                nodes_to_explore.append(neighbor.id)
            # Get the section between the actual node and the neighbor
            tracks = viriato_interface.get_section_tracks_between(actual_node, neighbor.id)
            # Get the section identity
            for track in tracks:
                section_identity = track.debug_string
                if section_identity not in added_sections:
                    if not cache_section_track_id_distance.__contains__(track.id):
                        cache_section_track_id_distance[track.id] = track.weight
                    # Define the edge
                    edge = (actual_node, neighbor.id, track.id)
                    edge_attribute = {'weight': track.weight, 'sectionTrackID': track.id,
                                      'debugString': track.debug_string}
                    # Add the edge attribute in the edges list
                    edges[edge] = edge_attribute
                    # Add the section debug string
                    added_sections.append(section_identity)

    for node, attributes in viriato_stations_attributes.items():
        infra_graph.add_node(node, **attributes)

    for edge, attributes in edges.items():
        infra_graph.add_edge(edge[0], edge[1], edge[2], **attributes)

    # Save the output in a pickle file
    if save_pickle:
        nx.write_gpickle(infra_graph, 'output/pickle/infra_graph.pickle')

    return infra_graph, sbb_nodes, node_codes, id_nodes
