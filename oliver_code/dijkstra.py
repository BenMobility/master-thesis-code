"""
Created on Mon Feb  10 2021

@author: BenMobility

Dijkstra algorithm
"""
# %% Usual imports
import sys
from collections import defaultdict
import numpy as np


# %% Dijkstra DAG

# Python program to find single source shortest paths
# for Directed Acyclic Graphs Complexity :OV(V+E)
# Graph is represented using adjacency list. Every
# node of adjacency list contains vertex number of
# the vertex to which edge connects. It also contains
# weight of the edge
class Graph:
    def __init__(self, vertices):

        self.V = vertices  # No. of vertices

        # dictionary containing adjacency List
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def add_edge(self, u, v, w):
        self.graph[u].append((v, w))

    # A recursive function used by shortestPath
    def topological_sort_util(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        if v in self.graph.keys():
            for node, weight in self.graph[v]:
                if not visited[node]:
                    self.topological_sort_util(node, visited, stack)

        # Push current vertex to stack which stores topological sort
        stack.append(v)

    ''' The function to find shortest paths from given vertex. 
    It uses recursive topologicalSortUtil() to get topological 
    sorting of given graph.'''

    def shortest_path(self, source_node):

        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from source vertices
        for i in range(self.V):
            if not visited[i]:
                self.topological_sort_util(source_node, visited, stack)

        # Initialize distances to all vertices as infinite and
        # distance to source as 0
        dist = [float("Inf")] * self.V
        dist[source_node] = 0

        # Process vertices in topological order
        while stack:

            # Get the next vertex from topological order
            i = stack.pop()

            # Update distances of all adjacent vertices
            for node, weight in self.graph[i]:
                if dist[node] > dist[i] + weight:
                    dist[node] = dist[i] + weight

        # Print the calculated shortest distances
        for i in range(self.V):
            print("%d" % dist[i]) if dist[i] != float("Inf") else "Inf",


g = Graph(6)
g.add_edge(0, 1, 5)
g.add_edge(0, 2, 3)
g.add_edge(1, 3, 6)
g.add_edge(1, 2, 2)
g.add_edge(2, 4, 4)
g.add_edge(2, 5, 2)
g.add_edge(2, 3, 7)
g.add_edge(3, 4, -1)
g.add_edge(4, 5, -2)

# source = 1
s = 1

print("Following are shortest distances from source %d " % s)
g.shortest_path(s)


# %% Dijkstra algorithm
# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

class Graph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def print_solution(self, dist):
        print("Vertex \tDistance from Source")
        for node in range(self.V):
            print(node, "\t", dist[node])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree (spt)
    def min_distance(self, dist, spt_set):

        # Initialize minimum distance for next node
        min_dist_next_node = sys.maxsize

        # Search not nearest vertex not in the
        # shortest path tree
        min_index = 0
        for v in range(self.V):
            if dist[v] < min_dist_next_node and spt_set[v] is False:
                min_dist_next_node = dist[v]
                min_index = v

        return min_index

        # Function that implements Dijkstra's single source

    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):

        dist = [sys.maxsize] * self.V
        dist[src] = 0
        spt_set = [False] * self.V

        for cut in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.min_distance(dist, spt_set)

            # Put the minimum distance vertex in the
            # shortest path tree
            spt_set[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0 and spt_set[v] is False and \
                        dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]

        self.print_solution(dist)

    # Driver program


g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
           [4, 0, 8, 0, 0, 0, 0, 11, 0],
           [0, 8, 0, 7, 0, 4, 0, 0, 2],
           [0, 0, 7, 0, 9, 14, 0, 0, 0],
           [0, 0, 0, 9, 0, 10, 0, 0, 0],
           [0, 0, 4, 14, 10, 0, 2, 0, 0],
           [0, 0, 0, 0, 0, 2, 0, 1, 6],
           [8, 11, 0, 0, 0, 0, 1, 0, 7],
           [0, 0, 2, 0, 0, 0, 6, 7, 0]
           ]

g.dijkstra(0)

# %% Dijkstra Stefano

# 8 X 8 infrastructure graph
infra_graph = [[0, 5, 7, 90, 2, 90, 90, 90],
               [3, 0, 8, 90, 90, 5, 11, 90],
               [90, 6, 0, 12, 4, 90, 90, 9],
               [8, 90, 90, 0, 3, 11, 4, 7],
               [90, 7, 90, 3, 0, 90, 6, 8],
               [90, 5, 90, 90, 20, 0, 8, 90],
               [90, 90, 5, 2, 5, 90, 0, 12],
               [14, 90, 9, 90, 90, 8, 11, 0]
               ]

# set the variables
dist = np.full(len(infra_graph), np.inf)
precedent = np.full(len(infra_graph), np.inf)
visited = np.zeros(len(infra_graph))
tent = np.zeros(len(infra_graph))

# ask the user the source node
start = int(input("Choose initial node:"))
dist[start] = 0

# initial number of unvisited nodes1
number_unvisited = len(infra_graph) - 1

# dijkstra algorithm
while number_unvisited > 0:
    # determine next node
    min_distance = np.inf
    current = 0
    for j in range(len(infra_graph)):
        if visited[j] == 0:
            if dist[j] < min_distance:
                min_distance = dist[j]
                current = j

    # set node as visited
    visited[current] = 1

    # update number of unvisited nodes
    number_unvisited = number_unvisited - 1

    # update shortest path of neighbors of current node
    for p in range(len(infra_graph)):
        if infra_graph[current][p] != 90:
            tent[p] = dist[current] + infra_graph[current][p]
            if tent[p] < dist[p]:
                dist[p] = tent[p]
                precedent[p] = current

# print solution
print("Solution of Dijkstra's algorithm for all node:\n")
for q in range(len(infra_graph)):
    print(f'Cost of shortest path to {q} = {dist[q]:.3f} \n Precedent = {precedent[q]:.3f}\n ----\n')
