import datetime
import networkx as nx

def sp_topological_order(G, source, adj_list_topo_order, dist, path, vertices_topo_order, target = None):
    '''
    1) Initialize dist[] = {INF, INF, } and dist[s] = 0 where s is the source vertex
    2) Create a toplogical order of all vertices
    3) Do following for every vertex u in topological order
        Do following for every adjacent vertex v of u
        if (dist[v] > dist[u] + weight(u, v))
        dist[v] = dist[u] + weight(u, v)
    '''
    dist[source] = 0
    # idx_source = vertices_topo_order.index(source)
    # vertices_topo_order = vertices_topo_order[idx_source:]
    # adj_list_topo_order = adj_list_topo_order[idx_source:]

    for idx in range(len(vertices_topo_order)):
        vertex = vertices_topo_order[idx]
        # Update distances of all adjacent vertices
        for node, weight in adj_list_topo_order[idx]:  # adj_dict[vertex].items():
            if dist[node] > dist[vertex] + weight['weight']:
                dist[node] = dist[vertex] + weight['weight']
                path[node] = path[vertex].copy()
                path[node].append(node)

    # print(dist[target])
    # print(path[target])
    if target is not None:
        if not dist[target] == float('inf'):
            dist = dist[target]
            path = path[target]
        else:
            dist = None
            path = None

    return dist, path  # dist[target], path[target]


def weight_from_datetime_to_str(weight, node, vertex):
    if isinstance(weight['weight'], datetime.timedelta):
        print(node)
        print(vertex)
        weight['weight'] = weight['weight'].seconds / 60  # weights in minutes
    return weight


def generate_topological_order_adj_list(G):
    vertices_topo_order = list(nx.topological_sort(G))  # (node, adjacency dictionary)
    adj_list = [(n, nbrdict) for n, nbrdict in G.adjacency()]  # (node, adjacency dictionary)
    adj_dict = {n: nbrdict for n, nbrdict in G.adjacency()}
    adj_list = [None] * len(vertices_topo_order)
    idx = 0

    for idx in range(len(vertices_topo_order)):
        adj_list[idx] = []
        for nbr, datadict in G.adj[vertices_topo_order[idx]].items():
            adj_list[idx].append([nbr, datadict])
    adj_list_topo_order = adj_list
    # adj_dict = adj_list
    return adj_dict, adj_list_topo_order, vertices_topo_order


def topologicalSortUtil(v, visited, stack, adj_dict):
    # Mark the current node as visited.
    visited[v] = True

    # Recur for all the vertices adjacent to this vertex
    if v in adj_dict.keys():
        for node, weight in adj_dict[v].items():
            if visited[node] == False:
                stack, visited = topologicalSortUtil(node, visited, stack, adj_dict)
                # Push current vert
                stack.append(v)
    return stack, visited


def find_path_topo_sort(G_topo):
    # Python program to find single source shortest paths
    # for Directed Acyclic Graphs Complexity :OV(V+E)
    from collections import defaultdict

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
        def addEdge(self, u, v, w):
            self.graph[u].append((v, w))

        # A recursive function used by shortestPath
        def topologicalSortUtil(self, v, visited, stack):

            # Mark the current node as visited.
            visited[v] = True

            # Recur for all the vertices adjacent to this vertex
            if v in self.graph.keys():
                for node, weight in self.graph[v]:
                    if visited[node] == False:
                        self.topologicalSortUtil(node, visited, stack)

                    # Push current vertex to stack which stores topological sort
            stack.append(v)

        ''' The function to find shortest paths from given vertex. 
            It uses recursive topologicalSortUtil() to get topological 
            sorting of given graph.'''

        def shortestPath(self, s):

            # Mark all the vertices as not visited
            visited = [False] * self.V
            stack = []

            # Call the recursive helper function to store Topological
            # Sort starting from source vertice
            for i in range(self.V):
                if visited[i] == False:
                    self.topologicalSortUtil(s, visited, stack)
                # Initialize distances to all vertices as infinite and
            # distance to source as 0
            dist = [float("Inf")] * (self.V)
            dist[s] = 0

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
    g.addEdge(0, 1, 5)
    g.addEdge(0, 2, 3)
    g.addEdge(1, 3, 6)
    g.addEdge(1, 2, 2)
    g.addEdge(2, 4, 4)
    g.addEdge(2, 5, 2)
    g.addEdge(2, 3, 7)
    g.addEdge(3, 4, -1)
    g.addEdge(4, 5, -2)

    # source = 1
    s = 1

    print("Following are shortest distances from source %d " % s)
    g.shortestPath(s)
    # https: // www.geeksforgeeks.org / shortest - path - for -directed - acyclic - graphs /
    # This code is contributed by Neelam Yadav
