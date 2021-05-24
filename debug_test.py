import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
infra_graph = np.load('output/pickle/infra_graph.pickle', allow_pickle=True)
nx.draw(infra_graph)
plt.show()
