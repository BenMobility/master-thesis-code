import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
path = 'input/2010_Distanz_Systemfahrplan_B.txt'
od_distance = np.genfromtxt(path)
od_distance = np.core.records.fromarrays(od_distance.transpose(),
                                         names='fromZone, toZone, distance',
                                         formats='i8, i8, f8')

