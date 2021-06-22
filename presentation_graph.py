"""
Created on Tue Jun  22 2021

@author: BenMobility

Presentation graphs for report
"""

import plotly
import plotly.graph_objs as go
import numpy as np


# Read z pickles
z_pickle = np.load('output/pickle/z_pickle.pkl', allow_pickle=True)

# Get the list of symbol
symbols = []
for i in range(len(z_pickle['z_cur_archived'])):
    if z_pickle['z_cur_archived'][i]:
        symbols.append(0)
    else:
        symbols.append(34)

# Set marker properties
markersize = [item / 125 for item in z_pickle['z_de_reroute_acc']]
markercolor = [item / 60 for item in z_pickle['z_de_cancel_acc']]

# Make Plotly figure
fig2 = go.Scatter(x=[item/60 for item in z_pickle['z_tt_acc']],
                  y=z_pickle['z_op_acc'],
                  marker=dict(size=markersize,
                              color=markercolor,
                              symbol=symbols,
                              line=dict(width=2,
                                        color='DarkSlateGrey'),
                              opacity=0.9,
                              reversescale=True,
                              colorbar=dict(title='z_deviation_cancel [min]'),
                              colorscale='Mint'),
                  mode='markers')

# Make Plotly Layout
mylayout = go.Layout(xaxis=dict(title="z_travel_time [min]"),
                     yaxis=dict(title="z_operation_cost [km]"))

# Plot and save html
plotly.offline.plot({"data": [fig2],
                     "layout": mylayout},
                    auto_open=True)


