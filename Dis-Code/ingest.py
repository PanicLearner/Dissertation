"""
File for importing route data from a json file
"""

import json
import os

# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.multiarray import ndarray
import pandas as pd


def get_data(file_name):
    """
    method to retrieve JSON data from "file"
    :param file_name: string representing file in which JSON data is stored
    :return data: Pyhonic data created from JSON file information
    """
    with open(os.path.join(os.sys.path[0], file_name), "r") as data_file:
        data = json.load(data_file)  # load data from JSON file
        return data


if __name__== "__main__":
    file_name = 'json_data.json'
    routes = get_data(file_name)  # routes is a list of routes


""" creating lists that will store distance travelled, overall elevation gain and elevation climbed from edge to edge 
    this will be used as X and Y axes in elevation plot (hopefully)
"""
distance = []
elev_gain = []
climb = []
gradients = []
gradient_elev = []


""" want to be able to add edge values of distance and elev_ gain to the empty lists [.append()] ??"""


def routeplot(index):
    for edge in routes[index]["edges"]:
        # accessing edgeLocation dictionary
        end_location = edge["endLocation"]
        start_location = edge["startLocation"]
        gradient = edge["gradient"]

        # accessing edgeDistance key to add to distance list
        # making running total of distance travelled  by adding on d-edge to previous value in distance
        d_edge = edge["edgeDistance"]

        if len(distance)== 0:
            distance.append(d_edge)

        else:
            distance.append(distance[len(distance) - 1] + d_edge)

        # accessing elevation at start and end points to calculate elev_gain for each edge
        # making running total of distance travelled  by adding on d-elev to previous value in elev_gian
        elev_stop = end_location["alt"]
        elev_start = start_location["alt"]
        d_elev = elev_stop - elev_start

        if len(elev_gain) == 0:
            elev_gain.append(d_elev)

        else:
            elev_gain.append(elev_gain[len(elev_gain) - 1] + d_elev)

    # just a nice statistic to have, same idea as elev-gain but only adding positive d_elev values
        if len(climb) == 0:
            if d_elev > 0:
                climb.append(d_elev)

        else:
            if d_elev > 0:
                climb.append(climb[len(climb) - 1] + d_elev)

        """ manipulating the gradient data  rather than using d_elev to get elevation 
            gains for each segment this might be more useful to get comparable features"""

        if len(gradient_elev) == 0:
            gradient_elev.append((np.array(gradient/100))* np.array(d_edge))

        else:
            gradient_elev.append(gradient_elev[len(gradient_elev) - 1] + (np.array(gradient/100)) * np.array(d_edge))

        gradients.append(gradient)

    return distance, elev_gain, climb, gradient_elev, gradients


""" defining method for trailing running average of elevation gain list """


def trailingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas


""" attempting to get central running average of elevation gain list """


i = 3
routeplot(i)

# window size for running averages = 15% number of GPS readings taken
window = 0.10*len(distance)

# assigning variables alphabetic value so that its quicker to get plots
a = np.array(distance)

b = np.array(elev_gain)

c = np.array(trailingaverage(elev_gain, window))

d = trailingaverage(gradient_elev, window)

e = gradient_elev

""" trying to find turning points so that straight line segments may be drawn making comparison simplier"""
# need to use plot smoothed data?
# For each edge the gradient should be evaluated.
# if the the gradient == 0 then its a turning point lets say gradient < 2% is negligable

df1 = pd.DataFrame({"gradient": gradients, "distance": distance, "elevation": gradient_elev}, index=range(1, len(gradients) + 1))
# Adding initial zero point to the start of the series so that segments start from origin
df2 = pd.DataFrame({"gradient": [0], "distance": [0], "elevation": [0]})
#df3 = pd.DataFrame({"gradient":})
df21 = [df2, df1]
data = pd.concat(df21)

tp = data["gradient"] == 0



dt = pd.DataFrame(data[tp], index=range(0,len(data["distance"])))
dt = dt[pd.notnull(dt["distance"])]
dt = dt.reset_index()

#print dt # This data will be used to get segments for comparison where a segment is the edge between two points.

# Create dataframe for segments rather than points

seg_distance = [] # distance between points in tp
seg_elev_gain = [] # elevation gain between points in tp


for i in range(0, len(dt) - 1):
   seg_distance.append(dt.distance[i+1] - dt.distance[i])


for i in range(0, len(dt) -1):
    seg_elev_gain.append(dt.elevation[i+1] - dt.elevation[i])


segments = pd.DataFrame({"distance travelled": seg_distance, "elevation gained": seg_elev_gain})
segments["segment gradient"] = ((segments["elevation gained"]/segments["distance travelled"])*100)
segments["score"] = (segments["distance travelled"] * segments["segment gradient"])

#print segments["score"].max()


"""out = dt.plot(x="distance", y="elevation", title="Plot Using Turning Points - route %d" % (i+1), legend=False)

out.set_xlabel("Distance (m)")
out.set_ylabel("Elevation Gain (m)")"""


"""plt.title("Route %d" % (i+1))
plt.plot(a, b, 'r--', label = 'Unaltered')
#plt.plot(a[len(b) - len(d):], d, 'r-', label = 'Trailing Average - grad_elev')
plt.plot(a[len(b) - len(c):], c, 'g-', label = 'Trailing Average - d_elev')
#plt.plot(a, e, 'y--', label = 'grad_elev unaltered')
plt.xlabel("distance traveled (m)")
plt.ylabel("elevation gained (m)")
plt.legend(loc = 'upper left')"""

plt.show()