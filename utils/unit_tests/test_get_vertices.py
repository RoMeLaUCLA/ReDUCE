import os, sys
dir_utils = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_utils)
from get_vertices import get_vertices, plot_rectangle
import numpy as np

center = np.array([0, 0])

theta = -80.0/180.0*np.pi  # Rotation angle from world frame to item frame
R_bw = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

size = np.array([1.0, 2.0])

v = get_vertices(center, R_bw, size)
print(v)

plot_rectangle(v, color='blue', show=True)
