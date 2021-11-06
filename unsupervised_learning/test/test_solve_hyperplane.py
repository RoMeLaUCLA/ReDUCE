import os, sys
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_ReDUCE+"/unsupervised_learning")
from solve_separating_planes import solve_separating_plane
from get_vertices import get_vertices, plot_rectangle
import matplotlib.pyplot as plt
import numpy as np

vertex_0 = np.array([[13, 0.0], [-60, 0.0], [-60, 20], [13, 20]])
# vertex_1 = np.array([[3.0, 5.0], [3.0, 0.0], [2.0, 0.0], [2.0, 5.0]])
vertex_1 = np.array([[20, 60], [20, 25], [10, 25], [10, 60]])

bin_width = 176
bin_height = 110

bin_left = -bin_width / 2.0
bin_right = bin_width / 2.0
bin_ground = 0.0
bin_up = bin_height

v_bin = np.array([[bin_right, bin_up],
                  [bin_right, bin_ground],
                  [bin_left, bin_ground],
                  [bin_left, bin_up]])

ret = solve_separating_plane(vertex_0, vertex_1, bin_width, bin_height)

print(ret)

a = ret[1]
b = ret[2]

fig, ax = plt.subplots()

plot_rectangle(ax, v_bin, color='black', show=False)
plot_rectangle(ax, vertex_0, color='red', show=False)
plot_rectangle(ax, vertex_1, color='red', show=False)

yy = np.linspace(bin_ground-0.5, bin_up+0.5, 100)
xx = (b - a[1]*yy)/a[0]
plt.plot(xx, yy, 'g')

plt.show()
