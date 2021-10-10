import os, sys
dir_utils = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_utils)
import gurobipy as go
import numpy as np
from add_McCormick_envelope_constraint import add_bilinear_constraint_gurobi, limit2vertex
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

#ax = Axes3D(fig)

ll = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

# Construct a 4 x 2 grid, y direction count up first
x = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
y = [[-5.0, 0.0], [0.0, 5.0]]

len_sections_x = len(x)
len_sections_y = len(y)

num_of_polygons = len_sections_x * len_sections_y
num_of_vertices = 4 * num_of_polygons

num_of_int = int(np.log2(num_of_polygons))

z0 = 1.0
z1 = 1.0
z2 = 0.0
# Add integer variables constraints ====================================================================================
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

m = go.Model("Bin_organization")
m.setParam('MIPGap', 1e-2)

pt = m.addVars(3, lb=-10000, ub=10000)
lam = m.addVars(num_of_vertices, lb=0.0, ub=1.0)
int_poly = m.addVars(num_of_int, vtype=go.GRB.BINARY)

m.update()

m.addConstr(int_poly[0] == z0)
m.addConstr(int_poly[1] == z1)
m.addConstr(int_poly[2] == z2)

v_all = np.zeros([3, num_of_vertices])

iter_polygon = 0

for iter_x in range(len_sections_x):
    for iter_y in range(len_sections_y):
        bilinear_limits = [x[iter_x][0], x[iter_x][1],
                           y[iter_y][0], y[iter_y][1]]

        vv = limit2vertex(bilinear_limits)

        for iter_v in range(4):
            ax1.scatter(vv[iter_v, 0], vv[iter_v, 1], vv[iter_v, 2], marker='.', color='blue')

        for iter_pair in range(len(ll)):
            ax1.plot3D([vv[ll[iter_pair][0], 0], vv[ll[iter_pair][1], 0]],
                      [vv[ll[iter_pair][0], 1], vv[ll[iter_pair][1], 1]],
                      [vv[ll[iter_pair][0], 2], vv[ll[iter_pair][1], 2]], color='blue')

        v_all[:, 4 * iter_polygon + 0] = np.array([vv[0, 0], vv[0, 1], vv[0, 2]])
        v_all[:, 4 * iter_polygon + 1] = np.array([vv[1, 0], vv[1, 1], vv[1, 2]])
        v_all[:, 4 * iter_polygon + 2] = np.array([vv[2, 0], vv[2, 1], vv[2, 2]])
        v_all[:, 4 * iter_polygon + 3] = np.array([vv[3, 0], vv[3, 1], vv[3, 2]])

        iter_polygon += 1

int_var = [int_poly[iter_zz] for iter_zz in range(num_of_int)]

ret_constr = add_bilinear_constraint_gurobi(m, pt, lam, int_var, num_of_polygons, v_all)

obj = pt[2]

m.setObjective(obj, go.GRB.MINIMIZE)

m.optimize()

print("Solution point:")
print([pt[0].X, pt[1].X, pt[2].X])
ax1.scatter(pt[0].X, pt[1].X, pt[2].X, marker='x', color='black')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Add constant integer values ==========================================================================================
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

m2 = go.Model("Bin_organization")
m2.setParam('MIPGap', 1e-2)

pt = m2.addVars(3, lb=-10000, ub=10000)
lam2 = m2.addVars(num_of_vertices, lb=0.0, ub=1.0)

m2.update()

int_poly = np.zeros(3)
int_poly[0] = z0
int_poly[1] = z1
int_poly[2] = z2

v_all = np.zeros([3, num_of_vertices])

iter_polygon = 0

for iter_x in range(len_sections_x):
    for iter_y in range(len_sections_y):
        bilinear_limits = [x[iter_x][0], x[iter_x][1],
                           y[iter_y][0], y[iter_y][1]]

        vv = limit2vertex(bilinear_limits)

        for iter_v in range(4):
            ax2.scatter(vv[iter_v, 0], vv[iter_v, 1], vv[iter_v, 2], marker='.', color='blue')

        for iter_pair in range(len(ll)):
            ax2.plot3D([vv[ll[iter_pair][0], 0], vv[ll[iter_pair][1], 0]],
                      [vv[ll[iter_pair][0], 1], vv[ll[iter_pair][1], 1]],
                      [vv[ll[iter_pair][0], 2], vv[ll[iter_pair][1], 2]], color='blue')

        v_all[:, 4 * iter_polygon + 0] = np.array([vv[0, 0], vv[0, 1], vv[0, 2]])
        v_all[:, 4 * iter_polygon + 1] = np.array([vv[1, 0], vv[1, 1], vv[1, 2]])
        v_all[:, 4 * iter_polygon + 2] = np.array([vv[2, 0], vv[2, 1], vv[2, 2]])
        v_all[:, 4 * iter_polygon + 3] = np.array([vv[3, 0], vv[3, 1], vv[3, 2]])

        iter_polygon += 1

int_var = [int_poly[iter_zz] for iter_zz in range(num_of_int)]

ret_constr = add_bilinear_constraint_gurobi(m2, pt, lam2, int_var, num_of_polygons, v_all, pinned=True)

obj = pt[2]

m2.setObjective(obj, go.GRB.MINIMIZE)

m2.optimize()

print("Solution point:")
print([pt[0].X, pt[1].X, pt[2].X])
ax2.scatter(pt[0].X, pt[1].X, pt[2].X, marker='x', color='black')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.show()
