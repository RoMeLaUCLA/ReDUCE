import os, sys
dir_Learning_MIP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_Learning_MIP+"/classification_tests/utils")

import gurobipy as go
import numpy as np
from add_McCormick_envelope_constraint import add_vertex_polytope_constraint_gurobi
import matplotlib.pyplot as plt

m = go.Model("Bin_organization")

num_of_item = 1
dim_2D = 2

R_sq_vertex = [-1.0, -0.5, 0.0, 0.5, 1.0]
R_sq_vertex_sq = [(-1.0)**2, (-0.5)**2, 0.0, 0.5**2, 1.0**2]
v_R_sq = np.array([R_sq_vertex, R_sq_vertex_sq])
num_of_int_from_R_sq_knots = 2
len_vertices_R_sq = len(R_sq_vertex)

int_R_stored_0000 = m.addMVar((num_of_item, num_of_int_from_R_sq_knots), vtype=go.GRB.BINARY)
lam_stored_0000 = m.addMVar((num_of_item, len_vertices_R_sq), lb=0.0, ub=1.0)

R_wb_stored = m.addMVar((num_of_item, dim_2D, dim_2D), lb=-1.0, ub=1.0)  # Rotation matrices for stored items
R_wb_stored_0000 = m.addMVar(num_of_item, lb=-1.0, ub=1.0)

for iter_item in range(num_of_item):
    # R_wb_stored_0000 = R_wb_stored[0, 0]*R_wb_stored[0, 0]
    int_list_0000 = [[  int_R_stored_0000[iter_item, 0],   int_R_stored_0000[iter_item, 1]],
                     [1-int_R_stored_0000[iter_item, 0],   int_R_stored_0000[iter_item, 1]],
                     [  int_R_stored_0000[iter_item, 0], 1-int_R_stored_0000[iter_item, 1]],
                     [1-int_R_stored_0000[iter_item, 0], 1-int_R_stored_0000[iter_item, 1]]]

    selection_0000 = [[0, 1], [1, 2], [2, 3], [3, 4]]
    lam_0000 = [lam_stored_0000[iter_item, iter_lam] for iter_lam in range(len_vertices_R_sq)]
    add_vertex_polytope_constraint_gurobi(m, [R_wb_stored[iter_item, 0, 0], R_wb_stored_0000[iter_item]],
                                          lam_0000, v_R_sq, int_list_0000, selection_0000)

m.addConstr(int_R_stored_0000[0, 0] == 1.0)
m.addConstr(int_R_stored_0000[0, 1] == 0.0)

obj = R_wb_stored_0000[0]

m.setObjective(obj, go.GRB.MINIMIZE)
m.optimize()

print("Result x variable")
print([int_R_stored_0000[0, iter_zz].X[0] for iter_zz in range(num_of_int_from_R_sq_knots)])

plt.figure()
x_range = np.linspace(-1.0, 1.0, 100)
plt.plot(x_range, x_range ** 2)
for iter_item in range(num_of_item):
    plt.plot(R_wb_stored[iter_item, 0, 0].X, R_wb_stored_0000[iter_item].X, 'x')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
