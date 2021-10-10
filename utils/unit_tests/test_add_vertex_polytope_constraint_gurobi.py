import os, sys
dir_utils = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_utils)
from add_McCormick_envelope_constraint import add_vertex_polytope_constraint_gurobi
import gurobipy as go
import numpy as np
import matplotlib.pyplot as plt

INF = go.GRB.INFINITY

m = go.Model("test_model")

x = m.addVars(2, lb=-INF, ub=INF)

set_int = False

if set_int:
    lam = m.addVars(8, lb=0.0, ub=1.0)

    z = m.addVars(2, vtype=go.GRB.BINARY)

    v = np.array([[-1, 1, 0,  0, 2, -2, 0,  0],
                  [ 0, 0, 1, -1, 0,  0, 2, -2]])

    arr_integer = [[  z[0],   z[1]],
                   [1-z[0],   z[1]],
                   [  z[0], 1-z[1]],
                   [1-z[0], 1-z[1]]]

    arr_selection = [[0, 1],
                     [2, 3],
                     [4, 5, 6],
                     [7]]

else:
    lam = m.addVars(4, lb=0.0, ub=1.0)

    v = np.array([[-1, 1, 0, 0],
                  [0, 0, 1, -1]])

    arr_integer = []

    arr_selection = []

m.update()

add_vertex_polytope_constraint_gurobi(m, x, lam, v, arr_integer, arr_selection)

if set_int:
    m.addConstr(z[0] == 1)
    m.addConstr(z[1] == 1)

#obj = x[0]*x[0] + x[1]*x[1]
obj = x[1]

m.setObjective(obj, go.GRB.MINIMIZE)
m.optimize()

x_sol = [x[0].X, x[1].X]

print(x_sol)
print(m.display())

plt.scatter(v[0, :], v[1, :])
plt.scatter(x_sol[0], x_sol[1], edgecolors='red')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
