import os, sys
dir_utils = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_utils)
from add_McCormick_envelope_constraint import add_vertex_polytope_constraint_pyomo
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
import pdb

inf = float('inf')

m = pyo.AbstractModel()

m.x = pyo.Var(pyo.RangeSet(0, 1), bounds=(-inf, inf))

m.lam = pyo.Var(pyo.RangeSet(0, 7), bounds=(0.0, 1.0))

m.z = pyo.Var(pyo.RangeSet(0, 1), domain=pyo.Binary)

v = np.array([[-1, 1, 0,  0, 2, -2, 0,  0],
              [ 0, 0, 1, -1, 0,  0, 2, -2]])

# Use rules to delay evaluation, see https://stackoverflow.com/a/59314459
# For the version of Gurobi, it is when the given z's are all 1, the lambda's are selected. This means the corresponding
# bit is 1 as that is when 1-z[bit] is added
arr_integer = [[1, 1],
               [0, 1],
               [1, 0],
               [0, 0]]

arr_selection = [[0, 1],
                 [2, 3],
                 [4, 5, 6],
                 [7]]

m = add_vertex_polytope_constraint_pyomo(m, 'x', 'lam', 'z', v, arr_integer, arr_selection)


def add_constr1(m):
    return m.z[0] == 1


def add_constr2(m):
    return m.z[1] == 0


m.con_add_constr1 = pyo.Constraint(rule=add_constr1)
m.con_add_constr2 = pyo.Constraint(rule=add_constr2)


# Objective function ===================================================================================================
def obj_expression(m):
    # return m.x[0]*m.x[0] + m.x[1]*m.x[1]
    return -m.x[0] - m.x[1]


m.OBJ = pyo.Objective(rule=obj_expression)
instance = m.create_instance()

opt = pyo.SolverFactory('bonmin', executable="/home/romelahex/Desktop/Bonmin-1.8.8/build/bin/bonmin")

results = opt.solve(instance)

feasible = np.all(results.Solver.termination_condition == 'optimal')

if feasible:
    print("This problem is Feasible !!!")
else:
    print("This problem is Infeasible !!!")

xx = instance.x.extract_values()
lamlam = instance.lam.extract_values()
zz = instance.z.extract_values()

x_sol = [xx[0], xx[1]]
lambda_sol = [lamlam[iter_lam] for iter_lam in pyo.RangeSet(0, 7)]
z_sol = [zz[iter_z] for iter_z in pyo.RangeSet(0, 1)]

print(x_sol)
print(lambda_sol)
print(z_sol)

plt.scatter(v[0, :], v[1, :])
plt.scatter(x_sol[0], x_sol[1], edgecolors='red')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
