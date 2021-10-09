from add_piecewise_linear_constraint import add_piecewise_linear_constraint
import gurobipy as go
import matplotlib.pyplot as plt
import numpy as np


x_lim = [[-1.0, -0.5],
         [-0.5,  0.0],
         [ 0.0,  0.5],
         [ 0.5,  1.0]]

k = [-0.75/0.5, -0.25/0.5, 0.25/0.5, 0.75/0.5]
b = [1-0.75/0.5, 0.0, 0.0, 1-0.75/0.5]
num_of_int = 2

bigM = 10000

z0_val = 1.0
z1_val = 1.0

# Add integer variables constraints ====================================================================================
m = go.Model("Test_add_piecewise_linear_constraint")

x = m.addVar(lb=-bigM, ub=bigM)
y = m.addVar(lb=-bigM, ub=bigM)
z = m.addVars(num_of_int, vtype=go.GRB.BINARY)

m.update()

int_list = [[1-z[0], 1-z[1]], [z[0], 1-z[1]], [1-z[0], z[1]], [z[0], z[1]]]
ret = add_piecewise_linear_constraint(m, x, y, x_lim, k, b, int_list, bigM)

m.addConstr(z[0] == z0_val)
m.addConstr(z[1] == z1_val)

obj = 0.0

m.update()

m.setObjective(obj, go.GRB.MINIMIZE)

m.optimize()

print([x.X, y.X])
print([z[0].X, z[1].X])

# Add constant integer values ==========================================================================================
m2 = go.Model("Test_add_piecewise_linear_constraint")

x2 = m2.addVar(lb=-bigM, ub=bigM)
y2 = m2.addVar(lb=-bigM, ub=bigM)
z2 = m2.addVars(num_of_int, vtype=go.GRB.BINARY)

m2.update()

z0 = z0_val
z1 = z1_val

int_list2 = [[1-z0, 1-z1], [z0, 1-z1], [1-z0, z1], [z0, z1]]
ret2 = add_piecewise_linear_constraint(m2, x2, y2, x_lim, k, b, int_list2, bigM, pinned=True)

obj = 0.0

m2.update()

m2.setObjective(obj, go.GRB.MINIMIZE)

m2.optimize()

print([x2.X, y2.X])

# ======================================================================================================================
x_plot = np.linspace(x_lim[0], x_lim[-1], 100)

fig, axs = plt.subplots(2)

# Plot result from int variables
axs[0].plot(x.X, y.X, 'x')
axs[0].plot(x_plot, x_plot**2)
for iter_piece in range(len(k)):
    axs[0].plot(x_plot, k[iter_piece]*x_plot + b[iter_piece])

# Plot result from int constants
axs[1].plot(x2.X, y2.X, 'x')
axs[1].plot(x_plot, x_plot**2)
for iter_piece in range(len(k)):
    axs[1].plot(x_plot, k[iter_piece]*x_plot + b[iter_piece])

plt.show()
