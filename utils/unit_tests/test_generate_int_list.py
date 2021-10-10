import os, sys
dir_utils = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_utils)
from generate_int_list import generate_int_list
import gurobipy as go

m = go.Model("test_generate_int_list")
test_x = m.addVars(3, vtype=go.GRB.BINARY)

m.update()

print([test_x[0], test_x[1], test_x[2]])

ret_list = generate_int_list(test_x)

print("----------------------------------------------------")
for iter_sect in range(len(ret_list)):
    print(ret_list[iter_sect])
