# This script solves the bookshelf problem with MINLP solver Bonmin

import os, sys
dir_ReDUCE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_ReDUCE+"/utils")
sys.path.append(dir_ReDUCE+"/bookshelf_generator")
sys.path.append(dir_ReDUCE+"/bookshelf_MIP_solver")
from fcn_solver_bonmin import solve_bin_pyomo
from bookshelf_generator import main as bin_generation_main
import numpy as np

new_data = bin_generation_main(1)

inst_shelf_geometry = new_data[0]["after"]["shelf"].shelf_geometry

bin_width = inst_shelf_geometry.shelf_width
bin_height = inst_shelf_geometry.shelf_height
num_of_item_stored = new_data[0]["after"]["shelf"].num_of_item

for iter_data in range(len(new_data)):

    print("################################# Data number {} #################################".format(iter_data))
    this_shelf = new_data[iter_data]["after"]["shelf"]

    solve_bin_pyomo(this_shelf, iter_data, iter_data)
