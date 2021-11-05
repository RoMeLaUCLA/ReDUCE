import os, sys
dir_ReDUCE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_ReDUCE+"/utils")
sys.path.append(dir_ReDUCE+"/bookshelf_generator")
sys.path.append(dir_ReDUCE+"/bookshelf_MIP_solver")
sys.path.append(dir_ReDUCE+"/unsupervised_learning")
sys.path.append(dir_ReDUCE+"/CoCo/solved_data_for_careful_comparison")
from fcn_clustered_solver_gurobi import solve_within_patch
from bookshelf_generator import main as bin_generation_main
from get_vertices import get_vertices, plot_rectangle
from book_problem_classes import BookProblemFeature, Item, ShelfGeometry, Shelf

import pickle, runpy
import numpy as np
import pdb

model_name = '1000_data/26_classes'

list_clusters = list(range(26))
num_of_classes = 26

# Load trained unsupervised classifier
with open(dir_ReDUCE + '/unsupervised_learning/models/' + model_name + '/xg_model.pkl', 'rb') as f:
    model_classifier = pickle.load(f)
    f.close()

# Use classified data to identify active patches for each class
with open(dir_ReDUCE + '/unsupervised_learning/models/' + model_name + '/combined_data.pkl', 'rb') as dd:
    all_data = pickle.load(dd)
    dd.close()

with open(dir_ReDUCE + '/unsupervised_learning/models/' + model_name + '/feature_scaler.pkl', 'rb') as dd:
    feature_scaler = pickle.load(dd)
    dd.close()

sample_data = bin_generation_main(1)

all_feature = all_data['feature']
all_solution = all_data['solution']
all_label = all_data['label']

num_of_labels = max(all_label)+1
num_of_data = len(all_label)

all_classified_solutions = []
all_classified_features = []

for iter_class in range(num_of_labels):
    this_class_solution = []
    this_class_feature = []
    for iter_data in range(num_of_data):
        if all_label[iter_data] == iter_class:
            this_class_solution.append(all_solution[iter_data])
            this_class_feature.append(all_feature[iter_data])
    all_classified_solutions.append(this_class_solution)
    all_classified_features.append(this_class_feature)

# Read data that were solved by CoCo, and re-solve it with time limit
# TODO: this parameter needs to be changed for different files
with open(dir_ReDUCE + '/CoCo/solved_data_for_careful_comparison/CoCo_solved.p', 'rb') as dd:
    data_solved = pickle.load(dd)

features_solved = data_solved['features']
time_consumed_solved = data_solved['time_consumed']
cost_solved = data_solved['cost']
count = data_solved['num_of_problem']

assert len(features_solved) == count, "Inconsistent length of data!"
assert len(time_consumed_solved) == count, "Inconsistent length of data!"
assert len(cost_solved) == count, "Inconsistent length of data!"

inst_shelf_geometry = sample_data[0]["after"]["shelf"].shelf_geometry

bin_width = inst_shelf_geometry.shelf_width
bin_height = inst_shelf_geometry.shelf_height
num_of_item_stored = sample_data[0]["after"]["shelf"].num_of_item

train_params = {'bin_width': bin_width, 'bin_height': bin_height, 'num_of_item': num_of_item_stored}

feature_all_classes = [[] for ii in range(num_of_classes)]
int_all_classes = [[] for ii in range(num_of_classes)]
X_all_classes = [[] for ii in range(num_of_classes)]
solve_times_all_classes = [[] for ii in range(num_of_classes)]
costs_all_classes = [[] for ii in range(num_of_classes)]
num_of_int_all_classes = [[] for ii in range(num_of_classes)]

cost_comparison = []

for iter_data in range(count):

    print("################################# Data number {} #################################".format(iter_data))
    this_feature = features_solved[iter_data].flatten()  # Read from CoCo_solve, should already be flattened

    rr = Shelf.flat_feature_to_init_list(this_feature, inst_shelf_geometry)
    this_shelf = Shelf(rr[0], rr[1], rr[2], rr[3], rr[4])

    this_feature_scaled = feature_scaler.transform(np.array([this_feature]))

    ret_class = model_classifier.predict(this_feature_scaled)[0]

    print("This data is classified as {}".format(ret_class))

    # Pick out data that has label of class 0, push features of data through to get integers
    if ret_class != -1:
        if ret_class in list_clusters:
            prob_success, X_ret, X_dict, Y_ret, cost_ret, time_ret, actual_num_of_int_var, data_out_range = solve_within_patch(
                          this_shelf, all_classified_solutions[ret_class], iter_data, iter_data, time_consumed_solved[iter_data])

            if (not data_out_range) and prob_success:
                feature_all_classes[ret_class].append(this_shelf.return_feature())
                int_all_classes[ret_class].append(Y_ret)
                X_all_classes[ret_class].append(X_ret)
                solve_times_all_classes[ret_class].append(time_ret)
                costs_all_classes[ret_class].append(cost_ret)
                num_of_int_all_classes[ret_class].append(actual_num_of_int_var)

            cost_comparison.append({'learner': cost_solved[iter_data], 'MIP': cost_ret})

print(cost_comparison)
diff = 0
for iter_data in range(count):
    diff += (cost_comparison[iter_data]['learner'] - cost_comparison[iter_data]['MIP'])

print("learner solution - MIP solution is on average".format(diff/count))