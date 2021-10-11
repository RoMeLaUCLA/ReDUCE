import os, sys
dir_ReDUCE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_ReDUCE+"/utils")
sys.path.append(dir_ReDUCE+"/bookshelf_generator")
sys.path.append(dir_ReDUCE+"/bookshelf_MIP_solver")
sys.path.append(dir_ReDUCE+"/unsupervised_learning")
from fcn_clustered_solver_gurobi import solve_within_patch
from bookshelf_generator import main as bin_generation_main
from get_vertices import get_vertices, plot_rectangle
import pickle, runpy
import numpy as np
import pdb

list_features = ['item_center_x_stored', 'item_center_y_stored', 'item_angle_stored',
                 'item_width_stored', 'item_height_stored', 'item_width_in_hand', 'item_height_in_hand']

model_name = 'models_100_classes'

prefix = '99'
#prefix = 'ALL_130_int'

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

# Generate more data, push through unsupervised classifier and load all clustered data
new_data = bin_generation_main(10)

inst_shelf_geometry = new_data[0]["after"]["shelf"].shelf_geometry

bin_width = inst_shelf_geometry.shelf_width
bin_height = inst_shelf_geometry.shelf_height
num_of_item_stored = new_data[0]["after"]["shelf"].num_of_item

num_of_classes = 100

train_params = {'bin_width': bin_width, 'bin_height': bin_height, 'num_of_item': num_of_item_stored}

feature_all_classes = [{iter_feature: [] for iter_feature in list_features} for ii in range(num_of_classes)]
int_all_classes = [[] for ii in range(num_of_classes)]
X_all_classes = [[] for ii in range(num_of_classes)]
solve_times_all_classes = [[] for ii in range(num_of_classes)]
costs_all_classes = [[] for ii in range(num_of_classes)]
num_of_int_all_classes = [[] for ii in range(num_of_classes)]

for iter_data in range(len(new_data)):

    print("############################################################## Data number {} ##############################################################".format(iter_data))
    this_shelf = new_data[iter_data]["after"]["shelf"]
    this_feature = this_shelf.return_flat_feature()
    assert len(this_feature) == 17, "Inconsistent feature length !!"

    this_feature_scaled = feature_scaler.transform(np.array([this_feature]))

    ret_class = model_classifier.predict(this_feature_scaled)[0]

    # if not np.any(ret_class_onehot):
    #     ret_class = -1
    # else:
    #     ret_class = np.argmax(ret_class_onehot, axis=1).tolist()[0]

        # # Begin: for debug
        # print("This data is classified as {}".format(ret_class))
        # print("Ground truth label {}".format(all_label[iter_data]))
        # assert ret_class == all_label[iter_data], "Incorrect classification !!"
        # # End: for debug

    print("This data is classified as {}".format(ret_class))

    # Pick out data that has label of class 0, push features of data through to get integers
    if ret_class == 68:
        prob_success, X_ret, X_dict, Y_ret, cost_ret, time_ret, actual_num_of_int_var, data_out_range = solve_within_patch(
                      this_shelf, all_classified_solutions[ret_class], iter_data, iter_data)

        if (not data_out_range) and prob_success:
            feature_all_classes[ret_class]['item_center_x_stored'].append(this_shelf.return_stored_item_center_x())
            feature_all_classes[ret_class]['item_center_y_stored'].append(this_shelf.return_stored_item_center_y())
            feature_all_classes[ret_class]['item_angle_stored'].append(this_shelf.return_stored_item_angles())
            feature_all_classes[ret_class]['item_width_stored'].append(this_shelf.return_stored_item_widths())
            feature_all_classes[ret_class]['item_height_stored'].append(this_shelf.return_stored_item_heights())
            feature_all_classes[ret_class]['item_width_in_hand'].append(this_shelf.item_width_in_hand)
            feature_all_classes[ret_class]['item_height_in_hand'].append(this_shelf.item_height_in_hand)
            int_all_classes[ret_class].append(Y_ret)
            X_all_classes[ret_class].append(X_ret)
            solve_times_all_classes[ret_class].append(time_ret)
            costs_all_classes[ret_class].append(cost_ret)
            num_of_int_all_classes[ret_class].append(actual_num_of_int_var)


for iter_class in range(num_of_classes):
    # Gather features and integers to train learner
    saves = {"train_params": train_params, "features": feature_all_classes[iter_class],
             "Integers": int_all_classes[iter_class], "X": X_all_classes[iter_class],
             "Solve_time": solve_times_all_classes[iter_class], "Cost": costs_all_classes[iter_class],
             "num_of_int": num_of_int_all_classes[iter_class]}

    folder_path = "clustered_dataset/" + "dataset_class_" + str(iter_class)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    save_path = folder_path + "/" + prefix + "_dataset_class_" + str(iter_class) + "_y_guess.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(saves, f)

# Separate dataset into training and testing sets
runpy.run_module('clustered_data_separate_train_test_data')
