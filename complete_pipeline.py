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

# # Begin: for debug
# with open('/home/romela/xuan/Learning_MIP/simulation2D/bin_problem_python/data/cluster4/dataset_unsupervised_learning_part4.p', 'rb') as f:
#     new_data = pickle.load(f)
#
# all_feature_for_debug = new_data['feature']
#
# bin_width = 176
# bin_height = 110
#
# bin_left = -88.0
# bin_right = 88.0
# bin_ground = 0.0
# bin_up = 110
#
# v_bin = np.array([[88., 110.],
#                   [88., 0.],
#                   [-88., 0.],
#                   [-88., 110.]])
#
# num_of_item = 3
# # End: for debug

bin_width = new_data[-1][0]
bin_height = new_data[-1][1]

bin_left = -bin_width / 2.0
bin_right = bin_width / 2.0
bin_ground = 0.0
bin_up = bin_height

v_bin = np.array([[bin_right, bin_up],
                  [bin_right, bin_ground],
                  [bin_left, bin_ground],
                  [bin_left, bin_up]])

num_of_item = 3

num_of_classes = 100

train_params = {'bin_width': bin_width, 'bin_height': bin_height, 'num_of_item': num_of_item}

print(bin_width)
print(bin_height)
print(bin_left)
print(bin_right)
print(bin_ground)
print(bin_up)
print(v_bin)

feature_all_classes = [[] for ii in range(num_of_classes)]
int_all_classes = [[] for ii in range(num_of_classes)]
X_all_classes = [[] for ii in range(num_of_classes)]
solve_times_all_classes = [[] for ii in range(num_of_classes)]
costs_all_classes = [[] for ii in range(num_of_classes)]
num_of_int_all_classes = [[] for ii in range(num_of_classes)]

# TODO: also make the following function an utility function
for iter_data in range(int((len(new_data)-1)/2)):

# # Begin: for debug
# for iter_data in range(len(all_feature_for_debug)):
#
#     this_feature = all_feature_for_debug[iter_data]
#
#     print("------------------------------------------------------------")
#     print("feature 0")
#     print(all_feature[iter_data])
#     print("feature 1")
#     print(this_feature)
#     assert all_feature[iter_data] == this_feature, "Wrong feature data !!"
#
#     item_width_stored = [this_feature[5 * 0 + 4], this_feature[5 * 1 + 4], this_feature[5 * 2 + 4]]
#     item_height_stored = [this_feature[5 * 0 + 3], this_feature[5 * 1 + 3], this_feature[5 * 2 + 3]]
#     item_center_stored = np.array([[this_feature[5 * 0 + 0], this_feature[5 * 0 + 1]],
#                                    [this_feature[5 * 1 + 0], this_feature[5 * 1 + 1]],
#                                    [this_feature[5 * 2 + 0], this_feature[5 * 2 + 1]]])
#     item_angle_stored = [this_feature[5 * 0 + 2], this_feature[5 * 1 + 2], this_feature[5 * 2 + 2]]
#     item_width_in_hand = this_feature[16]
#     item_height_in_hand = this_feature[15]
#     # End: for debug

    # TODO: There are 3 places that use the same box->feature function: here, bookshelf_generator and scene_dataset_generation.
    #  To fix this, just let the bookshelf generator give "shelf" object
    print("############################################################## Data number {} ##############################################################".format(iter_data))
    iter_1 = 2 * iter_data
    iter_2 = 2 * iter_data + 1

    mask_o = new_data[iter_2]['image']
    mask = np.zeros(np.shape(mask_o))
    for iter_row in range(np.shape(mask_o)[0]):
        for iter_column in range(np.shape(mask_o)[1]):
            mask[iter_row, iter_column] = mask_o[iter_row, iter_column]

    item_center_stored = []
    item_angle_stored = []
    item_width_stored = []
    item_height_stored = []
    vertices = []
    width_in_hand = 0.0; height_in_hand = 0.0
    this_feature = []

    remove = new_data[iter_2]['remove']
    item_width_in_hand = new_data[iter_1]['boxes'][remove][3][0]
    item_height_in_hand = new_data[iter_1]['boxes'][remove][3][1]

    for iter_box in range(len(new_data[iter_2]['boxes'])):  # iterr == iter_2, does not include item-in-hand

        # Note horizontally flipped
        center_pt = [0-(bin_width/2.0-new_data[iter_2]['boxes'][iter_box][0]), bin_height-new_data[iter_2]['boxes'][iter_box][1]-7]
        ang_pt = -new_data[iter_2]['boxes'][iter_box][2]
        R_bw_pt = np.array([[np.cos(ang_pt), -np.sin(ang_pt)],
                            [np.sin(ang_pt),  np.cos(ang_pt)]])

        size = np.array([new_data[iter_2]['boxes'][iter_box][3][1], new_data[iter_2]['boxes'][iter_box][3][0]])
        # First height (size[0] = [3][1]), then width (size[1] = [3][0])

        item_center_stored.append(center_pt)
        item_angle_stored.append(ang_pt)
        item_width_stored.append(new_data[iter_2]['boxes'][iter_box][3][0])
        item_height_stored.append(new_data[iter_2]['boxes'][iter_box][3][1])
        vv = get_vertices(center_pt, R_bw_pt, size)
        vertices.append(vv)
        # TODO: The above center/angle/width/height/vertices can be gathered in a single object named "item"

        this_feature.extend([center_pt[0], center_pt[1], ang_pt, size[0], size[1]])

    this_feature.extend([item_height_in_hand, item_width_in_hand])
    # TODO: put all information about items and in-hand-items into an object named "scene"

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
                      bin_width, bin_height, bin_left, bin_right, bin_ground, bin_up, v_bin,
                      num_of_item, np.array(item_width_stored), np.array(item_height_stored),
                      np.array(item_center_stored), np.array(item_angle_stored),
                      item_width_in_hand, item_height_in_hand,  # TODO: all the input above can be a single object named "scene"
                      all_classified_solutions[ret_class], iter_data, iter_data)

        if (not data_out_range) and prob_success:
            feat_item_width_stored = np.array([this_feature[5*iter_item+4] for iter_item in range(num_of_item)])
            feat_item_height_stored = np.array([this_feature[5*iter_item+3] for iter_item in range(num_of_item)])
            feat_item_center_x_stored = np.array([this_feature[5*iter_item+0] for iter_item in range(num_of_item)])
            feat_item_center_y_stored = np.array([this_feature[5*iter_item+1] for iter_item in range(num_of_item)])
            feat_item_angle_stored = np.array([this_feature[5*iter_item+2] for iter_item in range(num_of_item)])
            feat_item_height_in_hand = this_feature[15]
            feat_item_width_in_hand = this_feature[16]

            feature_dict = {'item_center_x_stored': feat_item_center_x_stored,
                            'item_center_y_stored': feat_item_center_y_stored,
                            'item_angle_stored': feat_item_angle_stored,
                            'item_width_stored': feat_item_width_stored,
                            'item_height_stored': feat_item_height_stored,
                            'item_width_in_hand': feat_item_width_in_hand,
                            'item_height_in_hand': feat_item_height_in_hand}

            feature_all_classes[ret_class].append(feature_dict)
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
