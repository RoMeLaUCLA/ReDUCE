import pickle, os, sys
dir_ReDUCE = os.path.dirname(os.path.realpath(__file__))
import numpy as np
from sklearn.model_selection import train_test_split

# TODO: Begin: Remember to make changes to the following parameters !!
dir_path = dir_ReDUCE + "/clustered_dataset"
# sss = "AllClusters"
sss = "cluster68_3000data"

train_fn = dir_path + '/train_separated_' + sss + '.p'
test_fn = dir_path + '/test_separated_' + sss + '.p'

#class_label_all = list(range(100))
class_label_all = [68]
len_class_label = len(class_label_all)

list_dataset = [[99] for iterr in range(len_class_label)]
# TODO: End: Remember to make changes to the following parameters !!

list_train_file = []
list_train_data = []

# Append all datasets from different clusters to list_train_file, list_train_data,
# and set num_dataset to the total number
num_dataset = len(list(np.concatenate(list_dataset)))

num_of_data_each_cluster = np.zeros(len_class_label, dtype=int)
num_of_int_each_cluster = np.zeros(len_class_label, dtype=int)

for iterrr in range(len_class_label):
    class_label = class_label_all[iterrr]
    for ct_dataset in range(len(list_dataset[iterrr])):
        file = open(dir_path + '/' + 'dataset_class_' + str(class_label) + '/' +
                                    str(list_dataset[iterrr][ct_dataset]) + '_dataset_class_' + str(class_label) +
                                    '_y_guess.pkl', 'rb')
        list_train_file.append(file)
        daaa = pickle.load(file)
        ll = len(daaa["X"])
        num_of_data_each_cluster[iterrr] += ll
        list_train_data.append(daaa)
        file.close()

        if daaa['num_of_int']:
            num_of_int_each_cluster[iterrr] = daaa['num_of_int'][0]

for iter_dataset in range(num_dataset - 1):
    assert np.all(list_train_data[iter_dataset]['train_params'] == list_train_data[num_dataset - 1]['train_params'])

print("Number of integers for each cluster is")
print(num_of_int_each_cluster)

# [ 77 0  0  0 46  0 43  0  0 40
#   41 0  0  0 50  0  0 47  0  0
#   0 47  0  0 47  0  0 49  0  0
#   0  0  0  0  0 66  0  0  0  0
#   0  0 52  0  0  0  0 50  0 47
#   46 0  0  0  0 68  0  0  0  0
#   0  0 52  0 45 42  0 41 68 77
#   0  0  0  0  0  0  0  0  0  0
#   54 0  0  0 50  0  0 49  0 50
#   0  0  0  0  0 48  0  0  0  0]

print("===============================================================================================================")
print("Separating data into training and testing dataset ...")
print("Using datasets: ")
for iter_dataset in range(len(list_train_file)):
    print(list_train_file[iter_dataset].name)
print("Number of data for each cluster: {}".format(num_of_data_each_cluster))
print("Save name is: {}".format(sss))
print("===============================================================================================================")

train_params = list_train_data[0]['train_params']

# --------------------------------------------------------------------
X_all = []
Y_all = []
features_all = []
solve_times_all = []
costs_all = []
len_prob = []
empty_dataset = []

for ct_dataset in range(num_dataset):
    if not list_train_data[ct_dataset]["X"]:
        empty_dataset.append(ct_dataset)

    X_all.append(list_train_data[ct_dataset]["X"])
    Y_all.append(list_train_data[ct_dataset]["Integers"])
    features_all.append(list_train_data[ct_dataset]["features"])
    solve_times_all.append(list_train_data[ct_dataset]["Solve_time"])
    costs_all.append(list_train_data[ct_dataset]["Cost"])

    assert np.shape(X_all[ct_dataset])[0] == np.shape(Y_all[ct_dataset])[0], "Inconsistent data length !!"
    assert np.shape(X_all[ct_dataset])[0] == np.shape(features_all[ct_dataset])[0], "Inconsistent data length !!"
    assert np.shape(X_all[ct_dataset])[0] == np.shape(solve_times_all[ct_dataset])[0], "Inconsistent data length !!"
    assert np.shape(X_all[ct_dataset])[0] == np.shape(costs_all[ct_dataset])[0], "Inconsistent data length !!"

    len_prob.append(np.shape(Y_all[ct_dataset])[0])

# --------------------------------------------------------------------
len_tot = sum(len_prob)
print("Total length is {}".format(len_tot))
print("Class labels are {}".format(class_label_all))

num_train_data = int(input("Please specify the size of training dataset:"))

feature = {'item_center_x_stored': [], 'item_center_y_stored': [],
           'item_angle_stored': [], 'item_width_stored': [],
           'item_height_stored': [], 'item_width_in_hand': [], 'item_height_in_hand': []}

# Features
for ct in range(num_dataset):
    if not ct in empty_dataset:
        len_this_dataset = len(features_all[ct])
        for iter_data in range(len_this_dataset):
            feature['item_center_x_stored'].append(features_all[ct][iter_data]['item_center_x_stored'])
            feature['item_center_y_stored'].append(features_all[ct][iter_data]['item_center_y_stored'])
            feature['item_angle_stored'].append(features_all[ct][iter_data]['item_angle_stored'])
            feature['item_width_stored'].append(features_all[ct][iter_data]['item_width_stored'])
            feature['item_height_stored'].append(features_all[ct][iter_data]['item_height_stored'])
            feature['item_width_in_hand'].append(features_all[ct][iter_data]['item_width_in_hand'])
            feature['item_height_in_hand'].append(features_all[ct][iter_data]['item_height_in_hand'])

feature['item_center_x_stored'] = np.array(feature['item_center_x_stored'])
feature['item_center_y_stored'] = np.array(feature['item_center_y_stored'])
feature['item_angle_stored'] = np.array(feature['item_angle_stored'])
feature['item_width_stored'] = np.array(feature['item_width_stored'])
feature['item_height_stored'] = np.array(feature['item_height_stored'])
feature['item_width_in_hand'] = np.array(feature['item_width_in_hand'])
feature['item_height_in_hand'] = np.array(feature['item_height_in_hand'])

# Solved states
X = np.vstack([X_all[ct] for ct in range(num_dataset) if not ct in empty_dataset])

# Solved integers
Y = np.vstack([Y_all[ct] for ct in range(num_dataset) if not ct in empty_dataset])

# Solve time
solve_times = np.hstack([solve_times_all[ct] for ct in range(num_dataset) if not ct in empty_dataset])

# Cost
costs = np.hstack([costs_all[ct] for ct in range(num_dataset) if not ct in empty_dataset])

# print("=======================================================================")
# print(feature)
# print(X)
# print(Y)
# print(solve_times)
# print(costs)

# ======================================================================================================================
XX, yy, indices = range(len_tot), range(len_tot), range(len_tot)

XX_train, XX_test, yy_train, yy_test, indices_train, indices_test = train_test_split(XX, yy, indices, test_size=1.0 - (
        1.0 * num_train_data / len_tot), random_state=42)

train_data = [train_params]
train_data += [{'item_center_x_stored': feature['item_center_x_stored'][indices_train],
                'item_center_y_stored': feature['item_center_y_stored'][indices_train],
                'item_angle_stored': feature['item_angle_stored'][indices_train],
                'item_width_stored': feature['item_width_stored'][indices_train],
                'item_height_stored': feature['item_height_stored'][indices_train],
                'item_width_in_hand': feature['item_width_in_hand'][indices_train],
                'item_height_in_hand': feature['item_height_in_hand'][indices_train]}]
train_data += [np.array(X[indices_train]), np.array(Y[indices_train])]
train_data += [np.array(costs[indices_train]), np.array(solve_times[indices_train])]

test_data = [train_params]
test_data += [{'item_center_x_stored': feature['item_center_x_stored'][indices_test],
               'item_center_y_stored': feature['item_center_y_stored'][indices_test],  # can do train,
               'item_angle_stored': feature['item_angle_stored'][indices_test],  # but just in case there is an overlap
               'item_width_stored': feature['item_width_stored'][indices_test],
               'item_height_stored': feature['item_height_stored'][indices_test],
               'item_width_in_hand': feature['item_width_in_hand'][indices_test],
               'item_height_in_hand': feature['item_height_in_hand'][indices_test]}]
test_data += [np.array(X[indices_test]), np.array(Y[indices_test])]
test_data += [np.array(costs[indices_test]), np.array(solve_times[indices_test])]

train_file = open(train_fn, 'wb')
pickle.dump(train_data, train_file)

test_file = open(test_fn, 'wb')
pickle.dump(test_data, test_file)
