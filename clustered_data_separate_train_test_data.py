import pickle, os, sys
dir_ReDUCE = os.path.dirname(os.path.realpath(__file__))
import numpy as np
from sklearn.model_selection import train_test_split

# TODO: Begin: Remember to make changes to the following parameters !!
dir_path = dir_ReDUCE + "/clustered_dataset"
sss = "AllClusters"
# sss = "cluster68_3000data"

train_fn = dir_path + '/train_separated_' + sss + '.p'
test_fn = dir_path + '/test_separated_' + sss + '.p'

class_label_all = list(range(100))
# class_label_all = [68]
len_class_label = len(class_label_all)

list_dataset = [[0] for iterr in range(len_class_label)]
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

feature = []

# Features
for ct in range(num_dataset):
    if not ct in empty_dataset:
        assert len(features_all[ct]) > 0, "Length of dataset should be larger than 0!"
        feature += features_all[ct]

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
        1.0 * num_train_data / len_tot), random_state=33)

# TODO: Maybe change this to dictionary (different from the original CoCo format)
train_data = [train_params]
train_data += [[feature[ii] for ii in indices_train]]
train_data += [np.array(X[indices_train]), np.array(Y[indices_train])]
train_data += [np.array(costs[indices_train]), np.array(solve_times[indices_train])]

test_data = [train_params]
test_data += [[feature[ii] for ii in indices_test]]
test_data += [np.array(X[indices_test]), np.array(Y[indices_test])]
test_data += [np.array(costs[indices_test]), np.array(solve_times[indices_test])]

train_file = open(train_fn, 'wb')
pickle.dump(train_data, train_file)

test_file = open(test_fn, 'wb')
pickle.dump(test_data, test_file)
