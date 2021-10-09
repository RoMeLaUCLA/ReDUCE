import pickle, os
import numpy as np

train_file = open('train_separated_cluster68_3000data.p', 'rb')
train_data = pickle.load(train_file)
train_file.close()

X = train_data[2]
Y = train_data[3]
train_params = train_data[0]
features = train_data[1]
solve_times = train_data[-1]
costs = train_data[-2]

len_prob = len(Y)

for ii in range(len_prob):
    print("========================== Problem {} =================================".format(ii))

    print("Features:")
    print([features['item_center_x_stored'][ii], features['item_center_y_stored'][ii], features['item_angle_stored'][ii],
           features['item_width_stored'][ii], features['item_height_stored'][ii],
           features['item_width_in_hand'][ii], features['item_height_in_hand'][ii]])

    print(np.hstack((features['item_center_x_stored'][ii], features['item_center_y_stored'][ii],
                     features['item_angle_stored'][ii],
                     features['item_width_stored'][ii], features['item_height_stored'][ii],
                     features['item_width_in_hand'][ii], features['item_height_in_hand'][ii])))

    print("Solved states:")
    print(X[ii])

    print("Solved integers:")
    print(Y[ii])

    print("Solve time:")
    print(solve_times[ii])

    print("Cost:")
    print(costs[ii])

