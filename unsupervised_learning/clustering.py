import os, sys
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_ReDUCE+"/utils")
path_dataset = os.path.dirname(os.path.realpath(__file__))

from get_vertices import get_vertices, plot_rectangle
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image as im
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from solve_separating_planes import solve_separating_plane
import pdb

# First, read the dataset
# data_file_list = ['zero_remove_angle_auto_generated_boxes_data_ba_035619.pkl']  # Part 1
# data_file_list = ['zero_remove_angle_auto_generated_boxes_data_ba_041221.pkl']  # Part 2
data_file_list = ['zero_remove_angle_auto_generated_boxes_data_ba_042455.pkl']  # Part 3
# data_file_list = ['zero_remove_angle_auto_generated_boxes_data_ba_135452.pkl']

# data_file_list = ['zero_remove_angle_auto_generated_boxes_data_ba_014932.pkl']  # Part4
# data_file_list = ['zero_remove_angle_auto_generated_boxes_data_ba_020159.pkl']  # Part5
# data_file_list = ['zero_remove_angle_auto_generated_boxes_data_ba_021450.pkl']  # Part6

len_files = len(data_file_list)

feature_list = []
sol_list = []

# TODO: this is data generator to add separating plane, change name!


def plot_ellipsoid(cent, cov, scale):
    num_pts = 200
    theta = np.linspace(0, 2 * np.pi, num_pts)
    list_pt_ellipse = []
    for iter in range(num_pts):
        pt_circle = [scale*np.cos(theta[iter]), scale*np.sin(theta[iter])]
        pt_ellipse = [cent[0] + cov[0, 0] * pt_circle[0] + cov[0, 1] * pt_circle[1],
                      cent[1] + cov[1, 0] * pt_circle[0] + cov[1, 1] * pt_circle[1]]
        list_pt_ellipse.append(pt_ellipse)

    return np.array(list_pt_ellipse)


iter_file = 0
assert iter_file == 0, "To prevent one bug, let's only use one .pkl file !!!"
with open(path_dataset + '/' + data_file_list[iter_file], 'rb') as f:
    data = pickle.load(f)
    bin_width = data[-1][0]
    bin_height = data[-1][1]

    bin_left = -bin_width / 2.0
    bin_right = bin_width / 2.0
    bin_ground = 0.0
    bin_up = bin_height

    v_bin = np.array([[bin_right, bin_up],
                      [bin_right, bin_ground],
                      [bin_left, bin_ground],
                      [bin_left, bin_up]])

    num_of_item_all = 4

    print("Reading dataset {}. Length of data is {}".format(data_file_list[iter_file], len(data)))

    for iter_data in range(int((len(data)-1)/2)):
        if iter_data >= -1:
            print("================================= Processing data {} ========================".format(iter_data))
            fig, ax = plt.subplots()
            plot_rectangle(ax, v_bin, color='black', show=False)

            # print("Data number {} --------------------------------------------".format(iter_data))

            iter_1 = 2 * iter_data
            iter_2 = 2 * iter_data + 1

            mask_1 = np.array(data[iter_1]['image'])
            mask_2 = np.array(data[iter_2]['image'])

            img_1 = im.fromarray(mask_1)
            img_2 = im.fromarray(mask_2)

            # img_1.show()
            # img_2.show()

            remove = data[iter_2]['remove']

            width_in_hand = data[iter_1]['boxes'][remove][3][0]
            height_in_hand = data[iter_1]['boxes'][remove][3][1]

            this_feature = []
            this_sol = []

            # Remember, we need a coordinate system to order the items. For now, just left to right
            for iterr in [iter_1, iter_2]:
                vertex_boxes = []

                for iter_box in range(len(data[iterr]['boxes'])):
                    # Note horizontally flipped
                    center_pt = [0 - (bin_width / 2.0 - data[iterr]['boxes'][iter_box][0]),
                                 bin_height - data[iterr]['boxes'][iter_box][1] - 7]
                    ang_pt = -data[iterr]['boxes'][iter_box][2]
                    R_bw_pt = np.array([[np.cos(ang_pt), -np.sin(ang_pt)],
                                        [np.sin(ang_pt), np.cos(ang_pt)]])
                    size = np.array([data[iterr]['boxes'][iter_box][3][1], data[iterr]['boxes'][iter_box][3][0]])
                    v_item = get_vertices(center_pt, R_bw_pt, size)

                    plot_rectangle(ax, v_item, color='blue', show=False)

                    if iterr == iter_1 and iter_box != remove:
                        vertex_boxes.append(v_item)  # Append in original scene
                        # Add vertex variables to make is ~30 dimension
                        this_sol.extend([ang_pt, v_item[0, 0], v_item[0, 1],
                                                 v_item[1, 0], v_item[1, 1],
                                                 v_item[2, 0], v_item[2, 1],
                                                 v_item[3, 0], v_item[3, 1]])
                    elif iterr == iter_2:
                        # TODO: standardize this format
                        this_feature.extend([0 - (bin_width / 2.0 - data[iterr]['boxes'][iter_box][0]),
                                             bin_height - data[iterr]['boxes'][iter_box][1] - 7,
                                             -data[iterr]['boxes'][iter_box][2],
                                             data[iterr]['boxes'][iter_box][3][1],
                                             data[iterr]['boxes'][iter_box][3][0]])  # Item height, then item width

                if iterr == iter_1:
                    # Append solution for removed items
                    center_pt_remove = [0 - (bin_width / 2.0 - data[iterr]['boxes'][remove][0]),
                                        bin_height - data[iterr]['boxes'][remove][1] - 7]
                    ang_pt_remove = -data[iterr]['boxes'][remove][2]
                    R_bw_pt_remove = np.array([[np.cos(ang_pt_remove), -np.sin(ang_pt_remove)],
                                               [np.sin(ang_pt_remove), np.cos(ang_pt_remove)]])
                    size_remove = np.array([data[iterr]['boxes'][remove][3][1], data[iterr]['boxes'][remove][3][0]])
                    v_item_remove = get_vertices(center_pt_remove, R_bw_pt_remove, size_remove)

                    vertex_boxes.append(v_item_remove)  # Append the removed item which should always be the last
                    this_sol.extend([ang_pt_remove, v_item_remove[0, 0], v_item_remove[0, 1],
                                                    v_item_remove[1, 0], v_item_remove[1, 1],
                                                    v_item_remove[2, 0], v_item_remove[2, 1],
                                                    v_item_remove[3, 0], v_item_remove[3, 1]])

                    # Append separating planes
                    N_planes = int(round(len(vertex_boxes) * (len(vertex_boxes) - 1)/2))
                    item_l = 0
                    item_r = 1
                    for iter_planes in range(N_planes):

                        success, aa, bb = solve_separating_plane(vertex_boxes[item_l], vertex_boxes[item_r],
                                                               bin_width, bin_height)

                        assert success, "Infeasible in solving separating plane !!"
                        # ==============================================================================================
                        yy = np.linspace(bin_ground - 0.5, bin_up + 0.5, 100)
                        xx = (bb - aa[1] * yy) / aa[0]
                        plt.plot(xx, yy, 'g')
                        plt.savefig('solved_figures/{}.png'.format(iter_data))
                        # ==============================================================================================

                        this_sol.extend([aa[0], aa[1]])

                        item_r += 1
                        if item_r >= num_of_item_all:
                            item_l += 1
                            item_r = item_l + 1

                    sol_list.append(this_sol)

                    # print(this_sol)

                elif iterr == iter_2:
                    this_feature.extend([height_in_hand, width_in_hand])  # TODO: this corresponds to the order of size
                                                                          # TODO: of each box, height first, then width
                    feature_list.append(this_feature)

            # Save dataset
            with open('dataset_unsupervised_learning.p', 'wb') as f:
                save_data = {'feature':feature_list, 'solution':sol_list}
                pickle.dump(save_data, f)

N_dim = len(sol_list[0])
print("The dimension of each solution is {}".format(N_dim))
# print(feature_list)
pdb.set_trace()

# Plot figure for the first 2 dimensions to demonstrate clustering
plt.figure()
plt.subplot(2, 1, 1)
for iter_sol in range(len(sol_list)):
    # print([sol_list[iter_sol][0], sol_list[iter_sol][1]])
    plt.plot(sol_list[iter_sol][0], sol_list[iter_sol][1], 'x')
plt.xlabel('Angle_item0')
plt.ylabel('Vertex0_item0')
plt.xlim([-3.2, 3.2])
plt.ylim([-100, 100])

pdb.set_trace()

# # GMM model for clustering
# n_classes = 20
# N_dim = len(sol_list[0])
# gmm = GaussianMixture(n_components=n_classes, covariance_type='full')
# gmm.fit(np.array(sol_list))

# OPTICS for clustering
print(f"===============CLUSTERING===============")
# Unsupervised clustering using OPTICS
min_samples = 50  # Number of samples in neighborhood for point to be considered as a core point
xi = 0.03  # Determines the minimum steepness on the reachability plot that constitutes a cluster boundary
min_cluster_size = 0.01  # Minimum number of samples in cluster, expressed as a fraction of the number of samples

data = np.array(sol_list)
print("Length of data is {}".format(len(data)))

clust = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
clust.fit(data)

n_classes = np.max(clust.labels_)+1
print(f"Number of clusters: {n_classes}")
print(f"Number of outliers: {np.sum(clust.labels_ == -1)}")

print(clust.labels_)

x_cutdown = bin_width/4.0
y_cutdown = bin_height/4.0
theta_cutdown = 0.35
print("x cutdown {}".format(x_cutdown))
print("y cutdown {}".format(y_cutdown))
print("theta cutdown {}".format(theta_cutdown))

# TODO: The right way to do this: grid the space first and predefine integer variables,
#  see what space the cluster crosses over, and activate certain integer variables
plt.subplot(2, 1, 2)
num_int = []
print("Means and covariances of components ===========================================================================")
for iter_peak in range(n_classes):
    print("-----------------------------------------------------------------------------------------------------------")

    this_cluster = []
    for iter_data in range(len(data)):
        if clust.labels_[iter_data] == iter_peak:
            this_cluster.append(sol_list[iter_data])

    arr_this_cluster = np.array(this_cluster)

    max_this_class = []
    min_this_class = []
    for iter_dim in range(N_dim):
        max_this_dim = max(arr_this_cluster[:, iter_dim])
        min_this_dim = min(arr_this_cluster[:, iter_dim])
        max_this_class.append(max_this_dim)
        min_this_class.append(min_this_dim)

    print("Max elements are {}".format(max_this_class))
    print("Min elements are {}".format(min_this_class))
    diff = np.array(max_this_class) - np.array(min_this_class)
    print("Differences are")
    this_class_num_int = 0
    for iterrr in range(N_dim):
        if iterrr == 0 or iterrr == 9 or iterrr == 18 or iterrr == 27:
            if diff[iterrr] >= theta_cutdown:
                print("{:.2f}, YES cut".format(diff[iterrr]))
                this_class_num_int += 1
            else:
                print("{:.2f}, NO cut".format(diff[iterrr]))

        elif iterrr == 1 or iterrr == 10 or iterrr == 19 or iterrr == 28:
            if diff[iterrr] >= x_cutdown:
                print("{:.2f}, YES cut".format(diff[iterrr]))
                this_class_num_int += 1
            else:
                print("{:.2f}, NO cut".format(diff[iterrr]))

        elif iterrr == 2 or iterrr == 11 or iterrr == 20 or iterrr == 29:
            if diff[iterrr] >= y_cutdown:
                print("{:.2f}, YES cut".format(diff[iterrr]))
                this_class_num_int += 1
            else:
                print("{:.2f}, NO cut".format(diff[iterrr]))

        elif iterrr == 3 or iterrr == 12 or iterrr == 21 or iterrr == 30:
            if diff[iterrr] >= x_cutdown:
                print("{:.2f}, YES cut".format(diff[iterrr]))
                this_class_num_int += 1
            else:
                print("{:.2f}, NO cut".format(diff[iterrr]))

        elif iterrr == 4 or iterrr == 13 or iterrr == 22 or iterrr == 31:
            if diff[iterrr] >= y_cutdown:
                print("{:.2f}, YES cut".format(diff[iterrr]))
                this_class_num_int += 1
            else:
                print("{:.2f}, NO cut".format(diff[iterrr]))

        elif iterrr == 5 or iterrr == 14 or iterrr == 23 or iterrr == 32:
            if diff[iterrr] >= x_cutdown:
                print("{:.2f}, YES cut".format(diff[iterrr]))
                this_class_num_int += 1
            else:
                print("{:.2f}, NO cut".format(diff[iterrr]))

        elif iterrr == 6 or iterrr == 15 or iterrr == 24 or iterrr == 33:
            if diff[iterrr] >= y_cutdown:
                print("{:.2f}, YES cut".format(diff[iterrr]))
                this_class_num_int += 1
            else:
                print("{:.2f}, NO cut".format(diff[iterrr]))

        elif iterrr == 7 or iterrr == 16 or iterrr == 25 or iterrr == 34:
            if diff[iterrr] >= x_cutdown:
                print("{:.2f}, YES cut".format(diff[iterrr]))
                this_class_num_int += 1
            else:
                print("{:.2f}, NO cut".format(diff[iterrr]))

        elif iterrr == 8 or iterrr == 17 or iterrr == 26 or iterrr == 35:
            if diff[iterrr] >= y_cutdown:
                print("{:.2f}, YES cut".format(diff[iterrr]))
                this_class_num_int += 1
            else:
                print("{:.2f}, NO cut".format(diff[iterrr]))

    num_int.append(this_class_num_int)
    plt.plot(arr_this_cluster[:, 0], arr_this_cluster[:, 1], '.')

    # print([round(gmm.means_[iter_peak, iter_dim], 2) for iter_dim in range(N_dim)])
    # print(gmm.covariances_[iter_peak])
    # plt.scatter(gmm.means_[iter_peak, 0], gmm.means_[iter_peak, 1], s=100, c='black')
    # scale = 3
    # ret_pts = plot_ellipsoid(np.array([gmm.means_[iter_peak, 0], gmm.means_[iter_peak, 1]]), gmm.covariances_[iter_peak][0:2, 0:2], scale)
    # plt.plot(ret_pts[:, 0], ret_pts[:, 1], 'blue')  # Plot ellipsoids

print("Required number of integers are {}".format(num_int))

plt.xlabel('Angle_item0')
plt.ylabel('Vertex0_item0')
plt.xlim([-3.2, 3.2])
plt.ylim([-100, 100])

plt.show()
