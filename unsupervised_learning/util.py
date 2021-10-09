import os, sys
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_ReDUCE+"/utils")

from get_vertices import get_vertices
import pickle
import numpy as np
from PIL import Image as im


def ints_to_one_hot(array_of_ints):
    one_hot = np.zeros((array_of_ints.size, array_of_ints.max()+1))
    one_hot[np.arange(array_of_ints.size), array_of_ints] = 1
    return one_hot


def load_data(list_of_data_paths):
    len_files = len(list_of_data_paths)

    feature_list = []
    sol_list = []

    for iter_file in range(len_files):
        with open(list_of_data_paths[iter_file], 'rb') as f:
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

            print("Reading dataset {}. Length of data is {}".format(list_of_data_paths[iter_file], len(data)))

            for iter_data in range(int((len(data)-1)/2)):
                # print("Data number {} --------------------------------------------".format(iter_data))

                iter_1 = 2 * iter_data
                iter_2 = 2 * iter_data + 1

                # mask_1 = np.array(data[iter_1]['image'])
                # mask_2 = np.array(data[iter_2]['image'])

                # img_1 = im.fromarray(mask_1)
                # img_2 = im.fromarray(mask_2)

                # img_1.show()
                # img_2.show()

                remove = data[iter_2]['remove']

                width_in_hand = data[iter_1]['boxes'][remove][3][0]
                height_in_hand = data[iter_1]['boxes'][remove][3][1]

                this_feature = []
                this_sol = []

                # Remember, we need a coordinate system to order the items. For now, just left to right
                for iterr in [iter_1, iter_2]:
                    for iter_box in range(len(data[iterr]['boxes'])):
                        # Note horizontally flipped
                        center_pt = [0 - (bin_width / 2.0 - data[iterr]['boxes'][iter_box][0]),
                                     bin_height - data[iterr]['boxes'][iter_box][1] - 7]
                        ang_pt = -data[iterr]['boxes'][iter_box][2]
                        R_bw_pt = np.array([[np.cos(ang_pt), -np.sin(ang_pt)],
                                            [np.sin(ang_pt), np.cos(ang_pt)]])
                        size = np.array([data[iterr]['boxes'][iter_box][3][1], data[iterr]['boxes'][iter_box][3][0]])
                        v_item = get_vertices(center_pt, R_bw_pt, size)

                        if iterr == iter_1 and iter_box != remove:
                            # Add vertex variables to make is ~30 dimension
                            this_sol.extend([ang_pt, v_item[0, 0], v_item[0, 1],
                                                     v_item[1, 0], v_item[1, 1],
                                                     v_item[2, 0], v_item[2, 1],
                                                     v_item[3, 0], v_item[3, 1]])
                        elif iterr == iter_2:
                            this_feature.extend(size)

                    if iterr == iter_1:
                        center_pt_remove = [0 - (bin_width / 2.0 - data[iterr]['boxes'][remove][0]),
                                            bin_height - data[iterr]['boxes'][remove][1] - 7]
                        ang_pt_remove = -data[iterr]['boxes'][remove][2]
                        R_bw_pt_remove = np.array([[np.cos(ang_pt_remove), -np.sin(ang_pt_remove)],
                                                   [np.sin(ang_pt_remove), np.cos(ang_pt_remove)]])
                        size_remove = np.array([data[iterr]['boxes'][remove][3][1], data[iterr]['boxes'][remove][3][0]])
                        v_item_remove = get_vertices(center_pt_remove, R_bw_pt_remove, size_remove)
                        this_sol.extend([ang_pt_remove, v_item_remove[0, 0], v_item_remove[0, 1],
                                                        v_item_remove[1, 0], v_item_remove[1, 1],
                                                        v_item_remove[2, 0], v_item_remove[2, 1],
                                                        v_item_remove[3, 0], v_item_remove[3, 1]])
                        sol_list.append(this_sol)

                        # print(this_sol)
                    elif iterr == iter_2:
                        this_feature.extend([height_in_hand, width_in_hand])  # TODO: this corresponds to the order of size
                                                                              # TODO: of each box, height first, then width
                        feature_list.append(this_feature)
    return sol_list