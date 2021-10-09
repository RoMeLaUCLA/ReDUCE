# Center positions and orientations of stored items
# Sizes of stored items
# Size of item in hand
# Mode
# Image (before)

import os, sys
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_ReDUCE+"/utils")
from get_vertices import get_vertices, plot_rectangle
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image as im
import pdb

data_file_list = ['zero_remove_angle_auto_generated_boxes_data_ba_104006.pkl']  # A list of dataset files
dataset_save_name = "0_scene_dataset.pkl"

bad_data_list = [[]]  # A list of bad data that will be excluded from the dataset

len_files = len(data_file_list)

data_all = []

for iter_file in range(len_files):
    with open("bookshelf_scene_data/" + data_file_list[iter_file], 'rb') as f:
        data = pickle.load(f)
        bin_width = data[-1][0]
        bin_height = data[-1][1]

        print("Reading dataset {}. Length of data is {}".format(data_file_list[iter_file], len(data)))

        for iter_data in range(int((len(data)-1)/2)):

            if not (iter_data in bad_data_list[iter_file]):
                print("Data number {} --------------------------------------------".format(iter_data))

                iter_1 = 2 * iter_data
                iter_2 = 2 * iter_data + 1

                mask_o = data[iter_2]['image']
                mask = np.zeros(np.shape(mask_o))
                for iter_row in range(np.shape(mask_o)[0]):
                    for iter_column in range(np.shape(mask_o)[1]):
                        mask[iter_row, iter_column] = mask_o[iter_row, iter_column]
                img = im.fromarray(mask)
                #img.show()

                center = []
                angle = []
                width = []
                height = []
                vertices = []

                # Only take data after removal - generated scenes
                print("Remove item {}".format(data[iter_2]['remove']))

                remove = data[iter_2]['remove']

                width_in_hand = data[iter_1]['boxes'][remove][3][0]
                height_in_hand = data[iter_1]['boxes'][remove][3][1]

                for iter_box in range(len(data[iter_2]['boxes'])):

                    # Note horizontally flipped
                    center_pt = [0-(bin_width/2.0-data[iter_2]['boxes'][iter_box][0]), bin_height-data[iter_2]['boxes'][iter_box][1]-7]
                    ang_pt = -data[iter_2]['boxes'][iter_box][2]
                    R_bw_pt = np.array([[np.cos(ang_pt), -np.sin(ang_pt)],
                                        [np.sin(ang_pt),  np.cos(ang_pt)]])

                    size = np.array([data[iter_2]['boxes'][iter_box][3][1], data[iter_2]['boxes'][iter_box][3][0]])

                    center.append(center_pt)
                    angle.append(ang_pt)
                    width.append(data[iter_2]['boxes'][iter_box][3][0])
                    height.append(data[iter_2]['boxes'][iter_box][3][1])

                    vv = get_vertices(center_pt, R_bw_pt, size)
                    vertices.append(vv)
 
                data_pt = {'center': center, 'angle': angle, 'width': width, 'height': height,
                           'width_in_hand': width_in_hand, 'height_in_hand': height_in_hand, 'image': img,
                           'bin_width': bin_width, 'bin_height': bin_height}

                data_all.append(data_pt)

with open("bookshelf_scene_data/" + dataset_save_name, "wb") as f:
    pickle.dump(data_all, f)
