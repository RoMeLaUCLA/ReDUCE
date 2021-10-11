# Center positions and orientations of stored items
# Sizes of stored items
# Size of item in hand
# Mode
# Image (before)

import os, sys
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_ReDUCE+"/utils")
sys.path.append(dir_ReDUCE+"/bookshelf_generator/bookshelf_scene_data")
from get_vertices import get_vertices, plot_rectangle
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image as im
import pdb

data_file_list = ['zero_remove_angle_auto_generated_boxes_data_ba_{ext}.pkl']  # A list of dataset files
dataset_save_name = "0_scene_dataset.pkl"

bad_data_list = [[]]  # A list of bad data that will be excluded from the dataset

len_files = len(data_file_list)

data_all = []

for iter_file in range(len_files):
    with open("bookshelf_scene_data/" + data_file_list[iter_file], 'rb') as f:
        data = pickle.load(f)

        inst_shelf_geometry = data[0]["after"]["shelf"].shelf_geometry

        bin_width = inst_shelf_geometry.shelf_width
        bin_height = inst_shelf_geometry.shelf_height
        num_of_item_stored = data[0]["after"]["shelf"].num_of_item

        print("Reading dataset {}. Length of data is {}".format(data_file_list[iter_file], len(data)))

        for iter_data in range(len(data)):

            if not (iter_data in bad_data_list[iter_file]):
                print("Data number {} --------------------------------------------".format(iter_data))

                mask_o = data[iter_data]["after"]['image']
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
                remove = data[iter_data]["after"]['remove']
                print("Remove item {}".format(remove))

                width_in_hand = data[iter_data]["after"]["shelf"].item_height_in_hand
                height_in_hand = data[iter_data]["after"]["shelf"].item_width_in_hand

                for iter_box in range(num_of_item_stored):
                    center.append(data[iter_data]["after"]["shelf"].return_stored_item_centers())
                    angle.append(data[iter_data]["after"]["shelf"].return_stored_item_angles())
                    width.append(data[iter_data]["after"]["shelf"].return_stored_item_widths())
                    height.append(data[iter_data]["after"]["shelf"].return_stored_item_heights())
 
                data_pt = {'center': center, 'angle': angle, 'width': width, 'height': height,
                           'width_in_hand': width_in_hand, 'height_in_hand': height_in_hand, 'image': img,
                           'bin_width': bin_width, 'bin_height': bin_height}

                data_all.append(data_pt)

with open("bookshelf_scene_data/" + dataset_save_name, "wb") as f:
    pickle.dump(data_all, f)
