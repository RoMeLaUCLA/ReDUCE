import numpy as np
import os, sys
dir_curr = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_curr)
from get_vertices import get_vertices
offset = 3  # To account for numerical issue


class Item:
    def __init__(self, center_x, center_y, angle, height, width):
        self.center_x = center_x
        self.center_y = center_y
        self.angle = angle
        self.R_bw = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle),  np.cos(angle)]])
        self.height = height
        self.width = width

    def return_center(self):
        return np.array([self.center_x, self.center_y])

    def return_flat_feature(self):
        return [self.center_x, self.center_y, self.angle, self.height, self.width]

    def get_vertices(self):
        return get_vertices(np.array([self.center_x, self.center_y]),
                            self.R_bw, np.array([self.height, self.width]))


class ShelfGeometry:
    def __init__(self, shelf_width, shelf_height):
        self.shelf_width = shelf_width
        self.shelf_height = shelf_height

        self.shelf_left = -shelf_width / 2.0
        self.shelf_right = shelf_width / 2.0
        self.shelf_ground = 0.0
        self.shelf_up = shelf_height

        self.v_bin = np.array([[self.shelf_right, self.shelf_up],
                               [self.shelf_right, self.shelf_ground],
                               [self.shelf_left, self.shelf_ground],
                               [self.shelf_left, self.shelf_up]])

    def item_inside(self, item):
        """
        Check if an given item is inside the shelf
        """
        assert isinstance(item, Item), "From ShelfGeometry class: item does not stay inside the shelf!"
        vertex_item = get_vertices(item.return_center(), item.R_bw, np.array([item.height, item.width]))
        # Item is inside the shelf as long as all vertices are inside the shelf
        for item_ct in range(4):
            assert self.shelf_right + offset >= vertex_item[item_ct, 0] >= self.shelf_left - offset, "Item not inside the shelf!"
            assert self.shelf_up + offset >= vertex_item[item_ct, 1] >= self.shelf_ground - offset, "Item not inside the shelf!"


class Shelf:
    def __init__(self, stored_item_list, num_of_stored_item, shelf_geometry, item_width_in_hand=-1, item_height_in_hand=-1):
        assert len(stored_item_list) == num_of_stored_item, "From shelf class: Inconsistent number of item!"
        for iter_item in range(num_of_stored_item):
            assert isinstance(stored_item_list[iter_item], Item), "From shelf class: item list must contain item objects!"
        assert isinstance(shelf_geometry, ShelfGeometry), "From shelf class: shelf geometry must be a ShelfGeometry object!"

        self.item_list = stored_item_list
        self.num_of_item = num_of_stored_item
        self.shelf_geometry = shelf_geometry

        # Check if each item is inside the shelf
        for iter_item in range(num_of_stored_item):
            shelf_geometry.item_inside(stored_item_list[iter_item])

        if item_width_in_hand > 0 and item_height_in_hand > 0:
            self.contain_in_hand_item = True
            self.item_width_in_hand = item_width_in_hand
            self.item_height_in_hand = item_height_in_hand
        else:
            self.contain_in_hand_item = False

    def return_flat_feature(self):
        ret = []

        for iter_item in range(self.num_of_item):
            ret += self.item_list[iter_item].return_flat_feature()

        if self.contain_in_hand_item:
            ret += [self.item_height_in_hand, self.item_width_in_hand]

        return np.array(ret)

    def return_stored_item_widths(self):
        ret = []
        for iter_item in range(self.num_of_item):
            ret.append(self.item_list[iter_item].width)
        return np.array(ret)

    def return_stored_item_heights(self):
        ret = []
        for iter_item in range(self.num_of_item):
            ret.append(self.item_list[iter_item].height)
        return np.array(ret)

    def return_stored_item_angles(self):
        ret = []
        for iter_item in range(self.num_of_item):
            ret.append(self.item_list[iter_item].angle)
        return np.array(ret)

    def return_stored_item_centers(self):
        ret = []
        for iter_item in range(self.num_of_item):
            ret.append([self.item_list[iter_item].center_x, self.item_list[iter_item].center_y])
        return np.array(ret)

    def return_stored_item_center_x(self):
        ret = []
        for iter_item in range(self.num_of_item):
            ret.append(self.item_list[iter_item].center_x)
        return np.array(ret)

    def return_stored_item_center_y(self):
        ret = []
        for iter_item in range(self.num_of_item):
            ret.append(self.item_list[iter_item].center_y)
        return np.array(ret)