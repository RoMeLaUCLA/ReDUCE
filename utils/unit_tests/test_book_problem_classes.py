import os, sys
dir_utils = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_utils)
import numpy as np
from book_problem_classes import Item, ShelfGeometry, Shelf

# 3 items
center_x = [1, 2, 3]
center_y = [54, 55, 56]
angle = [-1, -2, -3]
height = [10, 20, 30]
width = [5, 10, 15]
width_in_hand = 25
height_in_hand = 30

item_list = []
num_of_item = 3
for iter_item in range(num_of_item):
    item_list.append(Item(center_x[iter_item], center_y[iter_item], angle[iter_item], height[iter_item], width[iter_item]))

print("Item features ======================================================================")
for iter_item in range(num_of_item):
    print(item_list[iter_item].return_flat_feature())

assert item_list[0].return_flat_feature() == [1, 54, -1, 10, 5]
assert item_list[1].return_flat_feature() == [2, 55, -2, 20, 10]
assert item_list[2].return_flat_feature() == [3, 56, -3, 30, 15]

geom = ShelfGeometry(shelf_width=176, shelf_height=110)

shelf = Shelf(stored_item_list=item_list, num_of_stored_item=3, shelf_geometry=geom,
              item_width_in_hand=width_in_hand, item_height_in_hand=height_in_hand)

print("Item centers =======================================================================")
print(shelf.return_stored_item_centers())
assert np.all(np.all(shelf.return_stored_item_centers() == np.array([[1, 54], [2, 55], [3, 56]])))

print("Item center x ======================================================================")
print(shelf.return_stored_item_center_x())
assert np.all(shelf.return_stored_item_center_x() == np.array(center_x))

print("Item center y ======================================================================")
print(shelf.return_stored_item_center_y())
assert np.all(shelf.return_stored_item_center_y() == np.array(center_y))

print("Item heights =======================================================================")
print(shelf.return_stored_item_heights())
assert np.all(shelf.return_stored_item_heights() == height)

print("Item widths ========================================================================")
print(shelf.return_stored_item_widths())
assert np.all(shelf.return_stored_item_widths() == width)

print("Item angles ========================================================================")
print(shelf.return_stored_item_angles())
assert np.all(shelf.return_stored_item_angles() == angle)

print("Shelf feature ======================================================================")
print(shelf.return_feature().flatten())
assert np.all(shelf.return_feature().flatten() == [1, 54, -1, 10, 5, 2, 55, -2, 20, 10, 3, 56, -3, 30, 15,
                                                   height_in_hand, width_in_hand])

# Create shelf from flatten feature
rr = shelf.flat_feature_to_init_list(shelf.return_feature().flatten(), geom)
shelf_from_flat_feature = Shelf(rr[0], rr[1], rr[2], rr[3], rr[4])

for iter_item in range(num_of_item):
    assert shelf_from_flat_feature.item_list[iter_item].center_x == shelf.item_list[iter_item].center_x, "Error in recreated shelf from flat feature!"
    assert shelf_from_flat_feature.item_list[iter_item].center_y == shelf.item_list[iter_item].center_y, "Error in recreated shelf from flat feature!"
    assert shelf_from_flat_feature.item_list[iter_item].angle == shelf.item_list[iter_item].angle, "Error in recreated shelf from flat feature!"
    assert shelf_from_flat_feature.item_list[iter_item].height == shelf.item_list[iter_item].height, "Error in recreated shelf from flat feature!"
    assert shelf_from_flat_feature.item_list[iter_item].width == shelf.item_list[iter_item].width, "Error in recreated shelf from flat feature!"

assert shelf_from_flat_feature.num_of_item == shelf.num_of_item, "Error in recreated shelf from flat feature!"
assert shelf_from_flat_feature.shelf_geometry == shelf.shelf_geometry, "Error in recreated shelf from flat feature!"

print("Shelf feature dictionary ===========================================================")
dd = shelf.return_feature().flatten_to_dictionary()
print(dd)
assert np.all(dd['item_center_x_stored'] == np.array(center_x))
assert np.all(dd['item_center_y_stored'] == np.array(center_y))
assert np.all(dd['item_angle_stored'] == np.array(angle))
assert np.all(dd['item_width_stored'] == np.array(width))
assert np.all(dd['item_height_stored'] == np.array(height))
assert np.all(dd['item_width_in_hand'] == np.array(width_in_hand))
assert np.all(dd['item_height_in_hand'] == np.array(height_in_hand))


shelf2 = Shelf(stored_item_list=item_list, num_of_stored_item=3, shelf_geometry=geom,
              item_width_in_hand=-1, item_height_in_hand=-1)
assert np.all(np.all(shelf2.return_stored_item_centers() == np.array([[1, 54], [2, 55], [3, 56]])))
assert np.all(shelf2.return_stored_item_center_x() == np.array(center_x))
assert np.all(shelf2.return_stored_item_center_y() == np.array(center_y))
assert np.all(shelf2.return_stored_item_heights() == height)
assert np.all(shelf2.return_stored_item_widths() == width)
assert np.all(shelf2.return_stored_item_angles() == angle)

print("list of item angles and vertices ==================================================================")
for itt in range(shelf2.num_of_item):
    print(shelf2.item_list[itt].angle)
    print(shelf2.item_list[itt].v_item)

print("Return point on non-convex manifold ===============================================================")
nonconvex_point = shelf2.return_nonconvex_point(1)
print(nonconvex_point)

print("============================= The following test is supposed to fail: =============================")
fail = shelf2.item_width_in_hand
