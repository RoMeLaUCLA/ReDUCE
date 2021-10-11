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

item_list = []
num_of_item = 3
for iter_item in range(num_of_item):
    item_list.append(Item(center_x[iter_item], center_y[iter_item], angle[iter_item], height[iter_item], width[iter_item]))

for iter_item in range(num_of_item):
    print(item_list[iter_item].return_flat_feature())

assert item_list[0].return_flat_feature() == [1, 54, -1, 10, 5]
assert item_list[1].return_flat_feature() == [2, 55, -2, 20, 10]
assert item_list[2].return_flat_feature() == [3, 56, -3, 30, 15]

geom = ShelfGeometry(shelf_width=176, shelf_height=110)

shelf = Shelf(stored_item_list=item_list, num_of_stored_item=3, shelf_geometry=geom,
              item_width_in_hand=25, item_height_in_hand=25)

print(shelf.return_flat_feature())
assert np.all(shelf.return_flat_feature() == [1, 54, -1, 10, 5, 2, 55, -2, 20, 10, 3, 56, -3, 30, 15, 25, 25])
print(shelf.return_stored_item_centers())
assert np.all(np.all(shelf.return_stored_item_centers() == np.array([[1, 54], [2, 55], [3, 56]])))

print(shelf.return_stored_item_heights())
assert np.all(shelf.return_stored_item_heights() == height)

print(shelf.return_stored_item_widths())
assert np.all(shelf.return_stored_item_widths() == width)

print(shelf.return_stored_item_angles())
assert np.all(shelf.return_stored_item_angles() == angle)

shelf2 = Shelf(stored_item_list=item_list, num_of_stored_item=3, shelf_geometry=geom,
              item_width_in_hand=-1, item_height_in_hand=-1)
print(shelf2.return_flat_feature())
assert np.all(shelf2.return_flat_feature() == [1, 54, -1, 10, 5, 2, 55, -2, 20, 10, 3, 56, -3, 30, 15])
# print(shelf2.item_width_in_hand)
