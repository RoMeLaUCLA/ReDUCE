"""Simulates random arrangement of books then takes a book out--data generation"""
import numpy as np
import pygame
import random
import pickle
import pymunk
import pymunk.pygame_util
from classification import to_pandas
from sklearn import preprocessing
from datetime import datetime
from PIL import Image as im
import time
import os, sys
dir_curr_path = os.path.dirname(os.path.realpath(__file__))
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_ReDUCE+"/utils")
from book_problem_classes import Item, ShelfGeometry, Shelf

# TODO: documentation: pipeline: automatic_data_classification (this file) -> dataset_generation ->
#  solve_mode_0 -> separate_train_test_data

# TODO: If remove book can alter such that positions around that
#       are stable can also be used as possible examples


def img_to_pix(surf):
    img_data = pygame.surfarray.array2d(surf)
    img_data = img_data.swapaxes(0, 1)
    return img_data


def init_env(box_w, box_h, width, height, ang, x_pos, m, mu, num_boxes):
    seed = datetime.now()
    random.seed(seed)

    # Setup the base Pymunk Space.
    space = pymunk.Space()
    space.gravity = 0, 1000
    space.sleep_time_threshold = float("inf")  # Pause sim threshold

    # Draw outline of the shelf--top open
    # Note: Upper left corner is (0, 0)
    box = [(0, 0), (0, height), (width, height), (width, 0)]
    for i in range(len(box) - 1):
        l = pymunk.Segment(space.static_body, box[i], box[i+1], 5)  # last elm radius
        l.elasticity = 0.5
        l.friction = 1
        l.color = pygame.Color('black')
        space.add(l)
    
    rel_xpos = 0
    rel_ypos = height / 3
    for i in range(num_boxes):
        # Randomize box height width
        bw = random.randint(*box_w) / 2
        bh = random.randint(*box_h) / 2
        
        # Set up shape body
        bod = pymunk.Body()
        y_pos = random.randint(-20, 20)  # Can adjust and no roof
        x_spacing = random.randint(*x_pos)        # Can adjust but bounded
        bod.position = rel_xpos + x_spacing + bw / 2, rel_ypos + y_pos
        tilt = random.uniform(*ang)
        bod.angle = tilt        

        # Update relative position
        rel_xpos = bw / 2 + bod.position[0]
        rel_ypos = bod.position[1]
        
        # Build shape
        vert = [(-bw, -bh), (-bw, bh), (bw, bh), (bw, -bh)]
        box = pymunk.Poly(bod, vert, radius=2) # Radius adds outline and smooth corners
        box.mass = m
        box.friction = mu
        box.color = pygame.Color('green')
        box.wh = (2 * bw, 2 * bh)
        
        # Add shape to drawing
        space.add(box, bod)
    
    return space


def on_ground(boxes, ground):
    """Checks to see all boxes are touching ground"""
    for b in boxes:
        if len(b.shapes_collide(ground).points) == 0:
            return False
    return True


def moving(boxes, threshold=5e-25):
    """Checks to see if boxes have kinetic energy ie moving"""
    for b in boxes:
        if b.body.kinetic_energy >= threshold:
           return True
    return False


def stability_check(t, dt, max_threshold, space, screen, draw_options, surf, boxes, ground, count):
    energy_count = 0
    while t <= max_threshold:
        space.step(dt)
        screen.fill(pygame.Color("white"))  # Clear screen
        surf.fill(pygame.Color("white"))  # Draw
        space.debug_draw(draw_options)
        screen.blit(surf, (0, 0))  # Coordinates to place surface
        pygame.display.flip()  # Update full display surface
        t += dt
        if not moving(boxes):
            energy_count += 1
        # Added count since kinetic energy may be zero for instant
        if energy_count >= count:
            reset = False
            break
        else:
            reset = True
    
    # Check if books touching ground
    if not on_ground(boxes, ground):
        reset = True

    return reset


def main(num_data=50):
    ### Parameters to input
    # Boxes will be randomly generated
    num_data = num_data       # number of data points to generate
    percent_straight = 0.2
    box_w = (10, 40)    # box width range
    box_h = (50, 90)    # box height range
    ang = (-1.2, 1.2)   # tilt range for initialization
    m = 5               # mass
    mu = 0.1            # friction
    num_boxes = 4       # number of boxes
    save_data = False   # save data generated
    # Note: On initialization blocks may fall through boundary
    # Saves order: (x_pos, y_pos, angle, (bin_width, bin_height)), image
    #              at the very end the shelf dimensions are saved

    # Shelf/Screen size: function of above parameters
    bin_width, bin_height = (box_w[1] + 4) * num_boxes, box_h[1] + 20
    inst_shelf_geometry = ShelfGeometry(shelf_width=bin_width, shelf_height=bin_height)

    # Range for random spacing between position
    x_pos = (5, bin_width / num_boxes - 1.5 * num_boxes)

    # Setup pygame screen and surface
    pygame.init()
    screen = pygame.display.set_mode((bin_width, bin_height))
    surf = pygame.Surface((bin_width, bin_height))
    draw_options = pymunk.pygame_util.DrawOptions(surf)
    draw_options.shape_outline_color = (0, 0, 0, 255)

    # Init environment
    args = (box_w, box_h, bin_width, bin_height, ang, x_pos, m, mu, num_boxes)
    space = init_env(*args)

    # Find ground and boxes, same order as placed
    ground = space.shapes[1]
    boxes = []
    for s in space.shapes:
        if isinstance(s, pymunk.shapes.Poly):
            boxes.append(s)

    t = 0
    max_threshold = 5.0  # To terminate if blocks haven't settled
    count = 3  # For how many steps does kinetic energy threshold have to be lower than a number
    saves = []
    fps = 60     # Update physics rate
    dt = 1.0 / fps
    ct_straight = 0
    while len(saves) < num_data:
        # Draw new screen
        screen.fill(pygame.Color("white"))  # Clear screen
        surf.fill(pygame.Color("white"))  # Draw
        space.debug_draw(draw_options)
        screen.blit(surf, (0, 0))  # Coordinates to place surface
        pygame.display.flip()  # Update full display surface

        print("Resetting environment...")
        before = {}  # saves before removing book
        after = {}  # saves after removing book
        reset = False  # reset flag

        # Check to see that all books have settled
        reset = stability_check(t, dt, max_threshold, space, screen, draw_options, surf, boxes, ground, count)
        
        if not reset:
            # Save the before data
            before["boxes"] = []
            for b in boxes:
                # before["boxes"].append((b.body.position[0], b.body.position[1], b.body.angle, b.wh))
                # Note: that pos y-axis goes from top to bottom and difference of 7 because
                #       boundary of shelf (5) + box (2) = 7
                before["boxes"].append(Item(center_x=b.body.position[0] - bin_width / 2.0,
                                            center_y=bin_height - b.body.position[1] - 7,
                                               angle=-b.body.angle, height=b.wh[1], width=b.wh[0]))
            pix = img_to_pix(surf.copy())
            before["image"] = pix
            before["shelf"] = Shelf(before["boxes"], num_of_stored_item=num_boxes, shelf_geometry=inst_shelf_geometry,
                                    item_width_in_hand=-1, item_height_in_hand=-1)

            # Remove box and reset time
            t = 0
            angles = []
            for b in boxes:
                angles.append(abs(b.body.angle))
            min_angle = min(angles)
            remove_input = angles.index(min_angle)  # Remove the box with smallest tilting angle
            i = remove_input
            width_removed, height_removed = boxes[i].wh
            space.remove(boxes[i])
            boxes.pop(i)

            # Redo stability checks
            reset = stability_check(t, dt, max_threshold, space, screen, draw_options, surf, boxes, ground, count)

            # If the smallest tilting angle is not very close to zero, reset
            if min_angle > 0.2:
                reset = True

        # Check if all books are almost straight up. If so, throw it away if it is already more than certain percentage
        angles_check = []
        for b in boxes:
            angles_check.append(b.body.angle)

        if all([abs(angles_check[ii]) <= 0.05 for ii in range(len(angles_check))]):
            ct_straight += 1

        if ct_straight > percent_straight*(num_data/2.0):
            ct_straight -= 1
            reset = True

        # Save if after remove book it's stable
        if not reset:
            print("------------------")
            print(f"Sample: {len(saves)}")
            after["boxes"] = []
            for b in boxes:
                after["boxes"].append(Item(center_x=b.body.position[0] - bin_width / 2.0,
                                           center_y=bin_height - b.body.position[1] - 7,
                                              angle=-b.body.angle, height=b.wh[1], width=b.wh[0]))

            pix = img_to_pix(surf.copy())
            after["image"] = pix
            after["shelf"] = Shelf(after["boxes"], num_of_stored_item=num_boxes-1, shelf_geometry=inst_shelf_geometry,
                                    item_width_in_hand=width_removed, item_height_in_hand=height_removed)

            # Automatic mode selection
            num_boxes_stored = len(after['boxes'])
            center_list = []
            angle_list = []
            width_list = []
            height_list = []

            for iter_stored in range(num_boxes_stored):
                # Translate center and angle, directly from dataset_generation.py
                center_list.append([b.body.position[0] - bin_width / 2.0, bin_height - b.body.position[1] - 7])
                angle_list.append(-b.body.angle)
                width_list.append(b.wh[0])
                height_list.append(b.wh[1])

            # data_raw = {'center': center_list, 'angle': angle_list, 'width': width_list, 'height': height_list,
            #             'mode': 0, 'width_in_hand': width_in_hand, 'height_in_hand': height_in_hand}

            # if normalize:
            #     data_pandas = to_pandas([data_raw], normalize=normalize, use_normalization=normalization_scaler)
            # else:
            #     data_pandas = to_pandas([data_raw], normalize=normalize)
            #
            # # Remove the labels features
            # data_pandas = data_pandas.drop('mode0', axis=1)
            # data_pandas = data_pandas.drop('mode1', axis=1)
            #
            # # Change to numpy array
            # data_np = np.array(data_pandas)
            # prediction = rf_model.predict(data_np)
            # mode = 0 if prediction[0][0] == 1 else 1

            after["remove"] = remove_input
            assert (before["boxes"][remove_input].height == after["shelf"].item_height_in_hand) \
                   and (before["boxes"][remove_input].width == after["shelf"].item_width_in_hand), \
                   "From bookshelf generator: Inconsistent removed item width or height !!"

            print("------------------------SUCCESS------------------------------")

            saves.append({"before": before, "after": after})

        # Reset environment          
        t = 0
        space = init_env(*args)
        ground = space.shapes[1]
        boxes = []
        for s in space.shapes:
            if isinstance(s, pymunk.shapes.Poly):
                boxes.append(s)

    # Save shelf dimension
    if save_data:
        # print(saves)
        ext = datetime.now().strftime("%H%M%S")
        with open(dir_curr_path + "/bookshelf_scene_data/zero_remove_angle_auto_generated_boxes_data_ba_"+ext+".pkl", "wb") as f:
            pickle.dump(saves, f)

    return saves


if __name__ == "__main__":
    import sys

    sys.exit(main())
