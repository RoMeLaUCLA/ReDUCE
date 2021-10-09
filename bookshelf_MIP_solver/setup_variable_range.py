import os, sys
dir_Learning_MIP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_Learning_MIP + "/classification_tests/utils")
import numpy as np
import copy
from max_min_trig import find_min_max_cos, find_min_max_sin
from add_McCormick_envelope_constraint import limit2vertex
import pdb

global num_of_item, bin_width, bin_height, item_width_stored

bin_left = -bin_width / 2.0
bin_right = bin_width / 2.0
bin_ground = 0.0
bin_up = bin_height

num_of_pairs = int((num_of_item + 1) * num_of_item / 2)

dim_2D = 2
num_of_vertices = 4
num_of_states_stored = 5
num_of_states_in_hand = 3  # Let's consider only 3 cases of in hand item to simplify the problem

# Create a list of pairs, note the last item [num_of_item] is item-in-hand
list_pairs = np.zeros([num_of_pairs, 2])

ct_row = 0
for iter_pairs in range(num_of_item):
    for iter_pairs_next in range(iter_pairs + 1, num_of_item + 1):
        list_pairs[ct_row, :] = np.array([iter_pairs, iter_pairs_next])
        ct_row += 1


def get_kb_from_2_pts(p1, p2):
    # This function returns the approximated line given 2 points on it
    kkk = (p1 ** 2 - p2 ** 2) / (p1 - p2)
    bbb = p2 ** 2 - kkk * p2  # p^2 = k*p + b

    assert abs(p1 * kkk + bbb - p1 ** 2) <= 1e-6, "Error: point p_lo does not lie on the quadratic curve!"
    assert abs(p2 * kkk + bbb - p2 ** 2) <= 1e-6, "Error: point p_hi does not lie on the quadratic curve!"

    return kkk, bbb

# For integer output, indicate which orientation the specific item is. 0.0 is to right (negative) and 1.0 is to left
# item_ranges = [[[-np.pi / 2.0 - 0.05, -3*np.pi / 8.0], [-3*np.pi / 8.0, -np.pi / 4.0], [-np.pi / 4.0, -np.pi / 8.0], [-np.pi / 8.0, 0.0],
#                 [0.0, np.pi / 8.0], [np.pi / 8.0, np.pi / 4.0], [np.pi / 4.0, 3*np.pi / 8.0], [3*np.pi / 8.0, np.pi / 2.0 + 0.05]],
#
#                [[-np.pi / 2.0 - 0.05, -3*np.pi / 8.0], [-3*np.pi / 8.0, -np.pi / 4.0], [-np.pi / 4.0, -np.pi / 8.0], [-np.pi / 8.0, 0.0],
#                 [0.0, np.pi / 8.0], [np.pi / 8.0, np.pi / 4.0], [np.pi / 4.0, 3*np.pi / 8.0], [3*np.pi / 8.0, np.pi / 2.0 + 0.05]],
#
#                [[-np.pi / 2.0 - 0.05, -3*np.pi / 8.0], [-3*np.pi / 8.0, -np.pi / 4.0], [-np.pi / 4.0, -np.pi / 8.0], [-np.pi / 8.0, 0.0],
#                 [0.0, np.pi / 8.0], [np.pi / 8.0, np.pi / 4.0], [np.pi / 4.0, 3*np.pi / 8.0], [3*np.pi / 8.0, np.pi / 2.0 + 0.05]],
#
#                [[-np.pi / 2.0 - 0.05, -3*np.pi / 8.0], [-3*np.pi / 8.0, -np.pi / 4.0], [-np.pi / 4.0, -np.pi / 8.0], [-np.pi / 8.0, 0.0],
#                 [0.0, np.pi / 8.0], [np.pi / 8.0, np.pi / 4.0], [np.pi / 4.0, 3*np.pi / 8.0], [3*np.pi / 8.0, np.pi / 2.0 + 0.05]]]

angle_offset = 0.05  # To reduce the numerical issue

item_ranges = [[[  -np.pi / 2.0 - angle_offset, -3*np.pi / 8.0 + angle_offset],
                [-3*np.pi / 8.0 - angle_offset,   -np.pi / 4.0 + angle_offset],
                [  -np.pi / 4.0 - angle_offset,   -np.pi / 8.0 + angle_offset],
                [  -np.pi / 8.0 - angle_offset,            0.0 + angle_offset],
                [           0.0 - angle_offset,    np.pi / 8.0 + angle_offset],
                [   np.pi / 8.0 - angle_offset,    np.pi / 4.0 + angle_offset],
                [   np.pi / 4.0 - angle_offset,  3*np.pi / 8.0 + angle_offset],
                [ 3*np.pi / 8.0 - angle_offset,    np.pi / 2.0 + angle_offset]],

               [[  -np.pi / 2.0 - angle_offset, -3*np.pi / 8.0 + angle_offset],
                [-3*np.pi / 8.0 - angle_offset,   -np.pi / 4.0 + angle_offset],
                [  -np.pi / 4.0 - angle_offset,   -np.pi / 8.0 + angle_offset],
                [  -np.pi / 8.0 - angle_offset,            0.0 + angle_offset],
                [           0.0 - angle_offset,    np.pi / 8.0 + angle_offset],
                [   np.pi / 8.0 - angle_offset,    np.pi / 4.0 + angle_offset],
                [   np.pi / 4.0 - angle_offset,  3*np.pi / 8.0 + angle_offset],
                [ 3*np.pi / 8.0 - angle_offset,    np.pi / 2.0 + angle_offset]],

               [[  -np.pi / 2.0 - angle_offset, -3*np.pi / 8.0 + angle_offset],
                [-3*np.pi / 8.0 - angle_offset,   -np.pi / 4.0 + angle_offset],
                [  -np.pi / 4.0 - angle_offset,   -np.pi / 8.0 + angle_offset],
                [  -np.pi / 8.0 - angle_offset,            0.0 + angle_offset],
                [           0.0 - angle_offset,    np.pi / 8.0 + angle_offset],
                [   np.pi / 8.0 - angle_offset,    np.pi / 4.0 + angle_offset],
                [   np.pi / 4.0 - angle_offset,  3*np.pi / 8.0 + angle_offset],
                [ 3*np.pi / 8.0 - angle_offset,    np.pi / 2.0 + angle_offset]],

               [[  -np.pi / 2.0 - angle_offset, -3*np.pi / 8.0 + angle_offset],
                [-3*np.pi / 8.0 - angle_offset,   -np.pi / 4.0 + angle_offset],
                [  -np.pi / 4.0 - angle_offset,   -np.pi / 8.0 + angle_offset],
                [  -np.pi / 8.0 - angle_offset,            0.0 + angle_offset],
                [           0.0 - angle_offset,    np.pi / 8.0 + angle_offset],
                [   np.pi / 8.0 - angle_offset,    np.pi / 4.0 + angle_offset],
                [   np.pi / 4.0 - angle_offset,  3*np.pi / 8.0 + angle_offset],
                [ 3*np.pi / 8.0 - angle_offset,    np.pi / 2.0 + angle_offset]]]

# [[-1.6207963267948966, -1.1280972450961724],   000
#  [-1.2280972450961725, -0.7353981633974482],   001
#  [-0.8353981633974483, -0.34269908169872415],  010
#  [-0.44269908169872413, 0.05],                 011
#  [-0.05, 0.44269908169872413],                 100
#  [0.34269908169872415, 0.8353981633974483],    101
#  [0.7353981633974482, 1.2280972450961725],     110
#  [1.1280972450961724, 1.6207963267948966]]     111

# The last one is the item in hand

num_of_int_R = [int(np.log2(len(item_ranges[iter_item]))) for iter_item in range(len(item_ranges))]

# Variables related to R^2 =========================================================================================
# For each item i, R_knots_xxxx[i] is a 2D array with row = number of sections.
# k_R_sq_xxxx[i] and b_R_sq_xxxx[i] are 1D arrays

# Stored items
# R0000: cos*cos   -----------------------------------------------------------------------------------------------------
R_knots_0000 = []
k_R_sq_0000 = []
b_R_sq_0000 = []
for iter_item in range(num_of_item+1):  # Stored plus in hand
    secs = item_ranges[iter_item]
    len_secs = len(secs)

    # 00 is cos term
    R_knots_0000.append([[find_min_max_cos(item_ranges[iter_item][iter_sect][0],
                                           item_ranges[iter_item][iter_sect][1])[0],
                          find_min_max_cos(item_ranges[iter_item][iter_sect][0],
                                           item_ranges[iter_item][iter_sect][1])[1]] for iter_sect in range(len_secs)])

    kk = []
    bb = []
    for iter_sect in range(len_secs):
        p_lo, p_hi = find_min_max_cos(item_ranges[iter_item][iter_sect][0], item_ranges[iter_item][iter_sect][1])
        kkk, bbb = get_kb_from_2_pts(p_lo, p_hi)
        kk.append(kkk)
        bb.append(bbb)

    k_R_sq_0000.append(kk)
    b_R_sq_0000.append(bb)

# R1010: sin*sin   -----------------------------------------------------------------------------------------------------
R_knots_1010 = []
k_R_sq_1010 = []
b_R_sq_1010 = []
for iter_item in range(num_of_item+1):
    secs = item_ranges[iter_item]
    len_secs = len(secs)

    # 10 is sin term
    R_knots_1010.append([[find_min_max_sin(item_ranges[iter_item][iter_sect][0],
                                           item_ranges[iter_item][iter_sect][1])[0],
                          find_min_max_sin(item_ranges[iter_item][iter_sect][0],
                                           item_ranges[iter_item][iter_sect][1])[1]] for iter_sect in range(len_secs)])

    kk = []
    bb = []
    for iter_sect in range(len_secs):
        p_lo, p_hi = find_min_max_sin(item_ranges[iter_item][iter_sect][0], item_ranges[iter_item][iter_sect][1])
        kkk, bbb = get_kb_from_2_pts(p_lo, p_hi)
        kk.append(kkk)
        bb.append(bbb)

    k_R_sq_1010.append(kk)
    b_R_sq_1010.append(bb)

# Variables related to R cross ===================================================================================
# R_wb_stored_0001 = R_wb_stored[i, 0, 0]*R_wb_stored[i, 0, 1]

len_sections_R_0001 = []
num_of_polygons_R_0001 = []
num_of_vertices_R_0001 = []
v_all_R_0001 = []

for iter_item in range(num_of_item+1):
    secs = item_ranges[iter_item]
    len_secs = len(secs)

    # R_wb_stored_0001 is cos*sin
    # cos term
    cos_lim = [[find_min_max_cos(item_ranges[iter_item][iter_sect][0], item_ranges[iter_item][iter_sect][1])[0],
                find_min_max_cos(item_ranges[iter_item][iter_sect][0], item_ranges[iter_item][iter_sect][1])[1]] for
               iter_sect in range(len_secs)]
    # -sin term: 01 is -sin
    sin_lim = [[-find_min_max_sin(item_ranges[iter_item][iter_sect][0], item_ranges[iter_item][iter_sect][1])[1],
                -find_min_max_sin(item_ranges[iter_item][iter_sect][0], item_ranges[iter_item][iter_sect][1])[0]] for
               iter_sect in range(len_secs)]

    num_of_polygons_R_0001_temp = len_secs  # The number of polygons to choose is the same as number of sections
    num_of_vertices_R_0001_temp = num_of_polygons_R_0001_temp * 4  # Each polygon has 4 vertices
    v_all_R_0001_temp = np.zeros([3, num_of_vertices_R_0001_temp])

    len_sections_R_0001.append(len_secs)
    num_of_polygons_R_0001.append(num_of_polygons_R_0001_temp)
    num_of_vertices_R_0001.append(num_of_vertices_R_0001_temp)  # Each polygon has 4 vertices

    # Generate v_all_R_cross
    iter_polygon = 0  # TODO: after verifying iter_sect has no bug, remove this variable
    for iter_sect in range(len_secs):
        bilinear_limits = [cos_lim[iter_sect][0], cos_lim[iter_sect][1],
                           sin_lim[iter_sect][0], sin_lim[iter_sect][1]]

        # Use limit2vertex function along each bilinear subspace, and collect all vertices
        vv = limit2vertex(bilinear_limits)

        assert iter_polygon == iter_sect, "iter_sect value is incorrect !!"

        v_all_R_0001_temp[:, 4 * iter_sect + 0] = np.array([vv[0, 0], vv[0, 1], vv[0, 2]])
        v_all_R_0001_temp[:, 4 * iter_sect + 1] = np.array([vv[1, 0], vv[1, 1], vv[1, 2]])
        v_all_R_0001_temp[:, 4 * iter_sect + 2] = np.array([vv[2, 0], vv[2, 1], vv[2, 2]])
        v_all_R_0001_temp[:, 4 * iter_sect + 3] = np.array([vv[3, 0], vv[3, 1], vv[3, 2]])

        iter_polygon += 1

    assert iter_polygon == num_of_polygons_R_0001_temp, "The final value of iter_polygon is incorrect!"

    v_all_R_0001.append(v_all_R_0001_temp)

# Variables related to a^2 =========================================================================================
a_offset = 0.05

a_knots_00 = []
k_a_sq_00 = []
b_a_sq_00 = []

for iter_pair in range(num_of_pairs):

    # a_knots_00.append([[-1.0, -0.75], [-0.75, -0.5], [-0.5, -0.25], [-0.25, 0.0],
    #                    [0.0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]])

    a_knots_00.append([[ -1.0-a_offset, -0.75+a_offset],
                       [-0.75-a_offset,  -0.5+a_offset],
                       [ -0.5-a_offset, -0.25+a_offset],
                       [-0.25-a_offset,   0.0+a_offset],
                       [  0.0-a_offset,  0.25+a_offset],
                       [ 0.25-a_offset,   0.5+a_offset],
                       [  0.5-a_offset,  0.75+a_offset],
                       [ 0.75-a_offset,   1.0+a_offset]])

    len_sec_00 = len(a_knots_00[iter_pair])
    kk = []
    bb = []

    for iter_sec in range(len_sec_00):
        p_lo = a_knots_00[iter_pair][iter_sec][0]
        p_hi = a_knots_00[iter_pair][iter_sec][1]
        kkk, bbb = get_kb_from_2_pts(p_lo, p_hi)
        kk.append(kkk)
        bb.append(bbb)

    k_a_sq_00.append(kk)
    b_a_sq_00.append(bb)

num_of_int_a_00 = [int(np.log2(len(a_knots_00[iter_pair]))) for iter_pair in range(len(a_knots_00))]

# ------------------------------------------------------------------------------------------------------------------
a_knots_11 = []
k_a_sq_11 = []
b_a_sq_11 = []

for iter_pair in range(num_of_pairs):

    # a_knots_11.append([[-1.0, -0.75], [-0.75, -0.5], [-0.5, -0.25], [-0.25, 0.0],
    #                    [0.0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]])

    a_knots_11.append([[ -1.0-a_offset, -0.75+a_offset],
                       [-0.75-a_offset,  -0.5+a_offset],
                       [ -0.5-a_offset, -0.25+a_offset],
                       [-0.25-a_offset,   0.0+a_offset],
                       [  0.0-a_offset,  0.25+a_offset],
                       [ 0.25-a_offset,   0.5+a_offset],
                       [  0.5-a_offset,  0.75+a_offset],
                       [ 0.75-a_offset,   1.0+a_offset]])

    len_sec_11 = len(a_knots_11[iter_pair])
    kk = []
    bb = []

    for iter_sec in range(len_sec_11):
        p_lo = a_knots_11[iter_pair][iter_sec][0]
        p_hi = a_knots_11[iter_pair][iter_sec][1]
        kkk, bbb = get_kb_from_2_pts(p_lo, p_hi)
        kk.append(kkk)
        bb.append(bbb)

    k_a_sq_11.append(kk)
    b_a_sq_11.append(bb)

num_of_int_a_11 = [int(np.log2(len(a_knots_11[iter_pair]))) for iter_pair in range(len(a_knots_11))]

# Variables related to a x v cross =================================================================================
v_size = max((bin_up - bin_ground), (bin_right - bin_left))
horizontal_margin = bin_right - bin_left - sum(item_width_stored)  # Horizontal leftover spaces
bin_width = bin_right - bin_left
bin_height = bin_up - bin_ground

bin_offset = 1.5

v_knots_common_x = [[bin_left - bin_offset, bin_left + 0.25 * bin_width],  # Add 0.2 offset to cease numerical issue
                    [bin_left + 0.25 * bin_width, bin_left + 0.5 * bin_width],
                    [bin_left + 0.5 * bin_width, bin_left + 0.75 * bin_width],
                    [bin_left + 0.75 * bin_width, bin_left + bin_width + bin_offset]]
assert (bin_left + bin_width) == bin_right, "Bin width doesn't match bin left/right !!"

v_knots_common_y = [[bin_ground - bin_offset, bin_ground + 0.25 * bin_height],
                    [bin_ground + 0.25 * bin_height, bin_ground + 0.5 * bin_height],
                    [bin_ground + 0.5 * bin_height, bin_ground + 0.75 * bin_height],
                    [bin_ground + 0.75 * bin_height, bin_ground + bin_height + bin_offset]]

# [[-89.5, -44.0],  00
#  [-44.0, 0.0],    01
#  [0.0, 44.0],     10
#  [44.0, 89.5]]    11
#
# [[-1.5, 27.5],    00
#  [27.5, 55.0],    01
#  [55.0, 82.5],    10
#  [82.5, 111.5]]   11

v_knots_av = []
a_knots_av = []
num_of_polygons_av = []

num_of_integer_v = int(np.log2(len(v_knots_common_x)))
num_of_vertices_av = []
v_all_av = []

# a_times_v: #pair, which item in pair, 2D plane dimension, #vertex
for iter_pair in range(num_of_pairs):
    v_knots_av.append([])
    a_knots_av.append([])
    num_of_polygons_av.append([])
    num_of_vertices_av.append([])
    v_all_av.append([])

    for which_item in range(2):
        v_knots_av[iter_pair].append([])
        a_knots_av[iter_pair].append([])
        num_of_polygons_av[iter_pair].append([])
        num_of_vertices_av[iter_pair].append([])
        v_all_av[iter_pair].append([])

        for iter_dim in range(2):
            v_knots_av[iter_pair][which_item].append([])
            a_knots_av[iter_pair][which_item].append([])
            num_of_polygons_av[iter_pair][which_item].append([])
            num_of_vertices_av[iter_pair][which_item].append([])
            v_all_av[iter_pair][which_item].append([])

            for iter_vertex in range(num_of_vertices):

                if iter_dim == 0:
                    v_knots_av_temp = copy.deepcopy(v_knots_common_x)
                elif iter_dim == 1:
                    v_knots_av_temp = copy.deepcopy(v_knots_common_y)
                else:
                    assert False, "What is going on ??"

                # This section is same as above a direction so just use the same integer variables
                if iter_dim == 0:
                    a_knots_av_temp = copy.deepcopy(a_knots_00[iter_pair])
                elif iter_dim == 1:
                    a_knots_av_temp = copy.deepcopy(a_knots_11[iter_pair])
                else:
                    assert False, "Something is wrong !!"

                len_sections_a_temp = len(a_knots_av_temp)
                len_sections_v_temp = len(v_knots_av_temp)
                num_of_polygons_av_temp = len_sections_a_temp * len_sections_v_temp
                num_of_vertices_av_temp = num_of_polygons_av_temp * 4  # Each polygon has 4 vertices

                v_knots_av[iter_pair][which_item][iter_dim].append(v_knots_av_temp)
                a_knots_av[iter_pair][which_item][iter_dim].append(a_knots_av_temp)

                num_of_polygons_av[iter_pair][which_item][iter_dim].append(num_of_polygons_av_temp)
                num_of_vertices_av[iter_pair][which_item][iter_dim].append(num_of_vertices_av_temp)

                v_all_av_temp = np.zeros([3, num_of_vertices_av_temp])

                iter_polygon = 0
                for iter_a in range(len_sections_a_temp):
                    for iter_v in range(len_sections_v_temp):
                        bilinear_limits = [a_knots_av_temp[iter_a][0], a_knots_av_temp[iter_a][1],
                                           v_knots_av_temp[iter_v][0], v_knots_av_temp[iter_v][1]]
                        # Use limit2vertex function along each bilinear subspace, and collect all vertices
                        vv = limit2vertex(bilinear_limits)

                        v_all_av_temp[:, 4 * iter_polygon + 0] = np.array([vv[0, 0], vv[0, 1], vv[0, 2]])
                        v_all_av_temp[:, 4 * iter_polygon + 1] = np.array([vv[1, 0], vv[1, 1], vv[1, 2]])
                        v_all_av_temp[:, 4 * iter_polygon + 2] = np.array([vv[2, 0], vv[2, 1], vv[2, 2]])
                        v_all_av_temp[:, 4 * iter_polygon + 3] = np.array([vv[3, 0], vv[3, 1], vv[3, 2]])

                        iter_polygon += 1

                v_all_av[iter_pair][which_item][iter_dim].append(v_all_av_temp)
