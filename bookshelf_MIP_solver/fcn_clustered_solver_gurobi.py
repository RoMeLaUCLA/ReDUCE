import os, sys
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_ReDUCE + "/utils")

import gurobipy as go
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.get_vertices import get_vertices, plot_rectangle, plot_bilinear
from max_min_trig import find_min_max_cos, find_min_max_sin
from dec2bin import dec2bin
from create_int_char_list import create_int_char_list
from add_McCormick_envelope_constraint import add_vertex_polytope_constraint_gurobi, limit2vertex, \
    add_bilinear_constraint_gurobi
from add_piecewise_linear_constraint import add_piecewise_linear_constraint
from reassign_int_var import reassign_int_var
from generate_int_list import generate_int_list
from utils.get_pair_number import get_pair_number
import math, time, copy, runpy
import pickle
import pdb


# This script is for first mode where all items are kept the same integer state

# Since we assume easy order of items inside the bin, the input items should always be counting from left to
# right 0, 1, 2, 3, 4, 5, etc


def solve_within_patch(shelf_data, data_patch_indicator, count, iter_data, time_lim=-1, use_warm_start=False):

    bin_width = shelf_data.shelf_geometry.shelf_width
    bin_height = shelf_data.shelf_geometry.shelf_height
    bin_left = shelf_data.shelf_geometry.shelf_left
    bin_right = shelf_data.shelf_geometry.shelf_right
    bin_ground = shelf_data.shelf_geometry.shelf_ground
    bin_up = shelf_data.shelf_geometry.shelf_up
    v_bin = shelf_data.shelf_geometry.v_bin
    num_of_item = shelf_data.num_of_item
    item_width_stored = shelf_data.return_stored_item_widths()
    item_height_stored = shelf_data.return_stored_item_heights()
    item_center_stored = shelf_data.return_stored_item_centers()
    item_angle_stored = shelf_data.return_stored_item_angles()
    item_width_in_hand = shelf_data.item_width_in_hand
    item_height_in_hand = shelf_data.item_height_in_hand

    init_globals = {'num_of_item': num_of_item, 'bin_width': bin_width, 'bin_height': bin_height, 'item_width_stored': item_width_stored}
    ret_dict = runpy.run_module('setup_variable_range', init_globals=init_globals)

    dim_2D = ret_dict['dim_2D']
    num_of_vertices = ret_dict['num_of_vertices']
    num_of_pairs = ret_dict['num_of_pairs']
    list_pairs = ret_dict['list_pairs']
    num_of_states_stored = ret_dict['num_of_states_stored']
    num_of_states_in_hand = ret_dict['num_of_states_in_hand']

    # Rotation matrix
    item_angle_ranges = ret_dict['item_ranges']
    R_knots_0000 = ret_dict['R_knots_0000']
    k_R_sq_0000 = ret_dict['k_R_sq_0000']
    b_R_sq_0000 = ret_dict['b_R_sq_0000']
    R_knots_1010 = ret_dict['R_knots_1010']
    k_R_sq_1010 = ret_dict['k_R_sq_1010']
    b_R_sq_1010 = ret_dict['b_R_sq_1010']
    v_all_R_0001 = ret_dict['v_all_R_0001']

    a_knots_00 = ret_dict['a_knots_00']
    k_a_sq_00 = ret_dict['k_a_sq_00']
    b_a_sq_00 = ret_dict['b_a_sq_00']
    a_knots_11 = ret_dict['a_knots_11']
    k_a_sq_11 = ret_dict['k_a_sq_11']
    b_a_sq_11 = ret_dict['b_a_sq_11']

    # Vertices
    v_knots_common_x = ret_dict['v_knots_common_x']
    v_knots_common_y = ret_dict['v_knots_common_y']

    # Get active patches from data =====================================================================================
    len_data = len(data_patch_indicator)
    #     The order of data is [ang_pt, v_item[0, 0], v_item[0, 1],
    #                                   v_item[1, 0], v_item[1, 1],
    #                                   v_item[2, 0], v_item[2, 1],
    #                                   v_item[3, 0], v_item[3, 1], (x 3)
    #                           ang_pt_remove, v_item_remove[0, 0], v_item_remove[0, 1],
    #                                   v_item_remove[1, 0], v_item_remove[1, 1],
    #                                   v_item_remove[2, 0], v_item_remove[2, 1],
    #                                   v_item_remove[3, 0], v_item_remove[3, 1],
    #                                   a[0], a[1] (x 6)]  - In total 48 dim

    # Make a list of all the regions
    region_list = []
    for iter_attach in range(4):
        region_list.append(item_angle_ranges[iter_attach])
        for iter_attach2 in range(4):
            region_list.append(v_knots_common_x)
            region_list.append(v_knots_common_y)

    for iter_attach3 in range(num_of_pairs):
        region_list.append(a_knots_00[0])
        region_list.append(a_knots_11[0])

    assert len(region_list) == 48, "Error: wrong total length of regions!"

    active_region = []
    for iter_dim in range(48):
        active_region.append([])

    # Retrieve the active section list
    if use_warm_start:
        active_region_ws = []
        for iter_dim in range(48):
            active_region_ws.append([])

        # Begin: This section is to append all sections - solve complete problem
        for iter_dim in range(48):
            len_sections = len(region_list[iter_dim])
            for iter_sect in range(len_sections):
                active_region[iter_dim].append(iter_sect)
        # End: solve complete problem

    else:
        # Begin: This section normally find active regions for each data
        for iter_data in range(len_data):
            for iter_dim in range(48):
                len_sections = len(region_list[iter_dim])
                for iter_sect in range(len_sections):
                    tt = data_patch_indicator[iter_data][iter_dim]
                    if region_list[iter_dim][iter_sect][1] >= tt >= region_list[iter_dim][iter_sect][0]:
                        active_region[iter_dim].append(iter_sect)
                        break
        # End: This section normally find active regions for each data

    # Reassign integer variables according to the active region list
    data_out_range = False
    suppress_region = []
    num_of_int_reassigned = []
    list_sect_reassigned = []
    active_region_reassigned_with_repeat = []
    reassign_map = []

    for iter_dim in range(48):
        if active_region[iter_dim] == []:
            data_out_range = True
            break
        # Remove repeated items
        active_region[iter_dim] = list(set(active_region[iter_dim]))
        suppress_region.append([elem for elem in list(range(len(region_list[iter_dim]))) if not elem in active_region[iter_dim]])
        list_int = create_int_char_list(int(np.log2(len(region_list[iter_dim]))))  # Note the very first section is always '000'
        # Reassign integer variables
        nn, ll, aa, rr = reassign_int_var(region_list[iter_dim], list_int, suppress_region[iter_dim])
        num_of_int_reassigned.append(nn)
        list_sect_reassigned.append(ll)
        reassign_map.append(rr)
        active_region_reassigned_with_repeat.append(aa)

    if data_out_range:
        prob_success = False
        X_ret = []
        X_dict = []
        Y_ret = []
        cost_ret = 0
        time_ret = 0
        return prob_success, X_ret, X_dict, Y_ret, cost_ret, time_ret, data_out_range

    # print("============================ Active regions with repeat =================================================")
    # print(active_region_reassigned_with_repeat)

    # Compute number of vertices and polygons for bilinear approximations
    num_of_polygons_R_0001 = []
    num_of_vertices_R_0001 = []

    for iter_item in range(num_of_item+1):
        # The number of bilinear regions is identical to the number of sections for R
        num_of_polygons_R_0001.append(int(len(active_region_reassigned_with_repeat[int(iter_item * 9)])))
        num_of_vertices_R_0001.append(int(4 * len(active_region_reassigned_with_repeat[int(iter_item*9)])))

    # Redo integer variables, consider make this part a function =======================================================
    # Redo num_of_int_R
    num_of_int_R = []
    for iter_item in range(num_of_item+1):
        num_of_int_R.append(num_of_int_reassigned[int(iter_item * 9)])

    # Redo num_of_int_a00 and num_of_int_a11
    num_of_int_a00 = []
    num_of_int_a11 = []

    for iter_pair in range(num_of_pairs):
        num_of_int_a00.append(num_of_int_reassigned[int((num_of_item+1) * 9 + iter_pair * 2)])
        num_of_int_a11.append(num_of_int_reassigned[int((num_of_item+1) * 9 + iter_pair * 2 + 1)])

    # Redo k_a_sq, b_a_sq, a_knots
    k_a_sq_00_filtered = []
    b_a_sq_00_filtered = []
    a_knots_00_filtered = []
    k_a_sq_11_filtered = []
    b_a_sq_11_filtered = []
    a_knots_11_filtered = []

    for iter_pair in range(num_of_pairs):

        regions_dim0 = active_region_reassigned_with_repeat[int((num_of_item+1) * 9 + iter_pair * 2)]
        regions_dim1 = active_region_reassigned_with_repeat[int((num_of_item+1) * 9 + iter_pair * 2 + 1)]

        k_a_sq_00_filtered.append([k_a_sq_00[iter_pair][ii] for ii in regions_dim0])
        b_a_sq_00_filtered.append([b_a_sq_00[iter_pair][ii] for ii in regions_dim0])
        a_knots_00_filtered.append([a_knots_00[iter_pair][ii] for ii in regions_dim0])
        k_a_sq_11_filtered.append([k_a_sq_11[iter_pair][ii] for ii in regions_dim1])
        b_a_sq_11_filtered.append([b_a_sq_11[iter_pair][ii] for ii in regions_dim1])
        a_knots_11_filtered.append([a_knots_11[iter_pair][ii] for ii in regions_dim1])

    # Redo num_of_integer_v
    num_of_integer_v_filtered = []
    for iter_item in range(num_of_item+1):
        num_of_integer_v_filtered.append([])

        for iter_dim in range(dim_2D):
            num_of_integer_v_filtered[iter_item].append([])

            for iter_vertex in range(num_of_vertices):
                num_of_integer_v_filtered[iter_item][iter_dim].append(
                    num_of_int_reassigned[int(iter_item * 9 + 1 + iter_vertex * 2 + iter_dim)])

    # Redo num_of_vertices_av
    # Filter v_all_av
    v_all_av_filtered = []
    num_of_polygons_av_filtered = []
    num_of_vertices_av_filtered = []
    for iter_pair in range(num_of_pairs):
        v_all_av_filtered.append([])
        num_of_polygons_av_filtered.append([])
        num_of_vertices_av_filtered.append([])

        for iter_paired in range(2):
            v_all_av_filtered[iter_pair].append([])
            num_of_polygons_av_filtered[iter_pair].append([])
            num_of_vertices_av_filtered[iter_pair].append([])

            for iter_dim in range(dim_2D):
                v_all_av_filtered[iter_pair][iter_paired].append([])
                num_of_polygons_av_filtered[iter_pair][iter_paired].append([])
                num_of_vertices_av_filtered[iter_pair][iter_paired].append([])

                for iter_vertex in range(num_of_vertices):

                    iter_item = int(list_pairs[iter_pair, iter_paired])

                    # Get active regions for a and v respectively: iter_pair+iter_dim -> a,
                    # iter_pair+iter_paired_iter_dim+iter_vertex -> v
                    filter_active_a = active_region_reassigned_with_repeat[
                        int((num_of_item+1) * 9 + iter_pair * 2 + iter_dim)]
                    filter_active_v = active_region_reassigned_with_repeat[
                        int(iter_item * 9 + 1 + iter_vertex * 2 + iter_dim)]

                    num_of_polygons_av_temp = int(len(filter_active_a) * len(filter_active_v))
                    num_of_vertices_av_temp = int(4 * len(filter_active_a) * len(filter_active_v))
                    v_all_av_temp = np.zeros([3, num_of_vertices_av_temp])

                    num_of_polygons_av_filtered[iter_pair][iter_paired][iter_dim].append(num_of_polygons_av_temp)
                    num_of_vertices_av_filtered[iter_pair][iter_paired][iter_dim].append(num_of_vertices_av_temp)

                    if iter_dim == 0:
                        a_knots_av_temp = copy.deepcopy(a_knots_00[iter_pair])
                        v_knots_av_temp = copy.deepcopy(v_knots_common_x)
                    elif iter_dim == 1:
                        a_knots_av_temp = copy.deepcopy(a_knots_11[iter_pair])
                        v_knots_av_temp = copy.deepcopy(v_knots_common_y)
                    else:
                        assert False, "Something is wrong !!"

                    iter_polygon = 0
                    for iter_a in filter_active_a:
                        for iter_v in filter_active_v:
                            bilinear_limits = [a_knots_av_temp[iter_a][0], a_knots_av_temp[iter_a][1],
                                               v_knots_av_temp[iter_v][0], v_knots_av_temp[iter_v][1]]
                            # Use limit2vertex function along each bilinear subspace, and collect all vertices
                            vv = limit2vertex(bilinear_limits)

                            v_all_av_temp[:, 4 * iter_polygon + 0] = np.array([vv[0, 0], vv[0, 1], vv[0, 2]])
                            v_all_av_temp[:, 4 * iter_polygon + 1] = np.array([vv[1, 0], vv[1, 1], vv[1, 2]])
                            v_all_av_temp[:, 4 * iter_polygon + 2] = np.array([vv[2, 0], vv[2, 1], vv[2, 2]])
                            v_all_av_temp[:, 4 * iter_polygon + 3] = np.array([vv[3, 0], vv[3, 1], vv[3, 2]])

                            iter_polygon += 1

                    v_all_av_filtered[iter_pair][iter_paired][iter_dim].append(v_all_av_temp)

    bigM = 10000
    INF = go.GRB.INFINITY
    ct_item_in_hand = num_of_item

    # fig = plt.figure(1)
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(v_all_av[0, :], v_all_av[1, :], v_all_av[2, :], marker='.')
    # plt.show()

    begin_time = time.time()

    m = go.Model("Bin_organization")
    m.setParam('MIPGap', 1e-2)
    if time_lim != -1:
        m.setParam('TimeLimit', time_lim)

    x_item = m.addVars(num_of_item+1, dim_2D, lb=-bigM, ub=bigM)  # Positions for stored items
    R_wb = m.addVars(num_of_item+1, dim_2D, dim_2D, lb=-1.0, ub=1.0)  # Rotation matrices for stored items
    v_item = m.addVars(num_of_item+1, num_of_vertices, dim_2D, lb=-bigM, ub=bigM)  # Positions for vertices of stored items

    R_wb_0000 = m.addVars(num_of_item+1, lb=-1.0, ub=1.0)
    R_wb_1010 = m.addVars(num_of_item+1, lb=-1.0, ub=1.0)
    R_wb_0001 = m.addVars(num_of_item+1, lb=-1.0, ub=1.0)

    a_sep = m.addVars(num_of_pairs, dim_2D, lb=-1.0, ub=1.0)  # Variables to formulate a^{T}x<=b
    b_sep = m.addVars(num_of_pairs, lb=-bigM, ub=bigM)
    a_sep_sq = m.addVars(num_of_pairs, dim_2D, lb=-1.0, ub=1.0)

    # The meaning dimension for a_times_v: #pair, which item in pair, 2D plane dimension, #vertex
    a_times_v = m.addVars(num_of_pairs, dim_2D, dim_2D, num_of_vertices, lb=-bigM, ub=bigM)

    # Binary variables ================================================================================================
    int_all = 0

    int_item_state = m.addVars(num_of_item, num_of_states_stored, vtype=go.GRB.BINARY)
    int_item_state_in_hand = m.addVars(num_of_states_in_hand, vtype=go.GRB.BINARY)  # This is 5 digits insert state
    int_all += num_of_item*num_of_states_stored
    int_all += num_of_states_in_hand

    # Related to R
    # Add a list of integer variables
    int_R = []
    for iter_item in range(num_of_item+1):
        if num_of_int_R[iter_item] > 0:
            int_R.append(m.addVars(num_of_int_R[iter_item], vtype=go.GRB.BINARY))
            int_all += num_of_int_R[iter_item]
        else:
            int_R.append([])

    int_a_00 = []
    int_a_11 = []
    for iter_pair in range(num_of_pairs):
        if num_of_int_a00[iter_pair] > 0:
            int_a_00.append(m.addVars(num_of_int_a00[iter_pair], vtype=go.GRB.BINARY))
            int_all += num_of_int_a00[iter_pair]
        else:
            int_a_00.append([])

        if num_of_int_a11[iter_pair] > 0:
            int_a_11.append(m.addVars(num_of_int_a11[iter_pair], vtype=go.GRB.BINARY))
            int_all += num_of_int_a11[iter_pair]
        else:
            int_a_11.append([])

    int_a = [int_a_00, int_a_11]

    # No matter how many sections a has, the number of int for a*v is the same - determined by the sections of v
    int_v = []
    for iter_item in range(num_of_item+1):
        int_v.append([])
        for iter_vertex in range(num_of_vertices):
            int_v[iter_item].append([])
            for iter_dim in range(dim_2D):
                if num_of_integer_v_filtered[iter_item][iter_dim][iter_vertex] > 0:
                    int_v[iter_item][iter_vertex].append(m.addVars(num_of_integer_v_filtered[iter_item][iter_dim][iter_vertex], vtype=go.GRB.BINARY))
                    int_all += num_of_integer_v_filtered[iter_item][iter_dim][iter_vertex]
                else:
                    int_v[iter_item][iter_vertex].append([])

    print("================================================ Number of integers that are actually used are {}".format(int_all))
    actual_num_of_int_var = int_all

    # Lambda variables =================================================================================================
    lam_0001 = []
    for iter_item in range(num_of_item+1):
        lam_0001.append(m.addVars(num_of_vertices_R_0001[iter_item], lb=0.0, ub=1.0))  # For some reason elements
        # within numpy arrays cannot be used for indexing addVars

    lam_av = []
    for iter_pair in range(num_of_pairs):
        lam_av.append([])
        for iter_paired in range(2):
            lam_av[iter_pair].append([])
            for iter_dim in range(2):
                lam_av[iter_pair][iter_paired].append([])
                for iter_vertex in range(num_of_vertices):
                    lam_av[iter_pair][iter_paired][iter_dim].append(m.addVars(
                        num_of_vertices_av_filtered[iter_pair][iter_paired][iter_dim][iter_vertex], lb=0.0, ub=1.0))

    m.update()

    if use_warm_start:
        # Setup warm start
        m.NumStart = len_data

        # Set StartNumber
        for s in range(len_data):
            m.params.StartNumber = s

            # Get the feasible solution
            feas = data_patch_indicator[s]
            ret_bin_all = []

            # For each dimension, get the active region
            act_region_ws = []
            for iter_dim in range(48):
                tt = feas[iter_dim]
                len_sections = len(region_list[iter_dim])
                for iter_sect in range(len_sections):
                    if region_list[iter_dim][iter_sect][1] >= tt >= region_list[iter_dim][iter_sect][0]:
                        act_region_ws.append(iter_sect)
                        break

                # Retrieve the original integer variables for this dimension
                # Basically change the active region label to a binary value of length = original region length
                ret_bin = dec2bin(act_region_ws[iter_dim], int(np.log2(len(region_list[iter_dim]))))
                ret_bin_all.append(ret_bin.tolist())

            assert len(act_region_ws) == 48, "Incorrect length of active region list!"

            # Find the variable and set start value
            # Warm start integer variables, consider make this part a function =========================================
            # Since for warm start, the integer variables are not re-assigned,
            # the num_of_int arrays should have original values

            # TODO: clarify the definition of 48 dim vector and make it into a class
            # The format of 48 dim vector is for item 0, 1, 2 and item-in-hand, each one is of form:
            # [ang_pt_remove, v_item_remove[0, 0], v_item_remove[0, 1],
            #                 v_item_remove[1, 0], v_item_remove[1, 1],
            #                 v_item_remove[2, 0], v_item_remove[2, 1],
            #                 v_item_remove[3, 0], v_item_remove[3, 1]]
            # then append each of the separating plane with the form:
            # [aa[0], aa[1]]

            # Warm start int_R
            for iter_item in range(num_of_item+1):
                for iter_int in range(num_of_int_R[iter_item]):
                   int_R[iter_item][iter_int].Start = ret_bin_all[int(iter_item * 9)][iter_int]

            # Warm start int_a00 and int_a11
            for iter_pair in range(num_of_pairs):
                for iter_int in range(num_of_int_a00[iter_pair]):
                    int_a_00[iter_pair][iter_int].Start = ret_bin_all[int((num_of_item+1)*9+iter_pair*2)][iter_int]

                for iter_int in range(num_of_int_a11[iter_pair]):
                    int_a_11[iter_pair][iter_int].Start = ret_bin_all[int((num_of_item+1)*9+iter_pair*2+1)][iter_int]

            # Warm start v
            for iter_item in range(num_of_item+1):
                for iter_vertex in range(num_of_vertices):
                    for iter_dim in range(dim_2D):
                        for iter_int in range(num_of_integer_v_filtered[iter_item][iter_dim][iter_vertex]):
                            int_v[iter_item][iter_vertex][iter_dim][iter_int].Start = \
                                ret_bin_all[int(iter_item*9+1+iter_vertex*2+iter_dim)][iter_int]

    # Constraint: Vertices is related to center position and orientation ===============================================
    for iter_item in range(num_of_item+1):

        if iter_item == num_of_item:
            W = item_width_in_hand / 2.0
            H = item_height_in_hand / 2.0
        else:
            W = item_width_stored[iter_item] / 2.0
            H = item_height_stored[iter_item] / 2.0

        for iter_dim in range(dim_2D):
            m.addConstr(v_item[iter_item, 0, iter_dim] == (x_item[iter_item, iter_dim]
                                                       + R_wb[iter_item, iter_dim, 0] * W
                                                       + R_wb[iter_item, iter_dim, 1] * H))

            m.addConstr(v_item[iter_item, 1, iter_dim] == (x_item[iter_item, iter_dim]
                                                       + R_wb[iter_item, iter_dim, 0] * W
                                                       - R_wb[iter_item, iter_dim, 1] * H))

            m.addConstr(v_item[iter_item, 2, iter_dim] == (x_item[iter_item, iter_dim]
                                                       - R_wb[iter_item, iter_dim, 0] * W
                                                       - R_wb[iter_item, iter_dim, 1] * H))

            m.addConstr(v_item[iter_item, 3, iter_dim] == (x_item[iter_item, iter_dim]
                                                       - R_wb[iter_item, iter_dim, 0] * W
                                                       + R_wb[iter_item, iter_dim, 1] * H))

    # Constraint: all objects within bin ===============================================================================
    for iter_item in range(num_of_item+1):
        for iter_vertex in range(num_of_vertices):
            m.addConstr(v_item[iter_item, iter_vertex, 0] <= bin_right)
            m.addConstr(v_item[iter_item, iter_vertex, 0] >= bin_left)
            m.addConstr(v_item[iter_item, iter_vertex, 1] <= bin_up)
            m.addConstr(v_item[iter_item, iter_vertex, 1] >= bin_ground)

    # Rotation angles within -90 deg to 90 deg =========================================================================
    for iter_item in range(num_of_item+1):
        m.addConstr(R_wb[iter_item, 0, 0] >= 0.0)

    # Symmetric rotation matrix ========================================================================================
    for iter_item in range(num_of_item+1):
        # This implies orthogonality
        m.addConstr(R_wb[iter_item, 0, 0] == R_wb[iter_item, 1, 1])
        m.addConstr(R_wb[iter_item, 1, 0] + R_wb[iter_item, 0, 1] == 0)

    # Bilinear rotation matrix constraint ==============================================================================
    # First, bilinear variables become linear
    for iter_item in range(num_of_item+1):
        # Orthogonality is automatically satisfied
        # Determinant is 1
        m.addConstr((R_wb_0000[iter_item] + R_wb_1010[iter_item]) == 1)  # 0011 changed to 0000, 1001 changed to -1010

    # Squared terms
    # Stored items
    for iter_item in range(num_of_item+1):
        # R_wb_stored_0000 = R_wb_stored[0, 0]*R_wb_stored[0, 0]
        if len(int_R[iter_item]) > 0:
            int_list_0000 = generate_int_list(int_R[iter_item])
            int_list_1010 = generate_int_list(int_R[iter_item])
        else:
            int_list_0000 = []
            int_list_1010 = []

        filter_active_0000 = active_region_reassigned_with_repeat[int(iter_item * 9)]
        filter_active_1010 = active_region_reassigned_with_repeat[int(iter_item * 9)]

        # To reassign integer variables, filter R_knots, k, b. No need to filter int_list.
        add_piecewise_linear_constraint(m, R_wb[iter_item, 0, 0], R_wb_0000[iter_item],
                                        [R_knots_0000[iter_item][elem] for elem in filter_active_0000],
                                        [k_R_sq_0000[iter_item][elem] for elem in filter_active_0000],
                                        [b_R_sq_0000[iter_item][elem] for elem in filter_active_0000],
                                        int_list_0000, bigM)

        # R_wb_stored_1010 = R_wb_stored[1, 0]*R_wb_stored[1, 0]
        add_piecewise_linear_constraint(m, R_wb[iter_item, 1, 0], R_wb_1010[iter_item],
                                        [R_knots_1010[iter_item][elem] for elem in filter_active_1010],
                                        [k_R_sq_1010[iter_item][elem] for elem in filter_active_1010],
                                        [b_R_sq_1010[iter_item][elem] for elem in filter_active_1010],
                                        int_list_1010, bigM)

    # Cross terms
    for iter_item in range(num_of_item+1):
        # R_wb_stored_0001 = R_wb_stored[0, 0]*R_wb_stored[0, 1]
        # Filter v_all_R_0001.
        # No need to filter lam, num_of_polygons_R_0001, int_list - they are already based on the filtered vertices.
        x_0001 = [R_wb[iter_item, 0, 0], R_wb[iter_item, 0, 1], R_wb_0001[iter_item]]
        list_lam_0001 = [lam_0001[iter_item][iter_lam] for iter_lam in range(num_of_vertices_R_0001[iter_item])]
        int_list_0001 = [int_R[iter_item][iter_zz] for iter_zz in range(len(int_R[iter_item]))]

        filter_active = active_region_reassigned_with_repeat[int(iter_item * 9)]

        ll = [4*active + add for active in filter_active for add in range(4)]
        v_all_R_0001_filtered = np.array([v_all_R_0001[iter_item][:, elem] for elem in ll]).transpose()

        add_bilinear_constraint_gurobi(m, x_0001, list_lam_0001, int_list_0001, num_of_polygons_R_0001[iter_item],
                                       v_all_R_0001_filtered)

    # Item state constraint ============================================================================================
    bin_offset = 1.5  # To decrease numerical issue
    angle_offset = 0.05

    m.addConstr(go.quicksum(int_item_state_in_hand[iter_in_hand_state]
                            for iter_in_hand_state in range(num_of_states_in_hand)) == 1.0)

    # State z0 - left fall, sin(theta) = 1
    m.addConstr(R_wb[ct_item_in_hand, 1, 0] >= 1.0 - angle_offset - bigM * (1 - int_item_state_in_hand[0]))
    m.addConstr(x_item[ct_item_in_hand, 1] >= item_width_in_hand / 2.0 - bin_offset - bigM * (1 - int_item_state_in_hand[0]))
    m.addConstr(x_item[ct_item_in_hand, 1] <= item_width_in_hand / 2.0 + bin_offset + bigM * (1 - int_item_state_in_hand[0]))

    # State z1 - upright, sin(theta) = 0
    m.addConstr(R_wb[ct_item_in_hand, 1, 0] >= 0.0 - angle_offset - bigM * (1 - int_item_state_in_hand[1]))
    m.addConstr(R_wb[ct_item_in_hand, 1, 0] <= 0.0 + angle_offset + bigM * (1 - int_item_state_in_hand[1]))
    m.addConstr(x_item[ct_item_in_hand, 1] >= item_height_in_hand / 2.0 - bin_offset - bigM * (1 - int_item_state_in_hand[1]))
    m.addConstr(x_item[ct_item_in_hand, 1] <= item_height_in_hand / 2.0 + bin_offset + bigM * (1 - int_item_state_in_hand[1]))

    # State z2 - right fall, sin(theta) = -1
    m.addConstr(R_wb[ct_item_in_hand, 1, 0] <= -1.0 + angle_offset + bigM*(1 - int_item_state_in_hand[2]))
    m.addConstr(x_item[ct_item_in_hand, 1] >= item_width_in_hand / 2.0 - bin_offset - bigM * (1 - int_item_state_in_hand[2]))
    m.addConstr(x_item[ct_item_in_hand, 1] <= item_width_in_hand / 2.0 + bin_offset + bigM * (1 - int_item_state_in_hand[2]))

    # States of stored items
    for iter_item in range(num_of_item):
        # TODO: We have one issue here: stored object cannot lean onto the object in hand.
        #  We can fix that but maybe this is a reasonable assumption.
        #  In fact, logically any item (including in hand), if it is left tilting, it needs to lean on some other object
        #  and we can just set the separating plane between this item and the other item to cross those 2 vertices.
        #  Same thing for right tilting.

        m.addConstr(go.quicksum(int_item_state[iter_item, iter_state] for iter_state in range(num_of_states_stored)) == 1.0)

        # State 0: Left fall: the item is 90 degrees on the ground =====================================================
        m.addConstr(R_wb[iter_item, 1, 0] >= 1.0 - angle_offset - bigM * (1-int_item_state[iter_item, 0]))
        m.addConstr(x_item[iter_item, 1] >= bin_ground - bin_offset + item_width_stored[iter_item] / 2.0 - bigM * (1-int_item_state[iter_item, 0]))
        m.addConstr(x_item[iter_item, 1] <= bin_ground + bin_offset + item_width_stored[iter_item] / 2.0 + bigM * (1-int_item_state[iter_item, 0]))

        # State 1: the item is 0~90 degrees leaning towards left =======================================================
        if iter_item == 0:
            # TODO: Another way to do this is to make the left/right wall also items in the bin
            m.addConstr(v_item[iter_item, 3, 0] >= bin_left - bin_offset - bigM*(1-int_item_state[iter_item, 1]))
            m.addConstr(v_item[iter_item, 3, 0] <= bin_left + 7 + bin_offset + bigM*(1-int_item_state[iter_item, 1]))  # The thickness of wall is 7

        # Otherwise, enforce leaning left constraint
        else:
            # iter_pair == 0 means the item on the left
            item_left = iter_item - 1
            this_item = iter_item

            ct_pair = get_pair_number(list_pairs, item_left, this_item)

            # The meaning dimension for a_times_v: #pair, which item in pair, 2D plane dimension, #vertex
            # the plane goes across the vertex 3 on the right object, and vertex 0 on the left object
            m.addConstr((a_times_v[ct_pair, 1, 0, 3] + a_times_v[ct_pair, 1, 1, 3]) <= b_sep[ct_pair] + bigM*(1-int_item_state[iter_item, 1]))
            m.addConstr((a_times_v[ct_pair, 1, 0, 3] + a_times_v[ct_pair, 1, 1, 3]) >= b_sep[ct_pair] - bigM*(1-int_item_state[iter_item, 1]))
            m.addConstr((a_times_v[ct_pair, 0, 0, 0] + a_times_v[ct_pair, 0, 1, 0]) <= b_sep[ct_pair] + bigM*(1-int_item_state[iter_item, 1]))
            m.addConstr((a_times_v[ct_pair, 0, 0, 0] + a_times_v[ct_pair, 0, 1, 0]) >= b_sep[ct_pair] - bigM*(1-int_item_state[iter_item, 1]))

            # Stability x[left_item] <= x[this_item] <= v2_x[this_item]
            m.addConstr(x_item[this_item, 0] >= x_item[item_left, 0] - bigM*(1-int_item_state[iter_item, 1]))
            m.addConstr(x_item[this_item, 0] <= v_item[this_item, 2, 0] + bigM*(1-int_item_state[iter_item, 1]))

        # v2_y[this_item] touches the ground
        m.addConstr(v_item[iter_item, 2, 1] <= bin_ground + bin_offset + bigM*(1-int_item_state[iter_item, 1]))
        m.addConstr(v_item[iter_item, 2, 1] >= bin_ground - bin_offset - bigM*(1-int_item_state[iter_item, 1]))

        # Angle > 0
        m.addConstr(R_wb[iter_item, 1, 0] >= 0.0 - bigM*(1-int_item_state[iter_item, 1]))

        # State 2: the item is upright =================================================================================
        m.addConstr(R_wb[iter_item, 1, 0] >= 0.0 - angle_offset - bigM * (1-int_item_state[iter_item, 2]))
        m.addConstr(R_wb[iter_item, 1, 0] <= 0.0 + angle_offset + bigM * (1-int_item_state[iter_item, 2]))
        m.addConstr(x_item[iter_item, 1] >= bin_ground - bin_offset + item_height_stored[iter_item] / 2.0 - bigM * (1-int_item_state[iter_item, 2]))
        m.addConstr(x_item[iter_item, 1] <= bin_ground + bin_offset + item_height_stored[iter_item] / 2.0 + bigM * (1-int_item_state[iter_item, 2]))

        # State 3: the item is -90~0 degrees leaning towards right =====================================================
        # If it is the last item, special treatment
        if iter_item == num_of_item - 1:
            # TODO: Another way to do this is to make the left/right wall also items in the bin
            m.addConstr(v_item[iter_item, 0, 0] >= bin_right - bin_offset - bigM*(1-int_item_state[iter_item, 3]))
            m.addConstr(v_item[iter_item, 0, 0] <= bin_right + bin_offset + bigM*(1-int_item_state[iter_item, 3]))

        # Otherwise, enforce leaning left constraint
        else:
            # iter_pair == 0 means the item on the left
            this_item = iter_item
            item_right = iter_item + 1

            ct_pair = get_pair_number(list_pairs, this_item, item_right)

            # The meaning dimension for a_times_v: #pair, which item in pair, 2D plane dimension, #vertex
            # the plane goes across the vertex 3 on the right object, and vertex 0 on the left object
            m.addConstr((a_times_v[ct_pair, 0, 0, 0] + a_times_v[ct_pair, 0, 1, 0]) <= b_sep[ct_pair] + bigM*(1-int_item_state[iter_item, 3]))
            m.addConstr((a_times_v[ct_pair, 0, 0, 0] + a_times_v[ct_pair, 0, 1, 0]) >= b_sep[ct_pair] - bigM*(1-int_item_state[iter_item, 3]))
            m.addConstr((a_times_v[ct_pair, 1, 0, 3] + a_times_v[ct_pair, 1, 1, 3]) <= b_sep[ct_pair] + bigM*(1-int_item_state[iter_item, 3]))
            m.addConstr((a_times_v[ct_pair, 1, 0, 3] + a_times_v[ct_pair, 1, 1, 3]) >= b_sep[ct_pair] - bigM*(1-int_item_state[iter_item, 3]))

            # Stability v1_x[this_item] <= x[this_item] <= x[right_item]
            m.addConstr(x_item[this_item, 0] >= v_item[this_item, 1, 0] - bigM*(1-int_item_state[iter_item, 3]))
            m.addConstr(x_item[this_item, 0] <= x_item[item_right, 0] + bigM*(1-int_item_state[iter_item, 3]))

        # v1_y[this_item] touches the ground
        m.addConstr(v_item[iter_item, 1, 1] <= bin_ground + bin_offset + bigM*(1-int_item_state[iter_item, 3]))
        m.addConstr(v_item[iter_item, 1, 1] >= bin_ground - bin_offset - bigM*(1-int_item_state[iter_item, 3]))

        # Angle < 0
        m.addConstr(R_wb[iter_item, 1, 0] <= 0 + bigM*(1-int_item_state[iter_item, 3]))

        # State 4: Right fall: the item is -90 degrees on the ground ===================================================
        m.addConstr(R_wb[iter_item, 1, 0] <= -1.0 + angle_offset + bigM*(1-int_item_state[iter_item, 4]))
        m.addConstr(x_item[iter_item, 1] >= bin_ground - bin_offset + item_width_stored[iter_item] / 2.0 - bigM*(1-int_item_state[iter_item, 4]))
        m.addConstr(x_item[iter_item, 1] <= bin_ground + bin_offset + item_width_stored[iter_item] / 2.0 + bigM*(1-int_item_state[iter_item, 4]))

    # Collision free constraints =======================================================================================
    for iter_pair in range(num_of_pairs):
        # a is a unit vector
        m.addConstr(1.0 == (a_sep_sq[iter_pair, 0] + a_sep_sq[iter_pair, 1]))

        item_l = int(list_pairs[iter_pair, 0])
        item_r = int(list_pairs[iter_pair, 1])
        assert item_l != num_of_item, "Error: the first item cannot be the item in hand !"

        # a_sq terms are squared of a components
        # a_sep_sq[iter_pair, 0] = a_sep[iter_pair, 0]*a_sep[iter_pair, 0] ---------------------------------------------
        int_aaa00 = generate_int_list(int_a_00[iter_pair])
        # To reassign integer variables, filter R_knots, k, b. No need to filter int_list.

        add_piecewise_linear_constraint(m, a_sep[iter_pair, 0], a_sep_sq[iter_pair, 0],
                                        a_knots_00_filtered[iter_pair],
                                        k_a_sq_00_filtered[iter_pair],
                                        b_a_sq_00_filtered[iter_pair], int_aaa00, bigM)

        # a_sep_sq[iter_pair, 1] = a_sep[iter_pair, 1]*a_sep[iter_pair, 1] ---------------------------------------------
        int_aaa11 = generate_int_list(int_a_11[iter_pair])
        # To reassign integer variables, filter R_knots, k, b. No need to filter int_list.
        add_piecewise_linear_constraint(m, a_sep[iter_pair, 1], a_sep_sq[iter_pair, 1],
                                        a_knots_11_filtered[iter_pair],
                                        k_a_sq_11_filtered[iter_pair],
                                        b_a_sq_11_filtered[iter_pair], int_aaa11, bigM)

        # --------------------------------------------------------------------------------------------------------------
        for iter_vertex in range(num_of_vertices):
            # a*v on one side of the plane
            # The meaning dimension for a_times_v: #pair, which item in pair, 2D plane dimension, #vertex
            # TODO: Note! The order (<= or >=) doesn't matter. If the objects flip sides, the optimizer will get (-a, -b)
            #  instead of (a, b). Since we are getting global optimal, the sides won't be flipped.
            m.addConstr(a_times_v[iter_pair, 0, 0, iter_vertex] + a_times_v[iter_pair, 0, 1, iter_vertex] <= b_sep[iter_pair])
            m.addConstr(a_times_v[iter_pair, 1, 0, iter_vertex] + a_times_v[iter_pair, 1, 1, iter_vertex] >= b_sep[iter_pair])

            # Cross terms
            # Use filtered v_all_av
            for iter_dim in range(dim_2D):
                # Pair left item ---------------------------------------------------------------------------------------
                x = [a_sep[iter_pair, iter_dim], v_item[item_l, iter_vertex, iter_dim], a_times_v[iter_pair, 0, iter_dim, iter_vertex]]
                lam = [lam_av[iter_pair][0][iter_dim][iter_vertex][iter_lam] for iter_lam in
                       range(num_of_vertices_av_filtered[iter_pair][0][iter_dim][iter_vertex])]

                if int_v[item_l][iter_vertex][iter_dim] == [] and int_a[iter_dim][iter_pair] == []:
                    int_var = []
                elif int_a[iter_dim][iter_pair] == []:
                    int_var = [int_v[item_l][iter_vertex][iter_dim][iter_zz] for iter_zz in range(num_of_integer_v_filtered[item_l][iter_dim][iter_vertex])]
                elif int_v[item_l][iter_vertex][iter_dim] == []:
                    int_var = [int_a[iter_dim][iter_pair][iter_zz] for iter_zz in range(num_of_int_reassigned[int((num_of_item+1)*9+iter_pair*2+iter_dim)])]
                else:
                    int_var = [int_v[item_l][iter_vertex][iter_dim][iter_zz] for iter_zz in range(num_of_integer_v_filtered[item_l][iter_dim][iter_vertex])] + \
                              [int_a[iter_dim][iter_pair][iter_zz] for iter_zz in range(num_of_int_reassigned[int((num_of_item+1)*9+iter_pair*2+iter_dim)])]

                add_bilinear_constraint_gurobi(m, x, lam, int_var,
                                               num_of_polygons_av_filtered[iter_pair][0][iter_dim][iter_vertex],
                                               v_all_av_filtered[iter_pair][0][iter_dim][iter_vertex])

                # Pair right item --------------------------------------------------------------------------------------
                x = [a_sep[iter_pair, iter_dim], v_item[item_r, iter_vertex, iter_dim], a_times_v[iter_pair, 1, iter_dim, iter_vertex]]
                lam = [lam_av[iter_pair][1][iter_dim][iter_vertex][iter_lam] for iter_lam in
                       range(num_of_vertices_av_filtered[iter_pair][1][iter_dim][iter_vertex])]

                if int_v[item_r][iter_vertex][iter_dim] == [] and int_a[iter_dim][iter_pair] == []:
                    int_var = []
                elif not int_a[iter_dim][iter_pair]:
                    int_var = [int_v[item_r][iter_vertex][iter_dim][iter_zz] for iter_zz in range(num_of_integer_v_filtered[item_r][iter_dim][iter_vertex])]
                elif not int_v[item_r][iter_vertex][iter_dim]:
                    int_var = [int_a[iter_dim][iter_pair][iter_zz] for iter_zz in range(num_of_int_reassigned[int((num_of_item+1)*9+iter_pair*2+iter_dim)])]
                else:

                    int_var = [int_v[item_r][iter_vertex][iter_dim][iter_zz] for iter_zz in range(num_of_integer_v_filtered[item_r][iter_dim][iter_vertex])] + \
                      [int_a[iter_dim][iter_pair][iter_zz] for iter_zz in range(num_of_int_reassigned[int((num_of_item+1)*9+iter_pair*2+iter_dim)])]

                add_bilinear_constraint_gurobi(m, x, lam, int_var,
                                               num_of_polygons_av_filtered[iter_pair][1][iter_dim][iter_vertex],
                                               v_all_av_filtered[iter_pair][1][iter_dim][iter_vertex])

    obj1 = go.quicksum(go.quicksum((x_item[iter_item, iter_dim] * x_item[iter_item, iter_dim] -
                    2 * item_center_stored[iter_item, iter_dim] * x_item[iter_item, iter_dim]) for iter_item in
                   range(num_of_item)) for iter_dim in range(dim_2D))

    obj2 = go.quicksum((R_wb[iter_item, 0, 0] * R_wb[iter_item, 0, 0] - 2 * np.cos(item_angle_stored[iter_item]) *
                R_wb[iter_item, 0, 0]) for iter_item in range(num_of_item))

    obj = obj1 + obj2

    m.setObjective(obj, go.GRB.MINIMIZE)

    m.optimize()
    end_time = time.time()

    if m.SolCount > 0:
        prob_success = True
    else:
        prob_success = False

    # Plot original bin
    fig, (ax1, ax2) = plt.subplots(2, 1)

    plot_rectangle(ax1, v_bin, color='black', show=False)

    for iter_item in range(num_of_item):
        theta = item_angle_stored[iter_item]
        R_wb_original = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])

        v_item_original = get_vertices(item_center_stored[iter_item, :],
                                       R_wb_original,
                                       np.array([item_height_stored[iter_item], item_width_stored[iter_item]]))

        plot_rectangle(ax1, v_item_original, color='red', show=False)

    ax1.set_xlim([bin_left - 10, bin_right + 10])
    ax1.set_ylim([bin_ground - 10, bin_up + 10])
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    if prob_success:
        print("========================= Begin: item angles ===================================")
        for iter_item in range(num_of_item+1):
            print(math.atan2(R_wb[iter_item, 1, 0].X, R_wb[iter_item, 0, 0].X))

        print("========================= Begin: check orthogonality for rotation matrices ============================")
        for iter_item in range(num_of_item+1):
            R_stored = np.array([[R_wb[iter_item, 0, 0].X, R_wb[iter_item, 0, 1].X],
                                 [R_wb[iter_item, 1, 0].X, R_wb[iter_item, 1, 1].X]])
            print(R_stored)
            print(R_stored.dot(R_stored.transpose()))
            print("---------------------------------------------------------------------------------------------------")
        print("========================= End: check orthogonality for rotation matrices ==============================")

        print("======================== Item states ================================")
        for iter_item in range(num_of_item):
            print([int_item_state[iter_item, iter_int].X for iter_int in range(num_of_states_stored)])

        print([int_item_state_in_hand[iter_int].X for iter_int in range(num_of_states_in_hand)])
        print("======================== End: Item states ================================")

        print("=============================== Begin: Check Separating planes ========================================")
        for iter_pair in range(num_of_pairs):
            print("---------------------------------------------------------------------------------------------------")
            print([a_sep[iter_pair, 0].X, a_sep[iter_pair, 1].X])
            print([a_sep[iter_pair, 0].X ** 2, a_sep[iter_pair, 1].X ** 2])
            print([a_sep_sq[iter_pair, 0].X, a_sep_sq[iter_pair, 1].X])
            print(np.sqrt(a_sep[iter_pair, 0].X ** 2 + a_sep[iter_pair, 1].X ** 2))
        print("=============================== End: Check Separating planes ==========================================")

        # Plot solved bin
        plot_rectangle(ax2, v_bin, color='black', show=False)

        v_item_stored_sol = []
        for iter_item in range(num_of_item):
            v_item_stored_sol.append(np.array([[v_item[iter_item, 0, 0].X, v_item[iter_item, 0, 1].X],
                                               [v_item[iter_item, 1, 0].X, v_item[iter_item, 1, 1].X],
                                               [v_item[iter_item, 2, 0].X, v_item[iter_item, 2, 1].X],
                                               [v_item[iter_item, 3, 0].X, v_item[iter_item, 3, 1].X]]))

            plot_rectangle(ax2, v_item_stored_sol[iter_item], color='red', show=False)

        v_item_in_hand_sol = np.array([[v_item[ct_item_in_hand, 0, 0].X, v_item[ct_item_in_hand, 0, 1].X],
                                       [v_item[ct_item_in_hand, 1, 0].X, v_item[ct_item_in_hand, 1, 1].X],
                                       [v_item[ct_item_in_hand, 2, 0].X, v_item[ct_item_in_hand, 2, 1].X],
                                       [v_item[ct_item_in_hand, 3, 0].X, v_item[ct_item_in_hand, 3, 1].X]])

        plot_rectangle(ax2, v_item_in_hand_sol, color='blue', show=False)

        # Plot separating planes
        for iter_pair in range(num_of_pairs):
            yy = np.linspace(bin_ground, bin_up, 100)
            xx = (b_sep[iter_pair].X - a_sep[iter_pair, 1].X * yy) / a_sep[iter_pair, 0].X
            plt.plot(xx, yy, 'green')

        ax2.set_xlim([bin_left - 10, bin_right + 10])
        ax2.set_ylim([bin_ground - 10, bin_up + 10])
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid()
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # states are: item_center_stored, item_angle_stored, item_center_in_hand, item_angle_in_hand, a_sep, b_sep
        X_dict = {'x_stored': np.array([[x_item[iterr, 0].X, x_item[iterr, 1].X] for iterr in range(num_of_item)]),
                  'x_in_hand': np.array([x_item[ct_item_in_hand, 0].X, x_item[ct_item_in_hand, 1].X]),
                  'angle_stored': np.array([math.atan2(R_wb[iter_item, 1, 0].X, R_wb[iter_item, 0, 0].X)
                                            for iter_item in range(num_of_item)]),
                  'angle_in_hand': math.atan2(R_wb[ct_item_in_hand, 1, 0].X, R_wb[ct_item_in_hand, 0, 0].X),
                  'a_separating_plane': np.array([[a_sep[iter_pair, 0].X, a_sep[iter_pair, 1].X] for iter_pair in range(num_of_pairs)]),
                  'b_separating_plane': np.array([b_sep[iter_pair].X for iter_pair in range(num_of_pairs)]),
                  'vertices_stored': [v_item_stored_sol[iter_item] for iter_item in range(num_of_item)],
                  'vertices_in_hand': v_item_in_hand_sol}

        X_ret = []
        # Centers for stored items
        for iter_item in range(num_of_item):
            for iter_dim in range(dim_2D):
                X_ret.append(x_item[iter_item, iter_dim].X)

        # Angles for stored items
        for iter_item in range(num_of_item):
            X_ret.append(math.atan2(R_wb[iter_item, 1, 0].X, R_wb[iter_item, 0, 0].X))

        # Center for item-in-hand
        for iter_dim in range(dim_2D):
            X_ret.append(x_item[ct_item_in_hand, iter_dim].X)

        # Angle for item-in-hand
        X_ret.append(math.atan2(R_wb[ct_item_in_hand, 1, 0].X, R_wb[ct_item_in_hand, 0, 0].X))

        # Separating planes a
        for iter_pair in range(num_of_pairs):
            for iter_dim in range(dim_2D):
                X_ret.append(a_sep[iter_pair, iter_dim].X)

        # Separating planes b
        for iter_pair in range(num_of_pairs):
            X_ret.append(b_sep[iter_pair].X)

        # integer variables are: int_item_state, int_R_stored, int_a0_in_hand, int_v
        # Since regions have overlap, sometimes one integer will switch and that is fine
        Y_ret = []

        # TODO: Also make the following a big utility function,
        #  maybe make it something like "form transform" with defined integer form

        # Int state integers
        for iter_item in range(num_of_item):
            for iter_int in range(num_of_states_stored):
                Y_ret.append(int(round(int_item_state[iter_item, iter_int].X)))

        for iter_int in range(num_of_states_in_hand):
            Y_ret.append(int(round(int_item_state_in_hand[iter_int].X)))

        # Int R integers
        for iter_item in range(num_of_item+1):
            this_map = reassign_map[int(iter_item * 9)]
            this_int_R = [int(round(int_R[iter_item][iter_int].X)) for iter_int in range(num_of_int_R[iter_item])]

            if len(this_int_R) == 0:
                char_this_int_R = '0'
                tt = this_map[char_this_int_R]  # tt has small digits on the right
                for iter_int in range(len(tt)):
                    Y_ret.append(int(tt[-(iter_int+1)]))  # When append, small digits should be appended to the left

            else:
                this_int_R.reverse()  # The way this_int_R is printed has small digit on the left, thus needs reverse
                char_this_int_R = ''.join(map(str, this_int_R))
                tt = this_map[char_this_int_R]  # tt has small digits on the right
                for iter_int in range(len(tt)):
                    Y_ret.append(int(tt[-(iter_int+1)]))  # When append, small digits should be appended to the left

        # Int a integers. Int_a_00 first
        for iter_pair in range(num_of_pairs):
            this_map = reassign_map[int((num_of_item+1) * 9 + iter_pair * 2)]
            this_int_a00 = [int(round(int_a_00[iter_pair][iter_int].X)) for iter_int in range(num_of_int_a00[iter_pair])]

            if len(this_int_a00) == 0:
                char_this_int_R = '0'
                tt = this_map[char_this_int_R]  # tt has small digits on the right
                for iter_int in range(len(tt)):
                    Y_ret.append(int(tt[-(iter_int+1)]))  # When append, small digits should be appended to the left

            else:
                this_int_a00.reverse()  # The way this_int_R is printed has small digit on the left, thus needs reverse
                char_this_int_R = ''.join(map(str, this_int_a00))
                tt = this_map[char_this_int_R]  # tt has small digits on the right
                for iter_int in range(len(tt)):
                    Y_ret.append(int(tt[-(iter_int+1)]))  # When append, small digits should be appended to the left

        # Int a integers. Int_a_11 second
        for iter_pair in range(num_of_pairs):
            this_map = reassign_map[int((num_of_item+1) * 9 + iter_pair * 2 + 1)]
            this_int_a11 = [int(round(int_a_11[iter_pair][iter_int].X)) for iter_int in range(num_of_int_a11[iter_pair])]

            if len(this_int_a11) == 0:
                char_this_int_R = '0'
                tt = this_map[char_this_int_R]  # tt has small digits on the right
                for iter_int in range(len(tt)):
                    Y_ret.append(int(tt[-(iter_int+1)]))  # When append, small digits should be appended to the left

            else:
                this_int_a11.reverse()  # The way this_int_R is printed has small digit on the left, thus needs reverse
                char_this_int_R = ''.join(map(str, this_int_a11))
                tt = this_map[char_this_int_R]  # tt has small digits on the right
                for iter_int in range(len(tt)):
                    Y_ret.append(int(tt[-(iter_int+1)]))  # When append, small digits should be appended to the left

        # Int v integers.
        for iter_item in range(num_of_item + 1):
            for iter_vertex in range(num_of_vertices):
                for iter_dim in range(dim_2D):
                    this_map = reassign_map[int(iter_item * 9 + 1 + iter_vertex * 2 + iter_dim)]
                    this_int_v = [int(round(int_v[iter_item][iter_vertex][iter_dim][iter_int].X)) for iter_int in
                                    range(num_of_integer_v_filtered[iter_item][iter_dim][iter_vertex])]

                    if len(this_int_v) == 0:
                        char_this_int_R = '0'
                        tt = this_map[char_this_int_R]  # tt has small digits on the right
                        for iter_int in range(len(tt)):
                            Y_ret.append(
                                int(tt[-(iter_int + 1)]))  # When append, small digits should be appended to the left

                    else:
                        this_int_v.reverse()  # The way this_int_R is printed has small digit on the left, thus needs reverse
                        char_this_int_R = ''.join(map(str, this_int_v))
                        tt = this_map[char_this_int_R]  # tt has small digits on the right
                        for iter_int in range(len(tt)):
                            Y_ret.append(
                                int(tt[-(iter_int + 1)]))  # When append, small digits should be appended to the left

        # TODO: Note! y direction sometimes has weird switch of vy = 0 -> int=[1, 0] and vy = 1.5 -> int=[0, 1]
        cost_ret = obj.getValue()

    else:
        X_ret = []
        X_dict = []
        Y_ret = []
        cost_ret = 0

    time_ret = end_time - begin_time

    # plt.show(block=False)
    plt.savefig(dir_current + '/solved_figures/{}_{}.png'.format(count, iter_data))

    return prob_success, X_ret, X_dict, Y_ret, cost_ret, time_ret, actual_num_of_int_var, data_out_range


def main():

    bin_width = 176
    bin_height = 110

    bin_left = -88.0
    bin_right = 88.0
    bin_ground = 0.0
    bin_up = 110

    v_bin = np.array([[88., 110.],
                      [88., 0.],
                      [-88., 0.],
                      [-88., 110.]])

    num_of_item = 3

    with open('for_solver_test_dataset_unsupervised_learning_7_and_32_are_weird.p', 'rb') as f:
        data_test = pickle.load(f)

    features_test = data_test['feature']
    solutions_test = data_test['solution']
    len_data = len(features_test)
    assert len_data == len(solutions_test), "Inconsistent feature and solution length !!"

    all_success = []

    features_to_save = []
    y_guess_to_save = []

    for iter_data in range(len_data):
    # for iter_data in [0]:
        assert len(solutions_test[iter_data]) == 48, "Wrong data length!"

        # data_patch_indicator = [solutions_test[iter_data],
        #                         data_patch_indicator_extra[0],
        #                         data_patch_indicator_extra[1]]

        data_patch_indicator = [solutions_test[iter_data]]

        this_feature = features_test[iter_data]

        item_width_stored = [this_feature[5*iter_item+4] for iter_item in range(num_of_item)]
        item_height_stored = [this_feature[5*iter_item+3] for iter_item in range(num_of_item)]
        item_center_stored = np.array([[this_feature[5*iter_item+0], this_feature[5*iter_item+1]]
                                       for iter_item in range(num_of_item)])
        item_angle_stored = [this_feature[5*iter_item+2] for iter_item in range(num_of_item)]
        item_height_in_hand = this_feature[15]
        item_width_in_hand = this_feature[16]

        print("Features are:")
        print("Width {}".format(item_width_stored))
        print("Height {}".format(item_height_stored))
        print("Center {}".format(item_center_stored))
        print("Angle {}".format(item_angle_stored))
        print("Width in hand {}".format(item_width_in_hand))
        print("Height in hand {}".format(item_height_in_hand))
        print("Solutions are:")
        print(data_patch_indicator)

        prob_success, X_ret, X_dict, Y_ret, cost_ret, time_ret, actual_num_of_int_var, data_out_range = solve_within_patch(bin_width, bin_height,
                           bin_left, bin_right, bin_ground, bin_up, v_bin, num_of_item,
                           item_width_stored, item_height_stored, item_center_stored, item_angle_stored,
                           item_width_in_hand, item_height_in_hand, data_patch_indicator, iter_data, iter_data)

        if not data_out_range:
            all_success.append(prob_success)

            if prob_success:
                features_to_save.append(this_feature)
                y_guess_to_save.append(Y_ret)

    saves = [features_to_save, y_guess_to_save]
    with open(f"dataset_with_y_guess_for_debugging.pkl", "wb") as f:
        pickle.dump(saves, f)

    print(all_success)


if __name__ == "__main__":
    main()
