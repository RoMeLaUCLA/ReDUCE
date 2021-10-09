import os, sys
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_ReDUCE+"/utils")

import gurobipy as go
import numpy as np
from add_piecewise_linear_constraint import add_piecewise_linear_constraint
from add_McCormick_envelope_constraint import add_bilinear_constraint_gurobi, limit2vertex
import pdb


def get_kb_from_2_pts(p1, p2):
    # This function returns the approximated line given 2 points on it
    kkk = (p1 ** 2 - p2 ** 2) / (p1 - p2)
    bbb = p2 ** 2 - kkk * p2  # p^2 = k*p + b

    assert abs(p1 * kkk + bbb - p1 ** 2) <= 1e-6, "Error: point p_lo does not lie on the quadratic curve!"
    assert abs(p2 * kkk + bbb - p2 ** 2) <= 1e-6, "Error: point p_hi does not lie on the quadratic curve!"

    return kkk, bbb


def solve_separating_plane(vertex_0, vertex_1, bin_width, bin_height):

    bin_width += 2.0
    bin_height += 2.0

    bin_left = -bin_width / 2.0
    bin_right = bin_width / 2.0
    bin_ground = -1.0
    bin_up = bin_height - 1.0

    num_of_vertices = 4

    # Variables related to a squared ===================================================================================
    a_knots_00 = [[-1.0, -0.75], [-0.75, -0.5], [-0.5, -0.25], [-0.25, 0.0],
                  [0.0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
    k_a_sq_00 = []
    b_a_sq_00 = []

    num_of_int_a0 = int(np.log2(len(a_knots_00)))

    len_sec_00 = len(a_knots_00)
    for iter_sec in range(len_sec_00):
        p_lo = a_knots_00[iter_sec][0]
        p_hi = a_knots_00[iter_sec][1]
        kk, bb = get_kb_from_2_pts(p_lo, p_hi)
        k_a_sq_00.append(kk)
        b_a_sq_00.append(bb)

    a_knots_11 = [[-1.0, -0.75], [-0.75, -0.5], [-0.5, -0.25], [-0.25, 0.0],
                  [0.0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
    k_a_sq_11 = []
    b_a_sq_11 = []

    num_of_int_a1 = int(np.log2(len(a_knots_11)))

    len_sec_11 = len(a_knots_11)
    for iter_sec in range(len_sec_11):
        p_lo = a_knots_11[iter_sec][0]
        p_hi = a_knots_11[iter_sec][1]
        kk, bb = get_kb_from_2_pts(p_lo, p_hi)
        k_a_sq_11.append(kk)
        b_a_sq_11.append(bb)

    # Variables related to a x v cross =================================================================================
    v_knots_common_x = [[bin_left, bin_left + 0.25 * bin_width],
                        [bin_left + 0.25 * bin_width, bin_left + 0.5 * bin_width],
                        [bin_left + 0.5 * bin_width, bin_left + 0.75 * bin_width],
                        [bin_left + 0.75 * bin_width, bin_left + bin_width]]
    assert (bin_left + bin_width) == bin_right, "Bin width doesn't match bin left/right !!"

    v_knots_common_y = [[bin_ground, bin_ground + 0.25 * bin_height],
                        [bin_ground + 0.25 * bin_height, bin_ground + 0.5 * bin_height],
                        [bin_ground + 0.5 * bin_height, bin_ground + 0.75 * bin_height],
                        [bin_ground + 0.75 * bin_height, bin_ground + bin_height]]

    num_of_polygons_av_x = len(a_knots_00) * len(v_knots_common_x)
    num_of_polygons_av_y = len(a_knots_11) * len(v_knots_common_y)

    # Despite the pair and vertex, the number of integers are the same
    num_of_integer_v_x = int(np.log2(len(v_knots_common_x)))  # a already has its integer variables
    num_of_integer_v_y = int(np.log2(len(v_knots_common_y)))

    num_of_vertices_av_x = num_of_polygons_av_x * 4
    num_of_vertices_av_y = num_of_polygons_av_y * 4

    v_all_av_x = np.zeros([3, num_of_vertices_av_x])
    v_all_av_y = np.zeros([3, num_of_vertices_av_y])

    iter_polygon = 0
    for iter_a in range(len(a_knots_00)):
        for iter_v in range(len(v_knots_common_x)):
            a_knots_av_temp = a_knots_00[iter_a]
            v_knots_av_temp = v_knots_common_x[iter_v]

            bilinear_limits = [a_knots_av_temp[0], a_knots_av_temp[1], v_knots_av_temp[0], v_knots_av_temp[1]]

            # Use limit2vertex function along each bilinear subspace, and collect all vertices
            vv = limit2vertex(bilinear_limits)

            v_all_av_x[:, 4 * iter_polygon + 0] = np.array([vv[0, 0], vv[0, 1], vv[0, 2]])
            v_all_av_x[:, 4 * iter_polygon + 1] = np.array([vv[1, 0], vv[1, 1], vv[1, 2]])
            v_all_av_x[:, 4 * iter_polygon + 2] = np.array([vv[2, 0], vv[2, 1], vv[2, 2]])
            v_all_av_x[:, 4 * iter_polygon + 3] = np.array([vv[3, 0], vv[3, 1], vv[3, 2]])

            iter_polygon += 1

    iter_polygon = 0
    for iter_a in range(len(a_knots_11)):
        for iter_v in range(len(v_knots_common_y)):
            a_knots_av_temp = a_knots_11[iter_a]
            v_knots_av_temp = v_knots_common_y[iter_v]

            bilinear_limits = [a_knots_av_temp[0], a_knots_av_temp[1], v_knots_av_temp[0], v_knots_av_temp[1]]

            # Use limit2vertex function along each bilinear subspace, and collect all vertices
            vv = limit2vertex(bilinear_limits)

            v_all_av_y[:, 4 * iter_polygon + 0] = np.array([vv[0, 0], vv[0, 1], vv[0, 2]])
            v_all_av_y[:, 4 * iter_polygon + 1] = np.array([vv[1, 0], vv[1, 1], vv[1, 2]])
            v_all_av_y[:, 4 * iter_polygon + 2] = np.array([vv[2, 0], vv[2, 1], vv[2, 2]])
            v_all_av_y[:, 4 * iter_polygon + 3] = np.array([vv[3, 0], vv[3, 1], vv[3, 2]])

            iter_polygon += 1

    m = go.Model("Bin_organization")
    m.setParam('MIPGap', 1e-2)

    bigM = 10000

    a_sep = m.addVars(2, lb=-1.0, ub=1.0)
    a_sep_sq = m.addVars(2, lb=-1.0, ub=1.0)

    b_sep = m.addVar(lb=-bigM, ub=bigM)

    int_a0 = m.addVars(num_of_int_a0, vtype=go.GRB.BINARY)
    int_a1 = m.addVars(num_of_int_a1, vtype=go.GRB.BINARY)

    num_paired = 2
    a_times_v_x = m.addVars(num_paired, num_of_vertices, lb=-bigM, ub=bigM)
    int_v_x = m.addVars(num_paired, num_of_vertices, num_of_integer_v_x, vtype=go.GRB.BINARY)
    lam_av_x = m.addVars(num_paired, num_of_vertices, num_of_vertices_av_x, lb=0.0, ub=1.0)

    a_times_v_y = m.addVars(num_paired, num_of_vertices, lb=-bigM, ub=bigM)
    int_v_y = m.addVars(num_paired, num_of_vertices, num_of_integer_v_y, vtype=go.GRB.BINARY)
    lam_av_y = m.addVars(num_paired, num_of_vertices, num_of_vertices_av_y, lb=0.0, ub=1.0)

    m.update()

    # a is a unit vector
    m.addConstr(1.0 == (a_sep_sq[0] + a_sep_sq[1]))

    int_a_0 = []
    for iter_int_a_0 in range(pow(2, num_of_int_a0)):
        bin_char = bin(iter_int_a_0)[2:].zfill(num_of_int_a0)  # Change the iter value to binary
        this_int = []
        for iter_digit in range(num_of_int_a0):
            if bin_char[-(iter_digit+1)] == '1':
                this_int.append(int_a0[iter_digit])
            elif bin_char[-(iter_digit+1)] == '0':
                this_int.append(1 - int_a0[iter_digit])
            else:
                assert False, "Something is wrong !!"
        int_a_0.append(this_int)

    # This way of coding, the very left section will be chosen when int_a0[0] = 0, int_a0[1] = 0, int_a0[2] = 0
    add_piecewise_linear_constraint(m, a_sep[0], a_sep_sq[0], a_knots_00, k_a_sq_00, b_a_sq_00, int_a_0, bigM)

    int_a_1 = []
    for iter_int_a_1 in range(pow(2, num_of_int_a1)):
        bin_char = bin(iter_int_a_1)[2:].zfill(num_of_int_a1)  # Change the iter value to binary
        this_int = []
        for iter_digit in range(num_of_int_a1):
            if bin_char[-(iter_digit+1)] == '1':
                this_int.append(int_a1[iter_digit])
            elif bin_char[-(iter_digit+1)] == '0':
                this_int.append(1 - int_a1[iter_digit])
            else:
                assert False, "Something is wrong !!"
        int_a_1.append(this_int)

    add_piecewise_linear_constraint(m, a_sep[1], a_sep_sq[1], a_knots_11, k_a_sq_11, b_a_sq_11, int_a_1, bigM)

    for iter_vertex in range(num_of_vertices):
        # a*v on one side of the plane
        # TODO: Note! The order (<= or >=) doesn't matter. If the objects flip sides, the optimizer will get (-a, -b)
        #  instead of (a, b). Since we are getting global optimal, the sides won't be flipped.
        # The second one should be always larger according to the 2 constraints below
        m.addConstr(a_times_v_x[0, iter_vertex] + a_times_v_y[0, iter_vertex] <= b_sep)
        m.addConstr(a_times_v_x[1, iter_vertex] + a_times_v_y[1, iter_vertex] >= b_sep)

    vertex = [vertex_0, vertex_1]

    for iter_pair in range(num_paired):
        for iter_vertex in range(num_of_vertices):
            x = [a_sep[0], vertex[iter_pair][iter_vertex, 0], a_times_v_x[iter_pair, iter_vertex]]
            lam_x = [lam_av_x[iter_pair, iter_vertex, iter_lam] for iter_lam in range(num_of_vertices_av_x)]
            int_var_x = [int_v_x[iter_pair, iter_vertex, iter_zz] for iter_zz in range(num_of_integer_v_x)] + [int_a0[0], int_a0[1], int_a0[2]]
            add_bilinear_constraint_gurobi(m, x, lam_x, int_var_x, num_of_polygons_av_x, v_all_av_x)

            y = [a_sep[1], vertex[iter_pair][iter_vertex, 1], a_times_v_y[iter_pair, iter_vertex]]
            lam_y = [lam_av_y[iter_pair, iter_vertex, iter_lam] for iter_lam in range(num_of_vertices_av_y)]
            int_var_y = [int_v_y[iter_pair, iter_vertex, iter_zz] for iter_zz in range(num_of_integer_v_y)] + [int_a1[0], int_a1[1], int_a1[2]]
            add_bilinear_constraint_gurobi(m, y, lam_y, int_var_y, num_of_polygons_av_y, v_all_av_y)

    obj = a_sep[1]*a_sep[1]

    m.setObjective(obj, go.GRB.MINIMIZE)

    m.optimize()

    if m.SolCount > 0:
        prob_success = True
    else:
        prob_success = False

    a_sol = np.zeros(2)
    b_sol = 0.0

    if prob_success:
        a_sol = np.array([a_sep[0].X, a_sep[1].X])
        b_sol = b_sep.X

    return prob_success, a_sol, b_sol
