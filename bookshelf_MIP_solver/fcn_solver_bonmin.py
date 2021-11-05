import os, sys
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_ReDUCE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_ReDUCE + "/utils")

import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from utils.get_vertices import get_vertices, plot_rectangle, plot_bilinear
from dec2bin import dec2bin
from add_McCormick_envelope_constraint import add_vertex_polytope_constraint_pyomo_with_item_count, limit2vertex
import runpy
import pdb

# Important: since we assume easy order of items inside the bin, the input items should always be counting from left to
# right 0, 1, 2, 3, 4, 5, etc


def solve_bin_pyomo(shelf_data, count, iter_data, time_lim=-1):

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

    bigM = 1e8

    m = pyo.AbstractModel()

    num_of_pairs = int((num_of_item+1)*num_of_item/2)

    init_globals = {'num_of_item': num_of_item, 'bin_width': bin_width, 'bin_height': bin_height, 'item_width_stored': item_width_stored}
    ret_dict = runpy.run_module('setup_variable_range', init_globals=init_globals)
    list_pairs = ret_dict['list_pairs']

    m.range_items = pyo.RangeSet(0, num_of_item-1)
    m.range_dim = pyo.RangeSet(0, 1)
    m.range_vertices = pyo.RangeSet(0, 3)
    m.range_pairs = pyo.RangeSet(0, num_of_pairs-1)
    m.range_states = pyo.RangeSet(0, 4)  # 5 states for stored items
    m.range_in_hand_states = pyo.RangeSet(0, 2)  # 3 states for inserted item

    m.x_stored = pyo.Var(m.range_items, m.range_dim, bounds=(-bigM, bigM))  # Positions for stored items
    m.R_wb_stored = pyo.Var(m.range_items, m.range_dim, m.range_dim, bounds=(-1.0, 1.0))  # Rotation matrices for stored items
    m.v_stored = pyo.Var(m.range_items, m.range_vertices, m.range_dim, bounds=(-bigM, bigM))  # Positions for vertices of stored items

    m.x_in_hand = pyo.Var(m.range_dim, bounds=(-bigM, bigM))  # Positions for stored items
    m.R_wb_in_hand = pyo.Var(m.range_dim, m.range_dim, bounds=(-1.0, 1.0))  # Rotation matrices for stored items
    m.v_in_hand = pyo.Var(m.range_vertices, m.range_dim, bounds=(-bigM, bigM))  # Positions for vertices of stored items

    m.a_sep = pyo.Var(m.range_pairs, m.range_dim, initialize=1.0, bounds=(-1.0, 1.0))  # Variables to formulate a^{T}x<=b
    m.b_sep = pyo.Var(m.range_pairs, bounds=(-bigM, bigM))

    m.int_item_state = pyo.Var(m.range_items, m.range_states, domain=pyo.Binary)
    m.int_item_state_in_hand = pyo.Var(m.range_in_hand_states, domain=pyo.Binary)

    # Constraint: Vertices is related to center position and orientation ===============================================
    def center2vertex_stored(m, iter_item, iter_dim, iter_vertex):

        W = item_width_stored[iter_item] / 2.0
        H = item_height_stored[iter_item] / 2.0
        x = m.x_stored[iter_item, iter_dim]
        Ri1 = m.R_wb_stored[iter_item, iter_dim, 0]
        Ri2 = m.R_wb_stored[iter_item, iter_dim, 1]
        v0 = m.v_stored[iter_item, 0, iter_dim]
        v1 = m.v_stored[iter_item, 1, iter_dim]
        v2 = m.v_stored[iter_item, 2, iter_dim]
        v3 = m.v_stored[iter_item, 3, iter_dim]

        if iter_vertex == 0:
            return v0 == (x + Ri1*W + Ri2*H)
        elif iter_vertex == 1:
            return v1 == (x + Ri1*W - Ri2*H)
        elif iter_vertex == 2:
            return v2 == (x - Ri1*W - Ri2*H)
        elif iter_vertex == 3:
            return v3 == (x - Ri1*W + Ri2*H)

    def center2vertex_in_hand(m, iter_dim, iter_vertex):

        W = item_width_in_hand / 2.0
        H = item_height_in_hand / 2.0
        x = m.x_in_hand[iter_dim]
        Ri1 = m.R_wb_in_hand[iter_dim, 0]
        Ri2 = m.R_wb_in_hand[iter_dim, 1]
        v0 = m.v_in_hand[0, iter_dim]
        v1 = m.v_in_hand[1, iter_dim]
        v2 = m.v_in_hand[2, iter_dim]
        v3 = m.v_in_hand[3, iter_dim]

        if iter_vertex == 0:
            return v0 == (x + Ri1*W + Ri2*H)
        elif iter_vertex == 1:
            return v1 == (x + Ri1*W - Ri2*H)
        elif iter_vertex == 2:
            return v2 == (x - Ri1*W - Ri2*H)
        elif iter_vertex == 3:
            return v3 == (x - Ri1*W + Ri2*H)

    m.con_center2vertex_stored = pyo.Constraint(m.range_items, m.range_dim, m.range_vertices, rule=center2vertex_stored)
    m.con_center2vertex_in_hand = pyo.Constraint(m.range_dim, m.range_vertices, rule=center2vertex_in_hand)

    # Constraint: all objects within bin ===============================================================================
    def object_within_bin_stored(m, iter_item, iter_vertex, iter_cons):

        if iter_cons == 0:
            return m.v_stored[iter_item, iter_vertex, 0] <= bin_right
        elif iter_cons == 1:
            return m.v_stored[iter_item, iter_vertex, 0] >= bin_left
        elif iter_cons == 2:
            return m.v_stored[iter_item, iter_vertex, 1] <= bin_up
        elif iter_cons == 3:
            return m.v_stored[iter_item, iter_vertex, 1] >= bin_ground

    def object_within_bin_in_hand(m, iter_vertex, iter_cons):

        if iter_cons == 0:
            return m.v_in_hand[iter_vertex, 0] <= bin_right
        elif iter_cons == 1:
            return m.v_in_hand[iter_vertex, 0] >= bin_left
        elif iter_cons == 2:
            return m.v_in_hand[iter_vertex, 1] <= bin_up
        elif iter_cons == 3:
            return m.v_in_hand[iter_vertex, 1] >= bin_ground

    m.con_object_within_bin_stored = pyo.Constraint(m.range_items, m.range_vertices, pyo.RangeSet(0, 3), rule=object_within_bin_stored)
    m.con_object_within_bin_in_hand = pyo.Constraint(m.range_vertices, pyo.RangeSet(0, 3), rule=object_within_bin_in_hand)

    # Rotation angles within -90 deg to 90 deg =========================================================================
    def rotation_limit_stored(m, iter_item):
        return m.R_wb_stored[iter_item, 0, 0] >= 0.0

    def rotation_limit_in_hand(m):
        return m.R_wb_in_hand[0, 0] >= 0.0

    m.con_rotation_limit = pyo.Constraint(m.range_items, rule=rotation_limit_stored)
    m.con_rotation_limit_in_hand = pyo.Constraint(rule=rotation_limit_in_hand)

    # Bilinear rotation matrix constraint ==============================================================================
    def rotation_matrix_orthogonality_stored(m, iter_item, iter_cons):
        # Orthogonality
        if iter_cons == 0:
            return m.R_wb_stored[iter_item, 0, 0]*m.R_wb_stored[iter_item, 0, 0] + \
                   m.R_wb_stored[iter_item, 1, 0]*m.R_wb_stored[iter_item, 1, 0] == 1

        elif iter_cons == 1:
            return m.R_wb_stored[iter_item, 0, 1] * m.R_wb_stored[iter_item, 0, 1] + \
                   m.R_wb_stored[iter_item, 1, 1] * m.R_wb_stored[iter_item, 1, 1] == 1

        elif iter_cons == 2:
            return m.R_wb_stored[iter_item, 0, 0] * m.R_wb_stored[iter_item, 0, 1] + \
                   m.R_wb_stored[iter_item, 1, 0] * m.R_wb_stored[iter_item, 1, 1] == 0

        # Determinant is 1
        elif iter_cons == 3:
            return m.R_wb_stored[iter_item, 0, 0] * m.R_wb_stored[iter_item, 1, 1] - \
                   m.R_wb_stored[iter_item, 1, 0] * m.R_wb_stored[iter_item, 0, 1] == 1

    def rotation_matrix_orthogonality_in_hand(m, iter_cons):
        # Orthogonality
        if iter_cons == 0:
            return m.R_wb_in_hand[0, 0] * m.R_wb_in_hand[0, 0] + \
                   m.R_wb_in_hand[1, 0] * m.R_wb_in_hand[1, 0] == 1

        elif iter_cons == 1:
            return m.R_wb_in_hand[0, 1] * m.R_wb_in_hand[0, 1] + \
                   m.R_wb_in_hand[1, 1] * m.R_wb_in_hand[1, 1] == 1

        elif iter_cons == 2:
            return m.R_wb_in_hand[0, 0] * m.R_wb_in_hand[0, 1] + \
                   m.R_wb_in_hand[1, 0] * m.R_wb_in_hand[1, 1] == 0

        # Determinant is 1
        elif iter_cons == 3:
            return m.R_wb_in_hand[0, 0] * m.R_wb_in_hand[1, 1] - \
                   m.R_wb_in_hand[1, 0] * m.R_wb_in_hand[0, 1] == 1

    m.con_rotation_matrix_orthogonality = pyo.Constraint(m.range_items, pyo.RangeSet(0, 3), rule=rotation_matrix_orthogonality_stored)
    m.con_rotation_matrix_orthogonality_in_hand = pyo.Constraint(pyo.RangeSet(0, 3), rule=rotation_matrix_orthogonality_in_hand)

    # Collision free constraint ========================================================================================
    def normal_unit_vector(m, iter_pairs):
        return 1.0 == (m.a_sep[iter_pairs, 0]*m.a_sep[iter_pairs, 0] + m.a_sep[iter_pairs, 1]*m.a_sep[iter_pairs, 1])

    m.con_normal_unit_vector = pyo.Constraint(m.range_pairs, rule=normal_unit_vector)

    def collision_free(m, iter_pairs, iter_vertices, iter_cons):
        item_a = list_pairs[iter_pairs, 0]
        item_b = list_pairs[iter_pairs, 1]

        assert item_a != num_of_item, "Error: the first item cannot be the item in hand !"

        if iter_cons == 0:
            return m.a_sep[iter_pairs, 0] * m.v_stored[item_a, iter_vertices, 0] + \
                   m.a_sep[iter_pairs, 1] * m.v_stored[item_a, iter_vertices, 1] <= m.b_sep[iter_pairs]

        elif iter_cons == 1:
            if item_b != num_of_item:
                return m.a_sep[iter_pairs, 0] * m.v_stored[item_b, iter_vertices, 0] + \
                       m.a_sep[iter_pairs, 1] * m.v_stored[item_b, iter_vertices, 1] >= m.b_sep[iter_pairs]

            else:
                return m.a_sep[iter_pairs, 0] * m.v_in_hand[iter_vertices, 0] + \
                       m.a_sep[iter_pairs, 1] * m.v_in_hand[iter_vertices, 1] >= m.b_sep[iter_pairs]

    m.con_collision_free = pyo.Constraint(m.range_pairs, m.range_vertices, pyo.RangeSet(0, 1), rule=collision_free)

    # Each stored item has 5 different states
    def state_summation_stored(m, iter_item):
        return sum(m.int_item_state[iter_item, iter_int] for iter_int in m.range_states) == 1

    # In hand item has 3 different states
    def state_summation_in_hand(m):
        return sum(m.int_item_state_in_hand[iter_int] for iter_int in m.range_in_hand_states) == 1

    m.con_state_summation_stored = pyo.Constraint(m.range_items, rule=state_summation_stored)
    m.con_state_summation_in_hand = pyo.Constraint(rule=state_summation_in_hand)

    # Note bigM is 1e8, be careful not to make your values too large!
    # State 0: the item is 90 degrees on the ground --------------------------------------------------------------------
    def state_z0(m, iter_item, iter_cons):
        if iter_cons == 0:
            return m.R_wb_stored[iter_item, 1, 0] <= 1 + bigM * (1-m.int_item_state[iter_item, 0])
        elif iter_cons == 1:
            return m.R_wb_stored[iter_item, 1, 0] >= 1 - bigM * (1-m.int_item_state[iter_item, 0])
        elif iter_cons == 2:
            return m.x_stored[iter_item, 1] <= item_width_stored[iter_item]/2.0 + bigM*(1-m.int_item_state[iter_item, 0])
        elif iter_cons == 3:
            return m.x_stored[iter_item, 1] >= item_width_stored[iter_item]/2.0 - bigM*(1-m.int_item_state[iter_item, 0])

    m.con_state_z0 = pyo.Constraint(m.range_items, pyo.RangeSet(0, 3), rule=state_z0)

    # State 1: the item is 0~90 degrees leaning towards left -----------------------------------------------------------
    def state_z1_contact_separating_plane(m, iter_item, iter_cons):
        # If it is the first item, special treatment
        if iter_item == 0:
            if iter_cons == 0 or iter_cons == 1 or iter_cons == 2:
                return m.v_stored[iter_item, 3, 0] >= (bin_left - bigM*(1-m.int_item_state[iter_item, 1]))
            else:
                return m.v_stored[iter_item, 3, 0] <= (bin_left + bigM*(1-m.int_item_state[iter_item, 1]))

        # Otherwise, enforce leaning left constraint
        else:
            # Retrieve pair number
            found = False
            iter_pair = 0
            ct_pair = 0
            while not found:
                if list_pairs[iter_pair, 0] == iter_item-1 and list_pairs[iter_pair, 1] == iter_item:
                    ct_pair = iter_pair
                    found = True
                else:
                    iter_pair += 1

            assert found, "The separating plane is not founded !"

            if iter_cons == 0:
                return (m.a_sep[ct_pair, 0] * m.v_stored[iter_item, 3, 0] +
                        m.a_sep[ct_pair, 1] * m.v_stored[iter_item, 3, 1]) >= (m.b_sep[ct_pair] - bigM*(1-m.int_item_state[iter_item, 1]))
            elif iter_cons == 1:
                return (m.a_sep[ct_pair, 0] * m.v_stored[iter_item, 3, 0] +
                        m.a_sep[ct_pair, 1] * m.v_stored[iter_item, 3, 1]) <= (m.b_sep[ct_pair] + bigM*(1-m.int_item_state[iter_item, 1]))
            elif iter_cons == 2:
                return (m.a_sep[ct_pair, 0] * m.v_stored[iter_item-1, 0, 0] +
                        m.a_sep[ct_pair, 1] * m.v_stored[iter_item-1, 0, 1]) >= (m.b_sep[ct_pair] - bigM * (1 - m.int_item_state[iter_item, 1]))
            elif iter_cons == 3:
                return (m.a_sep[ct_pair, 0] * m.v_stored[iter_item-1, 0, 0] +
                        m.a_sep[ct_pair, 1] * m.v_stored[iter_item-1, 0, 1]) <= (m.b_sep[ct_pair] + bigM * (1 - m.int_item_state[iter_item, 1]))
            elif iter_cons == 4:
                return m.x_stored[iter_item, 0] >= (m.x_stored[iter_item-1, 0] - bigM*(1-m.int_item_state[iter_item, 1]))
            elif iter_cons == 5:
                return m.x_stored[iter_item, 0] <= (m.v_stored[iter_item, 2, 0] + bigM*(1-m.int_item_state[iter_item, 1]))
            else:
                assert False, "Constraint number out of bound!"

    def state_z1_on_ground(m, iter_item, iter_cons):
        if iter_cons == 0:
            return m.v_stored[iter_item, 2, 1] <= (bin_ground + bigM * (1 - m.int_item_state[iter_item, 1]))
        elif iter_cons == 1:
            return m.v_stored[iter_item, 2, 1] >= (bin_ground - bigM * (1 - m.int_item_state[iter_item, 1]))
        else:
            assert False, "Constraint number out of bound!"

    def state_z1_R21_range(m, iter_item):
        return m.R_wb_stored[iter_item, 1, 0] >= 0.0 - bigM * (1 - m.int_item_state[iter_item, 1])

    m.con_state_z1_contact_separating_plane = pyo.Constraint(m.range_items, pyo.RangeSet(0, 5), rule=state_z1_contact_separating_plane)
    m.con_state_z1_on_ground = pyo.Constraint(m.range_items, pyo.RangeSet(0, 1), rule=state_z1_on_ground)
    m.con_state_z1_R21_range = pyo.Constraint(m.range_items, rule=state_z1_R21_range)

    # State 2: the item is upright -------------------------------------------------------------------------------------
    def state_z2(m, iter_item, iter_cons):
        if iter_cons == 0:
            return m.R_wb_stored[iter_item, 1, 0] <= bigM * (1 - m.int_item_state[iter_item, 2])
        elif iter_cons == 1:
            return m.R_wb_stored[iter_item, 1, 0] >= - bigM * (1 - m.int_item_state[iter_item, 2])
        elif iter_cons == 2:
            return m.x_stored[iter_item, 1] <= item_height_stored[iter_item]/2.0 + bigM*(1-m.int_item_state[iter_item, 2])
        elif iter_cons == 3:
            return m.x_stored[iter_item, 1] >= item_height_stored[iter_item]/2.0 - bigM*(1-m.int_item_state[iter_item, 2])

    m.con_state_z2 = pyo.Constraint(m.range_items, pyo.RangeSet(0, 3), rule=state_z2)

    # State 3: the item is -90~0 degrees leaning towards right ---------------------------------------------------------
    def state_z3_contact_separating_plane(m, iter_item, iter_cons):
        # If it is the last item, special treatment
        if iter_item == num_of_item-1:
            if iter_cons == 0 or iter_cons == 1 or iter_cons == 2:
                return m.v_stored[iter_item, 0, 0] >= (bin_right - bigM*(1-m.int_item_state[iter_item, 3]))
            else:
                return m.v_stored[iter_item, 0, 0] <= (bin_right + bigM*(1-m.int_item_state[iter_item, 3]))

        # Otherwise, enforce leaning left constraint
        else:
            # Retrieve pair number
            found = False
            ct_pair = 0
            iter_pair = 0
            while not found:
                if list_pairs[iter_pair, 0] == iter_item and list_pairs[iter_pair, 1] == iter_item+1:
                    ct_pair = iter_pair
                    found = True
                else:
                    iter_pair += 1

            assert found, "The separating plane is not founded !"

            if iter_cons == 0:
                return (m.a_sep[ct_pair, 0] * m.v_stored[iter_item, 0, 0] +
                        m.a_sep[ct_pair, 1]*m.v_stored[iter_item, 0, 1]) >= (m.b_sep[ct_pair] - bigM*(1-m.int_item_state[iter_item, 3]))
            elif iter_cons == 1:
                return (m.a_sep[ct_pair, 0] * m.v_stored[iter_item, 0, 0] +
                        m.a_sep[ct_pair, 1]*m.v_stored[iter_item, 0, 1]) <= (m.b_sep[ct_pair] + bigM*(1-m.int_item_state[iter_item, 3]))
            elif iter_cons == 2:
                return (m.a_sep[ct_pair, 0] * m.v_stored[iter_item+1, 3, 0] +
                        m.a_sep[ct_pair, 1] * m.v_stored[iter_item+1, 3, 1]) >= (m.b_sep[ct_pair] - bigM * (1 - m.int_item_state[iter_item, 3]))
            elif iter_cons == 3:
                return (m.a_sep[ct_pair, 0] * m.v_stored[iter_item+1, 3, 0] +
                        m.a_sep[ct_pair, 1] * m.v_stored[iter_item+1, 3, 1]) <= (m.b_sep[ct_pair] + bigM * (1 - m.int_item_state[iter_item, 3]))
            elif iter_cons == 4:
                return m.x_stored[iter_item, 0] >= (m.v_stored[iter_item, 1, 0] - bigM*(1-m.int_item_state[iter_item, 3]))
            elif iter_cons == 5:
                return m.x_stored[iter_item, 0] <= (m.x_stored[iter_item+1, 0] + bigM*(1-m.int_item_state[iter_item, 3]))
            else:
                assert False, "Constraint number out of bound!"

    def state_z3_on_ground(m, iter_item, iter_cons):
        if iter_cons == 0:
            return m.v_stored[iter_item, 1, 1] <= (bin_ground + bigM * (1 - m.int_item_state[iter_item, 3]))
        elif iter_cons == 1:
            return m.v_stored[iter_item, 1, 1] >= (bin_ground - bigM * (1 - m.int_item_state[iter_item, 3]))
        else:
            assert False, "Constraint number out of bound!"

    def state_z3_R21_range(m, iter_item):
        return m.R_wb_stored[iter_item, 1, 0] <= 0 + bigM * (1 - m.int_item_state[iter_item, 3])

    m.con_state_z3_contact_separating_plane = pyo.Constraint(m.range_items, pyo.RangeSet(0, 5), rule=state_z3_contact_separating_plane)
    m.con_state_z3_on_ground = pyo.Constraint(m.range_items, pyo.RangeSet(0, 1), rule=state_z3_on_ground)
    m.con_state_z3_R21_range = pyo.Constraint(m.range_items, rule=state_z3_R21_range)

    # State 4: the item is -90 degrees on the ground -------------------------------------------------------------------
    def state_z4(m, iter_item, iter_cons):
        if iter_cons == 0:
            return m.R_wb_stored[iter_item, 1, 0] <= -1 + bigM * (1 - m.int_item_state[iter_item, 4])
        elif iter_cons == 1:
            return m.R_wb_stored[iter_item, 1, 0] >= -1 - bigM * (1 - m.int_item_state[iter_item, 4])
        elif iter_cons == 2:
            return m.x_stored[iter_item, 1] <= item_width_stored[iter_item]/2.0 + bigM*(1-m.int_item_state[iter_item, 4])
        elif iter_cons == 3:
            return m.x_stored[iter_item, 1] >= item_width_stored[iter_item]/2.0 - bigM*(1-m.int_item_state[iter_item, 4])

    m.con_state_z4 = pyo.Constraint(m.range_items, pyo.RangeSet(0, 3), rule=state_z4)

    # In hand item =====================================================================================================
    def state_z0_in_hand(m, iter_cons):
        if iter_cons == 0:
            return m.R_wb_in_hand[1, 0] <= 1 + bigM * (1 - m.int_item_state_in_hand[0])
        elif iter_cons == 1:
            return m.R_wb_in_hand[1, 0] >= 1 - bigM * (1 - m.int_item_state_in_hand[0])
        elif iter_cons == 2:
            return m.x_in_hand[1] <= item_width_in_hand / 2.0 + bigM * (1 - m.int_item_state_in_hand[0])
        elif iter_cons == 3:
            return m.x_in_hand[1] >= item_width_in_hand / 2.0 - bigM * (1 - m.int_item_state_in_hand[0])

    m.con_state_z0_in_hand = pyo.Constraint(pyo.RangeSet(0, 3), rule=state_z0_in_hand)

    def state_z1_in_hand(m, iter_cons):
        if iter_cons == 0:
            return m.R_wb_in_hand[1, 0] <= bigM * (1 - m.int_item_state_in_hand[1])
        elif iter_cons == 1:
            return m.R_wb_in_hand[1, 0] >= - bigM * (1 - m.int_item_state_in_hand[1])
        elif iter_cons == 2:
            return m.x_in_hand[1] <= item_height_in_hand/2.0 + bigM*(1-m.int_item_state_in_hand[1])
        elif iter_cons == 3:
            return m.x_in_hand[1] >= item_height_in_hand/2.0 - bigM*(1-m.int_item_state_in_hand[1])

    m.con_state_z1_in_hand = pyo.Constraint(pyo.RangeSet(0, 3), rule=state_z1_in_hand)

    def state_z2_in_hand(m, iter_cons):
        if iter_cons == 0:
            return m.R_wb_in_hand[1, 0] <= -1 + bigM * (1 - m.int_item_state_in_hand[2])
        elif iter_cons == 1:
            return m.R_wb_in_hand[1, 0] >= -1 - bigM * (1 - m.int_item_state_in_hand[2])
        elif iter_cons == 2:
            return m.x_in_hand[1] <= item_width_in_hand / 2.0 + bigM * (1 - m.int_item_state_in_hand[2])
        elif iter_cons == 3:
            return m.x_in_hand[1] >= item_width_in_hand / 2.0 - bigM * (1 - m.int_item_state_in_hand[2])

    m.con_state_z2_in_hand = pyo.Constraint(pyo.RangeSet(0, 3), rule=state_z2_in_hand)

    # Additional constraints for tuning
    def tune(m):
        return m.int_item_state[0, 1] == 1.0

    m.con_tune = pyo.Constraint(rule=tune)

    # Objective function ===============================================================================================
    def obj_expression(m):
        return sum( (m.x_stored[iter_item, iter_dim] - item_center_stored[iter_item, iter_dim]) * (m.x_stored[iter_item, iter_dim] - item_center_stored[iter_item, iter_dim])
                    + (m.R_wb_stored[iter_item, 0, 0] - np.cos(item_angle_stored[iter_item])) * (m.R_wb_stored[iter_item, 0, 0] - np.cos(item_angle_stored[iter_item]))
                    + (m.R_wb_stored[iter_item, 0, 1] + np.sin(item_angle_stored[iter_item])) * (m.R_wb_stored[iter_item, 0, 1] + np.sin(item_angle_stored[iter_item]))
                    + (m.R_wb_stored[iter_item, 1, 0] - np.sin(item_angle_stored[iter_item])) * (m.R_wb_stored[iter_item, 1, 0] - np.sin(item_angle_stored[iter_item]))
                    + (m.R_wb_stored[iter_item, 1, 1] - np.cos(item_angle_stored[iter_item])) * (m.R_wb_stored[iter_item, 1, 1] - np.cos(item_angle_stored[iter_item])) for iter_item in m.range_items for iter_dim in m.range_dim)

    m.OBJ = pyo.Objective(rule=obj_expression)
    instance = m.create_instance()

    opt = pyo.SolverFactory('bonmin', executable="/home/romela/xuan/Bonmin-1.8.8/build/bin/bonmin")

    results = opt.solve(instance, tee=True)

    feasible = np.all(results.Solver.termination_condition == 'optimal')

    if feasible:
        print("This problem is Feasible !!!")
    else:
        print("This problem is Infeasible !!!")

    x_stored_sol = instance.x_stored.extract_values()
    R_wb_stored_sol = instance.R_wb_stored.extract_values()
    v_stored_sol = instance.v_stored.extract_values()

    x_in_hand_sol = instance.x_in_hand.extract_values()
    R_wb_in_hand_sol = instance.R_wb_in_hand.extract_values()
    v_in_hand_sol = instance.v_in_hand.extract_values()

    a_sep_sol = instance.a_sep.extract_values()
    b_sep_sol = instance.b_sep.extract_values()

    int_item_state_sol = instance.int_item_state.extract_values()
    int_item_state_in_hand_sol = instance.int_item_state_in_hand.extract_values()

    print("====================================")
    for iter_item in range(num_of_item):
         print([int_item_state_sol[iter_item, iter_state] for iter_state in m.range_states])

    print("------------------------------------")
    print([int_item_state_in_hand_sol[iter_state] for iter_state in m.range_in_hand_states])

    # Plot original bin
    fig = plt.subplot(2, 1, 1)

    plot_rectangle(v_bin, color='black', show=False)

    for iter_item in range(num_of_item):
        theta = item_angle_stored[iter_item]
        R_wb = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

        v_item = get_vertices(item_center_stored[iter_item, :],
                              R_wb,
                              np.array([item_height_stored[iter_item], item_width_stored[iter_item]]))

        plot_rectangle(v_item, color='red', show=False)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot solved bin
    fig = plt.subplot(2, 1, 2)

    plot_rectangle(v_bin, color='black', show=False)

    # # Plot interior points of the polygons
    # for iter_x in range(100):
    #     xxx = bin_left + (bin_right-bin_left)*iter_x/100
    #     for iter_y in range(100):
    #         yyy = bin_ground + (bin_up-bin_ground)*iter_y/100
    #         for iter_ii in range(num_of_item+1):
    #             if np.all(A_matrix_sol[iter_ii, :, :].dot(np.array([[xxx], [yyy]])) <= B_matrix_sol[iter_ii, :]):
    #                 plt.plot(xxx, yyy, '.')

    for iter_item in range(num_of_item):
        R_wb_stored = np.array([[R_wb_stored_sol[iter_item, 0, 0], R_wb_stored_sol[iter_item, 0, 1]],
                                [R_wb_stored_sol[iter_item, 1, 0], R_wb_stored_sol[iter_item, 1, 1]]])

        print("------------------- verification of orthogonality -----------------------")
        print(R_wb_stored)
        print(R_wb_stored.transpose().dot(R_wb_stored))

        v_item_stored = np.array([[v_stored_sol[iter_item, 0, 0], v_stored_sol[iter_item, 0, 1]],
                                  [v_stored_sol[iter_item, 1, 0], v_stored_sol[iter_item, 1, 1]],
                                  [v_stored_sol[iter_item, 2, 0], v_stored_sol[iter_item, 2, 1]],
                                  [v_stored_sol[iter_item, 3, 0], v_stored_sol[iter_item, 3, 1]]])

        plot_rectangle(v_item_stored, color='red', show=False)

    R_wb_in_hand = np.array([[R_wb_in_hand_sol[0, 0], R_wb_in_hand_sol[0, 1]],
                             [R_wb_in_hand_sol[1, 0], R_wb_in_hand_sol[1, 1]]])

    print("------------------- verification of orthogonality -----------------------")
    print(R_wb_in_hand.transpose().dot(R_wb_in_hand))

    v_item_in_hand = np.array([[v_in_hand_sol[0, 0], v_in_hand_sol[0, 1]],
                               [v_in_hand_sol[1, 0], v_in_hand_sol[1, 1]],
                               [v_in_hand_sol[2, 0], v_in_hand_sol[2, 1]],
                               [v_in_hand_sol[3, 0], v_in_hand_sol[3, 1]]])

    print("===================== rotation matrix of item in hand =======================")
    print(R_wb_in_hand)

    print("===================== vertices of item in hand ==========================")
    print(v_item_in_hand)

    print("============================================================================================")
    for iter_a in m.range_pairs:
        print("---------------------------------------------------")
        print([a_sep_sol[iter_a, 0], a_sep_sol[iter_a, 1]])
        print(a_sep_sol[iter_a, 0]**2 + a_sep_sol[iter_a, 1]**2)

    plot_rectangle(v_item_in_hand, color='blue', show=False)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()
