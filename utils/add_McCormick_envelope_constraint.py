import numpy as np
import gurobipy as go


# ======================================================================================================================
def add_McCormick_envelope_constraint_from_limit_pf(model, p_component, force_component, moment_component, envelope_param,
                                      N_rounds, N_steps_per_round):

    # for MICP McCormick envelope, you only need to multiply the offset term of plane by z, thus [0,0,0] will satisfy the plane constraint
    for i in range(N_rounds):
        for j in range(N_steps_per_round):

            px_L = envelope_param[12*j+0]
            px_U = envelope_param[12*j+1]
            py_L = envelope_param[12*j+2]
            py_U = envelope_param[12*j+3]
            pz_L = envelope_param[12*j+4]
            pz_U = envelope_param[12*j+5]

            fx_L = envelope_param[12*j+6]
            fx_U = envelope_param[12*j+7]
            fy_L = envelope_param[12*j+8]
            fy_U = envelope_param[12*j+9]
            fz_L = envelope_param[12*j+10]
            fz_U = envelope_param[12*j+11]

            # xyz corresponding to the kkf index
            p_L = np.array([px_L, py_L, pz_L])
            p_U = np.array([px_U, py_U, pz_U])

            f_L = np.array([fx_L, fy_L, fz_L])
            f_U = np.array([fx_U, fy_U, fz_U])

            # print("================================ This is leg %d ====================================" % j)
            for kkp in range(3):
                for kkf in range(3):

                    # McCormick envelope constraints:
                    # m >= p_L*f + p*f_L - p_L*f_L
                    # m >= p_U*f + p*f_U - p_U*f_U
                    # m <= p_U*f + p*f_L - p_U*f_L
                    # m <= p_L*f + p*f_U - p_L*f_U

                    model.addConstr(moment_component[i, j, kkp, kkf] >=
                        (p_L[kkp]*force_component[i, j, kkf] + p_component[i, j, kkp]*f_L[kkf] - p_L[kkp]*f_L[kkf]))

                    model.addConstr(moment_component[i, j, kkp, kkf] >=
                        (p_U[kkp]*force_component[i, j, kkf] + p_component[i, j, kkp]*f_U[kkf] - p_U[kkp]*f_U[kkf]))

                    model.addConstr(moment_component[i, j, kkp, kkf] <=
                        (p_U[kkp]*force_component[i, j, kkf] + p_component[i, j, kkp]*f_L[kkf] - p_U[kkp]*f_L[kkf]))

                    model.addConstr(moment_component[i, j, kkp, kkf] <=
                        (p_L[kkp]*force_component[i, j, kkf] + p_component[i, j, kkp]*f_U[kkf] - p_L[kkp]*f_U[kkf]))


# ======================================================================================================================
def add_vertex_polytope_constraint_gurobi(model, x, lam, v, arr_integer, arr_selection, pinned=False):
    """
    A function that receives the x variable, the lambda variable, the vertices of polytopes, a set of integer selection
    variables, and a selection array, and enforces a constraint:
                                                 x = sum(lambda(i)*v(i))
    where lambda(i)s are subject to selection integers. This means, for each row of arr_integer, if all variables within
    that row equals 1, then the lambda(i)s that corresponds to the same row in arr_selection will sum up to 1, and other
    lambda(i)s will be 0.

    E.g. if lambda has 7 elements, arr_integer[0] = [z[1], 1-z[2]], arr_selection[0] = [2, 4, 5], then a constraint will
    be enforced: x = lambda(2)*v(2) + lambda(4)*v(4) + lambda(5)*v(5) if z[1] == 1 and 1-z[2] == 1
    Essentially, x needs to stay within the given polytope

    This function currently only supports vector of x and v (they need to be of same dimension). It may support multiple
    sets of x and v (matrix) later.

    :param model: Model of the problem.
    :param x: An 1D variable vector that will be confined within the given polytope provided different combination of z.
    :param lam:  An 1D variable vector to form the polytope. Sum(lambda(i)) == 1 and lambda(i) needs to satisfy
                0 <= lambda(i) <= 1.
    :param v: A ND array where each column is the vertices of the polytopes.
    :param arr_integer: Array of each integer variable. For each row, the corresponding vertices given by arr_selection
                        are chosen if all integer variables within arr_integer of this row equal to 1.
    :param arr_selection: If all the integer variables in the same row of arr_integer equals to 1, then the lambda(i)s
                        selected by this row of arr_selection sum up to 1.
    :return: If the constraint is successfully enforced.
    """

    constr_ret = []

    if not arr_integer:
        assert arr_selection == [], "Both arr_integer and arr_selection should be empty !"
        assert len(x) == np.shape(v)[0], "Error: Inconsistent dimension between x and vertices."
        assert np.shape(v)[1] == len(lam), "Error: Inconsistent dimension between number of vertices and number of lambda variables."

        len_dim = len(x)
        len_lambda = len(lam)

        # Constraint: lambda variables sum up to 1
        constr_ret.append(model.addConstr(go.quicksum(lam[i] for i in range(len_lambda)) == 1.0))

        # Constraint: lambda variables should be larger than or equal to 0
        for i in range(len_lambda):
            constr_ret.append(model.addConstr(lam[i] >= 0))

        # Constraint: x is the convex combination of vertices
        for iter_dim in range(len_dim):
            constr_ret.append(model.addConstr(x[iter_dim] == go.quicksum(
                lam[i] * v[iter_dim, i] for i in range(len_lambda))))  # vertices are given in form of column vectors

    else:
        assert len(x) == np.shape(v)[0], "Error: Inconsistent dimension between x and vertices."
        assert np.shape(v)[1] == len(lam), "Error: Inconsistent dimension between number of vertices and number of lambda variables."
        assert np.shape(arr_integer)[0] == np.shape(arr_selection)[0], "Error: Number of row for arr_integer and arr_selection should be identical."

        for iter_row in range(np.shape(arr_selection)[0]):
            assert max(arr_selection[iter_row]) < len(lam), "Error: selection indices should be less than number of lambda variables."
            assert min(arr_selection[iter_row]) >= 0, "Error: selection indices should be larger than or equal to zero."

        len_dim = len(x)
        len_lambda = len(lam)
        len_selection = np.shape(arr_integer)[0]

        # Constraint: lambda variables sum up to 1
        constr_ret.append(model.addConstr(go.quicksum(lam[i] for i in range(len_lambda)) == 1.0))

        # Constraint: lambda variables should be larger than or equal to 0
        for i in range(len_lambda):
            constr_ret.append(model.addConstr(lam[i] >= 0))

        # Constraint: x is the convex combination of vertices
        for iter_dim in range(len_dim):
            constr_ret.append(model.addConstr(x[iter_dim] == go.quicksum(
                lam[i] * v[iter_dim, i] for i in range(len_lambda))))  # vertices are given in form of column vectors

        # Constraint: compress the non-selected lambda variables to 0
        all_index = [i for i in range(len_lambda)]  # Generate a list of all indices

        # For each row of selection, generate a tuple that has all zero points
        for iter_row in range(len_selection):
            zz = []
            for index in all_index:
                if not (index in arr_selection[iter_row]):
                    zz.append(index)

            len_z = len(arr_integer[iter_row])  # Number of decision integers in the given row

            if pinned:
                select = True
                for iter_select in range(len_z):
                    if arr_integer[iter_row][iter_select] == 0:
                        select = False
                        break

                if select:
                    constr_ret.append(model.addConstr(go.quicksum(lam[iter_lambda] for iter_lambda in zz) <= 0 ))
                    break

            else:
                constr_ret.append(model.addConstr(go.quicksum(lam[iter_lambda] for iter_lambda in zz)
                                <= go.quicksum(((1 - arr_integer[iter_row][iter_z]) for iter_z in range(len_z)))))

    return True, constr_ret


# ======================================================================================================================
def add_bilinear_constraint_gurobi(model, x, lam, int_var, num_of_polygon, vertex_all, pinned=False):
    # TODO: make it so that it can receive only one section (no int choice)
    """
    The order to choose each section in order is like [z0=0, z1=0], [z0=0, z1=1], [z0=1, z1=0], [z0=1, z1=1] and so on
    :param model:
    :param x:
    :param lam:
    :param int_var:
    :param num_of_polygon:
    :param vertex_all:
    :return:
    """

    import numpy as np
    from dec2bin import dec2bin

    if not int_var:  # If integer variables don't exist, set up single set of constraint
        num_of_vertices = 4 * num_of_polygon
        num_of_integer = 0

        assert len(x) == 3, "The set of bilinear variables should have length 3."
        assert len(lam) == num_of_vertices, "The length of lambda variable should corresponds to the number of vertices."
        assert np.shape(vertex_all)[1] == num_of_vertices, "Inconsistent length for input vertex."
        assert len(int_var) == num_of_integer, "Inconsistent length for input integer variables."

        lam_in = [lam[iter_lam] for iter_lam in range(num_of_vertices)]
        feas, ret_constr = add_vertex_polytope_constraint_gurobi(model, x, lam_in, vertex_all, [], [], pinned)

    else:
        num_of_vertices = 4*num_of_polygon
        num_of_integer = int(np.log2(num_of_polygon))

        assert len(x) == 3, "The set of bilinear variables should have length 3."
        assert len(lam) == num_of_vertices, "The length of lambda variable should corresponds to the number of vertices."
        assert np.shape(vertex_all)[1] == num_of_vertices, "Inconsistent length for input vertex."
        assert len(int_var) == num_of_integer, "Inconsistent length for input integer variables."

        arr_integer = []
        arr_int_char = []  # TODO: for debug
        arr_selection = []
        count_binary = 0
        for iter_polygon in range(num_of_polygon):
            z_list_temp = dec2bin(count_binary, num_of_integer)
            arr_list_temp = []
            arr_list_char = []  # TODO: for debug
            for iter_zz in range(num_of_integer):
                if z_list_temp[iter_zz] == 0.0:
                    arr_list_temp.append(1 - int_var[iter_zz])
                    arr_list_char.append('1-z{}'.format(iter_zz))
                else:
                    arr_list_temp.append(int_var[iter_zz])
                    arr_list_char.append('z{}'.format(iter_zz))

            arr_integer.append(arr_list_temp)
            arr_int_char.append(arr_list_char)
            arr_selection.append(
                [iter_polygon * 4 + 0, iter_polygon * 4 + 1, iter_polygon * 4 + 2, iter_polygon * 4 + 3])

            count_binary += 1

        lam_in = [lam[iter_lam] for iter_lam in range(num_of_vertices)]
        feas, ret_constr = add_vertex_polytope_constraint_gurobi(model, x, lam_in, vertex_all, arr_integer, arr_selection, pinned)

    return ret_constr


# ======================================================================================================================
def add_vertex_polytope_constraint_pyomo(model, x, lam, int, v, arr_integer, arr_selection):
    """
    A function that receives the x variable, the lambda variable, the vertices of polytopes, a set of integer selection
    variables, and a selection array, and enforces a constraint:
                                                 x = sum(lambda(i)*v(i))
    where lambda(i)s are subject to selection integers. This means, for each row of arr_integer, if all variables within
    that row equals 1, then the lambda(i)s that corresponds to the same row in arr_selection will sum up to 1, and other
    lambda(i)s will be 0.

    E.g. if lambda has 7 elements, arr_integer[0] = [z[1], 1-z[2]], arr_selection[0] = [2, 4, 5], then a constraint will
    be enforced: x = lambda(2)*v(2) + lambda(4)*v(4) + lambda(5)*v(5) if z[1] == 1 and 1-z[2] == 1
    Essentially, x needs to stay within the given polytope

    This function currently only supports vector of x and v (they need to be of same dimension). It may support multiple
    sets of x and v (matrix) later.

    :param model: Model of the problem. This model should contain the following:
           x: An 1D variable vector that will be confined within the given polytope provided different combination of z.
           lam:  An 1D variable vector to form the polytope. Sum(lambda(i)) == 1 and lambda(i) needs to satisfy
                0 <= lambda(i) <= 1.
           int: An 1D variable vector that determines the selection among branches of lambda variables.

    :param x: The name of x variable in model. Type: str.
    :param lam: The name of lambda variable in model. Type: str.
    :param int: The name of integer variable in model. Type: str.
    :param v: A 2D array where each column is the vertices of the polytopes.
    :param arr_integer: Array of each integer variable. Each row corresponds to a binary list of integer variables that
                       if satisfy, the corresponding vertices given by arr_selection are chosen.
    :param arr_selection: If all the integer variables in the same row of arr_integer equals to 1, then the lambda(i)s
                        selected by this row of arr_selection sum up to 1.
    :return: If the constraint is successfully enforced.
    """

    assert np.shape(getattr(model, x).index_set())[0] == np.shape(v)[0], "Error: Inconsistent dimension between x and vertices."
    assert np.shape(v)[1] == np.shape(getattr(model, lam).index_set())[0], "Error: Inconsistent dimension between number of vertices and number of lambda variables."
    assert np.shape(arr_integer)[0] == np.shape(arr_selection)[0], "Error: Number of row for arr_integer and arr_selection should be identical."

    for iter_row in range(np.shape(arr_selection)[0]):
        assert max(arr_selection[iter_row]) < np.shape(getattr(model, lam).index_set())[0], "Error: selection indices should be less than number of lambda variables."
        assert min(arr_selection[iter_row]) >= 0, "Error: selection indices should be larger than or equal to zero."

    len_dim = np.shape(getattr(model, x).index_set())[0]
    len_lambda = np.shape(getattr(model, lam).index_set())[0]
    len_selection = np.shape(arr_integer)[0]
    len_z = np.shape(arr_integer)[1]  # The length of integer variables should be consistent in each row

    # Constraint: lambda variables sum up to 1
    def lambda_sum_to_one(model):
        return sum(getattr(model, lam)[i] for i in range(len_lambda)) == 1.0

    model.con_lambda_sum_to_one = pyo.Constraint(rule=lambda_sum_to_one)

    # Constraint: lambda variables should be larger than or equal to 0
    def lambda_greater_than_zero(model, iter_lam):
        return getattr(model, lam)[iter_lam] >= 0

    model.con_lambda_greater_than_zero = pyo.Constraint(pyo.RangeSet(0, len_lambda - 1),
                                                        rule=lambda_greater_than_zero)

    # Constraint: x is the convex combination of vertices
    def x_sum_lambda(model, iter_dim):
        return getattr(model, x)[iter_dim] == sum(getattr(model, lam)[i] * v[iter_dim, i] for i in range(len_lambda))

    model.con_x_sum_lambda = pyo.Constraint(pyo.RangeSet(0, len_dim - 1), rule=x_sum_lambda)

    # Constraint: compress the non-selected lambda variables to 0
    all_index = [i for i in range(len_lambda)]  # Generate a list of all indices

    z_shutdown = []
    # For each row of selection, generate a tuple that has all zero points
    for iter_row in range(len_selection):
        zz = []
        for index in all_index:
            if not (index in arr_selection[iter_row]):
                zz.append(index)
        z_shutdown.append(zz)

    def compress_lambda_zero(model, iter_row):
        left = sum(getattr(model, lam)[iter_lambda] for iter_lambda in z_shutdown[iter_row])
        # If arr_integer[iter_row][iter_z] == 0, add m.int[iter_z], otherwise add (1-m.int[int_z])
        right = 0
        for iter_z in range(len_z):
            if arr_integer[iter_row][iter_z] == 0:
                right += getattr(model, int)[iter_z]
            elif arr_integer[iter_row][iter_z] == 1:
                right += (1-getattr(model, int)[iter_z])
            else:
                assert False, "Incorrect value in arr_integer !!"

        return left <= right

    model.con_compress_lambda_zero = pyo.Constraint(pyo.RangeSet(0, len_selection-1), rule=compress_lambda_zero)

    return model


# ======================================================================================================================
def add_vertex_polytope_constraint_pyomo_with_item_count(model, x_attr, y_attr, z_attr, x_index, y_index, lam, int, v,
                                                         arr_integer, arr_selection,
                                                         num_of_item, len_dim, len_lambda, len_selection, len_z):
    """
    This is the same function as above, except that it uses range set number of items. This is not general but I don't
    know how it can be general yet ...
    """

    # Constraint: lambda variables sum up to 1
    def lambda_sum_to_one(model, iter_item):
        return sum(getattr(model, lam)[iter_item, i] for i in range(len_lambda)) == 1.0

    model.con_lambda_sum_to_one = pyo.Constraint(pyo.RangeSet(0, num_of_item-1), rule=lambda_sum_to_one)

    # Constraint: lambda variables should be larger than or equal to 0
    def lambda_greater_than_zero(model, iter_item, iter_lam):
        return getattr(model, lam)[iter_item, iter_lam] >= 0

    model.con_lambda_greater_than_zero = pyo.Constraint(pyo.RangeSet(0, num_of_item-1), pyo.RangeSet(0, len_lambda - 1),
                                                        rule=lambda_greater_than_zero)

    # TODO: this function has to be made so strange ...
    # Constraint: x is the convex combination of vertices
    def x_sum_lambda(model, iter_item, iter_dim):

        if iter_dim == 0:
            return getattr(model, x_attr)[iter_item, x_index[0], x_index[1]] == sum(
                getattr(model, lam)[iter_item, i] * v[iter_dim, i] for i in range(len_lambda))
        elif iter_dim == 1:
            return getattr(model, y_attr)[iter_item, y_index[0], y_index[1]] == sum(
                getattr(model, lam)[iter_item, i] * v[iter_dim, i] for i in range(len_lambda))
        elif iter_dim == 2:
            return getattr(model, z_attr)[iter_item] == sum(
                getattr(model, lam)[iter_item, i] * v[iter_dim, i] for i in range(len_lambda))
        else:
            assert False, "Dimension exceed 3 !!!"

    model.con_x_sum_lambda = pyo.Constraint(pyo.RangeSet(0, num_of_item-1), pyo.RangeSet(0, len_dim - 1), rule=x_sum_lambda)

    # Constraint: compress the non-selected lambda variables to 0
    all_index = [i for i in range(len_lambda)]  # Generate a list of all indices

    z_shutdown = []
    # For each row of selection, generate a tuple that has all zero points
    for iter_row in range(len_selection):
        zz = []
        for index in all_index:
            if not (index in arr_selection[iter_row]):
                zz.append(index)
        z_shutdown.append(zz)

    def compress_lambda_zero(model, iter_item, iter_row):
        left = sum(getattr(model, lam)[iter_item, iter_lambda] for iter_lambda in z_shutdown[iter_row])
        # If arr_integer[iter_row][iter_z] == 0, add m.int[iter_z], otherwise add (1-m.int[int_z])
        right = 0
        for iter_z in range(len_z):
            if arr_integer[iter_row][iter_z] == 0:
                right += getattr(model, int)[iter_item, iter_z]
            elif arr_integer[iter_row][iter_z] == 1:
                right += (1-getattr(model, int)[iter_item, iter_z])
            else:
                assert False, "Incorrect value in arr_integer !!"

        return left <= right

    model.con_compress_lambda_zero = pyo.Constraint(pyo.RangeSet(0, num_of_item-1), pyo.RangeSet(0, len_selection-1), rule=compress_lambda_zero)

    return model


# ======================================================================================================================
def limit2vertex(bilinear_limits):
    """
    Compute polytope vertices for McCormick envelopes from the limits of bilinear variables
    :param bilinear_limits: A vector containing limits for bilinear variables. The vector is composed of:
    [x_L, x_U, y_L, y_U], where x, y are for the bilinear constraint z=xy.
    :return: A 4x3 array containing 4 vertex points for the polytope.
    """

    x_L = bilinear_limits[0]
    x_U = bilinear_limits[1]
    y_L = bilinear_limits[2]
    y_U = bilinear_limits[3]

    # The 4 vertices are (x_L, y_L, x_L*y_L), (x_U, y_L, x_U*y_L), (x_L, y_U, x_L*y_U), (x_U, y_U, x_U*y_U)
    # Can verify by seeing any one of the 4 planes has 3 points on that plane

    V = np.array([[x_L, y_L, x_L*y_L],
                  [x_U, y_L, x_U*y_L],
                  [x_L, y_U, x_L*y_U],
                  [x_U, y_U, x_U*y_U]])

    return V
