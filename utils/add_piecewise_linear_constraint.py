import gurobipy as go


def add_piecewise_linear_constraint(model, x, y, x_lim, k, b, int_list, bigM, pinned=False):
    """
    Add piecewise linear constraint to the model. The specific section is selected when all variables in the corresponding
    row of int_list equal to one.
    :param model:
    :param x:
    :param y:
    :param x_lim:
    :param k:
    :param b:
    :param int_list:
    :param bigM:
    :return: A list of reference to constraints that can be removed using model.remove()
    """

    constrs = []

    if not int_list:
        assert len(k) == len(b) == len(x_lim) == 1, "Inconsistent number of section"

        constrs.append(model.addConstr(y == k[0] * x + b[0]))
        constrs.append(model.addConstr(x >= x_lim[0][0]))
        constrs.append(model.addConstr(x <= x_lim[0][1]))

    else:
        assert len(k) == len(b) == len(int_list) == (len(x_lim)), "Inconsistent number of section"

        if pinned:
            for iter_piece in range(len(k)):

                select = True
                # Only implement the section when all integer variables are 1
                for iter_select in range(len(int_list[iter_piece])):
                    if int_list[iter_piece][iter_select] == 0:
                        select = False
                        break

                if select:
                    constrs.append(model.addConstr(y == k[iter_piece] * x + b[iter_piece]))
                    constrs.append(model.addConstr(x >= x_lim[iter_piece][0]))
                    constrs.append(model.addConstr(x <= x_lim[iter_piece][1]))
                    break

        else:
            for iter_piece in range(len(k)):

                constrs.append(model.addConstr(y <= k[iter_piece] * x + b[iter_piece] +
                            go.quicksum(bigM * (1-int_list[iter_piece][iter_int]) for iter_int in range(len(int_list[iter_piece]))) ) )

                constrs.append(model.addConstr(y >= k[iter_piece] * x + b[iter_piece] -
                            go.quicksum(bigM * (1-int_list[iter_piece][iter_int]) for iter_int in range(len(int_list[iter_piece]))) ) )

                constrs.append(model.addConstr(x >= x_lim[iter_piece][0] -
                                               go.quicksum(bigM * (1-int_list[iter_piece][iter_int]) for iter_int in range(len(int_list[iter_piece])))) )

                constrs.append(model.addConstr(x <= x_lim[iter_piece][1] +
                                               go.quicksum(bigM * (1-int_list[iter_piece][iter_int]) for iter_int in range(len(int_list[iter_piece]))) ) )

    return constrs


def add_piecewise_linear_constraint_old(model, x, y, x_lim, k, b, int_list, bigM):
    """
    Add piecewise linear constraint to the model. The specific section is selected when all variables in the corresponding
    row of int_list equal to one.
    :param model:
    :param x:
    :param y:
    :param x_lim:
    :param k:
    :param b:
    :param int_list:
    :param bigM:
    :return:
    """

    assert len(k) == len(b) == len(int_list) == (len(x_lim)-1), "Inconsistent number of section"

    for iter_piece in range(len(k)):

        model.addConstr(y <= k[iter_piece] * x + b[iter_piece] +
                    sum(bigM * (1-int_list[iter_piece][iter_int]) for iter_int in range(len(int_list[iter_piece]))) )
        model.addConstr(y >= k[iter_piece] * x + b[iter_piece] -
                    sum(bigM * (1-int_list[iter_piece][iter_int]) for iter_int in range(len(int_list[iter_piece]))) )

        model.addConstr(x >= x_lim[iter_piece] - sum(bigM * (1 - int_list[iter_piece][iter_int]) for iter_int in range(len(int_list[iter_piece]))))
        model.addConstr(x <= x_lim[iter_piece+1] + sum(bigM * (1-int_list[iter_piece][iter_int]) for iter_int in range(len(int_list[iter_piece]))) )