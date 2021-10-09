from create_int_char_list import create_int_char_list


def generate_int_list(int_var):
    """
    A function that, given a list of integer variables, will return a list of combination of integer variables such that
    if all integer variables are 1, the first section will be all zeros, and so on. For example, for 2 integer variables,
    the return value will be:
    [[1 - z[0], 1 - z[1]],  # sect0 = [1, 1] when z0=0, z1=0
     [    z[0], 1 - z[1]],  # sect1 = [1, 1] when z0=1, z1=0
     [1 - z[0],     z[1]],  # sect2 = [1, 1] when z0=0, z1=1
     [    z[0],     z[1]]]  # sect3 = [1, 1] when z0=1, z1=1
     Note! This is little edian, left most should be z[0]
    This returned list can be used in combination of a section list, for constraint enforcer that will enforce sections
    when the associated section of combination integer list = [1, 1, ..., 1].
    :param int_var: A list of integer variables, can be gurobi addVars variables.
    :return: A list of combination of integer variables.
    """

    len_int = len(int_var)

    if len_int == 0:
        return []

    else:

        list_char = create_int_char_list(len_int)

        list_int = []
        for iter_sect in range(int(pow(2, len_int))):
            this_sect = []
            for iter_int in range(len_int):
                if list_char[iter_sect][-(iter_int+1)] == '0':
                    this_sect.append(1-int_var[iter_int])
                elif list_char[iter_sect][-(iter_int+1)] == '1':
                    this_sect.append(int_var[iter_int])
                else:
                    assert False, "Something is wrong !!"
            list_int.append(this_sect)

        return list_int
