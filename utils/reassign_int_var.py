from create_int_char_list import create_int_char_list


def reassign_int_var(list_sect, list_int, list_infeasible):
    """
    Given an list of sections, its corresponding integer variables, and a list of sections that needed to be ruled out,
    this function re-assigns the integer variables to the remaining sections.
    If the remaining sections don't use up the combination of integer variables, the redundant sections will repeat the
    last section.

    :param list_sect: A list of sections.
    :param list_int: A list of integer variables that corresponds to each of the section. Char type.
    :param list_infeasible: A list of numbers indicating sections that needed to be ruled out.
    :return:
    num_of_int_reassigned: The number of integer variables that is needed for the remaining sections.
    list_sect_reassigned: A list of sections that are re-assigned the integer variables (possibly repeated at the end).
    list_sect_label_reassigned_with_repeat: A list of labels of the reassigned sections corresponding to the list above.
    reassign_map: A dictionary that, given the re-assigned integer variables as keys, will output the original integer
                  variables.
    """

    assert len(list_sect) == len(list_int), "Inconsistent length of section list and integer list!"
    assert len(list_infeasible) < len(list_sect), "At least one section should remain active!"
    assert len(list_infeasible) == len(set(list_infeasible)), "Infeasible list should not have duplicates!"

    len_sect = len(list_sect)
    len_suppress = len(list_infeasible)
    len_sect_remain = len_sect - len_suppress

    # Compute the number of integer variables required
    num_of_int_reassigned = 0
    while pow(2, num_of_int_reassigned) < len_sect_remain:
        num_of_int_reassigned += 1

    sect_active = [elem for elem in list(range(len_sect)) if not elem in list_infeasible]
    len_active = len(sect_active)

    list_sect_reassigned = []
    list_sect_label_reassigned_with_repeat = []
    for iter_sect in range(pow(2, num_of_int_reassigned)):
        if iter_sect < len_active:
            list_sect_reassigned.append(list_sect[sect_active[iter_sect]])
            list_sect_label_reassigned_with_repeat.append(sect_active[iter_sect])
        else:
            list_sect_reassigned.append(list_sect[sect_active[-1]])
            list_sect_label_reassigned_with_repeat.append(sect_active[-1])

    reassign_map = {}

    int_active = create_int_char_list(num_of_int_reassigned)
    # num_of_int = num_of_int_reassigned
    # int_active = []
    # for iter_int in range(pow(2, num_of_int)):
    #     bin_char = bin(iter_int)[2:].zfill(num_of_int)  # Change the iter value to binary
    #     int_active.append(bin_char)

    for iter_int in range(pow(2, num_of_int_reassigned)):
        if iter_int < len_active:
            reassign_map[int_active[iter_int]] = list_int[sect_active[iter_int]]
        else:
            reassign_map[int_active[iter_int]] = list_int[sect_active[-1]]

    return num_of_int_reassigned, list_sect_reassigned, list_sect_label_reassigned_with_repeat, reassign_map
