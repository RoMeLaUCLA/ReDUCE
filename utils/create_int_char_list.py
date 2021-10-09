
def create_int_char_list(num_of_int):
    """
    Given the number of int variables, create a list of char list that has all combination of int variables
    For example, if num_of_int=3, the list will have ['000', '001', '010', '011', '100', '101', '110', '111']
    :param num_of_int: Number of integer variables (or number of digits)
    :return: A list of all combination of integer variables, char type.
    """

    int_active = []
    for iter_int in range(pow(2, num_of_int)):
        bin_char = bin(iter_int)[2:].zfill(num_of_int)  # Change the iter value to binary
        int_active.append(bin_char)

    return int_active
