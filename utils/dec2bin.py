import numpy as np


def dec2bin(dec, bin_digits):
    """
    Change decimal value to a list of binary values. little endian.
    :param dec:
    :param bin_digits:
    :return:
    """

    dec = int(dec)

    arr = np.zeros(bin_digits)

    ct = bin_digits - 1

    for iter in range(bin_digits):
        arr[ct] = dec//(2**ct)
        dec -= (2**ct)*arr[ct]

        ct -= 1

    return arr
