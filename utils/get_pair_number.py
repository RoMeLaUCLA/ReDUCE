def get_pair_number(list_pairs, item_left, item_right):
    # Retrieve pair number
    found = False
    ct_pair = 0
    iter_pair = 0
    while not found:
        if list_pairs[iter_pair, 0] == item_left and list_pairs[iter_pair, 1] == item_right:
            ct_pair = iter_pair
            found = True
        else:
            iter_pair += 1
    return ct_pair
