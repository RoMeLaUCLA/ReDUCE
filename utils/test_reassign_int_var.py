from reassign_int_var import reassign_int_var

list_sect = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
list_int = ['000', '001', '010', '011', '100', '101', '110', '111']
list_infeasible = [0, 2, 3, 4, 7]

num_of_int_reassigned, list_sect_reassigned, reassigned_region_with_repeat, reassign_map = reassign_int_var(list_sect, list_int, list_infeasible)

print(num_of_int_reassigned)
print(list_sect_reassigned)
print(reassigned_region_with_repeat)
print(reassign_map)

assert reassigned_region_with_repeat == [1, 5, 6, 6]
assert reassign_map == {'00': '001', '01': '101', '10': '110', '11': '110'}
