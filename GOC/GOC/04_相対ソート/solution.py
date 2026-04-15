def relativeSortArray(arr1, arr2):
    order = {v: i for i, v in enumerate(arr2)}
    
    # arr2にある → (0, arr2でのindex)
    # arr2にない → (1, 値) で末尾に昇順
    return sorted(arr1, key=lambda x: (0, order[x]) if x in order else (1, x))

print(relativeSortArray(
    [2,3,1,3,2,4,6,7,9,2,19],
    [2,1,4,3,9,6]
))
# [2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19]
