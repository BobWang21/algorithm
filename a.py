def partition2(nums, l, r):
    pivot = nums[l]
    lt, mt = l, r
    i = l
    while i <= mt:
        if nums[i] < pivot:
            nums[lt], nums[i] = nums[i], nums[lt]
            i += 1
            lt += 1
        elif nums[i] == pivot:
            i += 1
        else:
            nums[mt], nums[i] = nums[i], nums[mt]
            mt -= 1
    return lt, mt


def quick_sort(nums, l, r):
    if l < r:
        lt, mt = partition2(nums, l, r)
        quick_sort(nums, l, lt - 1)
        quick_sort(nums, mt + 1, r)


if __name__ == '__main__':
    nums = [5, 1, 2, 4, -1, 5, 9, 10]
    print(quick_sort(nums, 0, len(nums) - 1))
    print(nums)
