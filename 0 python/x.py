def large(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] <= target:
            l = mid + 1
        else:
            r = mid
    return r if nums[r] > target else -1


if __name__ == '__main__':
    print(large([1, 2, 3, 3, 3, 4], 0))
