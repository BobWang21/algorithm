#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def quick_sort(nums, lo, hi):
    if lo < hi:  # 此处为小于号 而不是等于号
        mid = partition(nums, lo, hi)
        quick_sort(nums, lo, mid - 1)
        quick_sort(nums, mid + 1, hi)
        return nums


def partition(nums, lo, hi):
    pivot = nums[lo]
    while lo < hi:
        while lo < hi and pivot <= nums[hi]:
            hi -= 1
        nums[lo] = nums[hi]  # 替换已保存的数据
        while lo < hi and nums[lo] <= pivot:
            lo += 1
        nums[hi] = nums[lo]  # 替换已保存的数据
    nums[lo] = pivot
    return lo


# 奇数在左边 偶数在右边
def sort_array_by_parity(nums):
    n = len(nums)
    if n < 2:
        return nums
    l, r = 0, n - 1
    pivot = nums[0]
    while l < r:
        while l < r and nums[r] % 2 == 1:
            r -= 1
        nums[l] = nums[r]
        while l < r and nums[l] % 2 == 0:
            l += 1
        nums[r] = nums[l]
    nums[l] = pivot
    return nums


if __name__ == '__main__':
    print('快排')
    nums = [4, 3, 8, 9, 7, 1]
    print(quick_sort(nums, 0, 5))

    print('奇偶分离')
    print(sort_array_by_parity(nums))
