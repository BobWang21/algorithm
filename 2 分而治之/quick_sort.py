#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def quick_sort(nums, lo, hi):
    if lo < hi:  # 长度为1不用排序
        mid = partition(nums, lo, hi)
        quick_sort(nums, lo, mid - 1)
        quick_sort(nums, mid + 1, hi)


def partition(nums, lo, hi):  # 原地修改
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


# 无序数组的前K个数
def get_least_num(nums, k):
    def partition(nums, l, r):
        if l >= r:
            return
        pivot = nums[l]
        while l < r:
            while l < r and nums[r] >= pivot:
                r -= 1
            nums[l] = nums[r]
            while l < r and nums[l] <= pivot:
                l += 1
            nums[r] = nums[l]
        nums[l] = pivot
        return l

    l, r = 0, len(nums) - 1
    p = partition(nums, l, r)
    while p != k - 1:
        if p + 1 < k:
            l = p + 1
        else:
            r = p - 1
        p = partition(nums, l, r)
    return nums[:k]


if __name__ == '__main__':
    print('快排')
    nums = [4, 3, 1, 9]
    quick_sort(nums, 0, 3)
    print(nums)

    print('奇偶分离')
    print(sort_array_by_parity(nums))

    print('无序数组的前K个数')
    print(get_least_num([10, 9, 8, 9, 1, 2, 0], 3))