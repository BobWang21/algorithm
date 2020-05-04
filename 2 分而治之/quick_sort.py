#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def sort_colors(nums):
    def swap(i, j):
        nums[i], nums[j] = nums[j], nums[i]

    if not nums:
        return []
    zero, two = 0, len(nums) - 1
    i = 0
    while i <= two:
        if nums[i] == 0:
            swap(i, zero)
            i += 1  # 保证i前为0或1
            zero += 1
        elif nums[i] == 1:
            i += 1  # 保证i前为0或1
        else:
            swap(two, i)
            two -= 1  # 后边缓过来的数 并不知道其数值 因此不移动i
    return nums


def quick_sort(nums, lo, hi):
    if lo < hi:  # 长度为1不用排序
        mid = partition(nums, lo, hi)
        quick_sort(nums, lo, mid - 1)
        quick_sort(nums, mid + 1, hi)


def partition(nums, lo, hi):  # 原地修改
    pivot = nums[lo]  # 可以随机选择pivot
    while lo < hi:
        while lo < hi and pivot <= nums[hi]:
            hi -= 1
        nums[lo] = nums[hi]  # 替换已保存的数据
        while lo < hi and nums[lo] <= pivot:
            lo += 1
        nums[hi] = nums[lo]  # 替换已保存的数据
    nums[lo] = pivot
    return lo


# 三个指针 partition
def triple_partition(nums, target):
    def swap(i, j):
        nums[i], nums[j] = nums[j], nums[i]

    if not nums:
        return []
    lt, gt = 0, len(nums) - 1
    i = 0
    while i <= gt:
        if nums[i] < target:
            swap(i, lt)
            i += 1
            lt += 1
        elif nums[i] == target:  # 保证i前为0或1
            i += 1
        else:
            swap(gt, i)
            gt -= 1  # 后边换过来的数 并不知道其数值 因此不移动i
    return lt, gt, nums


# 奇数在左边 偶数在右边
def sort_array_by_parity(nums):
    n = len(nums)
    if n < 2:
        return nums
    l, r = 0, n - 1
    pivot = nums[0]
    while l < r:
        while l < r and not nums[r] % 2:
            r -= 1
        nums[l] = nums[r]
        while l < r and nums[l] % 2:
            l += 1
        nums[r] = nums[l]
    nums[l] = pivot
    return nums


# 无序数组的前K个数T(n) = n + T(n/2) ->O(n)
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
    print('\n颜色排序')
    print(sort_colors([2, 0, 2, 1, 1, 0]))

    print('\n三路partition')
    print(triple_partition([-1, 2, 0, 5, 5, 4, 3, 2, 10], 5))

    print('\n快排')
    nums = [4, 3, 1, 9]
    quick_sort(nums, 0, 3)
    print(nums)

    print('\n奇偶分离')
    print(sort_array_by_parity(nums))

    print('\n无序数组的前K个数')
    print(get_least_num([10, 9, 8, 9, 1, 2, 0], 3))
