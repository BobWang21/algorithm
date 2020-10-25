#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random


def partition1(nums, l, r):  # 原地修改
    pivot = nums[l]  # 可以随机选择pivot
    while l < r:
        while l < r and pivot <= nums[r]:  # 右
            r -= 1
        nums[l] = nums[r]  # 替换已保存的数据
        while l < r and nums[l] <= pivot:  # 左 左右必须有一个包含等于号
            l += 1
        nums[r] = nums[l]  # 替换已保存的数据
    nums[l] = pivot
    return l


# 三个指针 partition
def partition2(nums, l, r):
    def swap(i, j):
        nums[i], nums[j] = nums[j], nums[i]

    if not nums:
        return []
    lt, mt = l, r
    i = l
    pivot = nums[l]
    while i <= mt:  # !!!
        if nums[i] < pivot:
            swap(i, lt)
            i += 1
            lt += 1
        elif nums[i] == pivot:  # nums[i-1] <= pivot!
            i += 1
        else:
            swap(mt, i)
            mt -= 1  # 后边换过来的数 并不知道其数值 因此不移动i
    return lt, mt


def quick_sort1(nums, l, r):
    if l < r:  # 长度为1不用排序
        mid = partition1(nums, l, r)
        quick_sort1(nums, l, mid - 1)
        quick_sort1(nums, mid + 1, r)


def quick_sort2(nums, l, r):
    if l < r:  # 长度为1不用排序
        lt, mt = partition2(nums, l, r)
        quick_sort2(nums, l, lt - 1)
        quick_sort2(nums, mt + 1, r)


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
def get_top_k(nums, k):
    if k <= 0:
        return []

    if len(nums) < k:
        return nums

    def partition(l, r):
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

    while True:
        p = partition(l, r)
        if p + 1 == k:
            return nums[:k]
        if p + 1 < k:
            l = p + 1
        else:
            r = p - 1


# 也可以计数排序
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


# 179 输入: [3, 30, 34, 5, 9] 输出: 9534330
def largest_number(nums):
    if not nums:
        return ''

    def bigger(s1, s2):
        return s1 + s2 > s2 + s1

    def partition(l, r):
        pivot = nums[l]
        while l < r:
            while l < r and bigger(pivot, nums[r]):
                r -= 1
            nums[l] = nums[r]

            while l < r and not bigger(pivot, nums[l]):  # 此处需要not
                l += 1
            nums[l], nums[r] = nums[r], nums[l]
        nums[l] = pivot
        return l

    def quick_sort(l, r):
        if l < r:
            pivot = partition(l, r)
            quick_sort(l, pivot - 1)
            quick_sort(pivot + 1, r)

    nums = [str(num) for num in nums]
    quick_sort(0, len(nums) - 1)
    s = ''.join(nums)
    return '0' if s[0] == '0' else s


if __name__ == '__main__':
    print('\n快排')
    nums = [4, 3, 1, 3, 9]
    # nums = [1, 1, 1, 1, 1]
    quick_sort1(nums, 0, 4)
    print(nums)

    print('\n三路partition')
    nums = [4, 3, 1, 3, 9]
    quick_sort2(nums, 0, 4)
    print(nums)

    print('\n奇偶分离')
    print(sort_array_by_parity(nums))

    print('\n无序数组的前K个数')
    print(get_top_k([10, 9, 8, 9, 1, 2, 0], 5))

    print('\n颜色排序')
    print(sort_colors([2, 0, 2, 1, 1, 0]))
