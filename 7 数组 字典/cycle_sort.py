#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 交换 访问标记正负 双指针


# 0-n-1之间的n个数
# 可能有多个数字重复
# 返回任意重复的数字
def find_duplicate_num1(nums):
    n = len(nums)
    for i in range(n):
        while i != nums[i]:  # 循环后保证i = nums[i] 如果不存在i == nums[i] 一定有一个数字是重复
            if nums[i] == nums[nums[i]]:
                return nums[i]
            else:
                tmp = nums[i]
                nums[i] = nums[tmp]
                nums[tmp] = tmp


# 使用负数标记已经访问的数
def find_duplicate_num2(nums):
    n = len(nums)
    i = 0
    for v in nums:
        if not v:
            i += 1
            if i > 1:
                return 0

    for i in range(n):
        v = abs(nums[i])
        if nums[v] > 0:
            nums[v] = -nums[v]
        elif nums[v] < 0:
            return v


# 数组中的重复数字
# 1 - n 的 n + 1 个数中 只有一个数重复一次或多次
# 要求 O(1)空间复杂度!!!
def find_duplicate_num3(nums):
    fast = slow = nums[0]
    # 证明有环 快慢两个指针
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if fast == slow:
            break
    # 入口
    ptr1 = nums[0]
    ptr2 = fast
    while ptr1 != ptr2:
        ptr1 = nums[ptr1]
        ptr2 = nums[ptr2]
    return ptr1


# 给你一个未排序的整数数组，请你找出其中没有出现的最小的正整数。
def first_missing_positive(nums):
    if not nums:
        return 1
    n = len(nums)
    min_p = n + 1
    for i in range(n):
        if nums[i] > 0:
            if nums[i] < min_p:
                min_p = nums[i]
            if nums[i] > n:
                nums[i] = -1

    if min_p > 1:
        return 1

    for i in range(n):
        if nums[i] == i + 1:
            continue
        while nums[i] > 0:
            j = nums[i] - 1
            if nums[j] == j + 1:
                break
            nums[i], nums[j] = nums[j], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


# 相当于使用idx做key的hash table
# Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array),
# some elements appear twice and others appear once.
# Find all the elements of [1, n] inclusive that do not appear in this array.
# Could you do it without extra space and in O(n) runtime?
# You may assume the returned list does not count as extra space.
def find_disappeared_numbers(nums):
    for i in range(len(nums)):
        v = abs(nums[i]) - 1
        if nums[v] > 0:
            nums[v] = -nums[v]
    res = []
    for i in range(len(nums)):
        if nums[i] > 0:
            res.append(i + 1)
    return res


# 1-n之间的n个数 有些出现1次 有些出现2次 出现2次的数字
def find_duplicates(nums):
    res = []
    for i in range(len(nums)):
        v = abs(nums[i]) - 1
        if nums[v] > 0:  # 第一次访问变成负数
            nums[v] = -nums[v]
        else:  # 变成负数 说明该数重复了两次
            res.append(v + 1)
    return res


# 26 原地删除升序数组中的重复数字 并返回非重复数组的长度
# Given nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
# Your function should return length = 5
def remove_duplicates(nums):
    n = len(nums)
    i = 0  # 第一个指针
    for j in range(1, n):  # 第二个指针
        if nums[j - 1] != nums[j]:
            nums[i + 1] = nums[j]
            i += 1
    return i + 1


# 不改变顺序 把0移到数组尾部
def move_zeros(nums):
    if not nums:
        return
    j = 0
    for i in range(len(nums)):
        if not nums[i]:
            continue
        nums[j] = nums[i]
        j += 1
    for i in range(j, len(nums)):
        nums[i] = 0

    return nums


def first_missing_positive(nums):
    if not nums:
        return 1
    min_p = float('inf')
    max_p = -float('inf')
    num = 0
    for v in nums:
        if v > 0:
            num += 1
            if v < min_p:
                min_p = v
            if v > max_p:
                max_p = v
    if not num or min_p > 1:
        return 1
    print(min_p, max_p, num)

    n = len(nums)
    for i in range(n):
        if nums[i] > n or nums[i] <= 0:
            nums[i] = -1

    for i in range(n):
        if nums[i] == i + 1:
            continue
        while nums[i] > 0:
            j = nums[i] - 1
            if nums[j] == j + 1:
                break
            nums[i], nums[j] = nums[j], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1


if __name__ == '__main__':
    print('\n找到数组中重复元素')
    print(find_duplicate_num1([1, 2, 3, 4, 2]))
    print(find_duplicate_num2([1, 2, 3, 2, 0]))

    print('\n 找到缺失的最小正数')
    print(first_missing_positive([3, 4, -1, 1]))

    print('\n删除排查数组中的重复数值')
    print(remove_duplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))

    print('\n移动0')
    print(move_zeros([0, -7, 0, 2, 3, 11]))

    print('\n缺失的最小正数')
    print(first_missing_positive([1, 3, 3]))
