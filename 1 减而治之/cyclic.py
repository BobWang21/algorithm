#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 循环排序: 适合于数值区间在一定范围内的数组
# 访问标记正负 适合于正数
# 双指针


def cyclic_sort(nums):
    if not nums:
        return []
    n = len(nums)
    for i in range(n):
        while nums[i] != i + 1:
            j = nums[i] - 1
            if nums[j] == j + 1:  # 说明已经存在
                break
            nums[i], nums[j] = nums[j], nums[i]
    return nums


# 0 ~ n-1之间的n个数 可能有多个数字重复 返回任意重复的数字
def find_duplicate_num1(nums):
    n = len(nums)
    for i in range(n):
        while i != nums[i]:
            j = nums[i]
            if j == nums[j]:  # 如果存在j == nums[j] 说明数字j重复了
                return j
            nums[i], nums[j] = nums[j], nums[i]


# 给你一个未排序的整数数组，请你找出其中没有出现的最小的正整数。
def first_missing_positive(nums):
    if not nums:
        return 1
    n = len(nums)

    for i in range(n):
        while 0 < nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            j = nums[i] - 1
            nums[i], nums[j] = nums[j], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1  # 如果数组是[1, 2, 3] !!!


# 使用负数标记已存在的数
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


# 使用负数标记已经访问的数
def find_duplicate_num2(nums):
    n = len(nums)
    i = 0
    for v in nums:  # 判断是否0位重复数字
        if not v:
            i += 1
            if i > 1:
                return 0

    for i in range(n):
        j = abs(nums[i])
        if nums[j] > 0:
            nums[j] = -nums[j]
        elif nums[j] < 0:
            return j


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


def rotate(nums, k):
    n = len(nums)
    k %= n
    if not k:
        return
    curr = 0
    tmp = nums[curr]
    cnt = 0
    while cnt < n:
        nxt = (curr + k) % n
        while nxt != curr:
            nums[nxt], tmp = tmp, nums[nxt]
            nxt = (nxt + k) % n
            cnt += 1
        nums[nxt] = tmp
        curr += 1
        tmp = nums[curr]
        cnt += 1


if __name__ == '__main__':
    print('\ncyclic sort')
    print(cyclic_sort([7, 5, 8, 1, 2, 9, 3, 4, 6, 10]))

    print('\n找到缺失的最小正数')
    print(first_missing_positive([3, 4, -1, 1]))

    print('\n找到数组中重复元素')
    print(find_duplicate_num1([1, 2, 3, 4, 2]))
    print(find_duplicate_num2([1, 2, 3, 2, 0]))
