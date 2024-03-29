#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数组重复和缺失值
- dic
- 循环排序 适合于数值区间在一定范围内的数组
- 数组可以看成index为key的特殊字典 标记正负 适合于正数
- 快慢双指针 不修改列表
- 异或操作
- 二分查找
"""


# 不包含重复数字的循环排序
def cyclic_sort(nums):
    if not nums:
        return []
    n = len(nums)
    for i in range(n):
        while nums[i] != i + 1:
            j = nums[i] - 1
            # if nums[j] == j + 1:  # 该位置已经排好
            #     break
            nums[i], nums[j] = nums[j], nums[i]
    return nums


# 0 ~ n-1之间的n个数 可能有多个数字重复 返回任意重复的数字
# 使用列表模拟字典
def find_duplicate_num1(nums):
    n = len(nums)
    for i in range(n):
        while i != nums[i]:
            j = nums[i]
            if j == nums[j]:  # 说明数字j重复了
                return j
            nums[i], nums[j] = nums[j], nums[i]


# 41 没有出现的最小的正整数, 可能含有重复数字
def first_missing_positive(nums):
    if not nums:
        return 1

    n = len(nums)
    for i in range(n):
        while 0 < nums[i] <= n and nums[i] != i + 1:
            j = nums[i] - 1
            if nums[j] == j + 1:  # nums[j] == nums[i] 防止无限循环
                break
            nums[i], nums[j] = nums[j], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1  # 如果数组是[1, 2, 3] !!!


# 448 使用负数标记 已存在的数
def find_disappeared_numbers(nums):
    for i in range(len(nums)):
        j = abs(nums[i]) - 1
        if nums[j] > 0:
            nums[j] = -nums[j]

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


# 1 - n 的 n + 1 个数中 只有一个数字重复 数字重复一次或多次
# 要求 O(1)空间复杂度!!!  不能修改列表
def find_duplicate_num3(nums):
    fast = slow = nums[0]
    # 证明有环
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


# 268 给定一个包含[0, n]中 n 个数的序列，不含重复数字
# 找出没有出现在序列中的那个数
def missing_number(nums):
    if not nums:
        return 0

    n = len(nums)

    miss_value = n
    for i in range(n):
        miss_value ^= nums[i] ^ i

    return miss_value


# 287. 寻找重复数
def find_duplicate(nums):
    n = len(nums)

    def count(target):
        return sum([1 if v <= target else 0 for v in nums])

    l, r = 0, n - 1
    while l < r:
        mid = l + (r - l) // 2
        if count(mid) <= mid:
            l = mid + 1
        else:
            r = mid
    return l


if __name__ == '__main__':
    print('\ncyclic sort')
    print(cyclic_sort([7, 5, 8, 1, 2, 9, 3, 4, 6, 10]))
    print(cyclic_sort([2, 2, 1, 3, 2]))

    print('\n找到缺失的最小正数')
    print(first_missing_positive([3, 4, -1, 1]))

    print('\n找到数组中重复元素')
    print(find_duplicate_num1([1, 2, 4, 3, 2]))
    print(find_duplicate_num2([1, 2, 3, 2, 0]))
