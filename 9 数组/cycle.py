#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 循环交换 访问标记正负 双指针


# 0 ~ n-1之间的n个数 可能有多个数字重复 返回任意重复的数字
def find_duplicate_num1(nums):
    n = len(nums)
    for i in range(n):
        while i != nums[i]:
            j = nums[i]
            if j == nums[j]:  # 如果存在j == nums[j] 说明数字j重复了
                return j
            else:
                nums[i], nums[j] = nums[j], nums[i]


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


# 给你一个未排序的整数数组，请你找出其中没有出现的最小的正整数。
def first_missing_positive(nums):
    if not nums:
        return 1
    n = len(nums)
    min_p = n + 1  # 最大的数为n
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
            if nums[j] == j + 1:  # 防止循环 [1, -1, 1, 2]
                break
            nums[i], nums[j] = nums[j], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


# 相当于使用idx做key的hash table
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
# nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
def remove_duplicates(nums):
    if not nums:
        return 0
    n = len(nums)
    i = 1  # 第一个指针 下一个不重复数字放置的位置 类似 三指针的lt
    for j in range(1, n):  # 第二个指针
        if nums[j - 1] != nums[j]:
            nums[i] = nums[j]
            i += 1
    return i


# 不改变顺序 把0移到数组尾部
def move_zeros(nums):
    if not nums:
        return
    k, n = 0, len(nums)
    for i in range(n):
        if nums[i]:
            nums[k] = nums[i]
            k += 1
    for i in range(k, n):
        nums[i] = 0
    return nums


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
    print('\n找到数组中重复元素')
    print(find_duplicate_num1([1, 2, 3, 4, 2]))
    print(find_duplicate_num2([1, 2, 3, 2, 0]))

    print('\n 找到缺失的最小正数')
    print(first_missing_positive([3, 4, -1, 1]))

    print('\n缺失的最小正数')
    print(first_missing_positive([1, 3, 3]))

    print('\n删除排查数组中的重复数值')
    print(remove_duplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))

    print('\n移动 0至数组尾部')
    print(move_zeros([0, -7, 0, 2, 3, 11]))
