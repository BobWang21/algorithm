#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict


# 众数 超过一半的数
def most_data(nums):
    if not nums:
        return
    n = len(nums)
    value = nums[0]
    count = 1
    for i in range(1, n):
        if count == 0:
            value = nums[i]
            count = 1
            continue
        if nums[i] == value:
            count += 1
        else:
            count -= 1
    return value


def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    i, j = 0, cols - 1
    while i < rows and j >= 0:
        num = matrix[i][j]
        if num == target:
            return True
        elif num < target:
            i += 1
        else:
            j -= 1
    return False


# 选择排序
def select_sort(nums):
    if not nums:
        return
    n = len(nums)
    for i in range(n):
        idx = i
        for j in range(i + 1, n):
            if nums[j] < nums[idx]:
                idx = j
        nums[idx], nums[i] = nums[i], nums[idx]
    return nums


# 计数排序


# 347 出现次数最多的K个数 类似计数排序
def top_k_frequent(nums, k):
    dic = defaultdict(int)
    for v in nums:
        dic[v] += 1

    fre = defaultdict(set)
    for k, v in dic.items():  # 将出现次数相同的数字放在一个列表中 类似链表
        fre[v].add(k)

    res = []
    for i in range(len(nums), 0, -1):  # 类似降序排列
        if i in fre:
            for v in fre[i]:
                res.append(v)
                if len(res) == k:
                    return res[:k]


# 最长连续数字 最小值的连续个数
def longest_consecutive(nums):
    if not nums:
        return 0
    s = set(nums)
    res = 1
    for v in s:
        if v - 1 in s:
            continue
        i = 1
        while v + 1 in s:
            i += 1
            v += 1
        res = max(res, i)

    return res


# 区间合并
def merge(intervals):
    if not intervals or not intervals[0]:
        return []
    intervals.sort()
    res = []
    pre_end = -float('inf')
    for start, end in intervals:
        if start > pre_end:
            res.append([start, end])
            pre_end = end
        elif end <= pre_end:
            continue
        else:
            res[-1][1] = end
            pre_end = end

    return res


def next_permutation(nums):
    if not nums:
        return
    n = len(nums)

    r = n - 1
    while r > 0 and nums[r - 1] >= nums[r]:
        r -= 1
    if not r:
        nums.reverse()
        return nums
    k = r - 1
    while r < n and nums[r] > nums[k]:  # 可以二分
        r += 1
    r -= 1
    nums[k], nums[r] = nums[r], nums[k]

    # reverse
    l, r = k + 1, n - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums


def findDisappearedNumbers(nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    n = len(nums)
    for i in range(n):
        if nums[i] == i - 1:
            continue
        while nums[i] != i - 1:
            j = nums[i]
            if nums[j - 1] == j:
                break
            nums[i], nums[j - 1] = nums[j - 1], nums[i]
    res = []
    for i in range(n):
        if i != nums[i] - 1:
            res.append(i + 1)
    return res


if __name__ == '__main__':
    print('\n选择排序')
    print(select_sort([3, 1, 4, 4, 10, -1]))

    print('\n众数')
    print(most_data([1, 3, 3, 3, 9]))

    print('\n矩阵查找')
    matrix = [
        [1, 3, 5, 7],
        [10, 11, 16, 20],
        [23, 30, 34, 50]
    ]
    print(search_matrix(matrix, 16))

    print('\n出现次数最多的K个数')
    print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))

    print('\n连续区间最大长度')
    print(longest_consecutive([1, 2, 3, 7, 8, 10, 11, 12, 13, 14]))

    print('\n区间合并')
    print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))

    print('\n下一个排列')
    print(next_permutation([1, 1, 3]))

    print('\n缺失的数据')
    nums = [4, 3, 2, 7, 8, 2, 3, 1]
    print(findDisappearedNumbers(nums))
