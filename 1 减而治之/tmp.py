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
        elif nums[i] == value:
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


# 计数排序 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，
# 原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列
# 使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
def sort_colors(nums):
    if not nums:
        return []
    count = [0] * 3
    for color in nums:
        count[color] += 1
    start = 0
    for i in range(3):
        for j in range(count[i]):
            nums[start] = i
            start += 1
    return nums


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


# 无序数组 最长连续区间 也可以使用并查集
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
    while r < n and nums[r] > nums[k]:  # 找到第一个小于等于该数的位置
        r += 1
    r -= 1  # 大于该数的最小值
    nums[k], nums[r] = nums[r], nums[k]

    # reverse
    l, r = k + 1, n - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums


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

    print('\n计数排序')
    print(sort_colors([2, 0, 2, 1, 1, 0]))

    print('\n出现次数最多的K个数')
    print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))

    print('\n连续区间最大长度')
    print(longest_consecutive([1, 2, 3, 7, 8, 10, 11, 12, 13, 14]))

    print('\n区间合并')
    print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))

    print('\n下一个排列')
    print(next_permutation([1, 1, 3]))
