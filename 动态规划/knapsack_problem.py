#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:32:29 2017

@author: wangbao
"""


# 0-1 背包问题
# matrix[i][j] 表示前 i 件物品恰放入一个容量为 j 的背包可以获得的最大价值
def knapsack1(cost, val, cap):
    n = len(cost)
    matrix = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost_i = cost[i - 1]
        val_i = val[i - 1]
        for j in range(1, cap + 1):
            matrix[i][j] = matrix[i - 1][j]  # 不装物品i
            if j - cost_i >= 0:  # 判断是否装物品i
                matrix[i][j] = max(matrix[i - 1][j], matrix[i - 1][j - cost_i] + val_i)
    return matrix[-1][-1]


# 完全背包问题
def knapsack2(cost, val, cap):
    n = len(cost)
    matrix = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost_i = cost[i - 1]
        val_i = val[i - 1]
        for j in range(1, cap + 1):
            matrix[i][j] = matrix[i - 1][j]
            max_value = matrix[i][j]
            k = 1
            while j - cost_i * k >= 0:  # 尝试多次
                max_value = max(max_value, matrix[i - 1][j - cost_i * k] + val_i * k)
                k += 1
            matrix[i][j] = max_value
    print(matrix)
    return matrix[-1][-1]


# Partition problem is to determine whether
# a given set can be partitioned into two subsets
# such that the sum of elements in both subsets is same.
# 化成背包问题, 背包的容量为整体重量的1/2,
# 物品的重量等于物品的价值
def find_partition(nums):
    mean = sum(nums) / 2
    if int(mean) != mean:
        return False
    mean = int(mean)
    values = []
    for val in nums:
        if val < mean:
            values.append(val)
        if val == mean:
            return True
    arr = [[0] * (mean + 1) for _ in range(len(values) + 1)]
    for i in range(1, len(values) + 1):
        val_i = values[i - 1]
        for j in range(1, mean + 1):
            arr[i][j] = arr[i - 1][j]
            if j >= val_i:
                arr[i][j] = max(arr[i][j], arr[i - 1][j - val_i] + val_i)
    return arr[-1][-1] == mean


if __name__ == '__main__':
    print(knapsack1([1, 2, 3, 4], [1, 3, 4, 8], 7))
    print(knapsack2([1, 2, 3, 4], [1, 3, 5, 8], 5))

    # print('找到子序列和相等的两个分区')
    print(find_partition([3, 1, 5, 9, 12]))
