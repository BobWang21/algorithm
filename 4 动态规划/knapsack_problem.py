#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:32:29 2017

@author: wangbao
"""


# 0-1 背包问题 回溯法
def knapsack_backtrack(cost, val, cap):
    def dfs(cap, idx, amount, res):
        for i in range(idx, len(cost)):
            if cap - cost[i] < 0:  # base 1 不能再装了
                res[0] = max(res[0], amount)
                continue
            elif cap - cost[i] == 0:  # base 2
                res[0] = max(res[0], amount + val[i])
                continue
            else:
                dfs(cap - cost[i], i + 1, amount + val[i], res)

    res = [-1]
    dfs(cap, 0, 0, res)
    return res[0]


# 动态规划 matrix[i][j] 表示前 i 件物品恰放入一个容量为 j 的背包可以获得的最大价值
def knapsack(cost, val, cap):
    n = len(cost)
    matrix = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        c = cost[i - 1]
        v = val[i - 1]
        for j in range(1, cap + 1):
            matrix[i][j] = matrix[i - 1][j]  # 不装物品i
            if j - c >= 0:  # # res[i - 1][j-cost_i]从 i-1, 如果从i开始 我们不知道是否之前用到过物品i
                matrix[i][j] = max(matrix[i - 1][j], matrix[i - 1][j - c] + v)
    return matrix[-1][-1]


# 完全背包问题 借鉴0-1背包 这种方式可以解决有件数约束的背包问题
def unbounded_knapsack1(cost, val, capacity):
    if not cost or not val or len(cost) != len(val):
        return
    n = len(cost)
    res = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        c = cost[i - 1]
        v = val[i - 1]
        for j in range(1, capacity + 1):
            res[i][j] = res[i - 1][j]
            if j >= c:
                k = 1
                while c * k <= j:
                    res[i][j] = max(res[i][j], res[i - 1][j - c * k] + v * k)
                    k += 1

    return res[-1][-1]


def unbounded_knapsack2(cost, val, capacity):
    if not cost or not val or len(cost) != len(val):
        return
    n = len(cost)
    res = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        c = cost[i - 1]
        v = val[i - 1]
        for j in range(1, capacity + 1):
            res[i][j] = res[i - 1][j]
            if j >= c:  # res[i][j] 索引从i开始 可以重复使用物品
                res[i][j] = max(res[i][j], res[i][j - c] + v)
    return res[-1][-1]


# 使用1维数组 换硬币
def unbounded_knapsack3(cost, val, capacity):
    if not cost or not val or len(cost) != len(val):
        return
    n = len(cost)
    res = [0] * (capacity + 1)
    for i in range(1, capacity + 1):  # bottom up
        for j in range(n):
            c, v = cost[j], val[j]
            if i >= c:
                res[i] = max(res[i], res[i - c] + v)

    return res[-1]


# 416. Partition Equal Subset Sum
# Partition problem is to determine whether
# a given set can be partitioned into two subsets
# such that the sum of elements in both subsets is same.
# 0-1背包问题, 背包的容量为整体重量的1/2,
# 物品的重量等于物品的价值 包的重量为mean
# 然后看可以装的物品的最大值是否等于mean
def can_partition(nums):
    mean = sum(nums) / 2
    if mean % 2:
        return False
    mean = int(mean)
    vals = []
    for val in nums:
        if val < mean:
            vals.append(val)
        if val == mean:
            return True
    res = [[0] * (mean + 1) for _ in range(len(vals) + 1)]
    for i in range(1, len(vals) + 1):
        v = vals[i - 1]
        for j in range(1, mean + 1):
            res[i][j] = res[i - 1][j]
            if j >= v:
                res[i][j] = max(res[i][j], res[i - 1][j - v] + v)
    return res[-1][-1] == mean


if __name__ == '__main__':
    print('0-1 背包问题')
    print(knapsack([1, 2, 3, 4], [1, 3, 4, 8], 7))

    print('\n完全背包问题')
    print(unbounded_knapsack1([5, 10, 15], [10, 30, 20], 100))
    print(unbounded_knapsack3([5, 10, 15], [10, 30, 20], 100))

    print('\n找到子序列和相等的两个分区')
    print(can_partition([3, 1, 5, 9, 12]))
