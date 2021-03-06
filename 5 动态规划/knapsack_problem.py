#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 动态规划 matrix[i][j] 表示前 i 件物品恰放入一个容量为 j 的背包可以获得的最大价值


# 0-1背包
def knapsack(costs, values, capacity):
    rows, cols = len(costs) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost = costs[i - 1]
        value = values[i - 1]
        for j in range(1, cols):
            dp[i][j] = dp[i - 1][j]  # 不装物品i
            if j - cost >= 0:  # # res[i - 1][j-cost_i]从 i-1, 如果从i开始 我们不知道是否之前用到过物品i
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cost] + value)
    return dp[-1][-1]


# 完全背包问题 借鉴0-1背包 这种方式可以解决有件数约束的背包问题
def unbounded_knapsack1(costs, val, capacity):
    if not costs or not val or len(costs) != len(val):
        return
    rows, cols = len(costs) + 1, capacity + 1
    res = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost = costs[i - 1]
        value = val[i - 1]
        for j in range(1, cols):
            res[i][j] = res[i - 1][j]
            k = 1
            while cost * k <= j:  # 从i - 1开始
                res[i][j] = max(res[i][j], res[i - 1][j - cost * k] + value * k)
                k += 1

    return res[-1][-1]


def unbounded_knapsack2(costs, values, capacity):
    rows, cols = len(costs) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost = costs[i - 1]
        value = values[i - 1]
        for j in range(1, cols):
            dp[i][j] = dp[i - 1][j]
            if j >= cost:  # 索引从i开始 可以重复使用物品
                dp[i][j] = max(dp[i][j], dp[i][j - cost] + value)
    return dp[-1][-1]


# 使用1维数组表示
def unbounded_knapsack3(costs, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(1, capacity + 1):
        for j, cost in enumerate(costs):
            if i >= cost:
                dp[i] = max(dp[i], dp[i - cost] + values[j])
    return dp[-1]


# 分组背包问题
def mckp1(costs, weights, capacity):
    rows, cols = len(costs) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):  # 分组
        for j in range(1, cols):  # 重量
            dp[i][j] = dp[i - 1][j]
            n = len(costs[i - 1])
            for k in range(n):
                if j >= costs[i - 1][k]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - costs[i - 1][k]] + weights[i - 1][k])
    return dp[-1][-1]


def mckp2(costs, weights, capacity):
    dp = [0] * (capacity + 1)

    for i in range(len(costs)):
        for cap in range(1, 1 + capacity):
            for j in range(len(costs[i])):
                if cap >= costs[i][j]:
                    dp[cap] = max(dp[cap], dp[cap - costs[i][j]] + weights[i][j])
    return dp[-1]


# 416. Partition Equal Subset Sum
# 0-1背包问题, 背包的容量为整体重量的1/2
def can_partition(nums):
    total = sum(nums)
    if total % 2:
        return False
    capacity = total // 2

    coins = []
    for val in nums:
        if val < capacity:
            coins.append(val)
        elif val == capacity:
            return True
        else:
            return False

    res = [[0] * (capacity + 1) for _ in range(len(coins) + 1)]
    for i in range(1, len(coins) + 1):
        v = coins[i - 1]
        for j in range(1, capacity + 1):
            res[i][j] = res[i - 1][j]
            if j >= v:
                res[i][j] = max(res[i][j], res[i - 1][j - v] + v)
    return res[-1][-1] == capacity


if __name__ == '__main__':
    print('\n0-1背包问题')
    print(knapsack([1, 2, 3, 4], [1, 3, 4, 8], 7))

    print('\n完全背包问题')
    print(unbounded_knapsack1([5, 10, 15], [10, 30, 20], 100))
    print(unbounded_knapsack2([5, 10, 15], [10, 30, 20], 100))
    print(unbounded_knapsack3([5, 10, 15], [10, 30, 20], 100))

    print('\n分组背包问题')
    costs = [[5, 6], [6, 5]]
    weights = [[6, 5], [6, 5]]
    capacity = 11
    print(mckp1(costs, weights, capacity))
    print(mckp2(costs, weights, capacity))

    print('\n找到子序列和相等的两个分区')
    print(can_partition([3, 1, 5, 9, 12]))
