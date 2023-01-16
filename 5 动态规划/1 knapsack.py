#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 动态规划 matrix[i][j] 表示前i件物品恰放入一个容量为j的背包可以获得的最大价值
# 一维背包
# 约束背包
## 0-1背包
## 无穷背包
# 无约束背包

# 0-1背包
def knapsack(costs, values, capacity):
    rows, cols = len(costs) + 1, capacity + 1  # 行列加1 作为base 初始状态都为0
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):  # 所有状态
        cost = costs[i - 1]
        value = values[i - 1]
        for j in range(1, cols):  # 所有状态
            dp[i][j] = dp[i - 1][j]  # 状态转移1 不装物品i
            if j - cost >= 0:  # 状态转移2 这里从i-1开始
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cost] + value)
    return dp[-1][-1]


# 完全背包问题 借鉴0-1背包
def unbounded_knapsack1(costs, val, capacity):
    if not costs or not val or len(costs) != len(val):
        return
    rows, cols = len(costs) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost = costs[i - 1]
        value = val[i - 1]
        for j in range(1, cols):
            dp[i][j] = dp[i - 1][j]
            k = j // cost  # 最多可以装的件数
            dp[i][j] = max(dp[i][j], dp[i - 1][j - cost * k] + value * k)

    return dp[-1][-1]


# 无穷背包问题
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


# 约束背包问题
def bounded_knapsack(costs, val, nums, capacity):
    if not costs or not val or len(costs) != len(val):
        return
    rows, cols = len(costs) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost = costs[i - 1]
        value = val[i - 1]
        for j in range(1, cols):
            dp[i][j] = dp[i - 1][j]
            k = min(j // cost, nums[i - 1])  # 物品件数约束
            dp[i][j] = max(dp[i][j], dp[i - 1][j - cost * k] + value * k)

    return dp[-1][-1]


# 分组背包问题
# 每一组你至多选择一个物品(也可以不选) , 每个物品都有自己的体积和价值，
# 现在给你一个容里为M的背包，让你用这个背包装物品，使得物品价值总和最大
def mckp1(costs, weights, capacity):
    rows, cols = len(costs) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):  # 容量
        for j in range(1, cols):  # 分组
            dp[i][j] = dp[i - 1][j]
            for k, cost in enumerate(costs[i - 1]):
                if j >= cost:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - cost] + weights[i - 1][k])
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
