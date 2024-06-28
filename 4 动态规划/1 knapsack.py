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
    rows, cols = len(costs) + 1, capacity + 1  # 行列加1
    dp = [[0] * cols for _ in range(rows)]  # 作为base 初始状态都为0
    for i in range(1, rows):  # 更新所有状态
        cost = costs[i - 1]
        value = values[i - 1]
        for j in range(1, cols):  # 所有状态
            dp[i][j] = dp[i - 1][j]  # 初始化为相同负载不装该物品
            if j - cost >= 0:  # 状态转移2 这里从i-1开始
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cost] + value)
    return dp[-1][-1]


# 无穷背包1 借鉴0-1背包
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
            k = 1
            while cost * k <= j:  # 从i - 1开始, 此处速度慢
                dp[i][j] = max(dp[i][j], dp[i - 1][j - cost * k] + value * k)
                k += 1
    return dp[-1][-1]


# 无穷背包2
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


# 无穷背包3 状态压缩 使用1维数组表示
def unbounded_knapsack3(costs, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(1, capacity + 1):
        for j, cost in enumerate(costs):
            if i >= cost:
                dp[i] = max(dp[i], dp[i - cost] + values[j])
    return dp[-1]


# 约束背包
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
            k = 1
            while k <= nums[i - 1] and cost * k <= j:  # 从i - 1开始
                dp[i][j] = max(dp[i][j], dp[i - 1][j - cost * k] + value * k)
                k += 1
    return dp[-1][-1]


# 分组背包
# 每一组至多选择一个物品(也可以不选)，每个物品有体积和价值，
# 容里为M的背包，用这个背包装物品，使得物品价值总和最大
def mckp1(costs, values, capacity):
    rows, cols = len(costs) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):  # 分组
        for j in range(1, cols):  # 容量
            dp[i][j] = dp[i - 1][j]
            for k, cost in enumerate(costs[i - 1]):
                if j >= cost:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - cost] + values[i - 1][k])
    return dp[-1][-1]


def mckp2(costs, weights, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(costs)):
        print(dp)
        for k in range(1, 1 + capacity):
            for j in range(len(costs[i])):
                if k >= costs[i][j]:
                    dp[k] = max(dp[k], dp[k - costs[i][j]] + weights[i][j])
    return dp[-1]


def group_knapsack(groups, capacity):
    # groups 是一个二维列表，其中每个子列表代表一个组，子列表中的元素是 (volume, value) 对
    # capacity 是背包的容量

    # dp[i] 表示容量为 i 的背包能装下的最大价值
    dp = [0] * (capacity + 1)

    # 遍历每个组
    for group in groups:
        # 临时数组用于记录当前组选择不同物品时的最大价值
        temp_dp = dp.copy()

        # 遍历当前组的每个物品
        for value, volume in group:
            # 逆序遍历容量，因为每个物品只能使用一次
            for i in range(capacity, volume - 1, -1):
                # 更新当前容量下的最大价值
                temp_dp[i] = max(temp_dp[i], temp_dp[i - volume] + value)

                # 更新 dp 数组为当前组的最大价值
        dp = temp_dp

        # dp[-1] 就是容量为 capacity 的背包能装下的最大价值
    return dp[-1]


# 416. Partition Equal Subset Sum
# 0-1背包, 背包的容量为整体重量的1/2
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
    values = [[6, 5, 11], [6, 5, 12]]
    costs = [[2, 1, 4], [3, 1, 5]]

    capacity = 5
    print(mckp1(costs, values, capacity))
    print(mckp2(costs, values, capacity))

    groups = [[(6, 2), (5, 1), (11, 4)],
              [(6, 3), (5, 1), (12, 5)]]
    print(group_knapsack(groups, capacity))

    print('\n找到子序列和相等的两个分区')
    print(can_partition([3, 1, 5, 9, 12]))
