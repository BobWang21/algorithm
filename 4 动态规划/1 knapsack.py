#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 动态规划 matrix[i][j] 表示前i件物品恰放入一个容量为j的背包可以获得的最大价值
'''
问题类型
- 最大价值/金额
- 最小组合数
- 多少种组合
- 能否组成金额
变量类型
- 0-1背包
- 无穷背包
约束
- 分组背包
'''


# 0-1背包 二维数组
def knapsack1(values, weights, capacity):
    rows, cols = len(weights) + 1, capacity + 1  # 行列加1
    dp = [[0] * cols for _ in range(rows)]  # 作为base 初始状态都为0
    for i in range(1, rows):  # 更新所有状态
        value, weight = values[i - 1], weights[i - 1]
        for j in range(1, cols):  # 所有状态
            dp[i][j] = dp[i - 1][j]  # 初始化为相同负载不装该物品
            if j >= weight:  # 状态转移2 这里从i-1开始
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight] + value)
    return dp[-1][-1]


# 0-1背包 一维数组 最优解
def knapsack2(values, weights, capacity):
    dp = [0] * (capacity + 1)
    for value, weight in zip(values, weights):
        # 每个物品只能用一次 防止物品使用多次 逆序排列
        for j in range(capacity, weight - 1, -1):
            dp[j] = max(dp[j], dp[j - weight] + value)
    return dp[-1]


# 无穷背包1 借鉴0-1背包
def unbounded_knapsack1(values, weights, capacity):
    if not weights or not values or len(weights) != len(values):
        return
    rows, cols = len(weights) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        value = values[i - 1]
        weight = weights[i - 1]
        for j in range(1, cols):
            dp[i][j] = dp[i - 1][j]
            k = 1
            while weight * k <= j:  # 从i - 1开始, 此处速度慢
                dp[i][j] = max(dp[i][j], dp[i - 1][j - weight * k] + value * k)
                k += 1
    return dp[-1][-1]


# 无穷背包2
def unbounded_knapsack2(values, weights, capacity):
    rows, cols = len(weights) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        weight = weights[i - 1]
        value = values[i - 1]
        for j in range(1, cols):
            dp[i][j] = dp[i - 1][j]
            if j >= weight:  # 索引从i开始 可以重复使用物品
                dp[i][j] = max(dp[i][j], dp[i][j - weight] + value)
    return dp[-1][-1]


# 无穷背包3 最优解 状态压缩 使用1维数组表示
def unbounded_knapsack3(values, weights, capacity):
    dp = [0] * (capacity + 1)
    n = len(weights)
    for i in range(n):
        weight, value = weights[i], values[i]
        for j in range(weight, capacity + 1):
            dp[j] = max(dp[j], dp[j - weight] + value)
    return dp[-1]


# 约束背包
def bounded_knapsack(values, weights, capacity):
    if not weights or not values or len(weights) != len(values):
        return
    rows, cols = len(weights) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        value = values[i - 1]
        weight = weights[i - 1]
        for j in range(1, cols):
            dp[i][j] = dp[i - 1][j]
            k = 1
            while k <= values[i - 1] and weight * k <= j:  # 从i - 1开始
                dp[i][j] = max(dp[i][j], dp[i - 1][j - weight * k] + value * k)
                k += 1
    return dp[-1][-1]


# 分组背包
# 每一组至多选择一个物品(也可以不选)，每个物品有体积和价值，
# 容里为M的背包，用这个背包装物品，使得物品价值总和最大
def mckp1(groups, capacity):
    rows, cols = len(groups) + 1, capacity + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):  # 分组
        for j in range(1, cols):  # 容量
            dp[i][j] = dp[i - 1][j]
            for val, vol in groups[i - 1]:
                if j >= vol:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - vol] + val)
    return dp[-1][-1]


'''
错误写法!
def mckp2(volumes, weights, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(volumes)):
        print(dp)
        for k in range(1, 1 + capacity):
            for j in range(len(volumes[i])):
                if k >= volumes[i][j]: # 一个分组中的物品可能被用多次
                    dp[k] = max(dp[k], dp[k - volumes[i][j]] + weights[i][j])
    return dp[-1]
'''


def mckp2(groups, capacity):
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
            for i in range(volume, capacity + 1):
                # 每个物品依赖于上一个分组结果 不依赖本组结果
                temp_dp[i] = max(dp[i], dp[i - volume] + value)

            # 更新 dp 数组为当前组的最大价值
        dp = temp_dp

        # dp[-1] 就是容量为 capacity 的背包能装下的最大价值
    print(dp)
    return dp[-1]


# 最优版本分组背包
def mckp3(groups, capacity):
    # groups 是一个二维列表，其中每个子列表代表一个组，子列表中的元素是 (volume, value) 对
    # capacity 是背包的容量
    # dp[i] 表示容量为 i 的背包能装下的最大价值
    dp = [0] * (capacity + 1)

    # 先遍历每个物品组，保证分组只使用一次
    for group in groups:
        # 逆序遍历 保证每个人容量引用上一分组的状态
        for i in range(capacity, -1, -1):
            for value, volume in group:
                if i >= volume:
                    dp[i] = max(dp[i], dp[i - volume] + value)
            # 更新 dp 数组为当前组的最大价值
    return dp[-1]


# 322 换硬币 最少硬币数表示金额 如果不能返回-1
# 复杂度高 cutting stock problem
def coin_change1(coins, amount):
    dp = [float('inf')] * (amount + 1)  # 初始状态
    dp[0] = 0  # 初始状态 coin=amount时使用
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1


# 518 多少种硬币换法 硬币可以使用多次
# 无穷背包问题的次数
def coin_change2(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:  # 保证coin之前没用过!
        for j in range(coin, amount + 1):
            dp[j] += dp[j - coin]
    return dp[-1]


# 279. Perfect Squares 12 = 4 + 4 + 4
def num_squares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # coin=amount时使用

    for coin in range(1, int(n ** 0.5) + 1):
        square = coin ** 2
        for i in range(square, n + 1):
            dp[i] = min(dp[i], dp[i - square] + 1)
    return dp[n]


# 494 给你一个非负整数数组 nums 和一个整数 target 。
# 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个表达式
# 0-1背包问题的次数
def find_target_sum_ways(nums, target):
    total = sum(nums)
    diff = total - target
    if diff % 2 != 0 or diff < 0:
        return 0

    capacity = (total - target) // 2
    dp = [0] * (capacity + 1)
    dp[0] = 1
    for num in nums:
        # 逆序遍历 每个数字只能使用一遍
        for j in range(capacity, num - 1, -1):
            dp[j] += dp[j - num]
    return dp[-1]


# 416. Partition Equal Subset Sum
# 0-1背包, 背包的容量为整体重量的1/2
# 集合的子集能组成固定数
def can_partition(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    total = sum(nums)
    if total % 2 == 1:
        return False
    capacity = total // 2
    values = []
    for v in nums:
        if v < capacity:
            values.append(v)
        elif v == capacity:
            return True
        else:
            return False

    dp = [0] * (capacity + 1)
    for value in values:
        for j in range(capacity, value - 1, -1):
            dp[j] = max(dp[j], dp[j - value] + value)
    # print(dp)
    return dp[-1] == capacity


# 先遍历金额 会重复计数
# def coin_change3(coins, amount):
#     if amount < 0:
#         return 0
#     dp = [0] * (amount + 1)
#     dp[0] = 1
#
#     for v in range(amount + 1):
#         for coin in coins:
#             if v >= coin:
#                 dp[v] += dp[v - coin]
#     return dp[-1]


if __name__ == '__main__':
    print('\n0-1背包问题')
    values = [1, 3, 4, 8]
    weights = [1, 2, 3, 4]
    print(knapsack1(values, weights, 7))
    print(knapsack2(values, weights, 7))

    print('\n完全背包问题')
    weights = [5, 10, 15]
    values = [10, 30, 20]
    capacity = 100
    print(unbounded_knapsack1(values, weights, capacity))
    print(unbounded_knapsack2(values, weights, capacity))
    print(unbounded_knapsack3(values, weights, capacity))

    print('\n分组背包问题')
    groups = [[(6, 2), (5, 1), (11, 4)],
              [(6, 3), (5, 1), (12, 5)],
              [(6, 3), (5, 1), (20, 1)]]
    capacity = 5
    print(mckp1(groups, capacity))
    print(mckp2(groups, capacity))
    print(mckp3(groups, capacity))

    print('\n换硬币')
    print(coin_change1([1, 2, 5, 10], 11))
    print(coin_change2([1, 2, 5], 5))
    print('\n找到子序列和相等的两个分区')
    print(can_partition([3, 1, 5, 9, 12]))
