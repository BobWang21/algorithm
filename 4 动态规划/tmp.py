#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def fib(n):
    result = [0, 1]
    if n <= 2:
        return result
    left, right = result
    for i in range(2, n):
        left, right = right, left + right
        result.append(right)
    return result


# top down
def fib2(n):
    dic = {0: 0, 1: 1}

    def helper(n):
        if n in dic:
            return dic[n]
        res = helper(n - 1) + helper(n - 2)
        dic.setdefault(n, res)
        return res

    return helper(n)


# 股票买卖 最佳时机
def max_profit1(nums):
    n = len(nums)
    if n < 2:
        raise Exception()
    buy = 0
    balance = 0
    for i in range(1, n):
        if nums[i] - nums[buy] > balance:
            balance = nums[i] - nums[buy]
            continue
        if nums[i] < nums[buy]:
            buy = i
    return balance


# 股票买卖 含冷冻期
def max_profit2(prices):
    n = len(prices)
    if n < 2:
        return 0
    hold, sold, rest = -float('inf'), 0, 0  # 因为不能出现hold 所以收益为负无穷
    for price in prices:
        hold, sold, rest = max(rest - price, hold), hold + price, max(rest, sold)

    return max(sold, rest)


# 最多交易两次
def max_profit3(prices):
    if not prices or len(prices) < 2:
        return 0
    n = len(prices)
    k = 2
    dp = [[[0, 0] for _ in range(k + 1)] for _ in range(n + 1)]

    # 处理base: -1 及 不可能状态的赋值
    for i in range(n + 1):  # 每天交易次数为0时
        dp[i][0][1] = -float('inf')
        dp[i][0][0] = 0

    for j in range(k + 1):  # 第0天
        dp[0][j][0] = 0
        dp[0][j][1] = -float('inf')

    for i in range(1, n + 1):
        for j in range(k, 0, -1):  # 倒序更新
            dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i - 1])
            dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i - 1])  # j-1

    return dp[-1][-1][0]


# 连续子序列 和最大
def max_sub_array(nums):
    res = include_max = nums[0]
    for val in nums[1:]:
        include_max = max(val, include_max + val)
        res = max(res, include_max)
    return res


# 连续子序列 乘积最大
def max_continuous_product(nums):
    res = include_min = include_max = nums[0]
    for val in nums[1:]:
        include_min, include_max = min(val, val * include_max, val * include_min), \
                                   max(val, val * include_max,
                                       val * include_min)
        res = max(include_max, res)
    return res


# dp[i] = max(dp[i-1], dp[i-2] + nums[i])
# 全为正数
def rob1(nums):
    if not nums:
        return 0
    n = len(nums)
    if n < 3:
        return max(nums)
    res = [0] * n
    res[0] = nums[0]
    res[1] = max(nums[:2])
    for i in range(2, n):
        res[i] = max(res[i - 2] + nums[i], res[i - 1])
    return res[-1]


# 滚动数组
def rob2(nums):
    include = exclude = 0
    for val in nums:
        include, exclude = max(val, exclude + val), max(exclude, include)
    return max(include, exclude)


# 第一个和最后一个连成环 不能同时选
def rob3(nums):
    if not nums:
        return 0
    n = len(nums)
    if n < 3:
        return max(nums)

    def helper(nums, lo, hi):
        include, exclude = 0, 0
        for i in range(lo, hi + 1):
            new_include = exclude + nums[i]
            exclude = max(include, exclude)
            include = new_include
        return max(include, exclude)

    return max(helper(nums, 0, n - 2), helper(nums, 1, n - 1))


# 最长上升子序列
def longest_increasing_subsequence(nums):
    dp = [1] * len(nums)
    res = -1
    for i, val in enumerate(nums):
        for j in range(i):
            if nums[j] < val:
                dp[i] = max(dp[j] + 1, dp[i])
        res = max(res, dp[i])
    return res


# res[i] 保存长度为i+1的子串的最小值 nlog(n)
def longest_increasing_subsequence2(nums):
    def binary_search(nums, tar):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == tar:
                return mid
            elif nums[mid] > tar:
                right -= 1
            else:
                left += 1
        return left

    if not nums:
        return 0
    res = [nums[0]]  # nums[i]表示长度为i+1上升子串结尾的最小值
    for v in nums[1:]:
        if v > res[-1]:
            res.append(v)
        else:
            idx = binary_search(res, v)
            res[idx] = v
    return len(res)


# 换硬币
# 您会得到不同面额的硬币和总金额。
# 编写一个函数来计算组成该数量所需的最少数量的硬币。
# 如果这笔钱不能用硬币的任何组合来弥补，请返回-1。
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 当coin = amount 时使用
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change2(coins, amount):
    if amount < 0:
        return 0
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:  # 保证之前没有重复的coin组合
        for amount in range(coin, amount + 1):
            dp[amount] += dp[amount - coin]
    return dp[-1]


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


# 给定一个正整数数组 求和为target的所有组合数
def combination_sum(coins, target):
    '''
    NUMS = [1，2，3]
    目标 = 4种的一种可能的组合方式有：
    （1，1，1，1）
    （1，1, 2）
    （1，2, 1）
    （1, 3）
    （2, 1，1）
    （2，2）
    （3，1）
    '''
    res = [0] * (target + 1)
    res[0] = 1
    for i in range(target + 1):  # 包含重复组合
        for v in coins:
            if i >= v:
                res[i] += res[i - v]
    return res[-1]


# 279. Perfect Squares 12 = 4 + 4 + 4
def num_squares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # 当coin = amount 时使用
    for i in range(1, n + 1):
        for coin in range(1, int(n ** 0.5) + 1):
            square = coin ** 2
            if i >= square:
                dp[i] = min(dp[i], dp[i - square] + 1)
    return dp[n]


# 割绳子
def max_product_after_cutting(m):
    if m == 1:
        return
    if m == 2:
        return 1
    if m == 3:
        return 2
    res = [0, 1, 2, 3]  # 0为了占位
    for i in range(4, m + 1):
        max_num = -1
        for j in range(1, i // 2 + 1):
            max_num = max(max_num, res[j] * res[i - j])
        res.append(max_num)
    return res[-1]


# 从格子中选出礼物的最大值
def max_gift(matrix):
    if not matrix:
        return
    row, col = len(matrix) + 1, len(matrix[0]) + 1
    res = [[0] * col for _ in range(row)]
    for i in range(1, row):
        for j in range(1, col):
            res[i][j] = max(res[i][j - 1], res[i - 1][j]) + matrix[i - 1][j - 1]
    return res[-1][-1]


if __name__ == '__main__':
    print('\nFibonacci sequence')
    print(fib(40)[39])
    print(fib2(39))

    print('\n股票买卖 ')
    print('最佳盈利')
    print(max_profit1([5, 10, 15, 1, 20]))

    print('包含冷冻期')
    print(max_profit2([1, 2, 3, 0, 2]))

    print('有交易次数限制')
    print(max_profit3([3, 3, 5, 0, 0, 3, 1, 4]))

    print('\n连续子序列和最大')
    print(max_sub_array([10, -5, 10]))

    print('\n连续子序列乘积最大')
    print(max_continuous_product([-1, 2, 3, 0.1, -10]))

    print('\n抢钱1')
    print(rob2([5, 3, -6, -5, 10]))
    print(rob1([5, 3, -6, -5, 10]))

    print('\n抢钱2')
    print(rob3([1, 2, 1, 1]))

    print('\n最长上升子序列')
    print(longest_increasing_subsequence([2, 5, 3, 4, 1, 7, 6]))

    print('\n换硬币')
    print(coin_change([1, 2, 5, 10], 11))
    print(coin_change2([1, 2, 5], 5))
    print(combination_sum([1, 2, 5], 5))

    print('\n数字由平方组合')
    print(num_squares(13))

    print('\n割绳子')
    print(max_product_after_cutting(8))

    print('\n最大礼物')
    matrix = [[1, 10, 3, 8],
              [12, 2, 9, 6],
              [5, 7, 4, 11],
              [3, 7, 16, 5]]
    print(max_gift(matrix))
