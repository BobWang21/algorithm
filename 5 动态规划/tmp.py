#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict


# 记忆化的 top down
def fib1(n):
    dic = {1: 1, 2: 2}

    def helper(n):
        if n in dic:
            return dic[n]
        res = helper(n - 1) + helper(n - 2)
        dic.setdefault(n, res)
        return res

    return helper(n)


# bottom up
# 可以记录所有状态 但当前状态只和最近前两个状态有关 因此只存最近两个状态
def fib2(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    left, right = 1, 2
    for i in range(3, n + 1):
        left, right = right, left + right
    return right


# dp[i] = max(dp[i-1], dp[i-2] + nums[i])
# 全为正数
def rob1(nums):
    if not nums:
        return 0
    n = len(nums)
    if n <= 2:
        return max(nums)
    res = [0] * n
    res[0] = nums[0]
    res[1] = max(nums[:2])
    for i in range(2, n):
        res[i] = max(res[i - 2] + nums[i], res[i - 1])
    return res[-1]


# 滚动数组
def rob2(nums):
    inc = exc = 0
    for num in nums:
        inc, exc = exc + num, max(exc, inc)
    return max(inc, exc)


# 第一个和最后一个连成环 不能同时选
def rob_with_cycle(nums):
    if not nums:
        return 0
    n = len(nums)
    if n <= 2:
        return max(nums)

    def helper(nums, lo, hi):
        include, exclude = 0, 0
        for i in range(lo, hi + 1):
            new_include = exclude + nums[i]
            exclude = max(include, exclude)
            include = new_include
        return max(include, exclude)

    return max(helper(nums, 0, n - 2), helper(nums, 1, n - 1))


# 连续子序列 和最大
def max_sub_array(nums):
    if not nums:
        return 0
    inc, exc = -float('inf'), -float('inf')  # 包含负数的最大值 初始值常用负无穷或
    for num in nums:
        inc, exc = max(inc + num, num), max(inc, exc)

    return max(inc, exc)


# 连续子序列 乘积最大
def max_continuous_product(nums):
    res = inc_min = inc_max = nums[0]
    for val in nums[1:]:
        inc_min, inc_max = min(val, val * inc_max, val * inc_min), max(val, val * inc_max, val * inc_min)
        res = max(inc_max, res)
    return res


# 最长上升子序列
def LIS(nums):
    dp = [1] * len(nums)
    res = 1
    for i, val in enumerate(nums):
        for j in range(i):
            if nums[j] < val:
                dp[i] = max(dp[j] + 1, dp[i])
        res = max(res, dp[i])
    return res


# res[i] 保存长度为i+1的子串的最小值 nlog(n)
def LIS2(nums):
    def binary_search(nums, target):
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid
        return l

    if not nums:
        return 0
    res = [nums[0]]  # nums[i]表示长度为i+1上升子串结尾的最小值
    for num in nums[1:]:
        if num > res[-1]:
            res.append(num)
        else:
            idx = binary_search(res, num)  # 第一个大于等于该数的位置
            res[idx] = num
    return len(res)


# 换硬币 最少硬币数 如果不能请返回-1。
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 当coin = amount 时使用
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1


# 多少种硬币换法
def coin_change2(coins, amount):
    if amount < 0:
        return 0

    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:  # 保证之前没有重复的coin组合
        for v in range(coin, amount + 1):
            dp[v] += dp[v - coin]
    return dp[-1]


# 注意这两种区别
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
        return 0
    rows, cols = len(matrix) + 1, len(matrix[0]) + 1
    res = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            res[i][j] = max(res[i][j - 1], res[i - 1][j]) + matrix[i - 1][j - 1]
    return res[-1][-1]


# 带权最小路径和
def min_path_sum(matrix):
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix) + 1, len(matrix[0]) + 1
    dp = [[0] * cols for _ in range(rows)]

    # 边界为无穷
    for i in range(rows):
        dp[i][0] = float('inf')

    for j in range(cols):
        dp[0][j] = float('inf')

    dp[0][1] = 0  # 入口

    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + matrix[i - 1][j - 1]
    return dp[-1][-1]


def minimum_total(nums):
    if not nums or not nums[0]:
        return 0

    n = len(nums)

    for i in range(1, n):
        for j in range(len(nums[i])):
            left = nums[i - 1][j - 1] if j > 0 else float('inf')
            right = nums[i - 1][j] if j < len(nums[i - 1]) else float('inf')
            nums[i][j] += min(left, right)

    return min(nums[-1])


# 最大正方形面积
def maximal_square(matrix):
    if not matrix or not matrix[0]:
        return 0
    rows, cols = len(matrix) + 1, len(matrix[0]) + 1

    dp = [[0] * (cols) for _ in range(rows)]

    res = 0
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i - 1][j - 1] == '1':
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                res = max(dp[i][j], res)
    return res * res


# 最长等差数列
def longest_arith_seq_length(nums):
    gap = max(nums) - min(nums)
    n = len(nums)
    if not gap:
        return n

    dp = [[1] * (2 * gap + 1) for _ in range(n)]  # 有部分数组用不到
    res = 1
    for i in range(n):
        for j in range(i):
            d = nums[i] - nums[j] + gap  # 可能出现负数 需要设计等差
            dp[i][d] = dp[j][d] + 1
            res = max(res, dp[i][d])
    return res


def longest_arith_seq_length2(nums):
    gap = max(nums) - min(nums)
    n = len(nums)
    if not gap:
        return n
    dp = defaultdict(int)  # 默认值为0
    res = 1
    for i in range(n):
        for j in range(i):
            d = nums[i] - nums[j]  # 可能出现负数 需要设计
            dp[(i, d)] = (dp[(j, d)] if dp[(j, d)] else 1) + 1
            res = max(res, dp[(i, d)])
    return res


# 二维矩阵中 0为空地 1为障碍物 是否可以从左上角到达右下角
# 求到达的路径数
def unique_paths_with_obstacles(grid):
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid) + 1, len(grid[0]) + 1
    dp = [[0] * cols for _ in range(rows)]
    dp[0][1] = 1  # 入口

    for i in range(1, rows):
        for j in range(1, cols):
            if not grid[i - 1][j - 1]:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]  # dp[1][1] = dp[0][1] + dp[1][0] 因此把 dp[0][1]设为1
    return dp[-1][-1]


# 股票买卖 最佳时机
def max_profit1(nums):
    n = len(nums)
    if n < 2:
        return 0
    i = 0
    balance = 0
    for j in range(1, n):
        if nums[j] - nums[i] > balance:
            balance = nums[j] - nums[i]
            continue
        if nums[j] < nums[i]:
            i = j
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


# 最大前缀 后缀
def trap1(height):
    if not height:
        return 0

    n = len(height)
    l = [0] * n
    r = [0] * n

    for i in range(n):  # 动态规划 左右
        if not i:
            l[i] = height[i]
        else:
            l[i] = max(l[i - 1], height[i])

    for i in range(n - 1, -1, -1):
        if i == n - 1:
            r[i] = height[i]
        else:
            r[i] = max(r[i + 1], height[i])

    res = 0
    for i in range(n):
        res += min(l[i], r[i]) - height[i]
    return res


# 238. 除自身以外数组的乘积 左右各扫描一遍
def product_except_self(nums):
    n = len(nums)

    l = [1] * n

    for i in range(1, n):
        l[i] = l[i - 1] * nums[i - 1]

    r = 1
    for i in range(n - 1, -1, -1):
        l[i] *= r
        r = r * nums[i]

    return l


# 鸡蛋
# 动态规划算法的时间复杂度就是子问题个数 × 函数本身的复杂度
# 函数本身的复杂度就是忽略递归部分的复杂度，这里 dp 函数中有一个 for 循环，
# 所以函数本身的复杂度是 O(N) 所以算法的总时间复杂度是 O(K*N^2)
def super_egg_drop(K, N):
    memo = dict()

    def dp(K, N):
        # base case
        if K == 1:
            return N
        if N == 0:
            return 0
        # 避免重复计算
        if (K, N) in memo:
            return memo[(K, N)]

        res = float('inf')

        # 穷举所有可能的选择
        for i in range(1, N + 1):
            res = min(max(dp(K, N - i), dp(K - 1, i - 1)) + 1, res)

        memo[(K, N)] = res
        return res

    return dp(K, N)


def super_egg_drop2(K, N):
    dic = {}

    def dp(k, n):
        if k == 1:
            return n
        if n == 0:
            return 0

        if (k, n) in dic:
            return dic[(k, n)]

        res = float('inf')
        l, r = 1, n
        while l + 1 < r:
            mid = l + (r - l) // 2
            v1 = dp(k, n - mid)
            v2 = dp(k - 1, mid - 1)
            if v1 > v2:
                l = mid + 1
            else:
                r = mid

        for x in [l, r]:
            res = min(res, max(dp(k - 1, x - 1), dp(k, n - x)) + 1)
        dic[(k, n)] = res
        return res

    return dp(K, N)


# 区间dp


if __name__ == '__main__':
    print('\nFibonacci sequence')
    print(fib1(39))
    print(fib2(39))

    print('\n抢钱1')
    print(rob2([5, 3, -6, -5, 10]))
    print(rob1([5, 3, -6, -5, 10]))

    print('\n抢钱2')
    print(rob_with_cycle([1, 2, 1, 1]))

    print('\n连续子序列和最大')
    print(max_sub_array([10, -5, 10]))

    print('\n连续子序列乘积最大')
    print(max_continuous_product([-1, 2, 3, 0.1, -10]))

    print('\n最长上升子序列')
    print(LIS([2, 5, 3, 4, 1, 7, 6]))

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

    print('\n最小路径和')
    matrix = [
        [1, 3, 1],
        [1, 5, 1],
        [4, 2, 1]
    ]
    print(min_path_sum(matrix))

    print('\n三角形最小路径和')
    print(minimum_total([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]))

    print('\n最大正方形面积')
    matrix = [["1", "0", "1", "0", "0"],
              ["1", "0", "1", "1", "1"],
              ["1", "1", "1", "1", "1"],
              ["1", "0", "0", "1", "0"]]
    print(maximal_square(matrix))

    print('\n股票买卖 ')
    print('最佳盈利')
    print(max_profit1([5, 10, 15, 1, 20]))

    print('包含冷冻期')
    print(max_profit2([1, 2, 3, 0, 2]))

    print('有交易次数限制')
    print(max_profit3([3, 3, 5, 0, 0, 3, 1, 4]))

    print('\n不包含自身的乘积')
    print(product_except_self([1, 2, 3, 4]))

    print('\n扔鸡蛋')
    print(super_egg_drop(3, 14))
