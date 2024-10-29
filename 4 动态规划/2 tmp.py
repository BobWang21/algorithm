#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from functools import lru_cache


# 记忆化的递归
@lru_cache(maxsize=None)  # maxsize=None 表示缓存可以无限制地增长
def fib1(n):
    if n <= 2:
        return 1  # 注意：斐波那契数列的前两个数字通常定义为1，但根据具体定义，可能需要稍作调整
    return fib1(n - 1) + fib1(n - 2)


# top down + 记忆
def fib2(n):
    dic = {1: 1, 2: 2}

    def helper(n):
        if n in dic:
            return dic[n]
        res = helper(n - 1) + helper(n - 2)
        dic.setdefault(n, res)
        return res

    return helper(n)


# bottom up
# 可以记录所有状态 但当前状态只与最近前两个状态有关 可只存两个状态
def fib3(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    left, right = 1, 2
    for i in range(3, n + 1):
        left, right = right, left + right
    return right


# 198 打家劫舍
# dp[i] = max(dp[i-1], dp[i-2] + nums[i])
# 全为正数
def rob1(nums):
    if not nums:
        return 0
    n = len(nums)
    if n <= 2:
        return max(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[:2])
    for i in range(2, n):
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
    return dp[-1]


# 滚动数组
def rob2(nums):
    inc = exc = 0  # 是否包括当前值 状态
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
        inc, exc = 0, 0
        for i in range(lo, hi):
            inc, exc = exc + nums[i], max(exc, inc)
        return max(inc, exc)

    # 约束的处理
    return max(helper(nums, 0, n - 1), helper(nums, 1, n))


# 53 最大连续子序列和
def max_sub_array(nums):
    if not nums:
        return 0
    inc = exc = -float('inf')  # 负无穷适用与目标值最大时的初始化
    for num in nums:
        inc, exc = max(inc + num, num), max(inc, exc)
    return max(inc, exc)


# 152 最大连续子序列乘积
def max_continuous_product(nums):
    res = inc_min = inc_max = nums[0]
    for val in nums[1:]:
        inc_min, inc_max = min(val, val * inc_max, val * inc_min), max(val, val * inc_max, val * inc_min)
        res = max(inc_max, res)
    return res


# 12 只能买卖一次 滑动窗口
def max_profit1(prices):
    n = len(prices)
    min_price = float('inf')
    max_profit = 0
    for j in range(n):
        max_profit = max(max_profit, prices[j] - min_price)
        min_price = min(prices[j], min_price)
    return max_profit


# 122 股票可以交易多次
# 你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。
def max_profit2(prices):
    n = len(prices)
    dp = [[0, 0] for _ in range(n)]
    dp[0][1] = -prices[0]

    for i in range(1, n):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
    return dp[-1][0]


# 状态只和前一天相关
def max_profit2_1(prices):
    hold, empty = -float('inf'), 0
    for price in prices:
        hold, empty = max(empty - price, hold), max(hold + price, empty)
    return max(hold, empty)


# 309 交易后有冷冻期
def max_profit3(prices):
    hold, empty_not, empty_can = -float('inf'), 0, 0  # 第一次不能出现hold, hold初始状态为负无穷
    for price in prices:
        hold, empty_not, empty_can = max(hold, empty_can - price), hold + price, max(empty_can, empty_not)
    return max(empty_not, empty_can)


# 最多交易两次
def max_profit4(prices):
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


# 400 最长上升子序列
def LIS1(nums):
    dp = [1] * len(nums)
    res = 1
    for i, val in enumerate(nums):
        for j in range(i):
            if nums[j] < val:
                dp[i] = max(dp[j] + 1, dp[i])
        res = max(res, dp[i])
    return res


# 400 res[i] 保存长度为i+1的子串的最小值 nlog(n)
def LIS2(nums):
    if not nums:
        return 0
    dp = [nums[0]]  # nums[i]表示长度为i+1上升子串结尾的最小值
    n = len(nums)

    def binary_search(target):
        l, r = 0, len(dp) - 1
        while l < r:
            mid = l + (r - l) // 2
            if dp[mid] < target:
                l = mid + 1
            else:
                r = mid
        return l

    for i in range(1, n):
        num = nums[i]
        if num > dp[-1]:
            dp.append(num)
        else:
            idx = binary_search(num)  # 大于等于该数的第一个数位置
            dp[idx] = num
    return len(dp)


# 279. Perfect Squares 12 = 4 + 4 + 4
def num_squares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # coin=amount时使用
    for i in range(1, n + 1):
        for coin in range(1, int(n ** 0.5) + 1):
            square = coin ** 2
            if i >= square:
                dp[i] = min(dp[i], dp[i - square] + 1)
    return dp[n]


# 整数拆分 343
def integer_break1(n):
    dp = [0] * (n + 1)
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
    return dp[n]


def integer_break2(n):
    if n <= 3:
        return n - 1

    dp = [0] * (n + 1)
    dp[2] = 1
    for i in range(3, n + 1):
        dp[i] = max(2 * (i - 2), 2 * dp[i - 2], 3 * (i - 3), 3 * dp[i - 3])

    return dp[n]


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


# 最小带权路径和 对比viterbi
def min_path_sum(matrix):
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix) + 1, len(matrix[0]) + 1
    dp = [[float('inf')] * cols for _ in range(rows)]
    dp[0][1] = 0  # 初始状态

    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = matrix[i - 1][j - 1] + min(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


# 120 三角形 自上至下路径和最小值
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


# 最大正方形面积 1277
def maximal_square(matrix):
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix) + 1, len(matrix[0]) + 1
    dp = [[0] * cols for _ in range(rows)]  # 我们用f[i][j]表示以(i, j)为右下角的正方形的最大边长

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


# 62. 不同路径
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


# 扔鸡蛋
# 给你k 枚相同的鸡蛋，并可以使用一栋从第 1 层到第 n 层共有 n 层楼的建筑。
# 已知存在楼层f，满足0 <=f<=n ，任何从高于f的楼层落下的鸡蛋都会碎，从f楼层或比它低的楼层落下的鸡蛋都不会破。
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
    print(LIS1([2, 5, 3, 4, 1, 7, 6]))

    print('\n数字由平方组合')
    print(num_squares(13))

    print('\n整数拆分')
    print(integer_break1(8))

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
    print(max_profit3([1, 2, 3, 0, 2]))
    print('有交易次数限制')
    print(max_profit4([3, 3, 5, 0, 0, 3, 1, 4]))

    print('\n扔鸡蛋')
    print(super_egg_drop(3, 14))
