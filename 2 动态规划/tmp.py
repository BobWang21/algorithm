# bottom up


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


# 股票买卖最佳时机
def stock(nums):
    n = len(nums)
    if n < 2:
        raise Exception()
    buy = sell = 0
    balance = -float('inf')
    for i in range(1, n):
        if nums[i] - nums[buy] > balance:
            balance = nums[i] - nums[buy]
            sell = i
            continue
        if nums[i] < nums[buy]:
            buy = i
    return buy, sell, balance


# 连续子序列和 最大
def max_continuous_sum(nums):
    max_sum = last_max_sum = nums[0]
    for val in nums[1:]:
        last_max_sum = max(val, last_max_sum + val)
        max_sum = max(max_sum, last_max_sum)
    return max_sum


# 连续子序列 乘积最大
def max_continuous_product(nums):
    max_pro = last_max_pro = last_min_pro = nums[0]
    for val in nums[1:]:
        new_last_min_pro = min(val, val * last_max_pro, val * last_min_pro)
        last_max_pro = max(val, val * last_max_pro, val * last_min_pro)
        max_pro = max(last_max_pro, max_pro)
        last_min_pro = new_last_min_pro

    return max_pro


# 给定一个整数的数组, 相邻的数不能同时选
# 求从该数组选取若干整数, 使得他们的和最大
def not_continuous_sum(nums):
    include = nums[0]
    exclude = 0
    for val in nums[1:]:
        new_include = max(val, exclude + val)
        exclude = max(exclude, include)
        include = new_include

    return max(include, exclude)


# 第一个 和 最后一个连成环 不能同时选
def rob2(nums):
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
    res = [1] * len(nums)
    l = -1
    for i, val in enumerate(nums):
        for j in range(i):
            if nums[j] < val:
                res[i] = max(res[j] + 1, res[i])
        l = max(l, res[i])
    return l


# 最长公共子序列
def longest_common_subsequence(str1, str2):
    l1, l2 = len(str1), len(str2)
    arr = [[0] * (l2 + 1) for _ in range(l1 + 1)]
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if str1[i - 1] == str2[j - 1]:
                arr[i][j] = arr[i - 1][j - 1] + 1
            else:
                arr[i][j] = max(arr[i - 1][j], arr[i][j - 1])
    return arr[-1][-1]


# 换硬币
# 您会得到不同面额的硬币和总金额。
# 编写一个函数来计算组成该数量所需的最少数量的硬币。
# 如果这笔钱不能用硬币的任何组合来弥补，请返回-1。
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 当coin = amount 时使用

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


# 给定一个正整数数组 求和为target的所有组合数
def combination_sum(nums, target):
    '''
    NUMS = [1，2，3]
    目标 = 4种的一种可能的组合方式有：
    （1，1，1，1）
    （1，1,2）
    （1，2,1）
    （1,3）
    （2,1 ，1）
    （2，2）
    （3，1）
    '''
    res = [0] * (target + 1)
    res[0] = 1
    for i in range(target + 1):
        for v in nums:
            if i >= v:
                res[i] += res[i - v]
    return res[-1]


# 割绳子
def max_product_after_cutting(m):
    res = [1, 1, 2]
    if m == 1:
        return
    if m == 2:
        return 1
    if m == 3:
        return 2
    res = [0, 1, 2, 3]
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
    print('Fibonacci sequence')
    print(fib(40)[39])
    print(fib2(39))

    print('股票的最佳买卖时机')
    print(stock([5, 10, 15, 1, 20]))

    print('连续子序列和最大')
    print(max_continuous_sum([10, -5, 10]))

    print('连续子序列乘积最大')
    print(max_continuous_product([-1, 2, 3, 0.1, -10]))

    print('非连续子序列和最大')
    print(not_continuous_sum([5, 3, -6, -5, 10]))

    print('抢钱')
    print(rob2([1, 2, 1, 1]))

    print('最长上升子序列')
    print(longest_increasing_subsequence([2, 5, 3, 4, 1, 7, 6]))

    print('最长公共子序列')
    print(longest_common_subsequence('aabcd', 'ad'))

    print('换硬币')
    print(coin_change([1, 2, 5, 10], 11))

    print('割绳子')
    print(max_product_after_cutting(8))

    print('最大礼物')
    matrix = [[1, 10, 3, 8],
              [12, 2, 9, 6],
              [5, 7, 4, 11],
              [3, 7, 16, 5]]
    print(max_gift(matrix))
