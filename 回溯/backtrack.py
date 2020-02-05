#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回溯法
"""


########### 组合
def dfs(n, k, index, res, path):
    if k < 0:
        return
    if k == 0:
        res.append(path)
        return
    for i in range(index, n):
        dfs(n, k - 1, i + 1, res, path + [i + 1])


def combine(n, k):
    res = []
    dfs(n, k, 0, res, [])
    return res


def subsets(num):
    path = []
    for i in range(num + 1):
        path.extend(combine(num, i))
    return path


########### 排序
def dfs2(n, k, index, res, path):
    if k < 0:
        return
    if k == 0:
        res.append(path)
        return
    for i in range(n):
        if i + 1 not in path:
            dfs2(n, k - 1, i + 1, res, path + [i + 1])


def combine2(n, k):
    res = []
    dfs2(n, k, 0, res, [])
    return res


######## 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集(幂集)。
def dfs3(nums, i, res, path):
    res.append(path)
    for i in range(i, len(nums)):
        dfs3(nums, i + 1, res, path + [nums[i]])


def subsets2(nums):
    res = []
    if not nums:
        return res.append([])
    dfs3(nums, 0, res, [])
    return res


############################### 路径最小的的path 所有节点为正数
def dfs4(root, total, result):
    if not root:
        return
    l, r = root.left, root.right
    if not l and not r:
        result[0] = min(result[0], total + root.val)
    if total + root.val < result[0]:
        dfs4(l, total + root.val, result)
        dfs4(r, total + root.val, result)


def min_path_sum(root):
    result = [float('inf')]
    dfs4(root, 0, result)
    return result[0]


def min_path_sum2(root):
    if not root:
        return 0
    l, r = root.left, root.right
    if not l and not r:
        return root.val
    return min(min_path_sum(l), min_path_sum(r)) + root.val


def dfs5(V, W, B, total, result):
    if not V:  # 选完了
        result[0] = max(total, result[0])
        return
    for i in range(len(V)):
        V_, W_ = V[:], W[:]
        v = V_.pop(i)
        w = W_.pop(i)
        if B - v >= 0:
            dfs5(V_, W_, B - v, total + w, result)
        else:
            result[0] = max(total, result[0])

        # v = [8, 6, 4, 3]


# w = [12, 11, 9, 8]
# b = 13
def bag(V, W, B):
    result = [0]
    dfs5(V, W, B, 0, result)
    return result[0]


# 硬币组合
class Solution(object):
    def __init__(self):
        self.result = -1

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        coins.sort(reverse=True)
        self.dfs(coins, amount, [])
        return self.result

    def dfs(self, coins, amount, path):
        if amount < 0:
            return
        if amount == 0:
            if self.result > -1:  # 已有结果
                self.result = min(self.result, len(path))
                return
            else:
                self.result = len(path)
                return
        for coin in coins:
            if path and coin > path[-1]:
                continue
            if self.result > -1 and len(path) + 1 > self.result:
                break
            self.dfs(coins, amount - coin, path + [coin])


coins = [1, 3, 5, 6]
amount = 11
s = Solution()
print(s.coinChange(coins, amount))


### n 皇后问题
def check(new_col, path):
    if not path:
        return True

    new_row = len(path)
    for row, col in enumerate(path):
        if col == new_col or abs(new_row - row) == abs(new_col - col):
            return False
    return True


def queen(n):
    result = []
    queen_dfs(0, [], result, n)
    print(len(result))


def queen_dfs(row, path, result, n):
    for col in range(n):
        if check(col, path):
            if row == n - 1:  # 出口
                result.append(path + [col])
            else:
                queen_dfs(row + 1, path + [col], result, n)


queen(8)


# coin change 1 dfs
class Solution(object):
    def coinChange(self, coins, amount):
        if amount <= 0:
            return 0
        f = {0: 0}
        for i in range(1, amount + 1):
            val = amount + 1
            for coin in coins:
                if i >= coin:
                    val = min(f[i - coin] + 1, val)
            f.setdefault(i, val)
        return f[amount] if f[amount] <= amount else -1


# 2 动态规划思想   
# bottom-up
class Solution(object):
    def coinChange(self, coins, amount):
        if amount <= 0:
            return 0
        f = {0: 0}  # 记录函数值
        for i in range(1, amount + 1):
            val = amount + 1
            for coin in coins:
                if i >= coin:
                    val = min(f[i - coin] + 1, val)
            f.setdefault(i, val)
        return f[amount] if f[amount] <= amount else -1


# top - down  
def coinChange(coins, amount):
    def f(amount):
        if amount == 0:
            return 0
        t = amount + 1
        for coin in coins:
            if amount - coin >= 0:
                t = min(f(amount - coin) + 1, t)
        return t

    if amount <= 0:
        return 0
    t = f(amount)
    return t if t <= amount else -1
