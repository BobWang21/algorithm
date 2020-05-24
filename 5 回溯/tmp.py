#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def n_sum1(nums, k, target):
    if not nums or len(nums) < k or target < 0:
        return []
    res = []
    n = len(nums)

    def helper(idx, k, target, path):
        if not k and not target:  # 满足条件 k个数的和为target
            res.append(path)
            return
        if not k:  # 不满足条件
            return
        if k > n - idx:  # 后面的数不够K个
            return
        for i in range(idx, n):
            if i > idx and nums[i] == nums[i - 1]:  # 去重
                continue
            # path + [nums[i]] 生成一个新的列表 因此不需要回溯
            helper(i + 1, k - 1, target - nums[i], path + [nums[i]])

    nums.sort()
    helper(0, k, target, [])
    return res


def n_sum2(nums, k, target):
    if not nums or len(nums) < k or target < 0:
        return []
    res = []
    n = len(nums)

    def helper(idx, k, target, path):
        if not k and not target:  # 满足条件 k个数的和为target
            res.append(path[:])  # 复制新的结果 !!!
            return
        if not k:  # 不满足条件
            return
        if k > n - idx:  # 后面的数不够K个
            return
        for i in range(idx, n):
            if i > idx and nums[i] == nums[i - 1]:  # 去重
                continue
            path.append(nums[i])
            helper(i + 1, k - 1, target - nums[i], path)
            path.pop(-1)

    nums.sort()
    helper(0, k, target, [])
    return res


# 698 10000 > nums[i] > 0
def can_partition_k_subsets(nums, k):
    if len(nums) < k:
        return False
    total = sum(nums)
    if total % k:  # 不可整除
        return False
    target = total // k

    candidate = []
    for v in nums:
        if v > target:
            return False
        elif v == target:
            k -= 1
        else:
            candidate.append(v)
    n = len(candidate)
    visited = [False] * n  # 已经使用过的数字 不能再次访问 如果没有重复数字可以使用set

    def helper(idx, k, tar):
        if k == 0:
            return True
        if not tar:
            return helper(0, k - 1, target)  # 从0开始 从target开始
        if tar < 0:
            return False
        for i in range(idx, n):
            if visited[i]:
                continue
            visited[i] = True
            if helper(i + 1, k, tar - candidate[i]):
                return True
            visited[i] = False  # 如果不满足条件回溯
        return False

    nums.sort()
    return helper(0, k, target)


# Given a set of candidate numbers (candidates) (without duplicates)
# and a target number (target), find all unique combinations in candidates
# where the candidate numbers sums to target.
# The same repeated number may be chosen from candidates unlimited number of times.
# Input: candidates = [2, 3, 6, 7], target = 7,
# A solution set is:
# [
#   [7],
#   [2, 2, 3]
# ]
def combination_sum(nums, target):
    if not nums or target == 0:
        return []
    res = []
    n = len(nums)

    def dfs(idx, target, path):
        if not target:
            res.append(path[:])
            return
        if target < 0:  # 不满足条件
            return
        for i in range(idx, n):
            # 索引从 i 开始表示数字可以用多次!!!
            path.append(nums[i])
            dfs(i, target - nums[i], path)
            path.pop(-1)

    nums.sort()
    dfs(0, target, [])
    return res


# 有重复数字的组合 每个数字只能用一次
def combination_sum2(nums, target):
    if not nums or target < 0:
        return

    res = []
    n = len(nums)

    def dfs(idx, target, path):
        if not target:
            res.append(path)
            return
        if target < 0:
            return
        for i in range(idx, n):  # 保证顺序
            if i > idx and nums[i] == nums[i - 1]:  # 排除相同的数字出现在同一层
                continue
            # 当前迭代索引为i 下一个迭代的索引为i+1
            dfs(i + 1, target - nums[i], path + [nums[i]])

    nums.sort()
    dfs(0, target, [])
    return res


# 全排列 输入数组中不含重复数字
def permutations(nums):
    """
    Given a collection of distinct integers, return all possible permutations.
    Example:
    Input: [1, 2, 3]
    Output:
    [
      [1, 2, 3],
      [1, 3, 2],
      [2, 1, 3],
      [2, 3, 1],
      [3, 1, 2],
      [3, 2, 1]
    ]
    """

    def dfs(candidates, path, res):
        if len(candidates) == len(path):
            res.append(path)
            return
        candidate = candidates - set(path)  # 从未访问过的集合中选取元素
        for val in candidate:
            dfs(candidates, path + [val], res)

    res = []
    dfs(set(nums), [], res)
    return res


# 输入数组中含重复数字
def permutations2(nums):
    if not nums:
        return []

    res = []

    def dfs(candidates, path):
        if not candidates:
            res.append(path)
            return
        for i, val in enumerate(candidates):
            if i > 0 and candidates[i - 1] == val:  # 不可出现在同一层
                continue
            # 候选集中去除访问过的元素
            dfs(candidates[:i] + candidates[i + 1:], path + [val])

    nums.sort()
    dfs(nums, [])
    return res


# 子集问题
def subset(candidates):
    res = []
    n = len(candidates)

    def dfs(idx, path):
        res.append(path)  # 把路径全部记录下来
        for i in range(idx, n):
            dfs(i + 1, path + [candidates[i]])

    dfs(0, [])
    return res


def subsets2(nums):
    res = [[]]
    for num in nums:
        res += [curr + [num] for curr in res]  # 之前的解并不包含当前数字
    return res


# 递增子序列
def find_sub_sequences(nums):
    if not nums:
        return []
    res = set()  # 使用集合去重
    n = len(nums)

    def helper(idx, path):
        if len(path) >= 2:
            res.add(tuple(path))  # 使用tuple hash list

        for i in range(idx, n):
            if not path or path[-1] <= nums[i]:
                helper(i + 1, path + [nums[i]])

    helper(0, [])
    return res


# 背包问题
def knapsack(costs, values, capacity):
    def dfs(capacity, idx, amount, res):
        for i in range(idx, len(costs)):
            cost, val = costs[i], values[i]
            if capacity - cost < 0:  # base 1
                res[0] = max(res[0], amount)
                continue
            elif capacity == cost:  # base 2
                res[0] = max(res[0], amount + val)
            else:
                dfs(capacity - cost, i + 1, amount + val, res)

    res = [-1]
    dfs(capacity, 0, 0, res)
    return res[0]


def pack(candidates, c1, c2):
    """
    # W=<90, 80, 40, 30, 20, 12, 10> c1 =152, c2 =130
    # 有n个集装箱，需要装上两艘载重分别为 c1 和 c2 的轮船。
    # wi 为第i个集装箱的重量，且 w1+w2+...+wn ≤ c1+c2。
    # 问是否存在一种合理的装载方 案把这n个集装箱装上船? 如果有，给出一种方案。
    # 算法思想: 令第一艘船的载入量为W1
    # 1. 用回溯法求使得c1 -W1 达到最小的装载方案
    # 2. 若满足 w1+w2+...+wn -W1 ≤ c2
    """
    candidates.sort()
    res = []

    def dfs(candidates, index, c1, path, res):
        if sum(candidates) - sum(path) <= c2:  # 加了新的后 开始小于0
            res.append(path)
            return
        for i in range(index, len(candidates)):
            if c1 - candidates[i] < 0:
                break
            dfs(candidates, i + 1, c1 - candidates[i], path + [candidates[i]], res)

    dfs(candidates, 0, c1, [], res)
    return res


def letter_case_permutation(s):
    if not s:
        return
    l = list(s)
    res = []

    def helper(l, idx, path):
        if len(path) == len(l):
            res.append(''.join(path))
            return
        for i in range(idx, len(l)):
            c = l[i]
            if c.isalpha():
                helper(l, i + 1, path + [c.lower()])
                helper(l, i + 1, path + [c.upper()])
            else:
                helper(l, i + 1, path + [c])

    helper(l, 0, [])
    return res


# 93. 复原IP地址 '010010' 恢复ip
def restore_ip_addresses(s):
    if not s:
        return []

    def valid(s):
        if len(s) > 1 and s[0] == '0':
            return False
        if int(s) > 255:
            return False
        return True

    res, n = [], len(s)

    def helper(idx, path, k):
        if k == 4 and idx == n:
            res.append(path)
            return
        if k == 4 or idx == n:
            return
        for i in range(1, 4):
            if idx + i <= n:
                string = s[idx: idx + i]
                if valid(string):
                    if not k:  # '.'的位置
                        helper(idx + i, string, 1)
                    else:
                        helper(idx + i, path + '.' + string, k + 1)

    helper(0, '', 0)
    return res


# Input:
# beginWord = "hit",
# endWord = "cog",
# wordList = ["hot","dot","dog","lot","log","cog"]
def word_ladder(begin_word, end_word, word_list):
    if not word_list:
        return 0
    queue = [(begin_word, 0)]  # 记录层数
    visited = {begin_word}  # 保存已经加入过队列的字符串 没有重复 可以使用集合
    word_set = set(word_list)  # 已经访问过的数字会被重复访问
    while queue:
        word, l = queue.pop(0)
        if word == end_word:
            return l + 1
        for i in range(len(word)):
            for j in range(26):  # 访问每个字符串的近邻
                c = chr(ord('a') + j)
                if word[i] == c:
                    continue
                new_word = word[:i] + c + word[i + 1:]
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, l + 1))

    return 0


def nested_list_weight_sum(nums):
    if not nums:
        return 0
    res = [0]

    def helper(nums, k):
        if not nums:
            return
        for v in nums:
            if isinstance(v, list):  # 判断是否为列表
                helper(v, k + 1)
            else:
                res[0] += v * k

    helper(nums, 1)
    return res[0]


def generate_parenthesis(n):
    if n < 0:
        return []
    res = []

    def helper(i, j, path, stack):
        if not i and not j and not stack:
            res.append(path)
            return
        if i < 0 or j < 0:
            return
        for c in ['(', ')']:
            if not stack:
                helper(i - 1, j, path + '(', ['('])
                break
            if c == ')' and stack[-1] == '(':
                stack.pop(-1)
                helper(i, j - 1, path + ')', stack)
                continue
            if c == '(':
                helper(i - 1, j, path + '(', stack + ['('])

    helper(n, n, '', [])
    return res


def max_profit(prices):
    if not prices and len(prices) == 1:
        return 0
    n = len(prices)

    cold, buy, sell = [0, 1, 2]
    res = [0]

    def helper(idx, buy_price, status, profit):
        if idx == n:
            res[0] = max(res[0], profit)
            return

        if status == cold:
            if not buy_price:
                helper(idx + 1, prices[idx], buy, profit)
            helper(idx + 1, buy_price, cold, profit)
            if buy_price:
                helper(idx + 1, None, sell, profit + prices[idx] - buy_price)
            return
        if status == sell:
            helper(idx + 1, None, cold, profit)
            return

        if status == buy:
            v = prices[idx] - buy_price
            if v > 0:
                helper(idx + 1, None, sell, profit + v)
            helper(idx + 1, buy_price, cold, profit)
            return

    helper(0, None, cold, 0)
    return res[0]


if __name__ == '__main__':
    print('\nn sum 回溯版')
    print(n_sum1([1, 1, 2, 3, 4], 3, 6))
    print(n_sum2([1, 1, 2, 3, 4], 3, 6))

    print('\nk个和相等的子数组')
    nums = [114, 96, 18, 190, 207, 111, 73, 471, 99, 20, 1037, 700, 295, 101, 39, 649]
    print(can_partition_k_subsets(nums, 4))

    print('\n数组中不包含重复数字 一个数字可以用无数次')
    print(combination_sum([2, 3, 5], 8))

    print('\n数组中包含重复数字 一个数字只能用一次')
    print(combination_sum2([1, 1, 2, 3, 4], 4))

    print('\n排列问题')
    print(permutations([1, 2, 3]))
    print(permutations2([1, 2, 1]))

    print('\n全集')
    print(subset([1, 2, 3]))

    print('\n背包问题')
    print(knapsack([1, 2, 3, 4], [1, 3, 5, 8], 5))

    print('\n两个轮船分集装箱')
    print(pack([90, 80, 40, 30, 20, 12, 10], 152, 130))

    print('\nLetter Case Permutation')
    print(letter_case_permutation("a1b2"))

    print('\n Word Ladder')
    a = "qa"
    b = "sq"
    c = ["si", "go", "se", "cm", "so", "ph", "mt", "db", "mb", "sb", "kr", "ln", "tm", "le", "av", "sm", "ar", "ci",
         "ca", "br", "ti", "ba", "to", "ra", "fa", "yo", "ow", "sn", "ya", "cr", "po", "fe", "ho", "ma", "re", "or",
         "rn", "au", "ur", "rh", "sr", "tc", "lt", "lo", "as", "fr", "nb", "yb", "if", "pb", "ge", "th", "pm", "rb",
         "sh", "co", "ga", "li", "ha", "hz", "no", "bi", "di", "hi", "qa", "pi", "os", "uh", "wm", "an", "me", "mo",
         "na", "la", "st", "er", "sc", "ne", "mn", "mi", "am", "ex", "pt", "io", "be", "fm", "ta", "tb", "ni", "mr",
         "pa", "he", "lr", "sq", "ye"]
    print(word_ladder(a, b, c))
    print("\nip恢复")
    print(restore_ip_addresses("010010"))

    print('\n嵌套链表权重和')
    print(nested_list_weight_sum([1, [4, [6]]]))

    print('\n生成括号')
    print(generate_parenthesis(2))

    print('\n股票买卖')
    print(max_profit([1, 2, 4]))
