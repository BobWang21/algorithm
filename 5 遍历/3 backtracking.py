# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict


# 78 子集问题 不回溯
def subset(nums):
    res = []
    n = len(nums)

    def dfs(i, path):
        res.append(path)

        for j in range(i, n):
            dfs(j + 1, path + [nums[j]])  # 生成新的路径,不需回溯

    dfs(0, [])
    return res


# 子集问题 回溯
def subsets_2(nums):
    n = len(nums)
    res = []
    path = []

    def helper(i):
        res.append(path[:])

        for j in range(i, n):  # i==n时跳出
            path.append(nums[j])
            helper(j + 1)
            path.pop(-1)

    helper(0)


# 90. 子集包含重复
def subsets_with_dup(nums):
    if not nums:
        return []

    n, res, path = len(nums), [], []

    def helper(i=0):
        res.append(path[:])
        for j in range(i, n):
            if j > i and nums[j] == nums[j - 1]:  # 相同数字不能出现在同一层
                continue
            path.append(nums[j])
            helper(j + 1)
            path.pop(-1)

    nums.sort()
    helper()
    return res


def n_sum1(nums, k, target):
    if not nums or len(nums) < k or target < 0:
        return []
    res, path = [], []
    n = len(nums)

    def helper(idx, total):
        if len(path) == k and total == target:  # 满足条件 k个数的和为target
            res.append(path[:])
            return
        if len(path) == k:  # 不满足条件
            return
        if total > target:  # 后面的数不够K个
            return
        for j in range(idx, n):
            if j > idx and nums[j] == nums[j - 1]:  # 去重
                continue
            path.append(nums[j])
            helper(j + 1, total + nums[j])  # total不需回溯
            path.pop(-1)

    nums.sort()
    helper(0, 0)
    return res


# 四个数组 每个数组中选一个数字 要求四个数和为0 返回和为0的数组个数
def four_sum_count(A, B, C, D):
    dic = defaultdict(int)
    n = len(A)
    for i in range(n):
        for j in range(n):
            s = A[i] + B[j]
            dic[s] += 1

    res = 0
    for i in range(n):
        for j in range(n):
            s = C[i] + D[j]
            if -s in dic:
                res += dic[-s]
    return res


# 494 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数
# 最优解使用动态规划
def find_target_sum_ways(nums, target):
    n = len(nums)
    res = [0]

    def helper(idx, total):
        if idx == n and total == target:
            res[0] = res[0] + 1
            return
        if idx == n:
            return

        helper(idx + 1, total + nums[idx])
        helper(idx + 1, total - nums[idx])

    helper(0, 0)
    return res[0]


# 39
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

    def dfs(i, total, path):
        if total > target:  # 不满足条件
            return
        if total == target:
            res.append(path[:])
            return

        for j in range(i, n):
            path.append(nums[j])
            # 索引不加1, 表示数字可以用多次!!!
            dfs(j, target + nums[j], path)
            path.pop(-1)

    nums.sort()
    dfs(0, target, [])
    return res


# 46 全排列 输入数组中不含重复数字
def permute1(nums):
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


# 46
def permute2(nums):
    if not nums:
        return []

    res = []
    n = len(nums)
    visited = [False] * n
    path = []

    def dfs():
        if len(path) == n:
            res.append(path[:])  # 复制

        for i in range(n):  # 从头开始
            if visited[i]:
                continue
            visited[i] = True
            path.append(nums[i])
            dfs()
            path.pop(-1)
            visited[i] = False

    dfs()
    return res


# 47. 全排列 输入数组中含重复数字
def permute_unique1(nums):
    nums.sort()

    n = len(nums)
    visited = [False] * n

    res = []

    def dfs(path):
        if len(path) == n:
            res.append(path[:])
            return

        for i in range(n):
            if visited[i]:
                continue
            # 前面的数字未使用时，nums[i-1] 和 nums[i] 不能用 [1, 2_1, 2_2, 3]
            # [1, 2_1, 2_2, ] 可以 [1, 2_2, 2_1, ]不可以
            if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]:
                continue
            visited[i] = True
            path.append(nums[i])
            dfs(path)
            path.pop(-1)
            visited[i] = False

    dfs([])

    return res


def permute_seq(n):
    if not n:
        return []
    res, path = [], []
    status = [2] * n

    def helper():
        if len(path) == 2 * n:
            res.append(path[:])
            return

        for i in range(n):
            if not status[i]:
                continue

            if status[i] == 2:
                path.append(str(i) + '_B')
            else:
                path.append(str(i) + '_C')
            status[i] -= 1

            helper()

            path.pop(-1)
            status[i] += 1

    helper()
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


"""
W=<90, 80, 40, 30, 20, 12, 10> c1 =152, c2 =130
有n个集装箱，需要装上两艘载重分别为 c1 和 c2 的轮船。
wi 为第i个集装箱的重量，且 w1+w2+...+wn ≤ c1+c2。
问是否存在一种合理的装载方 案把这n个集装箱装上船? 如果有，给出一种方案。
算法思想: 令第一艘船的载入量为W1
1. 用回溯法求使得c1 -W1 达到最小的装载方案
2. 若满足 w1+w2+...+wn -W1 ≤ c2
"""


def pack(candidates, c1, c2):
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
    def valid(string):
        if len(string) < 2:
            return True
        if string[0] == '0':
            return False
        return int(string) <= 255

    n = len(s)
    path = []
    res = []

    def helper(idx):
        if len(path) == 4 and idx == n:
            res.append('.'.join(path))

        for i in range(1, 4):
            end = idx + i
            if end > n:
                continue
            sub_string = s[idx:end]
            if not valid(sub_string):
                continue
            path.append(sub_string)
            helper(end)
            path.pop(-1)

    helper(0)

    return res


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


# 22. 括号生成
def generate_parenthesis(n):
    path, res = [], []

    def helper(left=n, right=n):
        if len(path) == 2 * n:
            res.append(''.join(path))
            return
        if left > 0:
            path.append('(')
            helper(left - 1, right)
            path.pop(-1)
        if right > left:
            path.append(')')
            helper(left, right - 1)
            path.pop(-1)

    helper()
    return res


# 301. 删除无效的括号
def remove_invalid_parentheses(s):
    if not s:
        return ['']

    l = r = 0  # 多余的括号
    for ch in s:
        if ch in {'(', ')'}:
            if ch == '(':
                l += 1
            elif l:
                l -= 1
            else:
                r += 1

    n, res = len(s), set()  # 因为可能存在重复值 所以使用set保存结果

    def dfs(i, l, r, left, path):  # left 表示未匹配的左括号
        if i == n and not l and not r and not left:
            res.add(path)
        if i == n:
            return
        if s[i] not in {'(', ')'}:
            dfs(i + 1, l, r, left, path + s[i])
            return
        if s[i] == '(':
            dfs(i + 1, l, r, left + 1, path + s[i])  # 加
            if l:
                dfs(i + 1, l - 1, r, left, path)  # 不加
        else:
            if left:
                dfs(i + 1, l, r, left - 1, path + s[i])  # 加
            if r:
                dfs(i + 1, l, r - 1, left, path)  # 不加

    dfs(0, l, r, 0, '')
    return list(res)


# 79 单词搜索
def exist(board, word):
    rows, cols = len(board), len(board[0])
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    n = len(word)

    def dfs(i, j, k):
        if k == n:
            return True

        if i < 0 or i == rows or j < 0 or j == cols or board[i][j] == '.':
            return False

        if board[i][j] != word[k]:
            return False

        char = board[i][j]
        board[i][j] = '.'
        for d in directions:
            row, col = i + d[0], j + d[1]
            if dfs(row, col, k + 1):
                return True
        board[i][j] = char
        return False

    for i in range(rows):
        for j in range(cols):
            if dfs(i, j, 0):
                return True
    return False


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


if __name__ == '__main__':
    print('\n全集')
    print(subset([1, 2, 3]))
    print(subsets_with_dup([1, 2, 2, 3]))

    print('\ncombination sum')
    print(n_sum1([1, 1, 2, 3, 4], 3, 6))

    print('\n数组中不包含重复数字 一个数字可以用无数次')
    print(combination_sum([2, 3, 5], 8))

    print('\nk个和相等的子数组')
    nums = [114, 96, 18, 190, 207, 111, 73, 471, 99, 20, 1037, 700, 295, 101, 39, 649]
    print(can_partition_k_subsets(nums, 4))

    print('\n数组中包含重复数字 一个数字只能用一次')
    print(combination_sum([1, 1, 2, 3, 4], 4))

    print('\n排列问题')
    print(permute1([1, 2, 3]))
    print(permute2([1, 2, 1]))
    print(permute_unique1([1, 2, 1]))

    print('\n背包问题')
    print(knapsack([1, 2, 3, 4], [1, 3, 5, 8], 5))

    print('\n两个轮船分集装箱')
    print(pack([90, 80, 40, 30, 20, 12, 10], 152, 130))

    print('\nLetter Case Permutation')
    print(letter_case_permutation("a1b2"))

    print("\nip恢复")
    print(restore_ip_addresses("010010"))

    print('\n嵌套链表权重和')
    print(nested_list_weight_sum([1, [4, [6]]]))

    print('\n生成括号')
    print(generate_parenthesis(2))

    print('\n删除无效的括号')
    print(remove_invalid_parentheses("(a)())()"))
