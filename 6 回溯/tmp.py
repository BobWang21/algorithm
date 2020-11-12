#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 78 子集问题
# 时间复杂度：O(n * 2^n) 2^n种可能 每种可能需要O(n)构造子集
def subset(nums):
    res = []
    n = len(nums)

    def dfs(idx, path):  # 防止重复 所以顺序加入
        res.append(path)  # 把路径全部记录下来
        for i in range(idx, n):
            dfs(i + 1, path + [nums[i]])  # 生成新的 不需要回溯

    dfs(0, [])
    return res


# 90. 子集 II 包含重复
def subsets_with_dup(nums):
    if not nums:
        return []

    n, res, path = len(nums), [], []

    def helper(idx=0):
        res.append(path[:])

        for i in range(idx, n):
            if i > idx and nums[i - 1] == nums[i]:
                continue
            path.append(nums[i])
            helper(i + 1)
            path.pop(-1)

    nums.sort()
    helper()
    return res


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
            path.append(nums[i])
            # 索引从 i 开始表示数字可以用多次!!!
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


def permute2(nums):
    if not nums:
        return []

    res = []
    n = len(nums)
    visited = [False] * n

    def dfs(path):
        if len(path) == n:
            res.append(path[:])

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            path.append(nums[i])
            dfs(path)
            path.pop(-1)
            visited[i] = False

    dfs([])
    return res


# 47. 全排列 输入数组中含重复数字
def permute_unique1(nums):
    if not nums:
        return []

    res = []

    def dfs(candidates, path):
        if not candidates:
            res.append(path)
            return
        for i in candidates:
            if i > 0 and candidates[i] == candidates[i - 1]:  # 不可出现在同一层
                continue
            # 候选集中去除访问过的元素
            dfs(candidates[:i] + candidates[i + 1:], path + [candidates[i]])

    nums.sort()
    dfs(nums, [])
    return res


def permute_unique2(nums):
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
            # nums[i-1] 和 nums[i] 不能在同一个位置
            if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]:
                continue
            visited[i] = True
            path.append(nums[i])
            dfs(path)
            path.pop(-1)
            visited[i] = False

    dfs([])

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
    if not s or len(s) > 12:
        return []

    def valid(s):
        if s[0] == '0':  # 0 开头
            if len(s) < 2:
                return True
            else:
                return False
        return int(s) <= 255

    res = []
    n = len(s)

    def dfs(idx, path):
        if idx == n and len(path) == 4:
            res.append('.'.join(path))
            return
        if idx == n or len(path) == 4:
            return
        for i in range(1, 4):  # 每次加 1-3 个数
            if idx + i <= n:
                string = s[idx: idx + i]
                if valid(string):
                    dfs(idx + i, path + [string])

    dfs(0, [])

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


# 22. 括号生成
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
        for c in '()':
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


def generate_parenthesis2(n):
    if not n:
        return []

    res = []

    def helper(path, i, left):  # i:未匹配的左括号 left:左括号数
        if not i and left == n:
            res.append(path)
            return

        if not i:
            helper(path + '(', 1, left + 1)
        elif left < n:
            helper(path + '(', i + 1, left + 1)
            helper(path + ')', i - 1, left)
        else:
            helper(path + ')', i - 1, left)

    helper('', 0, 0)
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


if __name__ == '__main__':
    print('\n全集')
    print(subset([1, 2, 3]))
    print(subsets_with_dup([1, 2, 2, 3]))

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
    print(permute1([1, 2, 3]))
    print(permute2([1, 2, 1]))
    print(permute_unique1([1, 2, 1]))

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

    print('\n删除无效的括号')
    print(remove_invalid_parentheses("(a)())()"))
