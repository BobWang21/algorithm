# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict


# 78 不包含重复元素回溯
def subset(nums):
    n = len(nums)
    res = []
    path = []

    def helper(i):
        res.append(path[:])

        for j in range(i, n):  # 所有选择
            path.append(nums[j])
            helper(j + 1)  # 新递归从i+1开始 保证不出现重复元素
            path.pop(-1)  # 回溯

    helper(0)
    return res


# 90. 包含重复元素子集问题
def subsets_with_dup(nums):
    n, res, path = len(nums), [], []

    def helper(i=0):
        res.append(path[:])
        for j in range(i, n):  # 所有选择
            if j > i and nums[j] == nums[j - 1]:  # 相同数字不能出现在同一层
                continue
            path.append(nums[j])
            helper(j + 1)
            path.pop(-1)

    nums.sort()
    helper()
    return res


# 46 全排列 输入数组中不含重复数字
def permute1(nums):
    if not nums:
        return []

    n = len(nums)
    path = []
    res = []
    visited = [False] * n

    def helper():
        if len(path) == n:
            res.append(path[:])  # 复制

        for i in range(n):  # 所有选择 从头开始遍历
            if visited[i]:
                continue
            visited[i] = True
            path.append(nums[i])
            helper()
            path.pop(-1)
            visited[i] = False

    helper()
    return res


# 47. 全排列 输入数组中含重复数字
def permute_unique1(nums):
    nums.sort()

    n = len(nums)
    visited = [False] * n

    res = []

    def helper(path):
        if len(path) == n:
            res.append(path[:])
            return

        for i in range(n):
            if visited[i]:
                continue
            # [1, 2_1, 2_2, ]可以; [1, 2_2, 2_1, ]不可以
            if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]:
                continue
            visited[i] = True
            path.append(nums[i])
            helper(path)
            path.pop(-1)
            visited[i] = False

    helper([])

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


# 39 数组子集和为target 数字可以重复使用
# 求组合个数为dp问题
def combination_sum(nums, target):
    if not nums or target == 0:
        return []
    res = []
    n = len(nums)
    path = []

    def helper(i, total):
        if total > target:  # 不满足条件
            return
        if total == target:
            res.append(path[:])
            return

        for j in range(i, n):
            path.append(nums[j])
            # 索引不加1, 表示数字可以重复使用!!!
            helper(j, total + nums[j])
            path.pop(-1)

    nums.sort()
    helper(0, 0)
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


# 454 四个数组 每个数组中选一个数字 要求四个数和为0 返回和为0的数组个数
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

        # 每一位有1-3三种选择
        for i in range(1, 4):
            end = idx + i
            if end > n:
                continue
            sub_string = s[idx:end]
            if not valid(sub_string):
                continue
            path.append(sub_string)
            helper(end)
            path.pop()

    helper(0)

    return res


# 22. 括号生成
def generate_parenthesis(n):
    res, path = [], []

    def helper(left, unmatched):
        if len(path) == 2 * n:
            res.append(''.join(path))
            return

        if left < n:
            path.append('(')
            helper(left + 1, unmatched + 1)
            path.pop()

        if unmatched > 0:
            path.append(')')
            helper(left, unmatched - 1)
            path.pop()

    helper(0, 0)
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

    def helper(i, l, r, left, path):  # left 表示未匹配的左括号
        if i == n and not l and not r and not left:
            res.add(path)
        if i == n:
            return
        if s[i] not in {'(', ')'}:
            helper(i + 1, l, r, left, path + s[i])
            return
        if s[i] == '(':
            helper(i + 1, l, r, left + 1, path + s[i])  # 加
            if l:
                helper(i + 1, l - 1, r, left, path)  # 不加
        else:
            if left:
                helper(i + 1, l, r, left - 1, path + s[i])  # 加
            if r:
                helper(i + 1, l, r - 1, left, path)  # 不加

    helper(0, l, r, 0, '')
    return list(res)


# 79 单词搜索
# 判断是否存在路径
def exist(board, word):
    rows, cols = len(board), len(board[0])
    n = len(word)
    visited = [[0] * cols for _ in range(rows)]

    def helper(i, j, idx):
        if idx == n:
            return True
        if i < 0 or i == rows or j < 0 or j == cols or visited[i][j]:
            return False

        visited[i][j] = True
        res = False
        if board[i][j] == word[idx]:
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if helper(i + di, j + dj, idx + 1):
                    res = True
                    break
        visited[i][j] = False
        return res

    for i in range(rows):
        for j in range(cols):
            if helper(i, j, 0):
                return True
    return False


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


# 494 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数
# 最优解为使用动态规划
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


def letter_case_permutation(s):
    if not s:
        return
    n = len(s)
    res = []
    path = []

    def helper(i):
        if len(path) == n:
            res.append(''.join(path))
            return
        for j in range(i, n):
            c = s[j]
            if c.isalpha():
                path.append(c.lower())
                helper(j + 1)
                path.pop()

                path.append(c.upper())
                helper(j + 1)
                path.pop()
            else:
                path.append(c)
                helper(j + 1)
                path.pop()

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

    print('\n排列问题')
    print(permute1([1, 2, 3]))
    print(permute_unique1([1, 2, 1]))

    print('\n数组中包含重复数字 一个数字只能用一次')
    print(combination_sum([1, 1, 2, 3, 4], 4))

    print('\ncombination sum')
    print(n_sum1([1, 1, 2, 3, 4], 3, 6))

    print('\n数组中不包含重复数字 一个数字可以用无数次')
    print(combination_sum([2, 3, 5], 8))

    print('\nk个和相等的子数组')
    nums = [114, 96, 18, 190, 207, 111, 73, 471, 99, 20, 1037, 700, 295, 101, 39, 649]
    print(can_partition_k_subsets(nums, 4))

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
