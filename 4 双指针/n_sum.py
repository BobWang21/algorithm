#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict


# 未排序数组 解唯一 o(N)空间复杂度
def two_sum1(nums, target):
    dic = dict()

    for i in range(len(nums)):
        if target - nums[i] in dic:
            return [dic[target - nums[i]], i]
        dic[nums[i]] = i


# 排序数组 解不唯一 O(N)
def two_sum2(nums, target):
    n = len(nums)
    if n < 2:
        return []
    l, r = 0, n - 1
    res = []
    while l < r:
        total = nums[l] + nums[r]
        if total < target:
            l += 1
        elif total > target:
            r -= 1
        else:
            res.append([nums[l], nums[r]])
            while l < r and nums[l] == nums[l + 1]:  # 左边界跳出相等部分即可
                l += 1
            l += 1
            # while l < r and nums[r] == nums[r - 1]:
            #     r -= 1
            # r -= 1
    return res


def three_sum(nums, target):
    n = len(nums)
    if n < 3:
        return
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:  # 防止重复!!!
            continue
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < target:
                l += 1
            elif s > target:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                l += 1
    return res


# 输入有重复数字, 每个数字只能用一次
# 输出不包含重复 回溯法
def n_sum(nums, target):
    if not nums:
        return
    res = []

    def dfs(nums, idx, k, target, path, res):
        if k == 4 and target == 0:
            res.append(path)
            return
        if k == 4:
            return
        for i in range(idx, len(nums)):
            if i > idx and nums[i] == nums[i - 1]:  # 不重复
                continue
            dfs(nums, i + 1, k + 1, target - nums[i], path + [i], res)

    dfs(nums, 0, 0, target, [], res)
    return res


def n_sum2(nums, target):
    def dfs(nums, idx, k, target, path, res):
        if len(nums[idx:]) < k or nums[idx] * k > target or nums[-1] * k < target:
            return
        elif k == 2:
            two_sum_paths = two_sum1(nums[idx:], target)
            for sum_paths in two_sum_paths:
                res.append(path + sum_paths)
        else:
            for i in range(idx, len(nums)):
                if i > idx and nums[i] == nums[i - 1]:
                    continue
                dfs(nums, i + 1, k - 1, target - nums[i], path + [nums[i]], res)

    if not len(nums) < 4:
        return
    res = []
    nums.sort()
    dfs(nums, 0, 4, target, [], res)
    return res


# 和与target最相近的三个数
def three_sum_closet(nums, target):
    n = len(nums)
    if n < 3:
        return
    closet_sum = None
    gap = float('inf')
    nums.sort()
    for i in range(n - 2):
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == target:
                return s
            new_gap = abs(s - target)
            if new_gap < gap:
                gap = new_gap
                closet_sum = s
            if s > target:
                r -= 1
            else:
                l += 1
    return closet_sum


# 和为s的连续正数序列 至少两个数 滑动窗口
def find_continuous_sequence(target):
    if target < 3:
        return
    l, r = 1, 2
    s = l + r
    res = []
    while r <= (1 + target) / 2:  # 缩减计算
        if s == target:
            res.append(list(range(l, r + 1)))
            r += 1
            s += r
        elif s < target:
            r += 1
            s += r
        else:
            s -= l
            l += 1
    return res


# 加减
def find_target_sum_ways(nums, S):
    n = len(nums)
    res = [0]

    def helper(idx, total):
        if idx == n and total == S:
            res[0] = res[0] + 1
            return
        if idx == n:
            return

        helper(idx + 1, total + nums[idx])
        helper(idx + 1, total - nums[idx])

    helper(0, 0)
    return res[0]


# 排序数组中距离不大于target的pair数 O(N)
def no_more_than(nums, target):
    n = len(nums)
    j = 1
    res = 0
    for i in range(n - 1):
        while j < n and nums[j] - nums[i] <= target:
            j += 1
        res += j - i - 1
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


if __name__ == '__main__':
    print('\n2 sum')
    print(two_sum1([7, 8, 2, 3], 9))
    print(two_sum2([1, 2, 7, 8, 11, 15], 9))

    print('\n3 sum')
    print(three_sum([2, 7, 7, 11, 15, 15, 20, 24, 24], 33))

    print('\n4 sum')
    print(n_sum2([1, 0, -1, 0, -2, 2], 0))

    print('\n3 sum closet')
    print(three_sum_closet([-1, 2, 1, -4], 1))

    print('\n和为S的连续子序列')
    print(find_continuous_sequence(15))

    print('\n差值小于等于target的对数')
    nums = [1, 3, 5, 7]
    print(no_more_than(nums, 1))
