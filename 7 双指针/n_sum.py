#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:20:38 2019

@author: wangbao
"""


# 有唯一解 O(N)
def two_sum(nums, target):
    size = len(nums)
    if size < 2:
        return []
    l, r = 0, size - 1
    res = []
    while l < r:
        s = nums[l] + nums[r]
        if s < target:
            l += 1
        elif s > target:
            r -= 1
        else:
            res.append([nums[l], nums[r]])
            while l < r and nums[l] == nums[l + 1]:  # 左边跳出相等部分即可
                l += 1
            l += 1
            # while l < r and nums[r] == nums[r - 1]:
            #     r -= 1
            # r -= 1
    return res


# 双指针
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
                while l < r and nums[l] == nums[l + 1]:  # 跳出时left为最后一个相同的数
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
            two_sum_paths = two_sum(nums[idx:], target)
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


# 和为s的(连续)正数序列 至少两个数 滑动窗口
def find_continuous_sequence(tar):
    if tar < 3:
        return
    l, r = 1, 2
    s = l + r
    res = []
    while r <= (1 + tar) / 2:  # 缩减计算
        if s == tar:
            res.append(list(range(l, r + 1)))
            r += 1
            s += r
        elif s < tar:
            r += 1
            s += r
        else:
            s -= l
            l += 1
    return res


if __name__ == '__main__':
    print('2 sum')
    print(two_sum([1, 2, 7, 8, 11, 15], 9))

    print('3 sum')
    print(three_sum([2, 7, 7, 11, 15, 15, 20, 24, 24], 33))

    print('4 sum')
    print(n_sum2([1, 0, -1, 0, -2, 2], 0))

    print('3 sum closet')
    print(three_sum_closet([-1, 2, 1, -4], 1))

    print('和为S的连续子序列')
    print(find_continuous_sequence(15))
