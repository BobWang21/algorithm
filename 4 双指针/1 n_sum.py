#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict


# 未排序数组 解唯一 O(N)空间复杂度
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
    return res


def three_sum(nums, target):
    n = len(nums)
    if n < 3:
        return
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:  # 防止第一个数重复
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

    print('\n3 sum closet')
    print(three_sum_closet([-1, 2, 1, -4], 1))

    print('\n差值小于等于target的对数')
    nums = [1, 3, 5, 7]
