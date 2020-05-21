#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter


# 左右指针
def sorted_squares(nums):
    if not nums:
        return []
    n = len(nums)
    res = [0] * n
    i = n - 1
    l, r = 0, n - 1
    while l < r:
        if abs(nums[l]) <= abs(nums[r]):
            res[i] = (nums[r]) ** 2
            r -= 1
        else:
            res[i] = (nums[l]) ** 2
            l += 1
        i -= 1
    return res


# 不改变顺序 把0移到数组尾部
def move_zeros(nums):
    if not nums:
        return
    j = 0
    for i in range(len(nums)):
        if nums[i]:
            nums[j] = nums[i]
            j += 1
    for i in range(j, len(nums)):
        nums[i] = 0
    return nums


# 相差为K的pair
def find_pairs(nums, k):
    if len(nums) < 2:
        return 0
    nums.sort()

    if nums[-1] - nums[0] < k:
        return 0

    n = len(nums)
    l, r = 0, 1
    num = 0
    while r < n and l < n:
        if l == r:
            r += 1
            continue
        s = nums[r] - nums[l]
        if s < k:
            r += 1
        elif s > k:
            l += 1
        else:
            num += 1
            while l + 1 < n and nums[l] == nums[l + 1]:
                l += 1
            l += 1
            r = max(l + 1, r + 1)
    return num


def find_pairs2(nums, k):
    res = 0
    c = Counter(nums)
    for i in c:
        if (k > 0 and i + k in c) or (k == 0 and c[i] > 1):
            res += 1
    return res


if __name__ == '__main__':
    print('\n平方排序')
    print(sorted_squares([-7, -3, 2, 3, 11]))

    print('\n移动0')
    print(move_zeros([0, -7, 0, 2, 3, 11]))

    print('\n相差为K的pair数目')
    print(find_pairs([1, 3, 1, 5, 4], 0))
    print(find_pairs2([1, 3, 1, 5, 4], 0))
