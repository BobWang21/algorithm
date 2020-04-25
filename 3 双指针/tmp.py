#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
        if not nums[i]:
            continue
        nums[j] = nums[i]
        j += 1
    for i in range(j, len(nums)):
        nums[i] = 0

    return nums


if __name__ == '__main__':
    print('\n平方排序')
    print(sorted_squares([-7, -3, 2, 3, 11]))

    print('\n移动0')
    print(move_zeros([0, -7, 0, 2, 3, 11]))
