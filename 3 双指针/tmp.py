#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:20:38 2019

@author: wangbao
"""


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


if __name__ == '__main__':
    print('平方排序')
    print(sorted_squares([-7, -3, 2, 3, 11]))
