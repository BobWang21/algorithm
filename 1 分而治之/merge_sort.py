#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:59:56 2017

@author: wangbao
"""


def merge_sort(nums):
    n = len(nums)
    if n <= 1:  # 递归基
        return nums
    mid = n // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge(left, right)


# 合并两个有序数组
def merge(a, b):
    if not a:
        return b
    if not b:
        return a
    i, j = 0, 0
    res = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    # 判断循环跳出时的状态
    if i < len(a):
        res += a[i:]
    if j < len(b):
        res += b[j:]
    return res


# 递归
def merge2(a, b):
    if not a:
        return b
    if not b:
        return a
    res = []
    if a[0] < b[0]:
        res.append(a[0])
        res += merge2(a[1:], b)
    else:
        res.append(b[0])
        res += merge2(a, b[1:])
    return res


if __name__ == '__main__':
    print('归并排序')
    print(merge_sort([1, 3, 2, 4]))

    print('合并两个有序数组')
    print(merge([1, 3], [2, 4]))
    print(merge2([1, 3], [2, 4]))
