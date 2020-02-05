#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:59:56 2017

@author: wangbao
"""


def merge_sort(lists):
    n = len(lists)
    # 递归基
    if n <= 1:
        return lists
    mid = n // 2
    left = merge_sort(lists[:mid])
    right = merge_sort(lists[mid:])
    return merge(left, right)


# 合并两个有序数组
def merge(a, b):
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return b
    if len_b == 0:
        return a
    i, j = 0, 0
    lists = []
    while i < len_a and j < len_b:
        if a[i] <= b[j]:
            lists.append(a[i])
            i += 1
        else:
            lists.append(b[j])
            j += 1
    # 判断循环跳出的状态
    if i < len_a:
        lists += a[i:]
    if j < len_b:
        lists += b[j:]
    return lists


# 递归
def merge_(a, b):
    len_a, len_b = len(a), len(b)
    if (len_a == 0):
        return b
    if (len_b == 0):
        return a
    lists = []
    if a[0] < b[0]:
        lists.append(a[0])
        lists += merge_(a[1:], b)
    else:
        lists.append(b[0])
        lists += merge_(a, b[1:])
    return lists


# 二分求和
def sum_(lists):
    len_ = len(lists)
    if len_ == 1:
        value = lists[0]
        return value
    mid = len_ >> 1
    left = sum_(lists[:mid])
    right = sum_(lists[mid:])
    return left + right
