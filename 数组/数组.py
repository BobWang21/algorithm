#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:20:38 2019

@author: wangbao
"""


# 超过一半的数
def most_data(data):
    size = len(data)
    if size == 0:
        raise Exception('')
    value = data[0]
    count = 1
    for i in range(1, size):
        if count == 0:
            value = data[i]
            count = 1
            continue
        if data[i] == value:
            count += 1
        else:
            count -= 1
    return value


# 1-n n+1个数中 只有一个数重复一次或多次
def find_duplicate_num(nums):
    fast = slow = nums[0]
    # 证明有环 快慢两个指针
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if fast == slow:
            break
    # 入口
    ptr1 = nums[0]
    ptr2 = fast
    while ptr1 != ptr2:
        ptr1 = nums[ptr1]
        ptr2 = nums[ptr2]
    return ptr1


print(find_duplicate_num([1, 2, 3, 4, 2]))
