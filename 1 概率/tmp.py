#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:50:36 2017

@author: wangbao
"""

import random as rd
from collections import Counter

import numpy as np


def shuffle(nums):
    n = len(nums)
    for i in range(n):
        idx = rd.randint(i, n - 1)  # 包含左右端点
        nums[i], nums[idx] = nums[idx], nums[i]
    return nums


def weighted_random(nums, weight):
    def binary_search(nums, target):
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return r

    total_weight = 0
    weights = [0]
    for w in weight:
        total_weight += w
        weights.append(total_weight)
    rdw = rd.random() * total_weight
    idx = binary_search(weights, rdw)
    return nums[idx]


def rand2to5():
    while True:
        data = rd.randint(0, 1) * 2 ** 2 + rd.randint(0, 1) * 2 + rd.randint(0, 1)
        if data <= 4:
            return data


def reservoir_sampling(nums, m):
    res = np.zeros(m)

    for i, value in enumerate(nums):
        # 保存前n个数，保证至少有n个数
        if i < m:
            res[i] = value
        else:
            if rd.randint(1, i + 1) <= m:  # 第i个数被选中概率为 m/(i+1)
                idx = rd.randint(0, m - 1)
                res[idx] = value
    return res


if __name__ == '__main__':
    print('\nshuffle')
    data = [rd.randint(0, 100) for _ in range(10)]
    print(shuffle(data))

    print('\n蓄水池')
    data = np.arange(100)
    print(reservoir_sampling(data, m=10))

    print('\n加权抽样')
    res = []
    for i in range(30000):
        res.append(weighted_random(list(range(3)), [2, 2, 4]))
    print(Counter(res))
