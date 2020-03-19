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
    """
    shuffle 一个数组
    n个数中抽取1个数放在第一位，从剩下n-1个数中抽取一个放在第二位
    某个数被放在第二位的概率为 (n-1 / n)*(1 / n-1) = 1/n
    """
    n = len(nums)
    for i in range(n):
        idx = rd.randint(i, n - 1)  # 包含左右端点
        nums[i], nums[idx] = nums[idx], nums[i]
    return nums


def weighted_random(nums, weight):
    weights = []
    total_weight = 0
    for w in weight:
        total_weight += w
        weights.append(total_weight)
    rdw = rd.random() * total_weight
    for i, w in enumerate(weights):
        if rdw <= w:
            return nums[i]
        rdw -= w


def rand2to5():
    """模拟其他发生器
    二进制思想:
    0-1 发生器 生成 0-4 发生器
    此外还有其他方法
    """
    while True:
        data = rd.randint(0, 1) * 2 ** 2 + rd.randint(0, 1) * 2 + rd.randint(0, 1)
        if data <= 4:
            return data


def reservoir_sampling(data, m):
    """蓄水池问题
    m 个数
    """
    res = np.zeros(m)

    for i, value in enumerate(data):
        # 保存前n个数，保证至少有n个数
        if i < m:
            res[i] = value
        else:
            if rd.randint(1, i + 1) <= m:  # 第i个数被选中概率为 m/(i+1)
                idx = rd.randint(0, m - 1)
                res[idx] = value
    return res


def exam(n):
    """
    随机生成n个不相邻的数
    """
    out = []
    data = [i + 1 for i in range(n)]
    prior = -1
    i = 0
    time = 0
    while len(out) < n:
        index = rd.randint(0, n - 1 - i)
        # print(index)
        if abs(prior - data[index]) != 1:
            prior = data[index]
            out.append(prior)
            data.remove(prior)
            i += 1
        time += 1
        if time > n ** 2:
            exam(n)
            break
    return out


if __name__ == '__main__':
    print('\nshuffle')
    data = [rd.randint(0, 100) for _ in range(10)]
    print(shuffle(data))

    print('\n蓄水池')
    data = np.arange(100)
    print(reservoir_sampling(data, m=10))

    print('\n加权抽样')
    res = []
    for i in range(100):
        res.append(weighted_random(list(range(3)), [2, 2, 4]))
    print(Counter(res))

    print(exam(100))
    data = [rand2to5() for _ in range(100000)]
