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


def reservoir_sampling(nums, m):
    res = np.zeros(m)
    for i, value in enumerate(nums):
        # 保存前n个数，保证至少有n个数
        if i < m:
            res[i] = value
        else:
            if rd.randint(1, i + 1) <= m:  # 第i个数被选中概率为 m/i i start from 1!!!
                idx = rd.randint(0, m - 1)
                res[idx] = value
    return res


# 随机返回数组中等于target的索引
def pick(nums, target):
    n = 1
    res = None
    for i, v in enumerate(nums):
        if v == target:
            if rd.randint(1, n) == 1:  # 第i元素被选中的概率为 1/i  m = 1的特殊场景
                res = i
            n += 1
    return res


# 根据生成1, 2随机数的发生器 生成1-5随机发生器
def rand2to5():
    while True:
        value = rd.randint(0, 1) * 2 ** 2 + rd.randint(0, 1) * 2 + rd.randint(0, 1)
        if value < 5:
            return value + 1


# 均匀分布生成正态分布 中心极限定理
def uniform_2_normal():
    m = 100
    s = 0
    for j in range(m):
        s += rd.uniform(0, 1)
    u = m * 0.5
    d = m / 12.0
    return (s - u) / (d ** 0.5)


# 把n个骰子扔在地上，所有骰子朝上一面的点数之和为S
# 输入n，打印出S的所有可能的值出现的概率
def probability(n):
    dic = dict()
    dic[1] = dict()
    for i in range(1, 7):
        dic[1][i] = round(1 / 6, 5)

    def helper(n):
        if n in dic:
            return dic[n]
        dic.setdefault(n, dict())
        for i in range(1, 7):
            last_dic = helper(n - 1)
            for v, p in last_dic.items():
                dic[n].setdefault(i + v, 0)
                dic[n][i + v] += round(1 / 6 * p, 5)
        return dic[n]

    return helper(n)


if __name__ == '__main__':
    print('\nshuffle')
    data = [rd.randint(0, 100) for _ in range(10)]
    print(shuffle(data))

    print('\n蓄水池')
    data = np.arange(100)
    print(reservoir_sampling(data, m=10))

    print('\n加权抽样')
    res = []
    for i in range(30):
        res.append(weighted_random(list(range(3)), [2, 2, 4]))
    print(Counter(res))

    print('\nrand2构造rand5')
    res = []
    for i in range(500):
        res.append(rand2to5())
    print(Counter(res))

    print('\n均匀分布生成正太分布')
    res = []
    n = 1000
    for i in range(n):
        res.append(uniform_2_normal())
    v = sum(res) / n
    s = sum([(v - i) ** 2 for i in res]) / n
    print(v, s)

    print('\nn个色子的和的概率')
    print(probability(2))
