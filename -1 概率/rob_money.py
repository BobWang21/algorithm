#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:18:21 2017

@author: wangbao
"""

import numpy as np


def rob(money, num_per):
    cents = 100 * money
    # 平均大于1分, 总金额大于1分
    if cents / num_per < 1  or cents < 1:
        return False
    # 生成概率为1 的矩阵
    weight = np.random.rand(num_per, )
    weight = weight / np.sum(weight)
    # 每个人分的钱
    per_cent = cents * weight
    
    # 舍弃小数部分, 不足1的, 补为1
    # 可能会出现0， 0， 0， max-1这种极端情况
    other = 0
    best = np.argmax(per_cent)
    for i, cent in enumerate (per_cent):
        if i == best:
            continue
        else:
            mycent = int(cent)
            if mycent == 0:
                per_cent[i] = 1
            else:
                per_cent[i] = mycent
            other += per_cent[i]
    # 总金额减去其他的
    per_cent[best] = cents - other
    per_money = per_cent / 100
    return per_money
    
    