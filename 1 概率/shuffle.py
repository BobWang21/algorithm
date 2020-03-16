#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:50:36 2017

@author: wangbao
"""

import numpy as np
import random as rd


def shuffle(data):
    '''
    n个数中抽取1个数放在第一位，从剩下n-1个数中抽取一个放在第二位
    某个数被放在第二位的概率为 (n-1 / n)*(1 / n-1) = 1/n
    '''
    n = len(data)
    for i in range(n-1, -1, -1):
        index = rd.randint(0, i)
        data[i], data[index] = data[index], data[i] 
    return data


def reservoir_sampling(data, m):
    """
    m 个数
    """
    out = np.zeros(m)
    
    for i, value in enumerate(data):
        # 保存前n个数，保证至少有n个数 
        if i < m: 
            out[i] = value
        else:
             # 第k个数被选中概率为 n/k 
            k = i + 1
            if rd.randint(1, k) <= m:
                index = rd.randint(0, m-1)
                out[index] = value
    return out


def exam(n):
    '''
    随机生成n个不相邻的数
    '''
    out = []
    data = [i+1 for i in range(n)] 
    prior = -1
    i = 0
    time = 0
    while len(out) < n:
        index = rd.randint(0, n-1-i)
        print(index)
        if abs(prior - data[index]) != 1:
            prior = data[index]
            out.append(prior)
            data.remove(prior)
            i += 1
        time += 1
        if time > n**2:
            exam(n)
            break             
    return out

def random_p(p):
    if rd.randint(0, 1000) / 1000 <= p:
       return 1
    return 0


    
if __name__ == '__main__':
    data = [rd.randint(0, 100) for _ in range(30)]
    print(data)
    print(shuffle(data))
    data = np.arange(100)
    print(reservoir_sampling(data, n=10))  
    print(exam(100))      

