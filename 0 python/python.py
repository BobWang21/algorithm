#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:33:13 2019

@author: wangbao
"""
import math
import random

if __name__ == '__main__':
    print('数字操作')
    print(4 / 2)  # 2.0,  2.0 =  2
    a = float('inf')
    math.isinf(a)
    print(isinstance([], list))  # 判断类型
    print(2 << 1)  # 左移

    print('随机')
    pro = random.random()  # 0-1
    print(pro)
    randint = random.randint(0, 10)  # 包括端点
    print(randint)
    print(random.shuffle([1, 2, 3]))
    print(random.choice([1, 2, 3]))  # 随机抽取一个点

    print('列表')
    a = [1, 2, 3, 4, 5]
    a.remove(1)  # 移除第一个值
    tail = a.pop()  # 按位置移除, 默认为尾部, 并返回
    print('list pop: ', tail)
    print('list.index', a.index(3))  # 查找数据在列表中的索引, 返回第一个
    max(a)
    a.sort(reverse=True)  # 原地更改
    b = [3, 4]
    a.insert(0, -1)  # 队首增加元素
    a.append(3)  # 增加一个数
    c = a + b  # 生成一个新的变量

    print('字典')
    dic = {1: 3}
    dic[1] = dic.get(2, 0) + 1
    for k, v in dic.items():
        print(k, v)

    print('字符串')
    s = 'abc'
    s.isalnum()  # 是否为数字或字母
    s.isalpha()  # 判断是否为字母
    s.isdigit()  # 判断是否为数字
    ''.join(['a', 'b', 'c'])  # 列表转换成字符串
    print(chr(97))  # 数字转字符
    print(ord('a'))  # 字母对应的数字

    print('集合')
    a = ['a', 'b', 'c']
    print(set(a))
    b = {'abc'}
    A = {1, 2, 3}
    B = {3, 4, 5}
    C = A | B  # A与B的并集
    D = A & B  # A与B的交集
    print(C, D)
    E = A - B  # B在A的补集 {1, 2}
    F = B - A  # A在B的补集 {4, 5}
    G = A ^ B  # A与B的差集 {1, 2, 4, 5}
    print(E, F, G)
