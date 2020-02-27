#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:33:13 2019

@author: wangbao
"""
import math
import random

pro = random.random()  # 0-1
print(pro)
randint = random.randint(0, 10)  # 包括端点
print(randint)
print(random.shuffle([1, 2, 3]))
print(random.choice([1, 2, 3]))  # 随机抽取一个点

a = [1, 2, 3, 4, 5]
a.remove(1)  # 移除第一个值
tail = a.pop()  # 按位置移除, 默认为尾部, 并返回
print('list pop: ', tail)
print('list.index', a.index(3))  # 查找数据在列表中的索引, 返回第一个
max(a)
a.sort(reverse=True)  # 原地更改
b = [3, 4]
a = a + b  # extend
a.insert(0, -1)  # 队首增加元素
a.append(3)  # 增加一个数
c = a + b  # 生成一个新的变量

dic = {1: 3}
dic[1] = dic.get(2, 0) + 1
a = float("inf")
math.isinf(a)

s = 'abc'
s.isalnum()  # 是否为数字或字母
s.isalpha()  # 判断是否为字母
s.isdigit()  # 判断是否为数字
''.join(['a', 'b', 'c'])  # 列表转换成字符串
