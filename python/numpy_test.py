#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:22:36 2017

@author: wangbao
"""

import numpy as np


a = np.array([1, 1, 1])
b = np.array([2, 2, 2])

# vertical 竖直方向
c = np.vstack((a, b))
print (c)

# horizontal 
d = np.hstack((a, b))
print (d)

# norm norm(b, )
print(np.linalg.norm(a))
print(np.linalg.norm(a, np.inf))

# sum
print(np.sum(a))
# 0 竖直方向, 1 水平方向
print(np.sum(a, axis=0))

print(np.sum(a, axis=0, keepdims=True))
###############

a = np.arange(12).reshape((3, 4))

# Return a contiguous flattened array.
print(a.ravel())
b = a.T
print('对应元素相乘: \n', a*a)
# array 要进行矩阵运算需要使用np.dot
print('矩阵相乘: \n', np.dot(a, b))

###############
print(1 / np.array([2, 3, 4]))

###############

 
################SORT
# numpy.argsort(a, axis=-1, kind='quicksort', order=None)

array = np.array([4, 2, 7, 1])
print('array: \n', array)
# 升序排序, 对应原序列的index
order = array.argsort()
print('order: \n', order)
rank = order.argsort()
print('rank: \n', rank)

# 复制
centers_old = np.array([[1, 2], [3, np.nan]]).copy()
# 是否为空
np.isnan(centers_old).any()
