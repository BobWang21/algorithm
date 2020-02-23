#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:04:19 2017

@author: wangbao
"""

import numpy as np
import time
# 缓存
dim = 9999
array = np.zeros((dim, dim))

start = time.time()

for  i in range(0, dim):
    for  j in range(0, dim):
        array[i][j] += 1
end = time.time()   
print(end - start)   

array = np.zeros((dim, dim))
start = time.time()
for  i in range(0, dim):
    for  j in range(0, dim):
        array[j][i] += 1

end = time.time()   
print(end - start)  
# 并行
array = np.zeros(dim)

start = time.time()
for  i in range(0, 999999):
     array[0] += 1
     array[1] += 1
end = time.time()   
print(end - start)   

array = np.zeros((dim, 1000))
start = time.time()
for  i in range(0, 999999):
     array[0] += 1
     array[0] += 1
end = time.time()   
print(end - start)  



