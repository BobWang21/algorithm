#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:38:12 2017

@author: wangbao
"""

from functools import wraps
import time


def logger(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = fn(*args, **kwargs)
        te = time.time()
        print ("function      = {0}".format(fn.__name__))
        print ("    arguments = {0} {1}".format(args, kwargs))
        print ("    return    = {0}".format(result))
        print ("    time      = %.6f sec" % (te-ts))
        return result
    return wrapper
 
@logger
def multipy(x, y):
    return x * y
 
@logger
def sum_num(n):
    s = 0
    for i in range(n+1):
        s += i
    return s
 
print (multipy(2, 10))
print (sum_num(100))
print (sum_num(10000000))