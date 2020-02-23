#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 18:23:44 2017

@author: wangbao
"""


class Iter():
    def __init__(self, n):
        self.i = 0
        self.n = n

    # The __iter__ method is what makes an object iterable.
    def __iter__(self):
        print('call __iter__')
        return self

    def __next__(self):
        print('call __next__')
        self.i += 1
        # 必须有终止条件
        if self.i > self.n:
            raise StopIteration
        return self.i


a = Iter(6)
print(next(a))


# 所有迭代环境, 优先使用__iter__ 然后再尝试__getitem__ 
# 可迭代对象
class IterThird():
    def __init__(self, lists):
        self.lists = lists
        self.len = len(lists)
        self.cursor = -1

    '''   
    def __iter__(self):
        return self
    
    def __next__(self):
        self.cursor += 1
        if self.cursor +1 > self.len:
            raise StopIteration
        return self.lists[self.cursor]
    '''

    def __getitem__(self, index):
        if index > self.len:
            raise StopIteration
        return self.lists[index] + 100


# 第二种方式
class IterSecond():
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        cursor = 1
        while cursor < self.n:
            yield cursor
            cursor += 1
