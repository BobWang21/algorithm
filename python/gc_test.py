#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:08:01 2017

@author: wangbao
"""
import gc


class ClassA():
    def __init__(self, a=10):
        self.a = a


print('=' * 10, 'Garbage collection', '=' * 10)
print('Garbage collection thresholds: ', gc.get_threshold())

gc.set_threshold(20, 10, 10)
print('Update collection thresholds: ', gc.get_threshold())

collected = gc.collect()
print("Garbage collector: collected %d objects." % (collected))

print('Active 10')
l = []
for i in range(20):
    x = ClassA()
    x.a = i
    l.append(x)
    print('Active:', gc.get_count())

print('=' * 10, 'Test class', '=' * 10)
print('类建立前: ', gc.get_count())
a = ClassA(2)
print('类建立后: ', gc.get_count())
del a
print('类删除后: ', gc.get_count())

print('=' * 10, 'Test cycle reference', '=' * 10)
for i in range(10):
    x = ClassA()
    y = ClassA()
    x.next = y
    y.next = x
    print('建立循环引用:', gc.get_count())

    del x
    del y
    print('删除循环引用', gc.get_count())

print('=' * 10, '清除垃圾', '=' * 10)
collected = gc.collect()
print("Garbage collector: collected %d objects." % (collected))

print('=' * 10, 'Test Interger', '=' * 10)
for i in range(10):
    s = 'a'
    print(gc.get_count())

print()
e = 7
print('e 建立后', gc.get_count())

collected = gc.collect()
print("Garbage collector: collected %d objects." % (collected))

print('=' * 10, '循环中一点', '=' * 10)
print('建立循环引用:', gc.get_count())
a = []
b = []
a.append(b)
b.append(a)
del b
print('试图删除b后', gc.get_count())

"""
执行上面这段代码，输出结果会让你大吃一惊，1的引用数量远超你的理解，可能多达几百个，
这是因为小整数和短字符串在python中会被缓存，以便重复使用，因此，他们的引用技术并不是
"""
