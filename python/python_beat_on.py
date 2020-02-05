#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:54:43 2017

https://eastlakeside.gitbooks.io/interpy-zh/content/args_kwargs/Usage_args.html
"""
#

'''    
*args, **kwargs
*(元组)  **(字典)
此外它也可以用来做猴子补丁(monkey patching)。猴子补丁的意思是在程序运行时(runtime)修改某些代码。 
打个比方，你有一个类，里面有个叫get_info的函数会调用一个API并返回相应的数据。
如果我们想测试它，可以把API调用替换成一些测试数据
'''


###############################################
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)


test_var_args('yasoob', 'python', 'eggs', 'test')


def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} => {1}".format(key, value))


greet_me(name="yasoob")


def test(f_arg, *argv, **kwargs):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)
    for key, value in kwargs.items():
        print("{0} => {1}".format(key, value))


test('yasoob', 'python', 'eggs', 'test', tom='tom', jack='jack')


def test_args_kwargs(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)


# 现在使用 **kwargs:
kwargs = {"arg3": 3, "arg2": "two", "arg1": 5}
test_args_kwargs(**kwargs)

###############################################
# debug
import pdb


def make_bread(x):
    pdb.set_trace()
    if (x > 10):
        x += 10
    if x > 20:
        x -= 5
    pdb.set_trace()
    return x


print(make_bread(21))

for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n / x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')

###############################################
from functools import lru_cache


@lru_cache(maxsize=32)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


print([fib(n) for n in range(200)])

# test dic
import time

import matplotlib.pyplot as plt

used_time = []

dic = {}
for i in range(90000):
    start = time.time()
    dic.setdefault(i, 1)
    end = time.time()
    used_time.append(end - start)

plt.plot(used_time)

###############################################
# test dic 
lists = []
for i in range(90000):
    start = time.time()
    lists.append(i)
    end = time.time()
    used_time.append(end - start)

plt.plot(used_time)


def generator(n):
    lists = []
    for i in range(n):
        lists.append(i)
    return lists


def list_comprehensions(n):
    return [i for i in range(n)]


start = time.time()
generator(999999)
end = time.time()
print(end - start)

start = time.time()
list_comprehensions(999999)
end = time.time()
print(end - start)

###############################################
# List Comprehension vs Generator 
import sys

l = [n * 2 for n in range(9000)]  # List comprehension
g = (n * 2 for n in range(9000))  # Generator expression

print(type(l))  # <type 'list'>
print(type(g))  # <type 'generator'>

print(sys.getsizeof(l))  # 9032
print(sys.getsizeof(g))  # 80

print(l[4])  # 8
# print(g[4])   # TypeError: 'generator' object has no attribute '__getitem__'

for v in l:
    pass
for v in g:
    pass

##################### 随机 ##########################
import random
from random import choice

# randint(a, b) 包括端点
print('0-100之间随机随机数: ', random.randint(0, 100))
lists = [20, 16, 10, 5]
print('Orgin lists: ', lists)
# shuffle 
random.shuffle(lists)
print('shuffle list: ', lists)
# random select K
print('random select 2 ', lists[:2])
# choice
print('随机选择一个数', choice(lists))


##################### 异常 ##########################
def test_(x):
    # 如出错程序不继续运行
    assert x > 0, 'x must bigger than 0'
    return x


def test_1(x):
    if x > 0:
        return x
    else:
        raise ValueError('init_center_method must be `random` or `kmeans++`')


print(test_(-1))
print(test_1(-1))

"""
and：
x and y 返回的结果是决定表达式结果的值。
如果 x 为真，则 y 决定结果，返回 y ；
如果 x 为假，x 决定了结果为假，返回 x。
即如果都为真，则返回最后一个变量值； 如果为假，则返回第一个假值

or：
x or y 跟 and 一样都是返回决定表达式结果的值。
即如果都为假则返回最后一个值；如果为真，则返回第一个真值
"""
