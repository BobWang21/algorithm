#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:57:36 2017

@author: wangbao
"""


# debug

def make_bread(x):
    # pdb.set_trace()
    if (x > 10):
        x += 10
    if x > 20:
        x -= 5
    # pdb.set_trace()
    return x


print(make_bread(21))

'''
试下保存上面的脚本后运行之。你会在运行时马上进入debugger模式。现在是时候了解下debugger模式
下的一些命令了。

命令列表：

c: 继续执行
w: 显示当前正在执行的代码行的上下文信息
a: 打印当前函数的参数列表
s: 执行当前代码行，并停在第一个能停的地方（相当于单步进入）
n: 继续执行到当前函数的下一行，或者当前行直接返回（单步跳过）
单步跳过（next）和单步进入（step）的区别在于， 单步进入会进入当前行调用的函数内部并停在里面， 
而单步跳过会（几乎）全速执行完当前行调用的函数，并停在当前函数的下一行。
'''
