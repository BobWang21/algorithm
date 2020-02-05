#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:28:25 2017

@author: wangbao
"""
import os

os.chdir('/Users/wangbao/Data')


class FileContextManager():
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.file = open(self.name, 'w')
        return self.file

    # 异常的类型, 值， trace_back
    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.file.close()
        if exc_type is None:
            print('Normally exit')
        else:
            print('Raise Error', exc_type)


# f = FileWriteManager.__enter__()
# f = self.file

with FileContextManager('hello.txt') as f2:
    f2.write('context manager..')

with FileContextManager('hello.txt') as f:
    f.write('context manager..')
    raise TypeError
