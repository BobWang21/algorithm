#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 22:26:55 2017

@author: wangbao
"""
import pdb


def yield_test():
    yield 'first content'
    yield 'second content'


pdb.set_trace()
for i in yield_test():
    pdb.set_trace()
    print(i)
    print('i am not yield')


####################################################   
def gen():
    print('--start')
    yield 1
    print('--middle')
    yield 2
    print('--stop')


for i in gen():
    print('_' * 10)
    print(i)

####################################################    
from contextlib import contextmanager


@contextmanager
def tag(name):
    print("<%s>" % name)
    yield
    print("</%s>" % name)


with tag("h1"):
    print("foo")


@contextmanager
def tag(name):
    print("<%s>" % name)
    yield 'hello'
    print("</%s>" % name)


with tag("h1") as h:
    print("foo")
