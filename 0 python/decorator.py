#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 13:31:04 2017

@author: wangbao
"""


def use_logging(level):
    def decorator(func):
        print('lever=%i' % level)
        print('awesome')
        func()
        print('done')

    return decorator


def foo():
    print("i am  tom")


use_logging(level=2)(foo)
