#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 20:59:23 2017

@author: wangbao
"""


def ten_2_binary(value):
    '''
    stack 
    '''
    stack = []
    while value != 0:
        remainder = value % 2
        stack.append(remainder)
        value = value >> 1
    for i in range(len(stack)-1, -1, -1):
        print(i)
    return stack
