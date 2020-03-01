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
    for i in range(len(stack) - 1, -1, -1):
        print(i)
    return stack


def valid_parentheses(s):
    pair_dic = {']': '[', '}': '{', ')': '('}
    stack = []
    for val in s:
        if val in pair_dic:
            if stack and stack.pop(-1) == pair_dic[val]:
                continue
            else:
                return False
        else:
            stack.append(val)
    return True if not stack else False


if __name__ == '__main__':
    print('表达式是否合法')
    print(valid_parentheses(['{', '}', '(', ')']))
