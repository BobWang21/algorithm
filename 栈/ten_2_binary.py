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


def valid_parentheses(vals):
    pair_dic = {']': '[', '}': '{', ')': '('}
    post_set = set(pair_dic.keys())
    stack = []
    for val in vals:
        if val in post_set:
            if stack and stack.pop(-1) == pair_dic[val]:
                continue
            else:
                return False
        else:
            stack.append(val)
    return True if not stack else False


def longest_valid_parentheses(s):
    if not s:
        return

    return


if __name__ == '__main__':
    print('符合表达式是否合法')
    print(valid_parentheses(['{', '}', '(', '(']))
