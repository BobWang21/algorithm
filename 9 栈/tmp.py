#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 20:59:23 2017

@author: wangbao
"""


def ten_2_binary(value):
    stack = []
    while value != 0:
        remainder = value % 2
        stack.append(remainder)
        value = value >> 1
    stack.reverse()
    return stack


# 数字reverse 123 -> 321
def reverse(x):
    if x < 0:
        return -reverse(-x)
    stack = []
    while x > 0:
        re = x % 10
        stack.append(re)
        x = x // 10
    res = 0
    i = 0
    while stack:
        v = stack.pop(-1)
        res += v * 10 ** i
        i += 1
    return res if res < 2 ** 31 else 0


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


def eval_RPN(tokens):
    stack = []
    for c in tokens:
        if c not in {'+', '-', '*', '/'}:
            stack.append(int(c))
        else:
            c2 = stack.pop(-1)
            c1 = stack.pop(-1)
            if c == '+':
                s = c1 + c2
            elif c == '-':
                s = c1 - c2
            elif c == '*':
                s = int(c1 * c2)
            else:
                s = int(c1 / c2)
            stack.append(s)
    return stack[-1]


if __name__ == '__main__':
    print('十进制转二进制')
    print(ten_2_binary(10))
    print('表达式是否合法')
    print(valid_parentheses(['{', '}', '(', ')']))

    print('波兰表达式')
    token = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    print(eval_RPN(token))
