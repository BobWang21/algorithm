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


# 波兰表达式
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


class Queue():
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, v):
        self.stack1.append(v)

    def pop(self):
        if not self.stack2:
            if not self.stack1:
                raise Exception('X')
            else:
                while self.stack1:
                    self.stack2.append(self.stack1.pop(-1))
        return self.stack2.pop(-1)


# 用两个栈实现最小栈
class MinStack():
    def __init__(self):
        self.stack = []
        self.min = []

    def push(self, v):
        self.stack.append(v)
        if self.min:
            if self.min[-1] <= v:
                self.min.append(self.min[-1])
            else:
                self.min.append(v)
        else:
            self.min.append(v)

    def pop(self):
        if self.stack and self.min:
            self.stack.pop(-1)
            self.min.pop(-1)

    def get_min(self):
        return self.min[-1]


# 判断一个路径是否为出栈路径
def validate_stack_sequences(pushed, popped):
    if len(pushed) != len(popped):
        return False
    stack = []
    while pushed:
        v = pushed.pop(0)  # 先进栈
        stack.append(v)
        while stack and stack[-1] == popped[0]:
            stack.pop(-1)
            popped.pop(0)
    return False if stack else True


# 删除字符串中的重复字母
def remove_duplicates(S):
    if not S:
        return S
    stack = []
    for c in S:
        if stack and stack[-1] == c:
            stack.pop(-1)
        else:
            stack.append(c)

    return ''.join(stack)


if __name__ == '__main__':
    print('\n十进制转二进制')
    print(ten_2_binary(10))

    print('\n表达式是否合法')
    print(valid_parentheses(['{', '}', '(', ')']))

    print('\n波兰表达式')
    token = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    print(eval_RPN(token))

    print('\n两个栈实现队列')
    queue = Queue()
    queue.push(3)
    queue.push(4)
    queue.push(5)

    queue.pop()
    queue.push(6)
    for _ in range(3):
        print(queue.pop())

    print('\n是否为出栈路径')
    print(validate_stack_sequences([1, 2, 3, 4, 5], [4, 3, 5, 1, 2]))
