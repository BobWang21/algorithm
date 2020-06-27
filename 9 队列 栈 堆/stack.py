#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
    k = 1
    while stack:
        v = stack.pop(-1)
        res += v * k
        k *= 10
    return res if res < 2 ** 31 else 0


def valid_parentheses(s):
    pair_dic = {']': '[', '}': '{', ')': '('}
    stack = []
    for c in s:
        if c in pair_dic:
            if stack and stack.pop(-1) == pair_dic[c]:
                continue
            else:
                return False
        else:
            stack.append(c)
    return True if not stack else False


# 判断括号是否合法 只含有()
def is_balanced_parentheses(s):
    balance = 0
    for char in s:
        if char == "(":
            balance = balance + 1
        if char == ")":
            balance = balance - 1
        if balance < 0:
            return False
    return balance == 0


# 多余的'(', ')'
def useless_parentheses(s):
    if not s:
        return 0, 0
    l = r = 0
    for ch in s:
        if ch == "(":
            l += 1  # '('的数目
        elif l:
            l -= 1  # 消去一个 '('
        else:
            r += 1  # 多一个 ')'
    return l, r


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


# 移除无效括号
def min_remove_to_make_valid(s):
    if not s:
        return
    n = len(s)
    valid = set()
    stack = []

    for i in range(n):
        c = s[i]
        if c.isalpha():
            valid.add(i)
        elif c == '(':
            stack.append(i)
        elif stack:
            j = stack.pop(-1)
            valid.add(i)
            valid.add(j)

    res = ''
    for i in range(n):
        if i in valid:
            res += s[i]
    return res


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


# 循环删除字符串中的连续重复字母
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


# 1- 2 / 50*4-9+45
# 1 + 2 - 3
def calculate(s):
    if not s:
        return

    lists = []
    v = None
    for c in s:
        if c in {' ', '+', '-', '*', '/'}:
            if c == ' ':
                if v is not None:
                    lists.append(v)
                    v = None
            elif v is not None:
                lists.append(v)
                lists.append(c)
                v = None
            else:
                lists.append(c)
        else:
            v = 10 * v + int(c) if v is not None else int(c)
    if v is not None:
        lists.append(v)
    # print(lists)
    res = 0
    stack1, stack2 = [], []

    for c in lists:
        if c in {'*', '/', '+', '-'}:
            stack2.append(c)
        else:
            if stack2 and stack2[-1] in {'*', '/'}:
                operate = stack2.pop(-1)
                if operate == '*':
                    v = int(c) * stack1.pop(-1)
                else:
                    v = stack1.pop(-1) // int(c)
                stack1.append(v)

            else:
                stack1.append(int(c))
    # print(stack1, stack2)
    while stack2:
        v = stack1.pop(-1)
        res += v if stack2.pop(-1) == '+' else -v
    res += stack1.pop(-1)
    return res


# 当遇到第二个运算符开始运算  1-2 / 50*4-9+45
def calculate2(s):
    if not s:
        return 0
    sign, stack, num, n = '+', [], 0, len(s)  # 保存之前的当符号 和 最近一个数值
    for i in range(n):
        if s[i].isdigit():
            num = num * 10 + int(s[i])
        if (not s[i].isdigit() and not s[i].isspace()) or i == n - 1:  # 符号 或 最后一个符号
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            elif sign == '*':
                stack.append(stack.pop(-1) * num)
            else:
                v = stack.pop(-1)
                if v // num < 0 and v % num:  # 负数
                    stack.append(v // num + 1)
                else:
                    stack.append(v // num)
            sign = s[i]
            num = 0
    return sum(stack)


if __name__ == '__main__':
    print('\n十进制转二进制')
    print(ten_2_binary(10))

    print('\n表达式是否合法')
    print(valid_parentheses(['{', '}', '(', ')']))

    print('\n波兰表达式')
    token = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    print(eval_RPN(token))

    print('\n计算器')
    print(calculate('1 + 1 * 5 * 4 - 9 + 45'))
    print(calculate2('1 + 1 * 5 * 4 - 9 + 45'))

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
