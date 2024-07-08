#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 进制转换
# 括号匹配
# 单调栈 适用于结果和数组当前值的前后缀相关的问题


def ten_2_binary(value):
    stack = []
    while value:
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


# 1047 循环删除字符串中的连续重复字母
# 类似俄罗斯方块的消去
# 相当于指针 用到了stack最后一个数
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


# 20 有效的括号
def valid_parentheses(s):
    pair_dic = {']': '[',
                '}': '{',
                ')': '('}
    stack = []
    for c in s:
        if c not in pair_dic:
            stack.append(c)
        elif stack and stack[-1] == pair_dic[c]:
            stack.pop(-1)
        else:
            return False
    return not stack


# 1249 移除无效括号
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


# 剑指Offer 09.
# 用两个栈实现队列
class CQueue:
    def __init__(self):
        self.A = []
        self.B = []

    def appendTail(self, value):
        self.A.append(value)

    def deleteHead(self):
        if self.B:
            return self.B.pop()
        if not self.A:
            return -1
        while self.A:
            self.B.append(self.A.pop())
        return self.B.pop()


# 155 用两个栈实现最小栈
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


# 单调栈
# 496. 下一个更大元素 I
# Input: nums1 = [4, 1, 2], nums2 = [1, 3, 4, 2].
# Output: [-1, 3, -1]
def next_greater_element(nums1, nums2):
    dic = dict()
    stack = []

    for i, v in enumerate(nums2):
        while stack and stack[-1] < v:
            v1 = stack.pop(-1)
            dic[v1] = v
        stack.append(v)

    res = [-1] * len(nums1)
    for i, v in enumerate(nums1):
        if v in dic:
            res[i] = dic[v]
    return res


# 739 需要多少天 气温会升高
def daily_temperatures(t):
    if not t:
        return
    res = [0] * len(t)
    stack = []
    for i, v in enumerate(t):
        while stack and t[stack[-1]] < v:
            j = stack.pop(-1)
            res[j] = i - j
        stack.append(i)
    return res


# 42 接雨水
def trap(height):
    n = len(height)
    if n <= 2:
        return 0
    stack = []
    res = 0
    for i, v in enumerate(height):
        while stack and height[stack[-1]] < v:
            j = stack.pop(-1)
            if stack:
                h = min(height[stack[-1]], v) - height[j]
                w = i - stack[-1] - 1
                res += h * w
        stack.append(i)
    return res


# 71. 简化路径
def simplify_path(path):
    if not path:
        return '/'
    stack, res = [], '/'
    string = path.split('/')
    for c in string:
        if not c or c == '.':
            continue
        if c == '..':
            if stack:
                stack.pop(-1)
        else:
            stack.append(c)
    res += '/'.join(stack)
    return res


# 150 波兰表达式
def eval_RPN(tokens):
    stack = []
    for c in tokens:
        if c not in {'+', '-', '*', '/'}:
            stack.append(int(c))
            continue
        c2 = stack.pop(-1)
        c1 = stack.pop(-1)
        if c == '+':
            s = c1 + c2
        elif c == '-':
            s = c1 - c2
        elif c == '*':
            s = c1 * c2
        else:
            s = int(c1 / c2)
        stack.append(s)
    return stack[-1]


# 227. 基本计算器 II
# 5-6*3 = 5+(-6)*3
def calculate1(s):
    if not s:
        return 0

    pre_sign, num, stack = '+', 0, []  # stack[-1](a) pre_sign(*) num(b) 保留了上一个操作
    s += '+'

    for ch in s:
        if ch.isspace():  # 排除空格
            continue
        if ch.isdigit():  # 可能大于10
            num = num * 10 + int(ch)
            continue
        if pre_sign == '+':
            stack.append(num)
        elif pre_sign == '-':
            stack.append(-num)
        elif pre_sign == '*':
            v = stack.pop(-1)
            stack.append(v * num)
        else:
            v = stack.pop(-1)
            stack.append(int(v / num))  # 注意负数的优先级

        num = 0
        pre_sign = ch

    return sum(stack)


# (4 + (1-2))
def calculate2(s):
    stack = []
    num = 0
    res = 0  # For the on-going result
    sign = 1  # 1 means positive, -1 means negative
    s += '+'
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == '+':
            res += sign * num
            sign = 1
            num = 0
        elif ch == '-':
            res += sign * num
            sign = -1
            num = 0
        elif ch == '(':
            stack.append(res)
            stack.append(sign)
            sign = 1
            res = 0
        elif ch == ')':
            res += sign * num
            res *= stack.pop()  # stack pop 1, sign
            res += stack.pop()  # stack pop 2, operand
            num = 0
    return res + sign * num


# 带括号的(+ - * /)
def calculate3(s):
    if not s:
        return 0

    def helper(queue):
        sign, num, stack = '+', 0, []
        while queue:
            ch = queue.pop(0)
            if ch.isdigit():
                num = num * 10 + int(ch)
                continue
            elif ch == '(':  # 递归
                num = helper(queue)  # 可以继续处理
            else:  # '+-*/)'
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    v = stack.pop(-1)
                    stack.append(v * num)
                else:
                    v = stack.pop(-1)
                    if v < 0:
                        stack.append(-((-v) // num))  # 注意负数的优先级
                    else:
                        stack.append(v // num)
                if ch == ')':
                    break
                num = 0
                sign = ch
        return sum(stack)

    queue = [c for c in s if not c.isspace()]
    queue.append('+')
    return helper(queue)


# 946 验证栈序列
def validate_stack_sequences(pushed, popped):
    stack = []
    while pushed:
        stack.append(pushed.pop(0))
        while stack and stack[-1] == popped[0]:
            stack.pop(-1)
            popped.pop(0)
    return False if stack else True


if __name__ == '__main__':
    print('\n下一个比其大数值')
    print(next_greater_element([4, 1, 2], [1, 3, 4, 2]))

    print('\n下一个天气比当前热')
    print(daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]))

    print('\n十进制转二进制')
    print(ten_2_binary(10))

    print('\n表达式是否合法')
    print(valid_parentheses(['{', '}', '(', ')']))

    print('\n波兰表达式')
    token = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    print(eval_RPN(token))

    print('\n计算器')
    print(calculate1('1 + 1 * 5 * 4 - 9 + 45'))
    print(calculate2('(1+(4+5+2)-3)+(6+8)'))
    print(calculate3('(2+6* 3+5- (3*14/7+2)*5)+3'))  # -12

    print('\n是否为出栈路径')
    print(validate_stack_sequences([1, 2, 3, 4, 5], [4, 3, 5, 1, 2]))

    print('\n两个栈实现队列')
    # queue = CQueue()
    # queue.push(3)
    # queue.push(4)
    # queue.push(5)
    #
    # queue.pop()
    # queue.push(6)
    # for _ in range(3):
    #     print(queue.pop())
