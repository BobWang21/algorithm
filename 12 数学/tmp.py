#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 不合法
0 幂
大于10进位
"""


def ten_2_binary(value):
    stack = []
    while value > 0:
        remainder = value % 2
        stack.append(remainder)
        value = value // 2
    stack.reverse()
    return stack


# 123 -> 321
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


# [1, 2, 3] -> [1, 2, 4]
def plus_one(digits):
    if digits[-1] < 9:
        digits[-1] += 1
        return digits
    if len(digits) == 1 and digits[0] == 9:  # base
        return [1, 0]
    return plus_one(digits[:-1]) + [0]


def my_atoi(str):
    def helper(str, ans):
        if not str:
            return 0
        if str[0] == ' ':
            return helper(str[1:], ans)
        if str[0].isalpha():
            return 0
        if str[0] == '-':
            if len(str) > 1:
                if str[1].isdigit():
                    return -helper(str[1:], ans)
                else:
                    return 0
            else:
                return 0

        for v in str:
            if v.isdigit():
                ans = ans * 10 + int(v)
            else:
                return ans
        return ans

    res = helper(str, 0)
    if res < -2147483648:
        return -2147483648
    if res > 2147483648:
        return 2147483648
    return res


def is_power_of_2(n):
    if n == 0:  # base
        return False
    if n == 1:
        return True
    if n % 2:
        return False
    return is_power_of_2(n // 2)


# 不用除法
def divide(dividend, divisor):
    sig = True if dividend * divisor > 0 else False  # 判断二者相除是正or负
    dividend, divisor = abs(dividend), abs(divisor)  # 将被除数和除数都变成正数
    res = 0  # 用来表示减去了多少个除数，也就是商为多少
    while divisor <= dividend:  # 当被除数小于除数的时候终止循环
        tmp_divisor, count = divisor, 1  # 倍增除数初始化
        while tmp_divisor <= dividend:  # 当倍增后的除数大于被除数重新，变成最开始的除数
            dividend -= tmp_divisor
            res += count
            count <<= 1  # 向左移动一位
            tmp_divisor <<= 1  # 更新除数(将除数扩大两倍)
    res_value = res if sig else -res  # 给结果加上符号
    return max(min(res_value, 2 ** 31 - 1), -2 ** 31)


# Input: 19
# Output: true
# Explanation:
# 12 + 92 = 82
# 82 + 22 = 68
# 62 + 82 = 100
# 12 + 02 + 02 = 1
# 如果不是1 则会出现循环
def is_happy(n):
    def helper(n):
        total = 0
        while n > 0:
            reminder = n % 10
            total += reminder ** 2
            n = n // 10
        return int(total)

    if n == 1:
        return True
    seen = set()
    while n != 1:
        if n in seen:
            return False
        seen.add(n)
        n = helper(n)
    return True


def add_digits(n):
    def helper(n):
        total = 0
        while n > 0:
            reminder = n % 10
            total += reminder
            n = n // 10
        return int(total)

    if n == 1:
        return True
    while n > 10:
        n = helper(n)
    return n


def is_ugly(num):
    if num <= 0:
        return False
    for x in [2, 3, 5]:
        while num % x == 0:
            num = num / x
    return num == 1


def is_ugly2(num):
    if num == 0:
        return False
    if num == 1:
        return True
    if num % 2 == 0:
        return is_ugly2(num % 2)
    elif num % 3 == 0:
        return is_ugly2(num % 3)
    elif num % 5 == 0:
        return is_ugly2(num % 3)
    else:
        return False


def get_ugly_number(n):
    res = [1]
    num = 1
    min2 = min3 = min5 = 0
    while num < n:
        value = min(2 * res[min2], 3 * res[min3], 5 * res[min5])
        res.append(value)
        while 2 * res[min2] <= res[-1]:
            min2 += 1
        while 3 * res[min3] <= res[-1]:
            min3 += 1
        while 5 * res[min5] <= res[-1]:
            min5 += 1
        num += 1
    return res[-1]


# 阶层后0的个数 因式分解 2^(m) * 3^(k) * 5^(n)... min(m, n) 含有5的个数
def zero_nums(n):
    res = 0
    while n:
        res += n // 5
        n = n // 5
    return res


def hamming_distance(x, y):
    return bin(x ^ y).count('1')


# 求平方根
def sqrt(n):
    if n == 1:
        return 1
    l, r = 1, n
    while l <= r:
        m = (l + r) / 2.0
        s = m * m
        if s == n:
            return m
        elif s < n:
            l = m + 0.1
        else:
            r = m - 0.1
    return r


def sqrt2(n):
    def fun(x):
        return x ** 2 - n  # 开口向上

    if n == 1:
        return 1
    x = n // 2
    f = fun(n)
    while abs(f - n) < 0.1:
        k = 2 * x
        b = f - k * x
        x = -b / k
        f = fun(x)
    return abs(x)


# 计算素数个数
def count_primes(n):
    if n < 2:
        return 0
    res = [True] * n
    res[0] = res[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        for j in range(2, (n - 1) // i + 1):
            res[i * j] = False
    return sum(res)


# 最大公约数
def common_greatest_divisor(a, b):
    if a < b:
        a, b = b, a
    if not a % b:
        return b
    return common_greatest_divisor(b, a % b)


if __name__ == '__main__':
    print('\n十进制转2进制')
    print(ten_2_binary(24))

    print('\nplus one')
    print(plus_one([9, 9]))

    print('\n字符串转数字')
    print(my_atoi('4193 with words'))

    print('\n求一个数的平方根')
    print(sqrt(17))

    print('\n计算素数个数')
    print(count_primes(10))

    print('\n数字翻转')
    print(reverse(123))

    print('\n丑数')
    print(get_ugly_number(10))

    print('\n阶层后0的个数')
    print(zero_nums(20))

    print('\n最大公约数')
    print(common_greatest_divisor(12, 38))
