#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:33:13 2019

@author: wangbao
"""


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
        return 0
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


if __name__ == '__main__':
    print('plus one')
    print(plus_one([9, 9]))

    print('字符串转数字')
    print(my_atoi('4193 with words'))
