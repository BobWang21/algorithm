#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 不合法
0 幂
大于10进位
"""

import heapq as hq


# 求平方根
def sqrt1(n):
    if n == 1:
        return 1
    l, r = 1, n
    while l <= r:
        mid = l + (r - l) // 2
        s = mid * mid
        if s == n:
            return mid
        if s < n:
            l = mid + 1
        else:
            r = mid - 1
    return r


# 带精度的求平方根
def sqrt2(n, precision):
    if n == 1:
        return 1
    l, r = 0.0, n * 1.0
    while l <= r:
        mid = l + (r - l) / 2.0
        s = mid * mid
        if abs(s - n) <= precision:
            return mid
        if s < n:
            l = mid
        else:
            r = mid
    return r


def sqrt3(n):
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


# 最大公约数 辗转相除法
# 294 = 84 * 3 + 42, 84 = 42 * 2 294 => 42 * 2 * 3 + 42
def common_greatest_divisor(a, b):
    if a < b:
        a, b = b, a
    if not a % b:
        return b
    return common_greatest_divisor(b, a % b)


# 166. 分数到小数
def fraction_to_decimal(numerator, denominator):
    if numerator == 0:
        return "0"
    # 首先判断结果正负, 异或作用就是 两个数不同 为 True 即 1 ^ 0 = 1 或者 0 ^ 1 = 1
    if (numerator > 0) ^ (denominator > 0):  # 防止溢出 不使用*,
        res = '-'
    else:
        res = ''
    numerator, denominator = abs(numerator), abs(denominator)
    # 判读到底有没有小数
    a, b = divmod(numerator, denominator)
    res += str(a)
    # 无小数
    if not b:
        return res

    res += "."
    # 处理余数
    # 把所有出现过的余数记录下来
    loc = {b: len(res)}  # 先把余数记录到词典中
    while b:
        b *= 10
        a, b = divmod(b, denominator)
        res += str(a)
        # 余数前面出现过,说明开始循环了,加括号
        if b in loc:
            i = loc[b]
            res = res[:i] + '(' + res[i:] + ')'
            break
        loc[b] = len(res)
    return res


# 204 计算小于n的素数个数
def count_primes(n):
    if n < 2:
        return 0
    res = [True] * n
    res[0] = res[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        for j in range(2, (n - 1) // i + 1):
            res[i * j] = False
    return sum(res)


def ten_2_binary(value):
    stack = []
    while value > 0:
        remainder = value % 2
        stack.append(remainder)
        value = value // 2
    stack.reverse()
    return stack


# 7. 整数反转 123->321
def reverse(x):
    y, pre = abs(x), 0
    # om 防止重复计算
    if x >= 0:
        om = (1 << 31) - 1  # 必须加()
    else:
        om = 1 << 31
    while y:
        pre = pre * 10 + y % 10  #
        if pre > om:
            return 0
        y //= 10
    return pre if x > 0 else -pre


def reverse2(x):
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


# 66. 加一 [1, 2, 3] -> [1, 2, 4]
def plus_one(digits):
    if len(digits) == 1 and digits[0] == 9:  # base
        return [1, 0]
    if digits[-1] < 9:
        digits[-1] += 1
        return digits
    return plus_one(digits[:-1]) + [0]


def plus_one2(digits):
    n = len(digits)
    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
        i -= 1
    return digits if digits[0] else [1] + digits


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


def is_power_of_two1(n):
    if n == 0:  # base
        return False
    if n == 1:
        return True
    if n % 2:
        return False
    return is_power_of_two1(n // 2)


# 29 两个数相除 不用除法
def divide(dividend, divisor):
    sign = 1
    if (dividend > 0) ^ (divisor > 0):
        sign = -1

    dividend, divisor = abs(dividend), abs(divisor)
    om = (1 << 31) - 1 if sign > 0 else (1 << 31)

    res = 0
    divisor_old = divisor
    while dividend >= divisor_old:
        divisor, k = divisor_old, 1
        while dividend >= divisor:
            dividend -= divisor
            res += k
            k <<= 1
            divisor <<= 1

    return res * sign if res <= om else (1 << 31) - 1


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

    seen = set(n)
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
    if num % 3 == 0:
        return is_ugly2(num % 3)
    if num % 5 == 0:
        return is_ugly2(num % 3)
    else:
        return False


# 三指针
def nth_ugly_number(n):
    if n == 1:
        return 1

    res = [1]
    i_2 = i_3 = i_5 = 0

    for i in range(n - 1):
        nxt_2 = res[i_2] * 2
        nxt_3 = res[i_3] * 3
        nxt_5 = res[i_5] * 5
        v = min(nxt_2, nxt_3, nxt_5)
        res.append(v)

        if nxt_2 == v:
            i_2 += 1
        if nxt_3 == v:
            i_3 += 1
        if nxt_5 == v:
            i_5 += 1
    return res[-1]


# 堆
def nth_ugly_number2(k):
    s = set()
    heap = [1]
    while True:
        v = hq.heappop(heap)
        if v not in s:
            s.add(v)
            hq.heappush(heap, v * 2)
            hq.heappush(heap, v * 3)
            hq.heappush(heap, v * 5)
        if len(s) == k:
            return v


# 阶层后0的个数 因式分解 2^(m) * 3^(k) * 5^(n)... min(m, n) 含有5的个数
def zero_nums(n):
    res = 0
    while n:
        res += n // 5
        n = n // 5
    return res


# 循环将最后一位数字1变成0
def hamming_weight(n):
    cnt = 0
    while n:
        n &= n - 1  # 每次去除最后的1 [10, 01] [01, 00]
        cnt += 1
    return cnt


def hamming_distance(x, y):
    return bin(x ^ y).count('1')


# 2的幂 100的形式
def is_power_of_two2(n):
    return n > 0 and not (n & (n - 1))  # n和n-1的区别为n的最后一个1


# 268 给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，
# 找出 0 .. n 中没有出现在序列中的那个数
def missing_number(nums):
    if not nums:
        return 0

    n = len(nums)

    miss_value = n
    for i in range(n):
        miss_value ^= nums[i] ^ i

    return miss_value




# 获取数值对应的二进制 第k位取值 idx从0开始
def find_bit_k_value(val, k):
    return (val >> k) & 1


if __name__ == '__main__':
    print('\n求一个数的平方根')
    print(sqrt1(17))
    x = sqrt2(17, 0.1)
    print(x ** 2)

    print('\n最大公约数')
    print(common_greatest_divisor(12, 38))

    print('\n十进制转2进制')
    print(ten_2_binary(24))

    print('\nplus one')
    print(plus_one([9, 9]))

    print('\n字符串转数字')
    print(my_atoi('4193 with words'))

    print('\n计算素数个数')
    print(count_primes(10))

    print('\n数字翻转')
    print(reverse(1463847412))

    print('\n丑数')
    print(nth_ugly_number(10))

    print('\n阶层后0的个数')
    print(zero_nums(20))

    print('\n二进制1的个数')
    print(hamming_weight(12))

    print('\n二进制k位对应的的取值')
    print(find_bit_k_value(8, 3))
