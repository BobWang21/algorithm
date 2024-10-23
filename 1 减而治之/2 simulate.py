#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict, Counter
import heapq as hq

'''
TopK 问题的几种解法
1 堆排序
2 对于未排序的 使用partition
3 二分查找 需要构造递增序列
4 桶排序
'''


# 魔术索引
# 有序数组, 可能含有重复值，找到nums[i]==i的最小索引
# 时间复杂度可能退化成o(n), 对比二分很好的例子
def find_magic_index(nums):
    def find(l, r):
        # 终止条件
        if l >= r:
            return -1
        mid = l + (r - l) // 2
        # 先左侧
        left = find(l, mid - 1)
        if left != -1:
            return left
        # 中间
        if nums[mid] == mid:
            return mid
        # 右侧
        return find(mid + 1, r)

    return find(0, len(nums) - 1)


# 计数排序
# 给定一个包含红色、白色和蓝色，n个元素的数组，
# 原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列
# 使用整数 0, 1 和 2 分别表示红色、白色和蓝色。
def sort_colors(nums):
    if not nums:
        return []
    count = [0] * 3
    for color in nums:
        count[color] += 1
    start = 0
    for i in range(3):
        for j in range(count[i]):
            nums[start] = i
            start += 1
    return nums


# 238. 除自身以外数组的乘积 左右各扫描一遍
def product_except_self(nums):
    n = len(nums)

    l = [1] * n

    for i in range(1, n):
        l[i] = l[i - 1] * nums[i - 1]

    r = 1
    for i in range(n - 1, -1, -1):
        l[i] *= r
        r = r * nums[i]

    return l


# 下一个排列
def next_permutation(nums):
    if not nums:
        return
    n = len(nums)

    i = n - 1
    while i > 0 and nums[i - 1] >= nums[i]:
        i -= 1
    if not i:
        nums.reverse()
        return nums

    k = i - 1
    while i < n and nums[i] > nums[k]:  # 找到最后一个大于该数的位置
        i += 1
    i -= 1  # 大于该数的最小值
    nums[k], nums[i] = nums[i], nums[k]

    # reverse
    l, r = k + 1, n - 1
    while l < i:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums


# 128 无序数组 最长连续区间
# 把数字及关系抽象为图
# 最大连通图问题 也可以使用并查集
def longest_consecutive(nums):
    if not nums:
        return 0

    dic = dict()
    for v in nums:
        dic[v] = False

    res = 1
    for v in dic.keys():
        if not dic[v]:
            continue
        cnt = 1
        while v in dic:  # 序
            dic[v] = True
            cnt += 1
            v += 1
        res = max(res, cnt)

    return res


# 众数 超过一半的数
# 摩尔投票法的核心就是一一抵消
def most_data(nums):
    if not nums:
        return
    n = len(nums)
    value = nums[0]
    cnt = 1
    for i in range(1, n):
        if cnt == 0:
            value = nums[i]
            cnt = 1
        if nums[i] == value:
            cnt += 1
        else:
            cnt -= 1
    return value


# n // 3
def majority_element(nums):
    if not nums:
        return []
    n = len(nums)

    v1 = v2 = nums[0]
    cnt1 = cnt2 = 0
    for num in nums:
        if num == v1:
            cnt1 += 1
            continue
        if num == v2:
            cnt2 += 1
            continue

        if not cnt1:
            cnt1 += 1
            v1 = num
            continue
        if not cnt2:
            cnt2 += 1
            v2 = num
            continue

        cnt1 -= 1
        cnt2 -= 1

    cnt1 = cnt2 = 0
    for num in nums:
        if num == v1:  # 可能v1 = v2
            cnt1 += 1
            continue
        if num == v2:
            cnt2 += 1
    res = []

    if cnt1 > n // 3:
        res.append(v1)

    if cnt2 > n // 3:
        res.append(v2)

    return res


def second_largest(nums):
    size = len(nums)
    if size < 2:
        return
    first = second = -float('inf')
    for i in range(size):
        if nums[i] > first:
            second = first
            first = nums[i]
            continue
        if nums[i] > second:
            second = nums[i]
    return second


# 区间合并
def merge(intervals):
    if not intervals or not intervals[0]:
        return []
    intervals.sort()
    res = []
    pre_end = -float('inf')
    for start, end in intervals:
        if start > pre_end:
            res.append([start, end])
            pre_end = end
        elif end <= pre_end:
            continue
        else:
            res[-1][1] = end
            pre_end = end

    return res


# 66. 加一 [1, 2, 3] -> [1, 2, 4]
def plus_one(digits):
    carry = 1
    n = len(digits)
    for i in range(n - 1, -1, -1):
        if digits[i] == 9 and carry == 1:
            digits[i] = 0
        else:
            digits[i] = digits[i] + carry
            carry = 0
    if carry == 1:
        digits = [1] + digits
    return digits


# 66. 加一 [1, 2, 3] -> [1, 2, 4]
def plus_one2(digits):
    if not digits:  # 0
        return [1]
    if digits[-1] < 9:  # 1-8
        digits[-1] += 1
        return digits
    return plus_one(digits[:-1]) + [0]  # 9


# num1 = "123", num2 = "456" 输出: "56088"
def add(num1, num2):
    m, n = len(num1), len(num2)
    i, j = m - 1, n - 1
    carry = 0
    s = ''
    while i >= 0 or j >= 0:
        v1 = int(num1[i]) if i >= 0 else 0
        v2 = int(num2[j]) if j >= 0 else 0
        v = v1 + v2 + carry
        s = str(v % 10) + s
        carry = v // 10
        i -= 1
        j -= 1
    if carry:
        s = str(carry) + s
    return s


def multiply(num1, num2):
    if not num1 or not num2:
        return
    m, n = len(num1), len(num2)

    res = '0'

    for i in range(m - 1, -1, -1):
        s = ''
        v1 = int(num1[i])
        carry = 0
        for j in range(n - 1, -1, -1):
            v2 = int(num2[j])
            v = v1 * v2 + carry
            s = str(v % 10) + s
            carry = v // 10
        if carry:
            s = str(carry) + s
        s += '0' * (m - 1 - i)  # 补0
        res = add(s, res)
    return res


def multiply2(num1, num2):
    if num1 == '0' or num2 == '0':
        return '0'

    m, n = len(num1), len(num2)
    res = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        v1 = int(num1[i])
        for j in range(n - 1, -1, -1):
            v2 = int(num2[j])
            v = res[i + j + 1] + v1 * v2
            res[i + j + 1] = v % 10
            res[i + j] += v // 10

    s = ''
    for i in range(m + n):
        if i == 0 and res[i] == 0:
            continue
        s += str(res[i])
    return s


# 14. 多个数组最长公共前缀
def longest_common_prefix(strs):
    if not strs:
        return ""

    def lcp(str1, str2):
        length, i = min(len(str1), len(str2)), 0
        while i < length and str1[i] == str2[i]:
            i += 1
        return str1[:i]

    prefix, n = strs[0], len(strs)
    for i in range(1, n):
        prefix = lcp(prefix, strs[i])
        if not prefix:
            break

    return prefix


# 43 字符串相乘
def multiply(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"

    m, n = len(num1), len(num2)
    res = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        x = int(num1[i])
        for j in range(n - 1, -1, -1):
            res[i + j + 1] += x * int(num2[j])

    for i in range(m + n - 1, 0, -1):
        res[i - 1] += res[i] // 10
        res[i] %= 10

    index = 1 if res[0] == 0 else 0
    ans = "".join(str(x) for x in res[index:])
    return ans


if __name__ == '__main__':
    print('\nmagic index')
    print(find_magic_index([2, 3, 4, 4, 5, 5, 5]))

    print('\n众数')
    print(most_data([1, 3, 3, 3, 9]))

    print('数组中第2大的数')
    print(second_largest([2, 3, 4, 10, 100]))

    print('\n计数排序')
    print(sort_colors([2, 0, 2, 1, 1, 0]))

    print('\n连续区间最大长度')
    print(longest_consecutive([1, 2, 3, 7, 8, 10, 11, 12, 13, 14]))

    print('\n区间合并')
    print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))

    print('\n下一个排列')
    print(next_permutation([1, 1, 3]))

    print('字符串相乘')
    print(add('123', '999'))

    print(multiply2('123', '456'))

    print('\n不包含自身的乘积')
    print(product_except_self([1, 2, 3, 4]))
