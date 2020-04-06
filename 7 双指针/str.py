#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:20:38 2019

@author: wangbao
"""
import collections


# 判断是否为回文
def is_palindrome(s):
    left = 0
    right = len(s) - 1
    while left < right:
        if not s[left].isalnum():  # isalnum() 数字或字符串
            left += 1
            continue
        if not s[right].isalnum():
            right -= 1
            continue

        if s[left].lower() != s[right].lower():  # 转换成小写!!
            return False

        left += 1
        right -= 1
    return True


def reverse_string(s):
    if len(s) < 2:
        return s
    left = 0
    right = len(s) - 1
    l = list(s)
    while left < right:  # 重合在一个就没有交换的必要了，因此是 left < right
        l[left], l[right] = l[right], l[left]
        left += 1
        right -= 1
    return ''.join(l)


# 72 最小覆盖子串 滑动窗口
def min_window(s, t):
    dic1 = dict()
    for c in t:
        dic1.setdefault(c, 0)
        dic1[c] += 1
    dic2 = dict()
    l, r = 0, 0
    match = 0
    min_len = float('inf')
    res = ''
    while r < len(s):
        c = s[r]
        if c in dic1:
            dic2.setdefault(c, 0)
            dic2[c] += 1
            if dic2[c] == dic1[c]:
                match += 1  # 字母匹配数

        while match == len(dic1):
            if r - l + 1 < min_len:
                min_len = r - l + 1
                res = s[l: r + 1]
            c = s[l]
            if c in dic1:
                dic2[c] -= 1
                if dic2[c] < dic1[c]:
                    match -= 1
            l += 1
        r += 1
    return res


# Given two strings s1 and s2,
# write a function to return true if s2 contains the permutation of s1.
def check_inclusion(s1, s2):
    if not s1 or not s2 or len(s1) > len(s2):
        return False
    d1 = dict()
    for c in s1:
        d1.setdefault(c, 0)
        d1[c] += 1

    match = 0
    d2 = dict()
    for i in range(len(s2)):
        if i >= len(s1):
            c = s2[i - len(s1)]
            if c in d2:
                if d1[c] == d2[c]:  # 只有以前match 现在不match 才减去1
                    match -= 1
                d2[c] -= 1
        c = s2[i]
        if c in d1:
            d2.setdefault(c, 0)
            d2[c] += 1
            if d2[c] == d1[c]:
                match += 1
        if match == len(d1):
            return True
    return False


# 最长非重复子串的长度 3
def length_of_longest_substring(s):
    l = 0  # 记录非重复开始
    res = 0
    dic = {}
    for r, c in enumerate(s):
        if c in dic and dic[c] >= l:
            l = dic[c] + 1
        res = max(res, r - l + 1)
        dic[c] = r

    return res


# 字符串1 包含 k个不同字符的最大长度
def max_k_char(s, k):
    if not s or k <= 0:
        return 0
    n = len(s)
    if n < k:
        return n
    dic = dict()
    dic[s[0]] = 1
    l, r = 0, 1
    max_len = 0
    while r < n:
        c = s[r]
        if c in dic:
            dic[c] += 1
            max_len = max(max_len, r - l + 1)
            r += 1
        else:
            if len(dic) < k:
                dic[c] = 1
                r += 1
            else:
                while len(dic) >= k:
                    c = s[l]
                    dic[c] -= 1
                    l += 1
                    if dic[c] == 0:
                        del dic[c]
    return max_len


def find_pairs(nums, k):
    if len(nums) < 2:
        return 0
    nums.sort()
    if nums[-1] - nums[0] < k:
        return 0
    n = len(nums)
    l, r = 0, 1
    num = 0
    while r < n and l < n:
        if l == r:
            r += 1
            continue
        s = nums[r] - nums[l]
        if s < k:
            r += 1
        elif s > k:
            l += 1
        else:
            num += 1
            while l + 1 < n and nums[l] == nums[l + 1]:
                l += 1
            l += 1
            r = max(l + 1, r + 1)
    return num


def find_pairs2(nums, k):
    res = 0
    c = collections.Counter(nums)
    for i in c:
        if (k > 0 and i + k in c) or (k == 0 and c[i] > 1):
            res += 1
    return res


if __name__ == '__main__':
    print('\n最小覆盖子串')
    print(min_window('aaaaaaaaaaaabbbbbcdd', 'abcdd'))

    print('\n最小非重复子串')
    print(length_of_longest_substring("abca"))

    print('\n一个字符串是否包含另外一个字符串的任一全排列')
    s1 = 'trinitrophenylmethylnitramine'
    s2 = 'dinitrophenylhydrazinetrinitrophenylmethylnitramine'
    print(check_inclusion(s1, s2))

    print('\n长度为K的最长子串')
    print(max_k_char('eceebaaaa', 2))

    print('\n相差为K的pair数目')
    print(find_pairs([1, 3, 1, 5, 4], 0))
    print(find_pairs2([1, 3, 1, 5, 4], 0))
