#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:20:38 2019

@author: wangbao
"""


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
    # 重合在一个就没有交换的必要了，因此是 left < right
    while left < right:
        l[left], l[right] = l[right], l[left]
        left += 1
        right -= 1
    return ''.join(l)


# 最小覆盖子串 滑动窗口
def min_window(s, t):
    str_dic = dict()
    for c in t:
        str_dic.setdefault(c, 0)
        str_dic[c] += 1
    window_dic = dict()
    left, right = 0, 0
    best_start_index = None
    match = 0
    min_len = float('inf')
    while right < len(s):
        c = s[right]
        if c in str_dic:
            window_dic.setdefault(c, 0)
            window_dic[c] += 1
            if window_dic[c] == str_dic[c]:
                match += 1  # 字母匹配数

        while match == len(str_dic):
            current_len = right - left + 1
            if current_len < min_len:
                min_len = current_len
                best_start_index = left
            c = s[left]
            if c in str_dic:
                window_dic[c] -= 1
                if window_dic[c] < str_dic[c]:
                    match -= 1
            left += 1
        right += 1
    return s[best_start_index: best_start_index + min_len] if best_start_index is not None else ''


def length_of_longest_substring(s):
    used_char = dict()
    l = 0
    left, right = 0, 0
    while right < len(s):
        c = s[right]
        if c not in used_char:
            used_char.setdefault(c, right)
        else:
            first_idx = used_char[c]
            while left <= first_idx:
                del used_char[s[left]]
                left += 1
            used_char[c] = right
        l = max(l, len(used_char))
        right += 1
    return l


def length_of_longest_substring2(s):
    left = max_length = 0
    used_char = {}

    for right in range(len(s)):
        if s[right] in used_char and left <= used_char[s[right]]:
            left = used_char[s[right]] + 1
        else:
            max_length = max(max_length, right - left + 1)

        used_char[s[right]] = right

    return max_length


if __name__ == '__main__':

    print('最小覆盖子串')
    print(min_window('aaaaaaaaaaaabbbbbcdd', 'abcdd'))

    print('最小非重复子串')
    print(length_of_longest_substring(" "))
