#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pprint import pprint


# 1143 最长公共子序列。子串连续 子序列不连续
def LCS(s1, s2):
    l1, l2 = len(s1) + 1, len(s2) + 1
    dp = [[0] * l2 for _ in range(l1)]
    for i in range(1, l1):
        for j in range(1, l2):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    pprint(dp)
    return dp[-1][-1]


# 最长公共子串
def longest_common_str(s1, s2):
    l1, l2 = len(s1), len(s2)
    dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

    pprint(dp)
    return dp[-1][-1]


# 516 最长回文子序列 top down
def longest_palindrome_subsequence1(s):
    if not s:
        return s
    n = len(s)
    dic = dict()

    def helper(l, r):
        if (l, r) in dic:
            return dic[(l, r)]
        # 回文问题考虑奇数和偶数
        if l == r:  # 奇数
            dic[(l, r)] = 1
            return 1
        if r - l == 1:  # 偶数
            dic[(l, r)] = 2 if s[l] == s[r] else 1
            return dic[(l, r)]
        if s[l] == s[r]:
            dic[(l, r)] = helper(l + 1, r - 1) + 2
            return dic[(l, r)]
        dic[(l, r)] = max(helper(l + 1, r), helper(l, r - 1))
        return dic[(l, r)]

    return helper(0, n - 1)


def longest_palindrome_subsequence2(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]


# 409 最长回文子串
def longest_palindrome(s):
    if not s:
        return s
    n = len(s)
    res = s[0]

    dp = [[False] * n for _ in range(n)]  # 两个位置之间是否为回文的二维数组

    for l in range(1, n + 1):  # bottom up 长度
        for i in range(n):
            j = l + i - 1
            if j >= n:
                break
            if l == 1 and s[i] == s[j]:
                dp[i][j] = True
                res = s[i:j + 1]
            elif l == 2 and s[i] == s[j]:
                dp[i][j] = True
                res = s[i:j + 1]
            elif s[i] == s[j] and dp[i + 1][j - 1]:  # dp[i, j] = dp[i+1, j-1] and s[i] == s[j]
                dp[i][j] = True
                res = s[i:j + 1]
    return res


# 编辑距离
def edit_distance(word1, word2):
    if not word1 and not word2:
        return 0
    if not word1:
        return len(word2)
    if not word2:
        return len(word1)

    rows, cols = len(word1) + 1, len(word2) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):  # base
        dp[i][0] = i
    for j in range(cols):  # base
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:  # delete dp[i][j - 1], dp[i - 1][j]  ;  dp[i - 1][j - 1]  replace
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


def word_break(s, word_dict):
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True  # base
    word_dict = set(word_dict)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if dp[i] and s[i:j] in word_dict:
                dp[j] = True
    return dp[-1]


# 10. 正则表达式匹配
# '.' 匹配任意单个字符
# '*' 匹配零个或多个前面的那一个元素
def is_match(s, p):
    if not s and not p:
        return True

    if not p:
        return False

    def match(i, j):
        if j == 0:  # 不存在字符
            return False
        if p[i - 1] == '.':  # . 匹配任何字符
            return True
        return p[i - 1] == s[j - 1]

    m, n = len(p), len(s)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for i in range(1, m + 1):
        for j in range(n + 1):
            if p[i - 1] == '*':
                dp[i][j] = dp[i - 2][j]  # 匹配0个
                if match(i - 1, j):
                    dp[i][j] |= dp[i][j - 1]  # 匹配一次
            else:
                if match(i, j):
                    dp[i][j] = dp[i - 1][j - 1]
    return dp[-1][-1]


if __name__ == '__main__':
    print('\n最长公共子序列')
    print(LCS('abcbdab', 'bdcaba'))

    print('\n最长公共子串')
    print(longest_common_str([0, 1, 1, 1, 1], [1, 0, 1, 0, 1]))

    print('\n最长回文子序列')
    print(longest_palindrome_subsequence1('babadada'))

    print('\n最长回文子串')
    print(longest_palindrome('babadada'))

    print('\n最短编辑距离')
    print(edit_distance("intention", "execution"))

    print('\n单词拆分')
    print(word_break('leetcode', ['leet', 'code']))
