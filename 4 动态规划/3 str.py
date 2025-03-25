#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pprint import pprint


# 1143 最长公共子序列。子序列不连续，子串连续
def LCS(s1, s2):
    l1, l2 = len(s1) + 1, len(s2) + 1
    dp = [[0] * l2 for _ in range(l1)]
    for i in range(1, l1):
        for j in range(1, l2):
            if s1[i - 1] == s2[j - 1]:  # 包含两个端点
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # 最多包含一个端点
    pprint(dp)
    return dp[-1][-1]


# 5 最长回文子串
def longest_palindrome1(s):
    if len(s) == 1:
        return s
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    # dp[i][j] 表示s[i:j+1]子串最长回文长度
    for i in range(n):
        dp[i][i] = 1  # 递归基

    # print(dp)
    res = s[0]
    max_len = 1
    # 长度 回文子串的长度先短后长
    for k in range(2, n + 1):
        # 子串起点
        for i in range(n):
            # j - i + 1 = k,
            j = k + i - 1
            if j >= n:
                continue
                # [0, 1, 2, 3]
            if s[i] != s[j]:
                continue
            if k == 2:
                dp[i][j] = 2  # 递归基
            elif dp[i + 1][j - 1]:  # dp[i+1][j-1]一定访问过
                dp[i][j] = dp[i + 1][j - 1] + 2

            # 区别dp[-1][-1]
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                res = s[i:j + 1]
    return res


# 5 最长回文子串
def longest_palindrome2(s):
    if len(s) < 2:
        return s

    n = len(s)
    dp = [[0] * n for _ in range(n)]
    max_len = 1
    res = s[0]
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            if j - i == 1 and s[i] == s[j]:
                dp[i][j] = 2  # 递归基
            # i依赖i+1, 因此i逆序
            elif s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = dp[i + 1][j - 1] + 2
                # print(dp[i][j])
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                res = s[i:j + 1]
    return res


# 516 最长回文子序列 bottom up
def longest_palindrome_subseq3(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    # 逆序访问
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                # i依赖i+1, 因此i逆序
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]


# 最长公共子串 子串连续
def longest_common_str(s1, s2):
    l1, l2 = len(s1), len(s2)
    dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

    pprint(dp)
    return dp[-1][-1]


# 编辑距离
def edit_distance(word1, word2):
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
            else:  # 删除j dp[i][j - 1], 删除i dp[i - 1][j]; 替换 dp[i - 1][j - 1]
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


def word_break(s, word_dict):
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True  # base
    word_dict = set(word_dict)
    for i in range(n):
        for j in range(i + 1, n + 1):
            # 当前字符和单词匹配，结尾状态与前一个字符状态相同
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

    print('\n最长回文子串')
    print(longest_palindrome1('babadada'))

    print('\n最短编辑距离')
    print(edit_distance("intention", "execution"))

    print('\n单词拆分')
    print(word_break('leetcode', ['leet', 'code']))
