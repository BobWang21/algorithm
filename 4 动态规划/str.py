#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 最长回文子序列 可以不连续 回文问题考虑奇数和偶数
def longest_palindrome_subseq(s):
    if not s:
        return s
    n = len(s)
    dic = dict()

    def helper(l, r):
        if (l, r) in dic:
            return dic[(l, r)]
        if l == r:  # 奇数
            dic[(l, r)] = 1
            return 1
        if r - l == 1:  # 偶数
            if s[l] == s[r]:
                dic[(l, r)] = 2
            else:
                dic[(l, r)] = 1
            return dic[(l, r)]
        if s[l] == s[r]:
            dic[(l, r)] = helper(l + 1, r - 1) + 2
            return dic[(l, r)]
        dic[(l, r)] = max(helper(l + 1, r), helper(l, r - 1))
        return dic[(l, r)]

    return helper(0, n - 1)


# 最长回文子串 连续
def longest_palindrome(s):
    if not s:
        return s
    n = len(s)
    max_str = s[0]
    # 初始化字符串两个位置之间是否为回文的二维数组
    dp = [[False] * n for _ in range(n)]

    for l in range(1, n + 1):  # bottom up
        for i in range(n):
            j = l + i - 1
            if j >= n:
                break
            if l == 1 and s[i] == s[j]:
                dp[i][j] = True
                max_str = s[i:j + 1]
            elif l == 2 and s[i] == s[j]:
                dp[i][j] = True
                max_str = s[i:j + 1]
            elif dp[i + 1][j - 1] and s[i] == s[j]:
                dp[i][j] = True
                max_str = s[i:j + 1]
    return max_str


def lp(s):
    if len(s) < 2 or s == s[::-1]:
        return s

    start, max_length = 0, 1
    for i in range(1, len(s)):
        odd = s[i - max_length - 1:i + 1]  # 检查l+2
        even = s[i - max_length:i + 1]  # 检查l+1
        if i - max_length - 1 >= 0 and odd == odd[::-1]:
            start = i - max_length - 1
            max_length += 2
        elif i - max_length >= 0 and even == even[::-1]:
            start = i - max_length
            max_length += 1
    return s[start:start + max_length]


# 最长公共子序列
def longest_common_subsequence(s1, s2):
    l1, l2 = len(s1), len(s2)
    res = [[0] * (l2 + 1) for _ in range(l1 + 1)]
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if s1[i - 1] == s2[j - 1]:
                res[i][j] = res[i - 1][j - 1] + 1
            else:
                res[i][j] = max(res[i - 1][j], res[i][j - 1])
    return res[-1][-1]


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
            else:  # dp[i][j - 1], dp[i - 1][j] delete ;  dp[i - 1][j - 1]  replace
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


if __name__ == '__main__':
    print('\n最长回文子序列')
    print(longest_palindrome_subseq('babadada'))

    print('\n最长回文子串')
    print(longest_palindrome('babadada'))

    print('\n最长公共子序列')
    print(longest_common_subsequence('aabcd', 'ad'))

    print('\n最短编辑距离')
    print(edit_distance("intention", "execution"))
