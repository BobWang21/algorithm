#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, Counter


# 判断是否为回文
def is_palindrome(s):
    l, r = 0, len(s) - 1
    while l < r:
        if not s[l].isalnum():  # isalnum() 数字或字符串
            l += 1
            continue
        if not s[r].isalnum():
            r -= 1
            continue

        if s[l].lower() != s[r].lower():  # 转换成小写!!
            return False
        else:
            l += 1
            r -= 1
    return True


def reverse_string(s):
    if len(s) < 2:
        return s
    l, r = 0, len(s) - 1
    lists = list(s)
    while l < r:  # 重合在一个就没有交换的必要了，因此是 left < right
        lists[l], lists[r] = lists[r], lists[l]
        l += 1
        r -= 1
    return ''.join(lists)


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


# 判断是否为子序列
def is_subsequence(s, t):
    if not s:
        return True
    i = j = 0
    m, n = len(s), len(t)
    num = 0
    while i < m and j < n:
        if s[i] == t[j]:
            i += 1
            j += 1
            num += 1
            if num == m:
                return True
        else:
            j += 1
    return False


def is_subsequence2(s, t):
    dic = defaultdict(list)
    for i, c in enumerate(t):
        dic[c].append(i)

    # print(dic)
    def binary_search(nums, target):
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] <= target:
                l = mid + 1
            else:
                r = mid
        return nums[l] if nums[l] > target else -1

    t = -1
    for c in s:
        if c in dic:
            idx = binary_search(dic[c], t)
            # print(dic[c], c, idx)
            if idx == -1:
                return False
            t = idx
        else:
            return False
    return True


# 最长前缀后缀
def get_nxt(s):
    n = len(s)
    lps = [0] * n
    l, i = 0, 1
    while i < n:
        if s[l] == s[i]:
            l += 1
            lps[i] = l
            i += 1
        elif l > 0:
            l = lps[l - 1]  # 尝试第二长的前缀和后缀，看是否能继续延续
        else:
            i += 1  # 没有匹配的元素 'abcd'
    return lps


# 判断是否为子串
# 在 s 字符串中找出 t 字符串出现的第一个位置 如果不存在返回-1
# 时间复杂度为O(M+N)
def kmp(s, t):
    m = len(s)
    n = len(t)

    if not n:
        return 0
    nxt = get_nxt(t)

    i = j = 0
    while i < m:
        if s[i] == t[j]:
            i += 1
            j += 1  # j 最多增加M次 时间复杂度为O(M)
            if j == n:
                return i - n
        elif j > 0:
            j = nxt[j - 1]  # 只移动j坐标 减少也最多O(M)次
        else:
            i += 1  # 没有匹配
    return -1


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


if __name__ == '__main__':
    print('\n最小覆盖子串')
    print(min_window('aaaaaaaaaaaabbbbbcdd', 'abcdd'))

    print('\n最小非重复子串')
    print(length_of_longest_substring("abca"))

    print('\n一个字符串是否包含另外一个字符串的任一全排列')
    s1 = 'trinitrophenylmethylnitramine'
    s2 = 'dinitrophenylhydrazinetrinitrophenylmethylnitramine'
    print(check_inclusion(s1, s2))

    print('\n长度为K的最长不重复子串')
    print(max_k_char('eceebaaaa', 2))

    print('\n最长前缀后缀长度')
    print(get_nxt('ababa'))

    print('\nKMP')
    print(kmp('hello', 'll'))

    print('字符串相乘')
    print(add('123', '999'))

    print(multiply2('123', '456'))
