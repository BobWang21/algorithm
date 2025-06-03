#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 判断是否为回文
def is_palindrome(s):
    if not s:
        return True
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
        l += 1
        r -= 1
    return True


# 977 左右指针 关注端点
def sorted_squares(nums):
    if not nums:
        return []
    n = len(nums)
    res = [0] * n
    i = n - 1
    l, r = 0, n - 1
    while l <= r:  # 此处为等号
        if abs(nums[l]) <= abs(nums[r]):
            res[i] = nums[r] ** 2
            r -= 1
        else:
            res[i] = nums[l] ** 2
            l += 1
        i -= 1
    return res


def reverse_string(s):
    if len(s) < 2:
        return s
    l, r = 0, len(s) - 1
    lists = list(s)  # 字符串不可原处更改
    while l < r:  # l = r 不交换
        lists[l], lists[r] = lists[r], lists[l]
        l += 1
        r -= 1
    return ''.join(lists)


# 392 s是否为t子序列 O(Max(m, n))
def is_subsequence(s, t):
    if not s:
        return True

    i = j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
            j += 1
        else:  # 下一个字符
            j += 1
    return i == len(s)


# 最长前缀后缀
# [0, j)中最大前缀和后缀长度
def get_nxt(s):
    n = len(s)
    nxt = [0] * n
    l, r = 0, 1
    while r < n:
        if s[l] == s[r]:
            l += 1
            nxt[r] = l
            r += 1
        elif l > 0:
            l = nxt[l - 1]  # 尝试第二长的前缀和后缀，看是否能继续延续
        else:
            r += 1  # 没有匹配的元素 'abcd'
    return nxt


# 判断是否为子串
# 在s字符串中找出t字符串出现的第一个位置
# 如果不存在返回-1。时间复杂度为O(M+N)
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
            j += 1  # j最多右移动m次 时间复杂度为O(m)
            if j == n:
                return i - n
        elif j > 0:  # s和t目前有匹配
            j = nxt[j - 1]  # 最多左移m次
        else:
            i += 1  # s和t未匹配
    return -1


# 189 循环移动K
def rotate(nums, k):
    n = len(nums)
    k %= n

    def reverse(nums, l, r):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1

    reverse(nums, 0, n - 1)
    reverse(nums, 0, k - 1)
    reverse(nums, k, n - 1)

    return nums


if __name__ == '__main__':
    print('\n平方排序')
    print(sorted_squares([-7, -3, 3, 11]))

    print('\n最长前缀后缀长度')
    print(get_nxt('abcab'))

    print('\nKMP')
    print(kmp('hello', 'll'))

    print('\n数组循环移动K位')
    print(rotate([1, 2, 3, 4], 2))
