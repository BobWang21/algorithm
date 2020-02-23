#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:20:38 2019

@author: wangbao
"""


# 超过一半的数
def most_data(data):
    size = len(data)
    if size == 0:
        raise Exception('')
    value = data[0]
    count = 1
    for i in range(1, size):
        if count == 0:
            value = data[i]
            count = 1
            continue
        if data[i] == value:
            count += 1
        else:
            count -= 1
    return value


# 移除重复数字
def remove_duplicates(nums):
    n = len(nums)
    if n < 2:
        return n
    l, r = 0, 1
    while r < n:
        if nums[l] == nums[r]:
            r += 1
        else:
            l += 1
            nums[l] = nums[r]
            r += 1
    return l + 1


def remove_duplicates2(nums):
    n = len(nums)
    if n < 2:
        return n
    pointer = 0  # 第一个指针
    for i in range(1, n):  # 第二个指针
        if nums[i] != nums[pointer]:
            pointer += 1
            nums[pointer] = nums[i]
    return pointer + 1


# 数组中的重复数字
# 1-n n+1个数中 只有一个数重复一次或多次
def find_duplicate_num(nums):
    fast = slow = nums[0]
    # 证明有环 快慢两个指针
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if fast == slow:
            break
    # 入口
    ptr1 = nums[0]
    ptr2 = fast
    while ptr1 != ptr2:
        ptr1 = nums[ptr1]
        ptr2 = nums[ptr2]
    return ptr1


# O(n) 剑指offer 1
def find_duplicate_num2(nums):
    for i in range(len(nums)):
        while i != nums[i]:  # 循环后保证i = nums[i]
            if nums[i] == nums[nums[i]]:
                return nums[i]
            else:
                tmp = nums[i]
                nums[i] = nums[tmp]
                nums[tmp] = tmp


'''
给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
说明:
你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
'''


# 有唯一解 O(N)
def two_sum(nums, target):
    size = len(nums)
    if size < 2:
        return []
    left, right = 0, size - 1
    res = []
    while 0 <= left < right < size:
        s = nums[left] + nums[right]
        if s == target:
            res.append([nums[left], nums[right]])
            while left < right and nums[left] == nums[left + 1]:
                left += 1
            while left < right and nums[right] == nums[right - 1]:
                right -= 1
            left += 1
            right -= 1
        elif s < target:
            left += 1
        else:
            right -= 1
    return res


# 双指针
def three_sum(nums, target):
    res = []
    nums.sort()
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:  # 防止重复
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s < target:
                left += 1
            elif s > target:
                right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:  # 跳出时left为最后一个相同的数
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return res


# 输入有重复数字, 每个数字只能用一次
# 输出不包含重复 回溯法
def n_sum(nums, target):
    if not nums:
        return
    res = []

    def dfs(nums, idx, k, target, path, res):
        if k == 4 and target == 0:
            res.append(path)
            return
        if k == 4:
            return
        for i in range(idx, len(nums)):
            if i > idx and nums[i] == nums[i - 1]:  # 不重复
                continue
            dfs(nums, i + 1, k + 1, target - nums[i], path + [i], res)

    dfs(nums, 0, 0, target, [], res)
    return res


def n_sum2(nums, target):
    if not nums:
        return

    def dfs(nums, left, k, target, path, res):
        if len(nums[left:]) < k or nums[left] * k > target or nums[-1] * k < target:
            return
        elif k == 2:
            two_sum_paths = two_sum(nums[left:], target)
            for sum_paths in two_sum_paths:
                res.append(path + sum_paths)
        else:
            for i in range(left, len(nums) - (k - 1)):
                if i > left and nums[i] == nums[i - 1]:
                    continue
                dfs(nums, i + 1, k - 1, target - nums[i], path + [nums[i]], res)

    res = []
    nums.sort()
    dfs(nums, 0, 4, target, [], res)
    return res


def three_sum_closet(nums, target):
    size = len(nums)
    if size < 3:
        return
    closet_sum = None
    gap = float('inf')
    nums.sort()
    for i in range(size - 2):
        l, r = i + 1, size - 1
        while i + 1 <= l < r <= size - 1:
            s = nums[i] + nums[l] + nums[r]
            if s == target:
                return s
            new_gap = abs(s - target)
            if new_gap < gap:
                gap = new_gap
                closet_sum = s
            if s > target:
                r -= 1
            else:
                l += 1
    return closet_sum


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
    print('找到数组中重复元素')
    print(find_duplicate_num([1, 2, 3, 4, 2]))
    print(find_duplicate_num2([1, 2, 3, 2, 0]))

    print('2 sum')
    print(two_sum([1, 2, 7, 8, 11, 15], 9))

    print('3 sum')
    print(three_sum([2, 7, 7, 11, 15, 15, 20, 24, 24], 33))

    print('4 sum')
    print(n_sum2([1, 0, -1, 0, -2, 2], 0))

    print('3 sum closet')
    print(three_sum_closet([-1, 2, 1, -4], 1))

    print('最小覆盖子串')
    print(min_window('aaaaaaaaaaaabbbbbcdd', 'abcdd'))

    print('最小非重复子串')
    print(length_of_longest_substring(" "))
