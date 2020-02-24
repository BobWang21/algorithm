#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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


# 数字在 0 - n 之间
# 可能有多个数字重复 返回任意重复的数字
# 数字从0开始, 把数字放在对应的位置上
def find_duplicate_num2(nums):
    for i in range(len(nums)):
        while i != nums[i]:  # 循环后保证i = nums[i]
            if nums[i] == nums[nums[i]]:
                return nums[i]
            else:
                tmp = nums[i]
                nums[i] = nums[tmp]
                nums[tmp] = tmp


# 数组中的重复数字
# 1 - n 的 n + 1 个数中 只有一个数重复一次或多次
# 要求 O(1)空间复杂度!!!
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


# 26 升序数组中的重复数字
# Given nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
# Your function should return length = 5,
# with the first five elements of nums being modified to 0, 1, 2, 3, and 4 respectively.
# It doesn't matter what values are set beyond the returned length.
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


if __name__ == '__main__':
    print('找到数组中重复元素')
    print(find_duplicate_num([1, 2, 3, 4, 2]))
    print(find_duplicate_num2([1, 2, 3, 2, 0]))
