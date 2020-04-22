#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 26 原地删除升序数组中的重复数字 并返回非重复数组的长度
# Given nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
# Your function should return length = 5
def remove_duplicates(nums):
    n = len(nums)
    i = 0  # 第一个指针
    for j in range(1, n):  # 第二个指针
        if nums[j - 1] != nums[j]:
            nums[i + 1] = nums[j]
            i += 1
    return i + 1


# 0 - n 之间可能有多个数字重复 返回任意重复的数字
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


if __name__ == '__main__':
    print('\n找到数组中重复元素')
    print(find_duplicate_num([1, 2, 3, 4, 2]))
    print(find_duplicate_num2([1, 2, 3, 2, 0]))

    print('\n删除排查数组中的重复数值')
    print(remove_duplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))
