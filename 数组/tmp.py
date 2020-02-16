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


# O(n)
def find_duplicate_num2(nums):
    for i in range(len(nums)):
        while i != nums[i]:  # 循环后保证i = nums[i]
            if nums[i] == nums[nums[i]]:
                return nums[i]
            else:
                tmp = nums[i]
                nums[i] = nums[tmp]
                nums[tmp] = tmp


# 2sum O(n) 减而治之
def two_sum(nums, target):
    if not nums:
        return
    dic = {}
    for i, v in enumerate(nums):
        if target - v in dic:
            return [dic[target - v], i]
        else:
            dic.setdefault(v, i)


def three_sum(nums, target):
    n = len(nums)
    for i in range(n):
        dic = {}
        for j in range(i + 1, n):
            new_target = target - nums[i] - nums[j]
            if new_target in dic:
                return [i, dic[new_target], j]
            dic.setdefault(nums[j], j)


# 输入有重复数字, 每个数字只能用一次
# 输出不包含重复
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
            if i > idx and nums[i] == nums[i - 1]:
                continue
            dfs(nums, i + 1, k + 1, target - nums[i], path + [i], res)

    dfs(nums, 0, 0, target, [], res)
    return res


if __name__ == '__main__':
    print('找到数组中重复元素')
    print(find_duplicate_num([1, 2, 3, 4, 2]))
    print(find_duplicate_num2([1, 2, 3, 2, 0]))

    print('2 sum')
    print(two_sum([2, 7, 11, 15], 9))

    print('3 sum')
    print(three_sum([2, 7, 11, 15], 33))

    print('4 sum')
    print(four_sum([1, 2, 7, 11, 15], 21))
