#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import heapq as hq

'''
kth 问题的几种解法
1 桶排序 
2 堆排序
3 二分查找 需要构造递增序列
4 对于未排序的 使用partition
'''


def find_unsorted_subarray(nums):
    if not nums or len(nums) == 1:
        return 0
    nums1 = nums[:]
    nums1.sort()
    n = len(nums)
    l1 = 0
    for i in range(n):
        if nums[i] == nums1[i]:
            l1 += 1
            continue
        else:
            break
    if l1 == n:
        return 0

    l2 = 0
    for i in range(n - 1, -1, -1):
        if nums[i] == nums1[i]:
            l2 += 1
            continue
        else:
            break
    return n - (l1 + l2)


def find_unsorted_subarray2(nums):
    if not nums or len(nums) == 1:
        return 0
    stack = []
    n = len(nums)
    for i in range(n):
        while stack and nums[stack[-1]] > nums[i]:
            stack.pop(-1)
        stack.append(i)
    l1 = 0
    for i, v in enumerate(stack):
        if i == v:
            l1 += 1
        else:
            break
    if l1 == n:
        return 0

    stack = []
    for i in range(n - 1, -1, -1):
        while stack and nums[stack[-1]] < nums[i]:
            stack.pop(-1)
        stack.append(i)

    l2 = 0
    for i, v in enumerate(stack):
        if nums[v] == nums[n - i - 1]:
            l2 += 1
        else:
            break
    return n - (l1 + l2)


# 连续数组和为K 前缀和
def subarray_sum(nums, k):
    dic = dict()
    res = 0
    total = 0
    dic[0] = 1  # 初始化 可能total = k
    for v in nums:
        total += v
        if total - k in dic:  # total - pre_total = k -> total - k in dic
            res += dic[total - k]
        dic[total] = dic.get(total, 0) + 1  # 后增加1 防止total - k = total
    return res


# 加油站 有点前缀和的意思
def can_complete_circuit(gas, cost):
    n = len(gas)
    if sum(gas) - sum(cost) < 0:
        return -1
    total = 0
    start = 0
    for i in range(n):
        if total + gas[i] - cost[i] >= 0:
            total += gas[i] - cost[i]
        else:
            total = 0
            start = i + 1
    if start == n:
        return -1
    return start


# 要求o(n) 左右扫描两次也是O(n)
def product_except_self1(nums):
    n = len(nums)
    left = [1] * n
    for i in range(1, n):
        left[i] = left[i - 1] * nums[i - 1]

    right = [1] * n
    for i in range(n - 2, -1, -1):
        right[i] = right[i + 1] * nums[i + 1]

    for i in range(n):
        right[i] = right[i] * left[i]
    return right


# o(1)空间 不算返回的空间
def product_except_self2(nums):
    n = len(nums)
    res = [1] * n
    k = nums[0]
    for i in range(1, n):
        res[i] *= k
        k *= nums[i]

    k = nums[-1]
    for i in range(n - 2, -1, -1):
        res[i] *= k
        k *= nums[i]

    return res


if __name__ == '__main__':
    print('\n找到未排序的部分')
    print(find_unsorted_subarray2([2, 6, 4, 8, 10, 9, 15]))

    print('\n连续数组和为K')
    print(subarray_sum([1, 2, 3, -3, 4], 0))

    print('\n加油站问题')
    print(can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]))

    print('\n除自身以外数组的乘积')
    print(product_except_self1([1, 2, 3, 4]))
    print(product_except_self2([1, 2, 3, 4]))
