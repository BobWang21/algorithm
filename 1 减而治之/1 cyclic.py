#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数组重复和缺失值
1 dic
2 修改列表: 将数组看成index为key的字典
 - 循环排序 适合于数值区间在一定范围内的数组
 - 标记正负 适合正数
3 不修改列表
 - 二分查找 构建单调函数
 - 快慢双指针 数组抽象为链表
 - 异或操作
"""


# 不包含重复数字的循环排序
# 使用列表模拟字典
def cyclic_sort(nums):
    if not nums:
        return []
    n = len(nums)
    for i in range(n):
        while nums[i] != i + 1:
            j = nums[i] - 1
            if nums[j] == j + 1:  # 该位置已经排好
                break
            nums[i], nums[j] = nums[j], nums[i]
    return nums


# 0 ~ n-1之间的n个数 可能有多个数字重复 返回任意重复的数字
def find_duplicate_num1(nums):
    n = len(nums)
    for i in range(n):
        while i != nums[i]:
            j = nums[i]
            if j == nums[j]:  # 说明数字j重复了
                return j
            nums[i], nums[j] = nums[j], nums[i]


# 41 没有出现的最小的正整数, 可能含有重复数字
def first_missing_positive(nums):
    if not nums:
        return 1

    n = len(nums)
    for i in range(n):
        while 0 < nums[i] <= n and nums[i] != i + 1:
            j = nums[i] - 1
            if nums[j] == j + 1:  # nums[j] == nums[i] 防止无限循环
                break
            nums[i], nums[j] = nums[j], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1  # 如果数组是[1, 2, 3] !!!


# 448 使用负数标记 已存在的数
def find_disappeared_numbers(nums):
    for i in range(len(nums)):
        j = abs(nums[i]) - 1
        if nums[j] > 0:
            nums[j] = -nums[j]

    res = []
    for i in range(len(nums)):
        if nums[i] > 0:
            res.append(i + 1)
    return res


# 1-n之间的n个数 有些出现1次 有些出现2次 出现2次的数字
def find_duplicates(nums):
    res = []
    for i in range(len(nums)):
        v = abs(nums[i]) - 1
        if nums[v] > 0:  # 第一次访问变成负数
            nums[v] = -nums[v]
        else:  # 变成负数 说明该数重复了两次
            res.append(v + 1)
    return res


# 使用负数标记已经访问的数
def find_duplicate_num2(nums):
    n = len(nums)
    i = 0
    for v in nums:  # 判断是否0位重复数字
        if not v:
            i += 1
            if i > 1:
                return 0

    for i in range(n):
        j = abs(nums[i])
        if nums[j] > 0:
            nums[j] = -nums[j]
        elif nums[j] < 0:
            return j


# 287. 寻找重复数
def find_duplicate(nums):
    n = len(nums)

    def count(target):
        return sum([1 if v <= target else 0 for v in nums])

    l, r = 0, n - 1
    while l < r:
        mid = l + (r - l) // 2
        if count(mid) <= mid:
            l = mid + 1
        else:
            r = mid
    return l


# 287
# 1 - n 的 n + 1 个数中 只有一个数字重复 数字重复一次或多次
# 要求 O(1)空间复杂度!!!
# 不能修改列表
# Floyd 判圈算法
def find_duplicate_num3(nums):
    fast = slow = 0
    # 证明有环
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if fast == slow:
            break
    # 入口
    ptr1 = 0
    ptr2 = fast
    while ptr1 != ptr2:
        ptr1 = nums[ptr1]
        ptr2 = nums[ptr2]
    return ptr1


# 268 给定一个包含[0, n]中 n 个数的序列，不含重复数字
# 找出没有出现在序列中的那个数
def missing_number(nums):
    if not nums:
        return 0

    n = len(nums)

    miss_value = n
    for i in range(n):
        miss_value ^= nums[i] ^ i

    return miss_value


# 560. 前缀和(和为K的子数组)
def subarray_sum(nums, k):
    res = 0
    total = 0
    dic = dict()
    dic[0] = 1  # 初始化 可能nums[0] = k
    for num in nums:
        total += num  # 以num为结尾的连续数组
        if total - k in dic:  # pre_sum+k=total -> total-k in dic
            res += dic[total - k]
        dic[total] = dic.get(total, 0) + 1  # 后增加1 防止total - k = total
    return res


# 523 给定一个包含 非负数 的数组和一个目标 整数k，
# 编写一个函数来判断该数组是否含有连续的子数组，其大小至少为 2，
# 且总和为k的倍数，即总和为 n*k，其中 n 也是一个整数。
def check_subarray_sum(nums, k):
    if len(nums) < 2:
        return False
    dic, total = {0: -1}, 0
    for i, num in enumerate(nums):
        total += num
        if k:
            total %= k
        j = dic.setdefault(total, i)
        if i - j >= 2:
            return True
    return False


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


# 581. 最短无序连续子数组 [2,4,8,|10,7,9|,15,20]
# 找到数组中需要排序的最小连续部分，对该部分排序后整个数组升序。
# 单调性 也可用单调栈
def find_unsorted_subarray(nums):
    n = len(nums)
    max_v, min_v = -float('inf'), float('inf')
    left, right = n - 1, 0
    # 从右向左遍历, 找到最后一个大于右侧最小值的数值位置
    for i in range(n - 1, -1, -1):
        if nums[i] <= min_v:  # 比右侧最小值大
            min_v = nums[i]
        else:
            left = i  # left左侧数数都比右侧最小值小
    # 从左向右遍历, 找到最后一个小于左侧最大值的数值位置
    for i in range(n):
        if nums[i] >= max_v:  # 比左侧最大值大
            max_v = nums[i]
        else:
            right = i  # right右侧的数都比左侧最大值大(递增)

    if left == n - 1 and right == 0:
        return 0
    return right - left + 1


if __name__ == '__main__':
    print('\ncyclic sort')
    print(cyclic_sort([7, 5, 8, 1, 2, 9, 3, 4, 6, 10]))
    print(cyclic_sort([2, 2, 1, 3, 2]))

    print('\n找到缺失的最小正数')
    print(first_missing_positive([3, 4, -1, 1]))

    print('\n连续数列中缺失的数')
    print(missing_number([0, 1, 2, 4]))

    print('\n找到数组中重复元素')
    print(find_duplicate_num1([1, 2, 4, 3, 2]))
    print(find_duplicate_num2([1, 2, 3, 2, 0]))

    print('\n数组连续和为K')
    print(subarray_sum([1, 2, 3, -3, 4], 0))

    print('\n加油站问题')
    print(can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]))

    print('\n未排序区间长度')
    print(find_unsorted_subarray([2, 6, 4, 8, 10, 9, 15]))
