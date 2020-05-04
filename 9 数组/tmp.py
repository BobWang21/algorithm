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
    dic[0] = 1  # 初始化
    for v in nums:
        total += v
        if total - k in dic:
            res += dic[total - k]
        dic[total] = dic.get(total, 0) + 1  # 后增加1 防止total - k = total
    return res


# 加油站 加油 有点前缀和的意思
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


# Find K-th Smallest Pair Distance
# 1 桶排序
def smallest_distance_pair_1(nums, k):
    nums.sort()

    v = max(nums) - min(nums)
    dp = [0] * (v + 1)
    n = len(nums)
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = nums[j] - nums[i]
            dp[d] += 1
    total = 0
    for i in range(v + 1):
        total += dp[i]
        if total >= k:
            return i


# 2 堆
def smallest_distance_pair_2(nums, k):
    heap = []
    n = len(nums)
    nums.sort()
    for i in range(n - 1):
        for j in range(i + 1, n):
            v = nums[j] - nums[i]
            if len(heap) < k:
                hq.heappush(heap, -v)
            elif v < -heap[0]:
                hq.heappop(heap)
                hq.heappush(heap, -v)
            else:
                break
    return -heap[0]


# 双指针 + 二分查找 差值小于等于某个数的pair数 为递增函数!!!
def smallest_distance_pair_3(nums, k):
    nums.sort()
    n = len(nums)

    def get_no_more(target):
        j = 1
        count = 0
        for i in range(n - 1):
            while j < n and nums[j] - nums[i] <= target:
                j += 1
            count += j - i - 1
        return count

    l, r = 0, nums[-1] - nums[0]
    while l < r:
        mid = (l + r) // 2
        count = get_no_more(mid)
        print(mid, count)
        if count >= k:
            r = mid
        else:
            l = mid + 1
    return l


if __name__ == '__main__':
    print('\n找到未排序的部分')
    print(find_unsorted_subarray2([2, 6, 4, 8, 10, 9, 15]))

    print('\n连续数组和为K')
    print(subarray_sum([1, 1, -1], 1))

    print('\n加油站问题')
    print(can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]))
