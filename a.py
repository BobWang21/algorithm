#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, deque

import heapq as hq

num = [1, 4, 10]

hq.heapify(num)

while num:
    print(hq.heappop(num))

queue = deque()
queue.append(1)
queue.append(10)
queue.append(20)

queue.pop()
queue.popleft()

print(queue)


def c_sort(nums):
    n = len(nums)
    for i in range(n):
        while nums[i] != i:
            j = nums[i]
            if nums[j] == j:
                break
            nums[i], nums[j] = nums[j], nums[i]
    return nums


nums = [3, 1, 0, 2, 2]
print(c_sort(nums))


def find_missing_value(nums):
    n = len(nums)

    for i in range(n):
        while 1 <= nums[i] <= n and nums[i] != i + 1:
            j = nums[i] - 1
            if nums[j] == j + 1:
                break
            nums[i], nums[j] = nums[j], nums[i]
    print(nums)
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


nums = [3, 1, 2, 4]
print(find_missing_value(nums))


def subarray_sum(nums, k):
    dic = defaultdict(int)
    dic[0] = 1

    total = 0
    res = 0
    for v in nums:
        total += v
        if total - k in dic:
            res += dic[total - k]
        dic[total] += 1
    return res


print(subarray_sum([2, 3, 1, 4], 5))


def us(nums):
    left = -1
    n = len(nums)
    # left 小于右侧的最小值的最后值的索引
    min_v = float('inf')
    for i in range(n - 1, -1, -1):
        if nums[i] < min_v:
            min_v = nums[i]
        else:
            left = i

    right = -1
    max_v = -float('inf')
    # 小于左侧的最大值的最后值的索引
    for i in range(n):
        if nums[i] > max_v:
            max_v = nums[i]
        else:
            right = i

    return right - left + 1 if left != right else 0


print(us([1]))


def find_magic_index(nums):
    def find_magic_index_(l, r):
        if l >= r:
            return -1
        mid = l + (r - l) // 2
        idx = find_magic_index_(l, mid - 1)
        if idx != -1:
            return idx
        if nums[mid] == mid:
            return mid
        return find_magic_index_(mid + 1, r)

    return find_magic_index_(0, len(nums) - 1)


print(find_magic_index([2, 3, 4, 4, 5, 5, 5]))

from collections import Counter
import heapq as hq


def top_k_frequent(nums, k):
    counter = Counter(nums)
    min_hq = []
    for value, cnt in counter.items():
        hq.heappush(min_hq, (cnt, value))
        if len(min_hq) > k:
            hq.heappop(min_hq)
    return [value for cnt, value in min_hq]


print('\n出现次数最多的K个数')
print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
