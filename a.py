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
