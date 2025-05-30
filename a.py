import heapq as hq


def top_k1(nums, k):
    res = []
    for i in range(k):
        hq.heappush(res, -nums[i])

    for i in range(k, len(nums)):
        if -res[0] > nums[i]:
            hq.heappop(res)
            hq.heappush(res, -nums[i])

    return -res[0]


def top_k2(nums, k):
    hq.heapify(nums)
    v = -1
    for _ in range(k):
        v = hq.heappop(nums)
    return v


print(top_k1([2, 20, -1, -2, 10, ], 4))

import random


def shuffle(nums):
    n = len(nums)

    for i in range(n):
        j = random.randint(i, n - 1)
        nums[i], nums[j] = nums[j], nums[i]

    return nums


print(shuffle([1, 2, 3, 4]))
