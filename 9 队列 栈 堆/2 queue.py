#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 单调队列 满足单调性的双端队列(可以同时弹出队首和队尾的元素的队列)
from collections import deque
import heapq as hq


# 面试题59 队列的最大值
# 单调队列
class MaxQueue():
    def __init__(self):
        self.enqueue = deque()
        self.dequeue = deque()

    def en(self, v):
        self.enqueue.append(v)
        while self.dequeue and self.dequeue[-1] < v:  # 此处为严格的小于号
            self.dequeue.pop()
        self.dequeue.append(v)  # <=

    def de(self):
        v = self.enqueue.popleft()
        if self.dequeue[0] == v:  # popleft
            self.dequeue.popleft()

    def get_max(self):
        return self.dequeue[0]


# 239. 滑动窗口最大值
def max_sliding_window1(nums, k):
    n = len(nums)
    res = []
    queue = deque()

    def enqueue(i):  #
        # 长度超过K, 窗口划出第一个值
        if queue and i - queue[0] == k:
            queue.popleft()  # o(1)
        # 比当前小的数字 都不可能是窗口中的最大值
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop()
        queue.append(i)

    for i in range(n):
        enqueue(i)
        if i + 1 < k:  # 未填满窗口, 不计算最值
            continue
        res.append(nums[queue[0]])
    return res


# 0(nlogn)
def max_sliding_window2(nums, k):
    n = len(nums)
    heap, res = [], []
    hq.heapify(heap)

    for i in range(n):
        hq.heappush(heap, (-nums[i], i))
        if i + 1 < k:
            continue
        while i - heap[0][1] >= k:  # 移除不满足的数值
            hq.heappop(heap)
        res.append(nums[heap[0][1]])
    return res


# 暴力法O(n^2)
def largest_rectangle_area1(heights):
    max_rec = 0
    n = len(heights)
    for i, h in enumerate(heights):
        l = r = i
        while l > 0 and heights[l - 1] >= h:  # 前面第一个小于该数
            l -= 1
        while r < n - 1 and heights[r + 1] >= h:  # 后面第一个小于该数
            r += 1
        max_rec = max(max_rec, (r - l - 1) * h)
    return max_rec


# 84. 柱状图中最大的矩形
def largest_rectangle_area2(height):
    height.append(0)  # 为了让剩余元素出栈
    stack = []
    res = 0
    n = len(height)
    for i in range(n):
        while stack and height[stack[-1]] > height[i]:
            h = height[stack.pop()]
            w = i - stack[-1] - 1 if stack else i  # 当前值为最小值长度为i
            res = max(res, h * w)
        stack.append(i)
    return res


# 42 接雨水
def trap1(height):
    if not height:
        return 0

    n = len(height)
    l = [0] * n
    r = [0] * n

    for i in range(n):  # 动态规划 左右
        if not i:
            l[i] = height[i]
        else:
            l[i] = max(l[i - 1], height[i])

    for i in range(n - 1, -1, -1):
        if i == n - 1:
            r[i] = height[i]
        else:
            r[i] = max(r[i + 1], height[i])

    res = 0
    for i in range(n):
        res += min(l[i], r[i]) - height[i]
    return res


def trap2(height):
    res = 0
    stack = []
    for i in range(len(height)):
        while stack and height[stack[-1]] < height[i]:
            j = stack.pop(-1)
            if not stack:
                break
            d = i - stack[-1] - 1
            h = min(height[i], height[stack[-1]]) - height[j]
            res += d * h  # 可以多加的水
        stack.append(i)

    return res


def max_area_min_sum_product(nums):
    if not nums:
        return 0
    nums.append(-1)  # 为了使栈中剩余元素出栈
    n = len(nums)
    stack = []
    total = [0] * n
    res = 0

    for i, v in enumerate(nums):
        v = v if v >= 0 else 0
        total[i] = total[i - 1] + v

        while stack and v < nums[stack[-1]]:
            j = stack.pop(-1)
            pre_total = 0
            if stack:
                pre_total = total[stack[-1]]
            res = max(res, (total[i - 1] - pre_total) * nums[j])
        stack.append(i)
    return res


if __name__ == '__main__':
    print('最大值值队列')
    maxQueue = MaxQueue()
    for i in range(10):
        maxQueue.en(i)
        maxQueue.de()
        maxQueue.en(i)
    print(maxQueue.get_max())

    print('\n滑动窗口的最大值')
    print(max_sliding_window1([9, 10, 9, -7, -4, -8, 2, -6], 5))

    print('\n柱状图最大矩形')
    print((largest_rectangle_area2([2, 1, 5, 6, 2, 3])))

    print('\n区间数字和与区间最小值乘积最大')
    print(max_area_min_sum_product([81, 87, 47, 59, 81, 18, 25, 40, 56, 0]))

    print('\n接雨水')
    print(trap1([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(trap2([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
