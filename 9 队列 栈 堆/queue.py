#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 单调栈适用于结果和数组当前值的前后缀相关的问题


# 496. 下一个更大元素 I
# Input: nums1 = [4, 1, 2], nums2 = [1, 3, 4, 2].
# Output: [-1, 3, -1]
def next_greater_element(nums1, nums2):
    if not nums1 or not nums2:
        return
    dic = dict()

    for i, v in enumerate(nums1):
        dic[v] = i
    res = [-1] * len(nums1)
    stack = []
    for v in nums2:
        while stack and stack[-1] < v:
            tail = stack.pop(-1)
            if tail in dic:
                res[dic[tail]] = v
        stack.append(v)
    return res


# 739 需要多少天 气温会升高
def daily_temperatures(t):
    if not t:
        return
    res = [0] * len(t)
    stack = []
    for i, v in enumerate(t):
        while stack and t[stack[-1]] < v:
            j = stack.pop(-1)
            res[j] = i - j
        stack.append(i)
    return res


# 可以返回最大值的队列
class MaxQueue():
    def __init__(self):
        self.queue1 = []
        self.queue2 = []

    def enqueue(self, v):
        self.queue1.append(v)
        while self.queue2 and self.queue2[-1] < v:
            self.queue2.pop(-1)
        self.queue2.append(v)

    def dequeue(self):
        v = self.queue1.pop(0)
        if self.queue2[0] == v:
            self.queue2.pop(0)

    def get_max(self):
        return self.queue2[0]


# 滑动窗口的最大值 数值由前后决定
def max_sliding_window(nums, k):
    def enqueue(queue, i):  #
        # 防止第一个划出窗口
        if queue and i - queue[0] == k:
            queue.pop(0)
        # 比当前小的数字 都不可能是窗口中的最大值
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop(-1)
        queue.append(i)

    n = len(nums)
    if n * k == 0:
        return nums
    res = []
    max_idx = 0
    queue = []
    for i in range(1, k):
        enqueue(queue, i)
        if nums[i] > nums[max_idx]:
            max_idx = i
    res.append(nums[max_idx])

    for i in range(k, n):
        enqueue(queue, i)
        res.append(nums[queue[0]])
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
        maxQueue.enqueue(i)
    maxQueue.dequeue()
    maxQueue.enqueue(11)
    print(maxQueue.get_max())

    print('\n滑动窗口的最大值')
    print(max_sliding_window([9, 10, 9, -7, -4, -8, 2, -6], 5))

    print('\n 下一个比其大数值')
    print(next_greater_element([4, 1, 2], [1, 3, 4, 2]))

    print('\n下一个天气比当前热')
    print(daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]))

    print('\n柱状图最大矩形')
    print((largest_rectangle_area2([2, 1, 5, 6, 2, 3])))

    print('\n区间数字和与区间最小值乘积最大')
    print(max_area_min_sum_product([81, 87, 47, 59, 81, 18, 25, 40, 56, 0]))

    print('\n接雨水')
    print(trap1([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(trap2([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
