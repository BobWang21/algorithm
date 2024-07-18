#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 单调队列 满足单调性的双端队列(可以同时弹出队首和队尾的元素的队列)

from collections import deque
import heapq as hq


# 225 使用两个队列实现栈
class MyStack:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue1 = deque()
        self.queue2 = deque()

    def push(self, x):
        """
        Push element x onto stack.
        """
        self.queue2.append(x)
        while self.queue1:
            self.queue2.append(self.queue1.popleft())
        self.queue1, self.queue2 = self.queue2, self.queue1

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.queue1.popleft()

    def top(self):
        """
        Get the top element.
        """
        return self.queue1[0]

    def empty(self):
        """
        Returns whether the stack is empty.
        """
        return not self.queue1


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
# 单调队列
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
