#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq as hq
from collections import defaultdict


# 发饼干 求最大满足的小孩的数目
def find_content_children(g, s):
    g.sort()  # 常排序
    s.sort()
    num = 0
    m, n = len(g), len(s)
    i = j = 0
    while i < m and j < n:
        if g[i] > s[j]:
            j += 1
        else:
            num += 1
            i += 1
            j += 1
    return num


def two_city_sched_cost(costs):
    costs.sort(key=lambda x: x[0] - x[1])  # 根据两个城市的费用差值排序 相当于2-OPT

    total = 0
    n = len(costs) // 2
    for i in range(n):  # 交换时 -cost[i][0] + cost[i][1]
        total += costs[i][0] + costs[i + n][1]
    return total


# 木头连接 费用为两根木头的长度和
# 哈夫曼数
def connect_sticks(sticks):
    hq.heapify(sticks)
    total = 0
    while len(sticks) > 1:
        v = hq.heappop(sticks) + hq.heappop(sticks)
        total += v
        hq.heappush(sticks, v)
    return total


# 最长摇摆序列
def wiggle_max_length(nums):
    if not nums:
        return 0
    if len(nums) < 2:
        return 1
    start, up, down = -1, 0, 1  # 状态机
    state = start
    num = 1
    for i in range(1, len(nums)):
        if state == start:
            if nums[i] > nums[i - 1]:
                state = up
                num += 1
            elif nums[i] < nums[i - 1]:
                state = down
                num += 1
        elif state == up:
            if nums[i] < nums[i - 1]:
                state = down
                num += 1
        else:
            if nums[i] > nums[i - 1]:
                state = up
                num += 1
    return num


# 判断是否为子序列 可以使用双指针
# 当有多个S的时候 可以使用2分查找
def is_subsequence(s, t):
    dic = defaultdict(list)
    for i, c in enumerate(t):
        dic[c].append(i)

    # print(dic)
    def binary_search(nums, target):
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] <= target:
                l = mid + 1
            else:
                r = mid
        return nums[l] if nums[l] > target else -1

    t = -1
    for c in s:
        if c in dic:
            idx = binary_search(dic[c], t)
            # print(dic[c], c, idx)
            if idx == -1:
                return False
            t = idx
        else:
            return False
    return True


# 是否可以跳到最后
# 如果某一个作为 起跳点 的格子可以跳跃的距离是 3，那么表示后面 3 个格子都可以作为 起跳点。
# 可以对每一个能作为 起跳点 的格子都尝试跳一次，把 能跳到最远的距离 不断更新
# 如果可以一直跳到最后，就成功了
def can_jump(nums):
    n = len(nums)
    total = 0
    for i in range(n - 1):
        if total >= i and i + nums[i] > total:
            total = i + nums[i]
    return total >= n - 1


def can_jump2(nums):
    n = len(nums)
    if len(nums) == 1:
        return 0
    l = r = 0
    num = 0
    while r < n:
        max_r = 0
        for j in range(l, r + 1):
            max_r = max(max_r, nums[j] + j)
        l, r = r + 1, max_r
        num += 1
    return num


def min_meeting_rooms(intervals):
    if not intervals:
        return 0
    intervals.sort()  # 按开始时间排序
    heap = [intervals.pop(0)[1]]  # 保存最早结束时间
    while intervals:
        s, e = intervals.pop(0)
        if s >= heap[0]:  # 所有房间的最早的结束时间
            hq.heappop(heap)
            hq.heappush(heap, e)
        else:
            hq.heappush(heap, e)
    return len(heap)


if __name__ == '__main__':
    print('\n两城调度')
    costs = [[10, 20], [30, 200], [400, 50], [30, 20]]
    print(two_city_sched_cost(costs))

    print('\n棍子连接')
    sticks = [2, 4, 3]
    print(connect_sticks(sticks))

    print('\n所需要最少会议室')
    print(min_meeting_rooms([[0, 30], [5, 10], [15, 20]]))
