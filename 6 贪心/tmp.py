#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq as hq


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


if __name__ == '__main__':
    print('\n两城调度')
    costs = [[10, 20], [30, 200], [400, 50], [30, 20]]
    print(two_city_sched_cost(costs))

    print('\n棍子连接')
    sticks = [2, 4, 3]
    print(connect_sticks(sticks))
