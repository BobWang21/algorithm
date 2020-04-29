from collections import defaultdict


# 拓扑排序
def can_finish(num_courses, prerequisites):
    dic = defaultdict(set)  # v -> u
    indegree = [0] * num_courses  # 入度
    for u, v in prerequisites:
        dic[v].add(u)
        indegree[u] += 1

    queue = []
    for i in range(num_courses):
        if not indegree[i]:
            queue.append(i)

    while queue:
        u = queue.pop(-1)
        for v in dic[u]:
            indegree[v] -= 1
            if not indegree[v]:
                queue.append(v)
        del dic[u]
    return False if dic else True


# !/usr/bin/env python3
# -*- coding: utf-8 -*-

def find_order(num_courses, prerequisites):
    dic = defaultdict(set)  # v -> u
    indegree = [0] * num_courses  # 入度
    for u, v in prerequisites:
        dic[v].add(u)
        indegree[u] += 1

    queue = []
    for i in range(num_courses):
        if not indegree[i]:
            queue.append(i)

    res = []
    while queue:
        u = queue.pop(-1)
        res.append(u)
        for v in dic[u]:
            indegree[v] -= 1
            if not indegree[v]:
                queue.append(v)
        del dic[u]
    return res if len(res) == num_courses else []


if __name__ == '__main__':
    print('\n是否可以完课')
    print(can_finish(2, [[1, 0]]))
