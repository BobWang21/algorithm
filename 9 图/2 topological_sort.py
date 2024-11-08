#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, deque


# 207
# 回溯版本 计算超时 节点会重复计算
def can_finish1(num_courses, prerequisites):
    visited = [False] * num_courses

    edges = defaultdict(set)

    for i, j in prerequisites:
        edges[i].add(j)

    def dfs(i):
        if visited[i]:
            return False

        visited[i] = True

        for j in edges[i]:
            if not dfs(j):
                visited[i] = False
                return False

        visited[i] = False
        return True

    for i, j in prerequisites:
        if not dfs(i):
            return False
    return True


# 深度优先|考虑出度
def can_finish2(num_courses, prerequisites):
    edges = defaultdict(list)
    visited = [0] * num_courses  # 0:未访问 1:正在访问 2:访问过节点及子节点

    visited = [0] * num_courses

    edges = defaultdict(set)

    for i, j in prerequisites:
        edges[i].add(j)

    def dfs(i):
        # 正在访问
        if visited[i] == 1:
            return False
        # 完全访问
        if visited[i] == 2:
            return True

        visited[i] = 1

        for j in edges[i]:
            if not dfs(j):
                return False
        visited[i] = 2
        return True

    for i, j in prerequisites:
        if not dfs(i):
            return False
    return True


# 拓扑排序 广度优先 | 考虑入度 时间复杂度为O(E+V)
def can_finish3(num_courses, prerequisites):
    dic = defaultdict(set)  # v -> u 邻接表
    indegree = [0] * num_courses  # 入度
    for u, v in prerequisites:  # O(E)
        dic[v].add(u)
        indegree[u] += 1

    queue = deque()  # 多向bfs O(E)
    for i in range(num_courses):
        if not indegree[i]:
            queue.append(i)

    while queue:
        v = queue.popleft()
        for u in dic[v]:
            indegree[u] -= 1
            if not indegree[u]:
                queue.append(u)
        del dic[v]
    return False if dic else True


def find_order(num_courses, prerequisites):
    dic = defaultdict(set)  # v -> u
    indegree = [0] * num_courses  # 入度
    for u, v in prerequisites:
        dic[v].add(u)
        indegree[u] += 1

    queue = deque()
    for i in range(num_courses):
        if not indegree[i]:
            queue.append(i)

    res = []
    while queue:
        u = queue.popleft()
        res.append(u)
        for v in dic[u]:
            indegree[v] -= 1
            if not indegree[v]:
                queue.append(v)
        del dic[u]
    return res if len(res) == num_courses else []


# 269 火星词典
def alien_order(words):
    if not words:
        return

    n = len(words)
    if n == 1:
        return words[0]

    chr_set = set()  # 所有的字母
    for word in words:
        for ch in word:
            chr_set.add(ch)

    dic = defaultdict(set)
    for i in range(n - 1):
        for j in range(i + 1, n):
            word1 = words[i]
            word2 = words[j]
            m_len = len(word1)
            n_len = len(word2)
            l = r = 0
            while l < m_len or r < n_len:
                c1 = word1[l] if l < m_len else None
                c2 = word2[r] if r < n_len else None
                if c1 == c2:
                    l += 1
                    r += 1
                elif not c1 or not c2:
                    break
                else:
                    dic[c1].add(c2)
                    break

    degree = defaultdict(int)  # 入度
    for _, v in dic.items():
        for u in v:
            degree[u] += 1

    queue = []
    for v in chr_set:
        if not degree[v]:
            queue.append(v)

    res = ''
    while queue:
        i = queue.pop(0)
        res += i
        for j in dic[i]:
            degree[j] -= 1
            if not degree[j]:
                queue.append(j)
        del dic[i]
    return res if not dic else ''


if __name__ == '__main__':
    print('\n是否可以完课')
    print(can_finish3(3, [[1, 0], [1, 2], [2, 0]]))

    print('\n火星字典')
    print(alien_order(['wrt', 'wrf', 'er', 'ett', 'rftt']))
