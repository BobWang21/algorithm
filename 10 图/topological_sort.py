#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict


# 拓扑排序
def can_finish(num_courses, prerequisites):
    dic = defaultdict(set)  # v -> u 原来你就是邻接表!!!
    indegree = [0] * num_courses  # 入度
    for u, v in prerequisites:
        dic[v].add(u)
        indegree[u] += 1

    queue = []  # 可能存在多个没有入度的节点 bfs
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
    print(can_finish(2, [[1, 0]]))

    print('\n火星字典')
    print(alien_order(['wrt', 'wrf', 'er', 'ett', 'rftt']))
