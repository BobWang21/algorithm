#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict


# 886 图是否二分 染色
def possible_bipartition(n, dislikes):
    dic = defaultdict(set)
    for u, v in dislikes:
        dic[u - 1].add(v - 1)
        dic[v - 1].add(u - 1)

    colors = [0] * n

    def dfs(i, color):
        if colors[i] == color:
            return True
        if colors[i] == -color:
            return False

        colors[i] = color  # 入口都未染色
        for j in dic[i]:
            if not dfs(j, -color):
                return False
        return True

    for i in range(n):
        if not colors[i] and not dfs(i, 1):
            return False
    return True


def bipartite_graphs(l2r):
    r2l = dict()
    for left, adj in l2r.items():
        for right, score in adj.items():
            r2l.setdefault(right, dict())
            r2l[right][left] = score

    visit_left = dict()
    visit_right = dict()
    res = []

    def helper(i, path):
        if visit_left.get(i, False):
            return
        visit_left[i] = True
        path[i] = l2r[i]
        for j in l2r[i]:
            if visit_right.get(i, False):
                continue
            visit_right[j] = True
            for k in r2l[j]:
                helper(k, path)

        return

    for i in l2r:
        if visit_left.get(i, False):
            continue
        path = dict()
        helper(i, path)
        res.append(path)

    return res


def hungarian(graph, m, n):
    match = [False] * n
    visit_y = [False] * n

    def helper(i):
        # 遍历y顶点
        for j, val in graph[i].items():
            # 每一轮匹配, 每个y顶点只尝试一次
            if visit_y[j]:
                continue
            visit_y[j] = True
            if match[j] == -1 or helper(match[j]):
                match[j] = i
                return True

        return True

    res = 0
    for i in range(m):
        visit_y = [False] * n
        if helper(i):
            res += 1
    return res


if __name__ == '__main__':
    print('\n是否可二分')
    print(possible_bipartition(4, [[1, 2], [1, 3], [2, 4]]))

    print('\n 二分图拆分连通图')
    left2right = {
        0: {0: 0.5},
        1: {0: 1},
        2: {1: 0.5, 2: 0.6},
        3: {4: 0},
        4: {3: 0},
    }
    print(bipartite_graphs(left2right))

    graph = {
        0: {0: 0, 2: 0},
        1: {0: 0, 1: 0},
        2: {1: 0}
    }
    print(hungarian(graph, 3, 3))
