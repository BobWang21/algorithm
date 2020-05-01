#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict


# 图是否二分 染色
def possible_bipartition(n, dislikes):
    dic = defaultdict(set)
    for u, v in dislikes:
        dic[u - 1].add(v - 1)
        dic[v - 1].add(u - 1)

    colors = [0] * n

    def dfs(node, color):
        if not colors[node]:  # 未染色
            colors[node] = color
            for child in dic[node]:
                if not dfs(child, -color):  # 邻接点染成相反色
                    return False
            return True
        if colors[node] != color:  # 已经染色 染色冲突
            return False
        return True

    for i in range(n):
        if not colors[i] and not dfs(i, 1):
            return False
    return True


if __name__ == '__main__':
    print('\n是否可二分')
    print(possible_bipartition(4, [[1, 2], [1, 3], [2, 4]]))
