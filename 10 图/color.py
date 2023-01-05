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
        colors[i] = color  # 入口都未染色
        for j in dic[i]:
            if not colors[j] and not dfs(j, -color):
                return False
            if colors[j] == color:  # 已经涂色的顶点, 发生冲突
                return False
        return True

    for i in range(n):
        if not colors[i] and not dfs(i, 1):
            return False
    return True


if __name__ == '__main__':
    print('\n是否可二分')
    print(possible_bipartition(4, [[1, 2], [1, 3], [2, 4]]))
