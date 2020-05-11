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

    def dfs(i, color):
        colors[i] = color
        for j in dic[i]:
            if not colors[j]:
                dfs(j, -color)
            elif colors[j] == color:  # 冲突一定是已经涂色的顶点
                return False
        return True

    for i in range(n):
        if not colors[i] and not dfs(i, 1):
            return False
    return True


if __name__ == '__main__':
    print('\n是否可二分')
    print(possible_bipartition(4, [[1, 2], [1, 3], [2, 4]]))
