#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, deque


# 207 课程表 拓扑排序 | 广度优先 | 考虑入度 时间复杂度为O(E+V)
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
    num = 0
    while queue:
        v = queue.popleft()
        num += 1
        for u in dic[v]:
            indegree[u] -= 1
            if not indegree[u]:
                queue.append(u)
    return num == num_courses


# 深度优先 | 考虑出度
def can_finish2(num_courses, prerequisites):
    edges = defaultdict(set)
    for i, j in prerequisites:
        edges[i].add(j)

    visited = [0] * num_courses  # 0:未访问 1:正在访问 2:访问过节点及子节点

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

    for i in range(num_courses):
        if not visited[i] and not dfs(i):
            return False
    return True


# 回溯版本 计算超时 节点会重复计算
def can_finish1(num_courses, prerequisites):
    edges = defaultdict(set)
    for i, j in prerequisites:
        edges[i].add(j)

    visited = [False] * num_courses

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


def find_order(num_courses, prerequisites):
    dic = defaultdict(set)  # v -> u
    indegree = [0] * num_courses  # 入度
    for u, v in prerequisites:
        dic[v].add(u)  # 指向的节点
        indegree[u] += 1  # 节点入度

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
# 从给定的词典中提取出字母间的相对顺序信息
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
    degree = defaultdict(int)  # 入度

    # 相邻单词对比
    for i in range(n - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        for j in range(min_len):
            c1, c2 = word1[j], word2[j]
            if c1 != c2:
                dic[c1].add(c2)
                degree[c2] += 1
                break
    queue = deque([k for k, v in degree.items() if v == 0])

    res = []
    while queue:
        i = queue.popleft()
        res.append(i)
        for j in dic[i]:
            degree[j] -= 1
            if not degree[j]:
                queue.append(j)

    return ''.join(res) if len(res) == len(chr_set) else ''


# 329.矩阵中的最长递增路径
# 将递增关系抽象为有向图
# 找到递增的点 该点度不为0
# 排序时从度为0的点逆序排列 找到的是递减序列
def longest_increasing_path(matrix):
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if not matrix:
        return 0

    rows, columns = len(matrix), len(matrix[0])
    outdegrees = [[0] * columns for _ in range(rows)]
    queue = deque()
    for i in range(rows):
        for j in range(columns):
            for dx, dy in DIRS:
                newRow, newColumn = i + dx, j + dy
                if 0 <= newRow < rows and 0 <= newColumn < columns and matrix[newRow][newColumn] > matrix[i][j]:
                    outdegrees[i][j] += 1
            if outdegrees[i][j] == 0:
                queue.append((i, j))

    ans = 0
    while queue:
        ans += 1
        size = len(queue)
        for _ in range(size):
            row, column = queue.popleft()
            for dx, dy in DIRS:
                newRow, newColumn = row + dx, column + dy
                if 0 <= newRow < rows and 0 <= newColumn < columns and matrix[newRow][newColumn] < matrix[row][
                    column]:
                    outdegrees[newRow][newColumn] -= 1
                    if outdegrees[newRow][newColumn] == 0:
                        queue.append((newRow, newColumn))

    return ans


# 310. 最小高度树 解的是无向图的问题
# 在无向图上利用拓扑排序思想
# 可选择树中任何一个节点作为根。
# 当选择节点 x 作为根节点时，设结果树的高度为 h 。
# 在所有可能的树中，具有最小高度的树（即，min(h)）被称为 最小高度树。
def find_min_height_trees(n, edges):
    if n == 1:
        return [0]

    # 1. 建图
    graph = [[] for _ in range(n)]
    degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        degree[u] += 1
        degree[v] += 1

    # 2. 叶子节点入队
    queue = deque([i for i in range(n) if degree[i] == 1])

    # 3. 逐层删除叶子
    remaining = n
    while remaining > 2:
        size = len(queue)
        remaining -= size
        # 叶子节点 同时出队列
        for _ in range(size):
            leaf = queue.popleft()
            for neighbor in graph[leaf]:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    queue.append(neighbor)

    # 4. 剩下的就是答案
    return list(queue)


if __name__ == '__main__':
    print('\n是否可以完课')
    print(can_finish3(3, [[1, 0], [1, 2], [2, 0]]))

    print('\n火星字典')
    print(alien_order(['wrt', 'wrf', 'er', 'ett', 'rftt']))
