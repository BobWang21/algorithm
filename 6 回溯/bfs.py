#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pprint import pprint


# 1表示墙壁 0表示空地
def maze_short_path(maze, start, destination):
    if not maze or not maze[0]:
        return False
    rows, cols = len(maze), len(maze[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    wall, empty = 1, 0

    queue = [(start[0], start[1], 0)]
    while queue:
        i, j, layer = queue.pop(0)
        for d in directions:
            x, y = i + d[0], j + d[1]
            if 0 <= x < rows and 0 <= y < cols and maze[x][y] == empty:  # 先判断是否安全
                if [x, y] == destination:
                    return layer + 1

                maze[x][y] = wall  # 标记已经访问过的点
                queue.append((x, y, layer + 1))
    return -1


# 286 多向BFS
# 时间复杂度: O(mn) 空间: O(mn)
# 首先考虑只有一个门的情况，bfs最多只需要mn步就能到达所有的房间，所以时间复杂度是 O(mn)。如果从k个门呢？
# 一旦我们到达了一个房间，并记录下它的距离时，这意味着我们也标记了这个房间已经被访问过了，这意味着每个房间最多会被访问一次。
# 因此，时间复杂度与门的数量无关，所以时间复杂度为 O(mn)
def walls_and_gates(rooms):
    if not rooms or not rooms[0]:
        return
    rows, cols = len(rooms), len(rooms[0])
    inf, queue, = 2147483647, []
    for i in range(rows):
        for j in range(cols):
            if not rooms[i][j]:
                queue.append((i, j, 0))

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    while queue:
        row, col, layer = queue.pop(0)
        for direction in directions:
            i, j = row + direction[0], col + direction[1]
            if i < 0 or i == rows or j < 0 or j == cols:
                continue
            if rooms[i][j] == inf:  # 过滤墙和已经访问过的位置
                rooms[i][j] = layer + 1
                queue.append((i, j, layer + 1))


if __name__ == '__main__':
    print('\n迷宫最短路径')
    maze = [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 1, 1, 0]]
    start = [0, 0]
    end = [3, 4]
    print(maze_short_path(maze, start, end) == 11)

    print('\nWalls and Gates')
    inf = 2147483647
    rooms = [[inf, -1, 0, inf],
             [inf, inf, inf, -1],
             [inf, -1, inf, -1],
             [0, -1, inf, inf]]
    print(walls_and_gates(rooms))
