#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 是否连通 1表示墙壁 0表示空地
def maze_can_reach(maze, start, destination):
    if not maze or not maze[0]:
        return False
    m, n = len(maze), len(maze[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    wall, empty = 1, 0

    def dfs(i, j):
        if i < 0 or j < 0 or i == m or j == n:
            return False

        if maze[i][j] == wall:  # 访问过 或是 墙 不需要回溯!!!
            return False

        # 到达目的地
        if [i, j] == destination[1]:
            return True

        # 标记已经访问的点
        maze[i][j] = wall

        for d in directions:
            if dfs(i + d[0], j + d[1]):  # 多路问题
                return True
        return False

    return dfs(start[0], start[1])


# 1表示墙壁 0表示空地
def maze_short_path(maze, start, destination):
    if not maze or not maze[0]:
        return False
    rows, cols = len(maze), len(maze[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    wall, empty = 1, 0

    def safe(i, j):
        if i < 0 or j < 0 or i == rows or j == cols:
            return False
        return True

    def bfs():
        queue = [(start[0], start[1], 0)]
        while queue:
            i, j, layer = queue.pop(0)
            for d in directions:
                x, y = i + d[0], j + d[1]
                if safe(x, y) and maze[x][y] == empty:  # 先判断是否安全
                    maze[x][y] = wall
                    if [x, y] == destination:
                        return layer + 1
                    else:
                        queue.append((x, y, layer + 1))
        return -1

    return bfs()


# 286 -1: 墙 0:大门 INF:空房间 2147483647来表示。
def walls_and_gates(rooms):
    if not rooms or not rooms[0]:
        return

    inf = 2147483647
    m, n = len(rooms), len(rooms[0])
    queue = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for row in range(m):
        for col in range(n):
            if not rooms[row][col] == 0:
                queue.append((row, col))

    while queue:
        point = queue.pop(0)
        row = point[0]
        col = point[1]
        for direction in directions:
            r = row + direction[0]
            c = col + direction[1]
            if 0 <= r < m and 0 <= c < n and rooms[r][c] == inf:  # 已经标记过 或者是墙
                rooms[r][c] = rooms[row][col] + 1
                queue.append((r, c))


# bfs
def find_cheapest_price(flights, src, dst, K):
    dic = dict()  # {s: {d:p}}

    for s, d, p in flights:
        dic.setdefault(s, dict())
        dic[s][d] = p
    print(dic)
    queue = [(src, -1, 0, {src})]
    amount = float('inf')
    while queue:
        node, stop, price, path = queue.pop(0)
        if stop > K:
            break
        if price >= amount:
            continue
        if node == dst and stop <= K:
            amount = min(amount, price)
        if node in dic:
            for new_node, new_price in dic[node].items():
                if new_node not in path:  # 记录路径
                    queue.append((new_node, stop + 1, price + new_price, path | {new_node}))

    return amount if amount < float('inf') else -1


# dfs
def find_cheapest_price2(flights, src, dst, K):
    dic = dict()  # {s: {d:p}}

    for s, d, p in flights:
        dic.setdefault(s, dict())
        dic[s][d] = p

    res = [float('inf')]

    def helper(node, path, amount):
        if len(path) > K + 2:
            return
        if node in dic:
            for new_node, price in dic[node].items():
                if new_node not in path:
                    if new_node == dst:
                        res[0] = min(res[0], amount + price)
                        return
                    else:
                        helper(new_node, path | {new_node}, amount + price)

    helper(src, set(), 0)
    return res[0]


# 岛屿数
def num_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])

    def helper(i, j):
        queue = [(i, j)]
        grid[i][j] = '0'
        while queue:
            i, j = queue.pop(0)
            if i - 1 >= 0 and grid[i - 1][j] == '1':
                grid[i - 1][j] = '0'
                queue.append((i - 1, j))
            if i + 1 < rows and grid[i + 1][j] == '1':
                grid[i + 1][j] = '0'
                queue.append((i + 1, j))
            if j - 1 >= 0 and grid[i][j - 1] == '1':
                grid[i][j - 1] = '0'
                queue.append((i, j - 1))
            if j + 1 < cols and grid[i][j + 1] == '1':
                grid[i][j + 1] = '0'
                queue.append((i, j + 1))
        return 1

    num = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                num += helper(i, j)
    return num


# 岛屿数 dfs
def num_islands2(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])

    def dfs(i, j):
        if i < 0 or j < 0 or i == rows or j == cols:
            return
        if grid[i][j] == '0':
            return
        grid[i][j] = '0'
        dfs(i - 1, j)
        dfs(i + 1, j)
        dfs(i, j - 1)
        dfs(i, j + 1)
        return

    num = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                dfs(i, j)
                num += 1
    return num


# 130 将围住的o变成x 逆向思维
def surrounded_regions(board):
    if not board:
        return
    rows, cols = len(board), len(board[0])

    def dfs(i, j):
        if i < 0 or j < 0 or i == rows or j == cols or board[i][j] == 'X':
            return
        if board[i][j] == 'A':
            return
        board[i][j] = 'A'
        dfs(i - 1, j)
        dfs(i + 1, j)
        dfs(i, j - 1)
        dfs(i, j + 1)

    for i in range(rows):
        dfs(i, 0)
        dfs(i, cols - 1)
    for j in range(cols):
        dfs(0, j)
        dfs(rows - 1, j)
    dic = {'A': 'O', 'O': 'X', 'X': 'X'}
    for i in range(rows):
        for j in range(cols):
            board[i][j] = dic[board[i][j]]
    return board


# 回溯思想
def surrounded_regions2(board):
    if not board:
        return
    rows, cols = len(board), len(board[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def back(path):
        for i, j in path:
            board[i][j] = 'O'

    def dfs(i, j, path):
        if i < 0 or j < 0 or i == rows or j == cols:  # 先判断是否有效
            return False
        if board[i][j] == 'X':
            return True
        board[i][j] = 'X'
        path.add((i, j))  # 不建议这么写
        for d in directions:
            if dfs(i + d[0], j + d[1], path):
                return True
        back(path)
        return False

    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                dfs(i, j, set())
    return board


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

    print('\nfind_cheapest_price')
    print(find_cheapest_price([[0, 1, 100], [1, 2, 100], [0, 2, 500]], 0, 2, 1))

    print('\nWalls and Gates')
    rooms = [[2147483647, -1, 0, 2147483647],
             [2147483647, 2147483647, 2147483647, -1],
             [2147483647, -1, 2147483647, -1],
             [0, -1, 2147483647, 2147483647]]
    print(walls_and_gates(rooms))

    print('\n岛屿数')
    grid = [['1', '1', '0', '0', '0'],
            ['1', '1', '0', '0', '0'],
            ['0', '0', '1', '0', '0'],
            ['0', '0', '0', '1', '1']]
    print(num_islands(grid))

    print('\n岛屿数2')
    grid = [['1', '1', '0', '0', '0'],
            ['1', '1', '0', '0', '0'],
            ['0', '0', '1', '0', '0'],
            ['0', '0', '0', '1', '1']]
    print(num_islands2(grid))
