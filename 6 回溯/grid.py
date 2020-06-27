#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def max_gift(board):
    if not board or not board[0]:
        return 0

    rows, cols = len(board), len(board[0])

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # ← → ↑ ↓
    res = [-float('inf')]

    def helper(i, j, total):
        if i < 0 or j < 0 or i == rows or j == cols:
            res[0] = max(res[0], total)
            return

        if board[i][j] == 'x':  # 访问过
            res[0] = max(res[0], total)
            return

        if isinstance(board[i][j], int):
            total = total + board[i][j]

        c = board[i][j]

        board[i][j] = 'x'
        print('+', board[0])
        for direction in directions:
            helper(i + direction[0], j + direction[1], total)  # 1

        board[i][j] = c  # 因为1没有返回动作 因此可以回溯
        print('-', board[0])

    print('s', board[0])
    helper(0, 2, 0)
    return res[0]


def surround(board):
    if not board or not board[0]:
        return []

    rows, cols = len(board), len(board[0])

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # ← → ↑ ↓

    def back(s):
        for i, j in s:
            board[i][j] = 'o'

    def helper(i, j, update_set):
        if i < 0 or j < 0 or i == rows or j == cols:
            return False

        if board[i][j] == 'x':  # 访问过
            return True

        c = board[i][j]
        if c == '👷':
            return True

        board[i][j] = '👷'
        update_set.add((i, j))

        for direction in directions:
            if not helper(i + direction[0], j + direction[1], update_set):
                back(update_set)  # 1 需要回溯所有的尝试
                return False

        return True  # 如果某一方向满足条件则不会回溯 因此1处需要回溯所有路径!!!

    return helper(1, 1, set())


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

        if maze[i][j] == wall:  # 访问过 或是 墙
            return False

        # 到达目的地
        if [i, j] == destination[1]:
            return True

        # 标记已经访问的点
        maze[i][j] = wall

        for d in directions:
            if dfs(i + d[0], j + d[1]):
                return True
        return False

    return dfs(start[0], start[1])  # 从某一个位置出发可以不用回溯


# 1表示墙壁 0表示空地
def maze_short_path(maze, start, destination):
    if not maze or not maze[0]:
        return False
    rows, cols = len(maze), len(maze[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    wall, empty = 1, 0

    def bfs():
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

    return bfs()


# 286 -1:墙 0:大门 INF:空房间 2147483647来表示
# 直接法 会超时
def walls_and_gates2(rooms):
    INF = 2147483647
    if not rooms or not rooms[0]:
        return
    rows, cols = len(rooms), len(rooms[0])

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def bfs(i, j, visited):
        queue = [(i, j, 0)]
        visited[i][j] = True
        while queue:
            r, c, layer = queue.pop(0)
            for direction in directions:
                i, j = r + direction[0], c + direction[1]
                if 0 <= i < rows and 0 <= j < cols:
                    if rooms[i][j] == -1:
                        continue
                    if rooms[i][j] == 0:
                        return layer + 1
                    if not visited[i][j]:
                        queue.append((i, j, layer + 1))
                        visited[i][j] = True
        return INF

    for i in range(rows):
        for j in range(cols):
            if rooms[i][j] == INF:
                visited = [[False] * cols for _ in range(rows)]
                rooms[i][j] = bfs(i, j, visited)


# 多向BFS
def walls_and_gates(rooms):
    if not rooms or not rooms[0]:
        return
    INF = 2147483647
    rows, cols = len(rooms), len(rooms[0])
    queue = []
    for i in range(rows):
        for j in range(cols):
            if not rooms[i][j]:
                queue.append((i, j, 0))

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    while queue:
        row, col, layer = queue.pop(0)
        for direction in directions:
            i, j = row + direction[0], col + direction[1]
            if 0 <= i < rows and 0 <= j < cols:
                if rooms[i][j] == INF:
                    rooms[i][j] = layer + 1
                    queue.append((i, j, layer + 1))


# 200. 岛屿数量 也可使用union and find
def num_islands(grid):
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


# 130 将围住的'O'变成'X' 任何边界上的'O'都不会被填充为'X' 逆向思维
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

    def back(path):
        for i, j in path:
            board[i][j] = 'O'

    def dfs(i, j, path):
        if i < 0 or j < 0 or i == rows or j == cols:
            return False
        if board[i][j] == 'X':
            return True

        board[i][j] = 'X'

        path.add((i, j))

        if dfs(i - 1, j, path) and dfs(i + 1, j, path) and dfs(i, j - 1, path) and dfs(i, j + 1, path):
            return True  # 正确的时候 不会回溯

        back(path)  # 如果前3个方向正确, 最后一个后1个错误时, 无法回溯正确的方向

        return False

    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                dfs(i, j, set())
    return board


# 1254 封闭岛屿的数量
def closed_island(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(i, j):
        if i < 0 or j < 0 or i == rows or j == cols or grid[i][j]:
            return
        grid[i][j] = 1
        for direction in directions:
            r, c = i + direction[0], j + direction[1]  # 联通!!!
            dfs(r, c)

    for row in range(rows):
        dfs(row, 0)
        dfs(row, cols - 1)

    for col in range(cols):
        dfs(0, col)
        dfs(rows - 1, col)

    num = 0
    for i in range(rows):
        for j in range(cols):
            if not grid[i][j]:
                num += 1
                dfs(i, j)
    return num


# dfs + 记忆 dp[i][j] = max(dp[x][y]) + 1
# (x, y)为(i, j)的相邻点 存在重复计算
def longest_increasing_path(matrix):
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dic = {}

    def dfs(i, j):
        if (i, j) in dic:
            return dic[(i, j)]
        tmp = matrix[i][j]
        matrix[i][j] = '.'
        res = 0
        for d in directions:
            r, c = i + d[0], j + d[1]
            if r < 0 or r == rows or c < 0 or c == cols or matrix[r][c] == '.':
                continue
            if matrix[r][c] > tmp:
                res = max(res, dfs(r, c))
        matrix[i][j] = tmp
        dic[(i, j)] = res + 1
        return res + 1

    res = 0
    for i in range(rows):
        for j in range(cols):
            res = max(res, dfs(i, j))

    return res


# 抢钱问题 记忆化深度搜索
def rob(tree):
    dic = {}

    def helper(root):
        if not root:
            return 0, 0  # inc, max
        if root in dic:
            return dic[root]
        left = helper(root.left)
        right = helper(root.right)
        include = root.val
        if root.left:
            include += helper(root.left.left)[1]  # 涉及到重复计算
            include += helper(root.left.right)[1]
        if root.right:
            include += helper(root.right.left)[1]
            include += helper(root.right.right)[1]
        exclude = left[1] + right[1]
        max_v = max(include, exclude)
        dic.setdefault(root, (include, max_v))
        return include, max_v

    return helper(tree)[1]


# 单词搜索 or
def exist(board, word):
    rows, cols = len(board), len(board[0])
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    n = len(word)

    def dfs(i, j, k):
        if k == n:
            return True

        if i < 0 or i == rows or j < 0 or j == cols or board[i][j] == '.':
            return False

        if board[i][j] != word[k]:
            return False

        w = board[i][j]
        board[i][j] = '.'
        for d in directions:
            r, c = i + d[0], j + d[1]
            if dfs(r, c, k + 1):
                return True
        board[i][j] = w
        return False

    for i in range(rows):
        for j in range(cols):
            if dfs(i, j, 0):
                return True
    return False


if __name__ == '__main__':
    print('\n路径中捡到的最多钱')
    board = [[100, 'o', 'o', 'o', 80]]
    print(max_gift(board))

    print('\n包围')
    board = [['o', '👷', '👷', 'o'],
             ['👷', 'o', 'o', '👷'],
             ['👷', 'o', '👷', 'o'],
             ['o', 'o', 'o', 'o']]

    for line in board:
        print(line)
    surround(board)
    print()
    for line in board:
        print(line)

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

    print('\n封闭岛屿数')
    grid = [[1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]]
    print(closed_island(grid))
