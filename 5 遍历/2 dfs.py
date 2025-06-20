#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pprint import pprint


# 200 岛屿数量
# union-find也可
def num_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def dfs(i, j):
        if i < 0 or j < 0 or i == rows or j == cols:
            return
        if grid[i][j] == '0':
            return
        grid[i][j] = '0'
        for di, dj in directions:
            dfs(i + di, j + dj)
        return

    num = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                dfs(i, j)
                num += 1
    return num


# 695 岛屿的最大面积
def max_area_of_island(grid):
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    water, land = 0, 1
    res = 0

    def helper(i, j):
        if i < 0 or j < 0 or i == rows or j == cols or grid[i][j] == water:
            return 0
        grid[i][j] = water
        res = 1
        for di, dj in directions:
            res += helper(i + di, j + dj)
        return res

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == land:
                res = max(res, helper(i, j))
    return res


# 从某一位置出发, 判断是否连通 不回溯
def has_path(maze, start, destination):
    if not maze or not maze[0]:
        return False
    rows, cols = len(maze), len(maze[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    empty, wall = 0, 1

    def dfs(i, j):
        if i < 0 or j < 0 or i == rows or j == cols:
            return False

        if maze[i][j] == wall:  # 访问过/墙
            return False

        # 到达目的地
        if [i, j] == destination:
            return True

        # 标记访问的点
        maze[i][j] = wall

        for di, dj in directions:
            if dfs(i + di, j + dj):
                return True
        return False

    print('before')
    pprint(maze)
    res = dfs(start[0], start[1])
    print('after')
    pprint(maze)
    return res


# 从某一位置出发 判断是否连通 回溯 不改变maze
def has_path2(maze, start, destination):
    if not maze or not maze[0]:
        return False
    rows, cols = len(maze), len(maze[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    empty, wall = 0, 1

    def dfs(i, j):
        if i < 0 or j < 0 or i == rows or j == cols:
            return False

        if maze[i][j] == wall:  # 访问过/墙
            return False

        # 到达目的地
        if [i, j] == destination:
            return True

        # 标记已经访问的点
        maze[i][j] = wall

        for di, dj in directions:
            if dfs(i + di, j + dj):
                maze[i][j] = empty  # 回溯 不改变数组
                return True
        maze[i][j] = empty  # 回溯
        return False

    print('before')
    pprint(maze)
    res = dfs(start[0], start[1])
    print('after')
    pprint(maze)
    return res


# 490 小球是否能在目的地停留 你可以假定迷宫的边缘都是墙壁
# 由空地（用 0 表示）和墙（用 1 表示）
def has_path_3(maze, start, destination):
    rows, cols = len(maze), len(maze[0])
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    def dfs(i, j):
        if maze[i][j] == -1:
            return False
        maze[i][j] = -1
        if [i, j] == destination:
            return True

        for dx, dy in directions:
            x, y = i, j
            while 0 <= x + dx < rows and 0 <= y + dy < cols and maze[x + dx][y + dy] != 1:
                x = x + dx
                y = y + dy

            if dfs(x, y):
                return True

        return False

    return dfs(start[0], start[1])


# 130 捕获 所有 被围绕的区域
# 将围住的'O'变成'X' 任何边界上的'O'都不会被填充为'X'
# 逆向思维
def surrounded_regions1(board):
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


# 329. 矩阵中的最长递增路径
# dfs + 记忆 dp[i][j]
# max(dp[x][y]) + 1
def longest_increasing_path(matrix):
    memory = {}
    rows, cols = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def helper(i, j):
        # 递归开始满足约束，此处约束放在了下面
        if (i, j) in memory:
            return memory[(i, j)]

        num = 0
        for direction in directions:
            dx, dy = i + direction[0], j + direction[1]
            # 将约束放在这
            if dx < 0 or dy < 0 or dx == rows or dy == cols:
                continue
            if matrix[i][j] <= matrix[dx][dy]:
                continue
            num = max(num, helper(dx, dy))

        num += 1  # 本身
        memory[(i, j)] = num
        return num

    res = 1
    for i in range(rows):
        for j in range(cols):
            res = max(res, helper(i, j))
    return res


# 抢钱问题 记忆化深度搜索
def rob(tree):
    dic = dict()

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


# 130 回溯思想
def surrounded_regions2(board):
    if not board:
        return
    rows, cols = len(board), len(board[0])

    def back(path):
        for i, j in path:
            board[i][j] = 'O'

    def dfs(i, j, s):
        if i < 0 or j < 0 or i == rows or j == cols:
            return False
        if board[i][j] == 'X':
            return True
        board[i][j] = 'X'
        s.add((i, j))
        if dfs(i - 1, j, s) and dfs(i + 1, j, s) and dfs(i, j - 1, s) and dfs(i, j + 1, s):
            return True  # 正确的时候 不会回溯
        back(s)  # 只要有一个错误 回溯所有
        return False

    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                dfs(i, j, set())
    return board


# 回溯过程
def max_gift(board):
    if not board:
        return 0

    cols = len(board)
    res = [-float('inf')]

    def helper(i, total):
        if i < 0 or i == cols:
            return

        if board[i] == 'x':  # 访问过
            return

        if isinstance(board[i], int):
            total = total + board[i]
            res[0] = max(res[0], total)

        c = board[i]
        board[i] = 'x'
        print('+', board)
        for direction in [-1, 1]:
            helper(i + direction, total)  # 1 此处没return 因此2处可回溯

        board[i] = c  # 2 回溯
        print('-', board)

    print('.', board)
    helper(2, 0)
    return res[0]


# 1254 封闭岛屿的数量
# 逆向思维
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
            dfs(i + direction[0], j + direction[1])

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


class Node(object):
    def __init__(self, val=0, neighbors=None):
        if neighbors is None:
            neighbors = []
        self.val = val
        self.neighbors = neighbors


# 133 复制图
def clone_graph(node):
    dic = {}

    def dfs(node):
        if not node:
            return
        if node in dic:
            return dic[node]

        new_node = Node(node.val)
        dic[node] = new_node  # 防止循环必须在此数标记状态

        new_node.neighbors = [dfs(n) for n in node.neighbors]
        return new_node

    return dfs(node)


if __name__ == '__main__':
    print('\n路径中捡到的最多钱')
    board = [100, 'o', 'o', 'o', 80]
    print(max_gift(board))

    print('\n是否可以到达')
    maze = [[1, 0, 0],
            [1, 0, 1],
            [0, 1, 1]]

    start = [0, 2]
    destination = [2, 0]
    print(has_path(maze, start, destination))

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
