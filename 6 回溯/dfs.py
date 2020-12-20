#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pprint import pprint


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


# 200. 岛屿数量 也可用 union find
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


# 从某一位置出发,判断是否连通 不回溯
def maze_can_reach(maze, start, destination):
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
        if [i, j] == destination[1]:
            return True

        # 标记访问的点
        maze[i][j] = wall

        for direction in directions:
            if dfs(i + direction[0], j + direction[1]):
                return True
        return False

    print('before')
    pprint(maze)
    res = dfs(start[0], start[1])
    print('after')
    pprint(maze)
    return res


# 从某一位置出发 判断是否连通 回溯 不改变maze
def maze_can_reach2(maze, start, destination):
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
        if [i, j] == destination[1]:
            return True

        # 标记已经访问的点
        maze[i][j] = wall

        for direction in directions:
            if dfs(i + direction[0], j + direction[1]):
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


# 130 将围住的'O'变成'X' 任何边界上的'O'都不会被填充为'X'
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


# 回溯思想
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


# 错误版# # # # # # # # # # # # # # # # # # # # # #
def surrounded_regions_wrong(board):
    if not board:
        return
    rows, cols = len(board), len(board[0])
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(i, j):
        if i < 0 or j < 0 or i == rows or j == cols:
            return False

        if board[i][j] == 'X':
            return True

        board[i][j] = 'X'
        for direction in directions:
            # 若之前的三个方向正确 则无法回溯
            if not dfs(i + direction[0], j + direction[1]):
                board[i][j] = 'O'
                return False

        return True

    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                dfs(i, j)
    return board


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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


# 329. 矩阵中的最长递增路径
# dfs + 记忆 dp[i][j] = max(dp[x][y]) + 1
def longest_increasing_path(matrix):
    if not matrix or not matrix[0]:
        return 0

    dic = {}
    rows, cols = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def helper(i, j):
        if (i, j) in dic:
            return dic[(i, j)]
        res = 0
        for direction in directions:
            row, col = i + direction[0], j + direction[1]
            if row < 0 or row == rows or col < 0 or col == cols:
                continue
            if matrix[i][j] > matrix[row][col]:  # 递增可以保证路径点不重复访问
                res = max(res, helper(row, col))

        res += 1
        dic[(i, j)] = res
        return res

    res = 1
    for i in range(rows):
        for j in range(cols):
            res = max(res, helper(i, j))
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


# 单词搜索
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

        char = board[i][j]
        board[i][j] = '.'
        for d in directions:
            row, col = i + d[0], j + d[1]
            if dfs(row, col, k + 1):
                return True
        board[i][j] = char
        return False

    for i in range(rows):
        for j in range(cols):
            if dfs(i, j, 0):
                return True
    return False


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
    print(maze_can_reach(maze, start, destination))

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
