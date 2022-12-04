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
                queue.append((x, y, layer + 1))
                maze[x][y] = wall  # 标记经访问过的点
    return -1


# 286 多向BFS
# 时间复杂度: O(mn) 空间: O(mn)
def walls_and_gates(rooms):
    if not rooms or not rooms[0]:
        return
    rows, cols = len(rooms), len(rooms[0])
    inf, queue, = 2147483647, []
    # 加入多个源
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


# 127 Input:
# beginWord = "hit",
# endWord = "cog",
# wordList = ["hot","dot","dog","lot","log","cog"]
def word_ladder(begin_word, end_word, word_list):
    if not word_list:
        return 0
    queue = [(begin_word, 0)]  # 记录层数
    visited = {begin_word}  # 保存已加入过队列的字符串
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    while queue:
        word, l = queue.pop(0)
        for i in range(len(word)):
            for j in range(26):  # 访问每个字符串的近邻
                new_word = word[:i] + chr(ord('a') + j) + word[i + 1:]
                if new_word in word_set and new_word not in visited:
                    if word == end_word:
                        return l + 1
                    visited.add(new_word)  # 如果出队列时 再判断会有重复
                    queue.append((new_word, l + 1))

    return 0


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

    print('\n Word Ladder')
    a = "qa"
    b = "sq"
    c = ["si", "go", "se", "cm", "so", "ph", "mt", "db", "mb", "sb", "kr", "ln", "tm", "le", "av", "sm", "ar", "ci",
         "ca", "br", "ti", "ba", "to", "ra", "fa", "yo", "ow", "sn", "ya", "cr", "po", "fe", "ho", "ma", "re", "or",
         "rn", "au", "ur", "rh", "sr", "tc", "lt", "lo", "as", "fr", "nb", "yb", "if", "pb", "ge", "th", "pm", "rb",
         "sh", "co", "ga", "li", "ha", "hz", "no", "bi", "di", "hi", "qa", "pi", "os", "uh", "wm", "an", "me", "mo",
         "na", "la", "st", "er", "sc", "ne", "mn", "mi", "am", "ex", "pt", "io", "be", "fm", "ta", "tb", "ni", "mr",
         "pa", "he", "lr", "sq", "ye"]
    print(word_ladder(a, b, c))
