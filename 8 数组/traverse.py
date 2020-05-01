#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def diagnose_traverse(data):
    rows, cols = len(data), len(data[0])
    res = []
    # 上三角
    for i in range(cols):
        row = 0
        col = i
        while col >= 0:
            res.append(data[row][col])
            row += 1
            col -= 1
    # 下三角
    for i in range(1, rows):
        row = i
        col = cols - 1
        while row < rows:
            res.append(data[row][col])
            row += 1
            col -= 1
    return res


def clockwise_traverse(data):
    cols, rows = len(data), len(data[0])
    iteration = min(cols + 1, rows + 1) >> 1
    res = []
    for i in range(iteration):
        row, col = i, i
        # left
        while col < cols - i - 1:
            res.append(data[row][col])
            col += 1
        # down
        while row < rows - i - 1:
            res.append(data[row][col])
            row += 1

        # right
        while col > i:
            res.append(data[row][col])
            col -= 1
        # up
        while row > i:
            res.append(data[row][col])
            row -= 1
    return res


# 顺时针旋转90度
def rotate(matrix):
    def reverse(l, h):
        while l < h:
            matrix[l], matrix[h] = matrix[h], matrix[l]
            l += 1
            h -= 1

    if not matrix or not matrix[0]:
        return matrix
    n = len(matrix)

    reverse(0, n - 1)

    for i in range(n):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


if __name__ == '__main__':
    print('\n逆时针访问')
    matrix = [[1, 10, 3, 8],
              [12, 2, 9, 6],
              [5, 7, 4, 11],
              [3, 7, 16, 5]]
    print(diagnose_traverse(matrix))

    print('\n顺时针访问')
    print(clockwise_traverse(matrix))

    print('\n顺时针旋转90度')
    matrix = [[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]]
    rotate(matrix)
    print(matrix)
