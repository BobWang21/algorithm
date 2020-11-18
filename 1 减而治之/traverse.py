#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pprint import pprint


def diagnose_traverse(matrix):
    rows, cols = len(matrix), len(matrix[0])
    res = []
    # 上三角
    for i in range(cols):
        row = 0
        col = i
        while col >= 0:
            res.append(matrix[row][col])
            row += 1
            col -= 1
    # 下三角
    for i in range(1, rows):
        row = i
        col = cols - 1
        while row < rows:
            res.append(matrix[row][col])
            row += 1
            col -= 1
    return res


# 54. 螺旋矩阵
def spiral_order(matrix):
    if not matrix:
        return []
    left, right, top, bottom = 0, len(matrix[0]) - 1, 0, len(matrix) - 1
    res = []
    while True:
        # left to right
        for i in range(left, right + 1):
            res.append(matrix[top][i])
        top += 1
        if top > bottom:
            break

        # top to bottom
        for i in range(top, bottom + 1):
            res.append(matrix[i][right])
        right -= 1
        if left > right:
            break

        # right to left
        for i in range(right, left - 1, -1):
            res.append(matrix[bottom][i])
        bottom -= 1
        if top > bottom:
            break

        # bottom to top
        for i in range(bottom, top - 1, -1):
            res.append(matrix[i][left])
        left += 1
        if left > right:
            break
    return res


def spiral_order2(matrix):
    if not matrix:
        return

    res = []
    while matrix:
        # top
        res.extend(matrix.pop(0))
        # right
        if matrix:
            for i in range(len(matrix)):
                if matrix[i]:
                    v = matrix[i].pop(-1)
                    res.append(v)
        # down
        if matrix:
            line = matrix.pop(-1)
            line.reverse()
            res.extend(line)

        # left
        if matrix:
            for i in range(len(matrix) - 1, -1, -1):
                if matrix[i]:
                    v = matrix[i].pop(0)
                    res.append(v)
    return res


def generate_matrix(n):
    left, right, top, bottom = 0, n - 1, 0, n - 1
    matrix = [[0] * n for _ in range(n)]
    num, tar = 1, n * n
    while num <= tar:
        for i in range(left, right + 1):  # left to right
            matrix[top][i] = num
            num += 1
        top += 1
        for i in range(top, bottom + 1):  # top to bottom
            matrix[i][right] = num
            num += 1
        right -= 1
        for i in range(right, left - 1, -1):  # right to left
            matrix[bottom][i] = num
            num += 1
        bottom -= 1
        for i in range(bottom, top - 1, -1):  # bottom to top
            matrix[i][left] = num
            num += 1
        left += 1
    return matrix


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


# 73. 矩阵置零 如果一个元素为0，则将其所在行和列的所有元素都设为0。
# 和使用负号标记 有异曲同工之妙
def set_zeroes(matrix):
    if not matrix or not matrix[0]:
        return

    rows, cols = len(matrix), len(matrix[0])
    first_row_zero = first_col_zero = False

    for j in range(cols):
        if not matrix[0][j]:
            first_row_zero = True
            break

    for i in range(rows):
        if not matrix[i][0]:
            first_col_zero = True
            break

    for i in range(1, rows):
        for j in range(1, cols):
            if not matrix[i][j]:
                matrix[i][0] = 0
                matrix[0][j] = 0

    for i in range(1, rows):
        for j in range(1, cols):
            if not matrix[i][0] or not matrix[0][j]:
                matrix[i][j] = 0

    if first_row_zero:
        matrix[0] = [0] * cols
    if first_col_zero:
        for i in range(rows):
            matrix[i][0] = 0


if __name__ == '__main__':
    print('\n对角线访问')
    matrix = [[1, 10, 3, 8],
              [12, 2, 9, 6],
              [5, 7, 4, 11],
              [3, 7, 16, 5]]
    print(diagnose_traverse(matrix))

    print('\n顺时针访问')
    print(spiral_order(matrix))

    print('\n顺时针生成数组')
    print(generate_matrix(3))

    print('\n顺时针旋转90度')
    matrix = [[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]]
    rotate(matrix)
    print(matrix)

    print('\n矩阵置0')
    matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    set_zeroes(matrix)
    pprint(matrix)
