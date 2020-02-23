#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def max_gift_1(matrix, row, col):
    """
    剑指offer 47题
    递归解法
    """
    rows, cols = matrix.shape
    if rows == 0 and cols == 0:
        return False
    if (row == rows - 1) and (col == cols - 1):
        return matrix[row][col]
    # last, row, turn right
    if (row == rows - 1) and (col < cols - 1):
        return matrix[row][col] + max_gift_1(matrix, row, col + 1)
    # last, col turn down
    if (row < rows - 1) and (col == cols - 1):
        return matrix[row][col] + max_gift_1(matrix, row + 1, col)
    # mid, turn right or down
    if (row < rows - 1) and (col < cols - 1):
        return matrix[row][col] + max(max_gift_1(matrix, row, col + 1),
                                      max_gift_1(matrix, row + 1, col))


def max_gift_2(matrix):
    """
    剑指offer 47题
    非递归解法
    """
    rows, cols = matrix.shape
    if rows == 0 and cols == 0:
        return False
    if rows == 1 or cols == 1:
        return np.sum(matrix)
    gift = np.zeros((rows, cols))

    gift[rows - 1][cols - 1] = matrix[rows - 1][cols - 1]
    # turn right 
    for col in range(cols - 2, -1, -1):
        gift[rows - 1][col] = gift[rows - 1][col + 1] + matrix[rows - 1][col]
    for row in range(rows - 2, -1, -1):
        gift[row][cols - 1] = gift[row + 1][cols - 1] + matrix[row][cols - 1]

    for col in range(cols - 2, -1, -1):
        for row in range(rows - 2, -1, -1):
            gift[col][row] = matrix[col][row] + max(gift[col][row + 1],
                                                    gift[col + 1][row])
    return gift[0][0]
