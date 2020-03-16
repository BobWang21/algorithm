#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 22:02:03 2017

@author: wangbao
"""


def diagnose_traverse(data):
    rows, cols = len(data), len(data[0])

    # 上三角
    for i in range(cols):
        row = 0
        col = i
        while col >= 0:
            print(data[row][col], sep=' ')
            row += 1
            col -= 1
    # 下三角
    for i in range(1, rows):
        row = i
        col = cols - 1
        while row < rows:
            print(data[row][col], sep=' ')
            row += 1
            col -= 1


def clockwise_traverse(data):
    cols, rows = len(data), len(data[0])
    iteration = min(cols + 1, rows + 1) >> 1

    for i in range(iteration):
        row, col = i, i
        # left
        while col < cols - i - 1:
            print(data[row][col], sep=' ')
            col += 1
        # down 
        while row < rows - i - 1:
            print(data[row][col], sep=' ')
            row += 1

        # right
        while col > i:
            print(data[row][col], sep=' ')
            col -= 1
        # up
        while row > i:
            print(data[row][col], sep=' ')
            row -= 1


if __name__ == '__main__':
    matrix = [[1, 10, 3, 8],
              [12, 2, 9, 6],
              [5, 7, 4, 11],
              [3, 7, 16, 5]]
    print(diagnose_traverse(matrix))
    print(clockwise_traverse(matrix))
