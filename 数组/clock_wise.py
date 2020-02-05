#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 22:02:03 2017

@author: wangbao
"""


def diagnose_visit(data):
    rows, cols = data.shape

    # 上三角
    for i in range(cols):
        row = 0
        col = i
        while col >= 0:
            print(data[row, col], sep=' ')
            row += 1
            col -= 1
    # 下三角
    for i in range(1, rows):
        row = i
        col = cols - 1
        while row < rows:
            print(data[row, col], sep=' ')
            row += 1
            col -= 1


def clock_visit(data):
    cols, rows = data.shape
    iteration = min(cols + 1, rows + 1) >> 1

    for i in range(iteration):
        row, col = i, i
        # ->
        while (col < cols - i - 1):
            print(data[row, col], sep=' ')
            col += 1
        # down 
        while (row < rows - i - 1):
            print(data[row, col], sep=' ')
            row += 1

        # left 
        while col > i:
            print(data[row, col], sep=' ')
            col -= 1

        while row > i:
            print(data[row, col], sep=' ')
            row -= 1


def spiralPrint(m, n, a):
    i, k, l = 0, 0, 0
    # *  k - starting row index
    # - ending row index
    # l - starting column index
    # n - ending column index
    # i - iterator
    while (k < m and l < n):

        # Print the first row from the remaining rows
        for i in range(l, n):
            print(a[k][i])

        k += 1

        # // Print the last column from the remaining columns
        for i in range(k, m):
            print(a[i][n - 1])
        n -= 1

        # Print the last row from the remaining rows */
        if (k < m):

            for i in range(n - 1, l - 1, -1):
                print(a[m - 1][i])
            m -= 1

        # // Print the first column from the remaining columns */
        if (l < n):
            for i in range(m - 1, k - 1, -1):
                print(a[i][l])
            l += 1
