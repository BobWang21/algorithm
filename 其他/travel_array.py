#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 16:44:47 2017

@author: wangbao
"""

import numpy as np


# Diagonal traversal array

def diagonal_tra(data):
    n = data.shape[0]

    # upper half
    for i in range(n):
        x = 0
        y = i
        while (y >= 0):
            print(data[x][y], end=' ')
            x += 1
            y -= 1
        print('\n')

    # lower half
    for i in range(1, n):
        x = i
        y = n - 1
        while (x <= n - 1):
            print(data[x][y], end=' ')
            x += 1
            y -= 1
        print('\n')


def clock_tra(data):
    n = data.shape[0]
    rd = n // 2
    r = 0
    while (r <= rd):
        x = r
        y = r

        # y++
        while (y < n - r - 1):
            print(data[x][y], end=' ')
            y += 1
            # x++
        while (x < n - r - 1):
            print(data[x][y], end=' ')
            x += 1

            # y--
        while (y > r):
            print(data[x][y], end=' ')
            y -= 1
        # x--
        while (x > r):
            print(data[x][y], end=' ')
            x -= 1
        # print('\n')
        r += 1
    if n % 2 == 1:
        print(data[r - 1][r - 1])


if __name__ == '__main__':
    data = np.arange(1, 26).reshape(5, 5)
    print(data)
    # diagonal_tra(data)
    clock_tra(data)
