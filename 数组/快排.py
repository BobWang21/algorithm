#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def quick_sort(lists, lo, hi):
    if lo < hi:  # 此处为小于号 而不是等于号
        mi = partition(lists, lo, hi)
        quick_sort(lists, lo, mi - 1)
        quick_sort(lists, mi + 1, hi)


def partition(lists, lo, hi):
    print('low:%i, hi:%i' % (lo, hi))
    pivot = lists[lo]
    while lo < hi:
        while lo < hi and pivot <= lists[hi]:
            hi -= 1
        lists[lo] = lists[hi]
        while lo < hi and lists[lo] <= pivot:
            lo += 1
        lists[hi] = lists[lo]
    lists[lo] = pivot
    return lo


if __name__ == '__main__':
    data = [6, 1, 4]
    quick_sort(data, 0, 2)
    print(data)
