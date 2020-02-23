#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def quick_sort(arr, lo, hi):
    if lo < hi:  # 此处为小于号 而不是等于号
        mid = partition(arr, lo, hi)
        quick_sort(arr, lo, mid - 1)
        quick_sort(arr, mid + 1, hi)


def partition(arr, lo, hi):
    pivot = arr[lo]
    while lo < hi:
        while lo < hi and pivot <= arr[hi]:
            hi -= 1
        arr[lo] = arr[hi]  # 替换已保持的数据
        while lo < hi and arr[lo] <= pivot:
            lo += 1
        arr[hi] = arr[lo]
    arr[lo] = pivot
    return lo


# 比 pivot小的全部放在左边
def partition2(arr, lo, hi):
    i = lo - 1  # index of smaller element
    pivot = arr[hi]  # pivot
    for j in range(lo, hi):  # 不包含hi
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1


if __name__ == '__main__':
    lists = [4, 3, 8, 9, 7]
    partition2(lists, 0, 4)
    # quick_sort(lists, 0, 2)
    print(lists)
