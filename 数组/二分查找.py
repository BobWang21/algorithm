#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:02:31 2017

@author: wangbao
"""


# 3次比较
def binary_search_1(lists, value):
    '''
     3log(n) 3次比较
    '''
    low = 0
    high = len(lists)
    while low < high:  # 一次比较, 查找区间为0
        mid = (low + high) >> 1
        # print(low, high, mid)
        if value < lists[mid]:  # 两次比较
            high = mid - 1
        elif lists[mid] < value:  # 两次比较
            low = mid + 1
        else:
            return mid
    return -1


# 1次比较
def binary_search_2(lists, value):
    low = 0
    high = len(lists)
    while high - low > 1: # 查找区间为1
        mid = (low + high) // 2
        print(low, high, mid)
        if value < lists[mid]:
            high = mid
        else:
            low = mid
    if lists[low] == value:
        return low
    else:
        return -1


def binary_search_3(lists, value):
    # 返回不大于value的最后一个数
    low, high = 0, len(lists)
    while low < high:
        mid = (low + high) >> 1
        print(low, high, mid)
        if lists[mid] <= value:
            low = mid + 1
        else:
            high = mid
    low -= 1
    return low


if __name__ == '__main__':
    data = [1, 3, 5, 9, 10, 16, 17]
    print(binary_search_2(data, 3))
