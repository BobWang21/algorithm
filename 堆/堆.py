#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 22:06:38 2017

@author: wangbao
"""


class Heap():
    def __init__(self, value):
        self.value = value
        self.length = len(value)

    def heapify(self):
        # 非叶子节点
        first_no_leaf = (self.length >> 1) - 1
        # 从下而上 下溢
        for index in range(first_no_leaf, -1, -1):
            # print(index)
            self.down_flow(index)

    # 下溢
    def down_flow(self, index):
        '''
        非叶子节点自下而上的下溢 来构建堆
        '''
        # 判断是否为非叶子节点 
        while 2 * index + 1 < self.length:

            left = self.lchild(index)
            right = self.rchild(index)
            # print('left, right', left, right)
            # 是否有右孩子
            if right > -1:
                # 3个 if 出错了！！！ 
                if self[left] > self[index] and self[left] >= self[right]:
                    # print(index, left)
                    self.swap(index, left)
                    index = left

                elif self[right] > self[index] and self[right] > self[left]:
                    # print(index, right)
                    self.swap(index, right)
                    index = right

                elif self[index] >= self[left] and self[index] >= self[right]:
                    break
            else:
                if self[left] > self[index]:
                    self.swap(index, left)
                    # print(index, left)
                    index = left
                else:
                    break

    # 上溢
    def up_flow(self, index):

        # 判断是否为根节点，如果存在子节点 那么子节点为2*index
        while index > 0:
            father = self.father(index)

            if self[father] < self[index]:
                self.swap(index, father)
                index = father
            else:
                break

    def swap(self, i, j):
        t = self.value[i]
        self.value[i] = self.value[j]
        self.value[j] = t

        # 父节点

    def father(self, index):
        if index == 0:
            return -1
        if index % 2 == 1:
            return (index - 1) >> 1
        # `>>` 优先级小于 减号
        return (index >> 1) - 1

    # 左孩子
    def lchild(self, index):
        if 2 * index + 1 < self.length:
            return 2 * index + 1
        return -1

    # 右孩子
    def rchild(self, index):
        if 2 * (index + 1) < self.length:
            return 2 * (index + 1)
        return -1

    def add(self, x):
        self.value.append(x)
        self.length += 1
        index = self.length - 1
        self.up_flow(index)

    def pop(self):
        top = self.value[0]
        index = self.length - 1
        self.value[0] = self.value[index]
        self.value.pop(-1)
        self.length -= 1
        self.down_flow(0)
        return top

    def __getitem__(self, index):
        return self.value[index]


if __name__ == '__mian__':
    data = []
