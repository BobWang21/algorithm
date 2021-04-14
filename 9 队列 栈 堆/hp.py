#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Heap():
    def __init__(self):
        self.__list = []

    def len(self):
        return len(self.__list)

    def _parent(self, child):  # 父亲, 没有父亲返回-1
        return (child - 1) // 2

    def _children(self, i):  # 获取孩子, 可返回空
        left, right = 2 * i + 1, 2 * (i + 1)
        return [i for i in (left, right) if i < self.len()]

    def _swap(self, i, j):
        self.__list[i], self.__list[j] = self.__list[j], self.__list[i]

    # 从父节点和孩子中选择最小值
    def _get_min(self, parent, children):
        min_v = self.get(parent)
        min_idx = parent
        for child in children:
            tmp = self.get(child)
            if tmp < min_v:
                min_v = tmp
                min_idx = child
        return min_idx

    def _up(self, child):  # 上滤
        while True:
            parent = self._parent(child)
            if parent == -1 or self.get(parent) <= self.get(child):
                break
            self._swap(parent, child)
            child = parent
        return

    def _down(self, parent):  # 下滤
        while True:
            children = self._children(parent)
            min_idx = self._get_min(parent, children)
            if min_idx == parent:
                break
            self._swap(min_idx, parent)
            parent = min_idx
        return

    def get(self, i):  # 取值
        if self.__list:
            return self.__list[i]

    def push(self, v):
        self.__list.append(v)  # 1 插入尾部
        i = self.len() - 1  # 2 上滤
        self._up(i)

    def pop(self):  # 删除堆顶元素 将队尾元素放入堆首, 下滤
        if not self.len():
            return
        if self.len() == 1:
            return self.__list.pop(0)
        v = self.get(0)
        tail = self.__list.pop(-1)
        self.__list[0] = tail
        self._down(0)
        return v

    def heapify(self, lists):  # 批量建堆, 自下而上的下滤 o(n)
        self.__list = [v for v in lists]
        n = self.len() // 2 - 1
        for i in range(n, -1, -1):
            self._down(i)


if __name__ == '__main__':
    heap = Heap()
    heap.push(7)
    heap.push(2)
    heap.push(10)
    heap.push(-1)
    print(heap.pop(), end=',')
    heap.push(4)
    for i in range(4):
        print(heap.pop(), end=',')

    print()
    heap.heapify([7, 2, 10, -1, 4])
    for i in range(5):
        print(heap.pop(), end=',')
