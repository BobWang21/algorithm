#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
线段树 区间最大值/最小值/和
"""


class SegmentTree():
    def __init__(self, sum, start, end, left=None, right=None):
        self.sum = sum
        self.left = left
        self.right = right
        self.start = start  # 索引起点
        self.end = end  # 索引终点


def build_tree(start, end, nums):
    if not nums:
        return
    if start == end:
        return SegmentTree(nums[start], start, end)
    mid = (start + end) // 2
    left = build_tree(start, mid, nums)
    right = build_tree(mid + 1, end, nums)
    node = SegmentTree(left.sum + right.sum, start, end, left, right)
    return node


def update_tree(root, index, val):
    if root.start == root.end == index:
        root.sum = val
        return
    mid = (root.start + root.end) // 2
    if index <= mid:
        update_tree(root.left, index, val)
    else:
        update_tree(root.right, index, val)

    root.sum = root.left.sum + root.right.sum


def range_sum(root, i, j):
    if root.start == i and root.end == j:
        return root.sum
    mid = (root.start + root.end) // 2
    if j <= mid:
        return range_sum(root.left, i, j)
    if i > mid:
        return range_sum(root.right, i, j)
    return range_sum(root.left, i, mid) + range_sum(root.right, mid + 1, j)


def level(root):
    queue = [root]
    while queue:
        node = queue.pop(0)
        print((node.start, node.end), node.sum)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)


if __name__ == '__main__':
    nums = [2, 1, 5, 3, 4]
    tree = build_tree(0, 4, nums)

    update_tree(tree, 1, 2)
    level(tree)
    # nums = [2, 2, 5, 3, 4]
    print(range_sum(tree, 0, 3))
