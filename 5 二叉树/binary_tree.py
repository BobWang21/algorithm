#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:02:41 2017

@author: wangbao
"""


class TreeNode():
    def __init__(self):
        self.val = None
        self.left = None
        self.right = None


# 创建完全二叉树
def create_full_binary_tree(nums):
    tree = TreeNode()
    tree.val = nums[0]
    queue = [tree]
    for val in nums[1:]:
        head = queue[0]
        if not head.left:
            node = TreeNode()
            node.val = val
            head.left = node
            queue.append(node)
        elif not head.right:
            node = TreeNode()
            node.val = val
            head.right = node
            queue.append(node)
            queue.pop(0)  # 出队列
    return tree


# 先序遍历(递归)
def preorder_traversal(tree):
    if tree:
        print(tree.val, end=' ')
        preorder_traversal(tree.left)
        preorder_traversal(tree.right)


# 先序遍历(非递归) dfs
def preorder2(tree):
    if not tree:
        return
    stack = [tree]
    res = []
    while stack:
        node = stack.pop(-1)
        res.append(node.val)
        left, right = node.left, node.right
        # 右子树先进栈 
        if right:
            stack.append(right)  # 右子树先进栈
        # 左子树进栈
        if left:
            stack.append(left)
    return res


# 中序遍历 左子树一直向下
# 访问到头就出栈
def inorder_traversal(tree):
    if not tree:
        return
    res = []
    node, stack = tree, []
    while node or stack:
        while node:  # 一直往左
            stack.append(node)
            node = node.left
        node = stack.pop(-1)
        res.append(node.val)
        node = node.right
    return res


# 层次遍历二叉树 使用一个queue bfs
def level_traversal(tree):
    if not tree:
        return
    queue = [tree]
    res = []
    while queue:
        node = queue.pop(0)
        res.append(node.val)
        left, right = node.left, node.right
        if left:
            queue.append(left)
        if right:
            queue.append(right)
    return res


# 层次遍历2 使用2个queue
def level_traversal2(root):
    if not root:
        return []
    curr_level = [root]
    res = []
    while curr_level:
        next_level = []
        for node in curr_level:  # 访问当前层的每个节点, 并将叶节点放入下一层
            res.append(node.val)
            left, right = node.left, node.right
            if left:
                next_level.append(left)
            if right:
                next_level.append(right)
        curr_level = next_level
    return res


def level_num(tree):
    if not tree:
        return 0
    return max(level_num(tree.left), level_num(tree.right)) + 1


# 最小深度 根节点到叶节点! 也可以使用bfs
def min_depth(root):
    if not root:
        return 0
    l, r = root.left, root.right
    if not l:  # 没有左子树
        return min_depth(r) + 1
    if not r:
        return min_depth(l) + 1
    return min(min_depth(l), min_depth(r)) + 1


if __name__ == '__main__':
    nums = [i for i in range(7)]
    tree = create_full_binary_tree(nums)
    print('\n先序遍历')
    preorder_traversal(tree)
    print('\n中序遍历')
    print(inorder_traversal(tree))
    print('\n层次遍历')
    level_traversal(tree)

    print('\n数的最小深度')
    root = TreeNode()
    root.val = 1
    a = TreeNode()
    a.val = 2
    b = TreeNode()
    b.val = 3
    root.left = a
    a.left = b
    print(min_depth(root))
