#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:02:41 2017

@author: wangbao
"""


class BiTree():
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None


# 创建完全二叉树
def create_full_binary_tree(lists):
    tree = BiTree()
    queue = []
    tree.data = lists[0]
    queue.append(tree)

    for data in lists[1:]:
        head = queue[0]
        if not head.left:
            new_tree = BiTree()
            new_tree.data = data
            head.left = new_tree
            queue.append(queue[0].left)
        elif not head.right:
            head.right = BiTree()
            head.right.data = data
            queue.append(head.right)
            queue.pop(0)
    return tree


def pre_order(tree):
    # 先序遍历(递归)
    if tree:
        print(tree.data)
        pre_order(tree.left)
        pre_order(tree.right)


def pre_order_1(tree):
    # 先序遍历(非递归先)
    stack = []
    if tree:
        stack.append(tree)

    while stack:
        tail = stack.pop(-1)
        print(tail.data, end=' ')  # 访问根节点
        # 右子树先进栈 
        if tail.right:
            stack.append(tail.right)  # 右子树先进栈
        # 左子树进栈
        if tail.left:
            stack.append(tail.left)


def traversal_left(tree):
    # 一直往左访问
    if not tree:
        return
    stack = []
    print(tree.data)

    while tree.left is not None:
        print(tree.left.data)
        if tree.right is not None:
            stack.append(tree.right)
        tree = tree.left
    return stack


def pre_order_2(tree):
    # 非递归遍历2
    stack = traversal_left(tree)

    while len(stack):
        tail = stack[-1]
        stack.pop(-1)
        new_stack = traversal_left(tail)
        stack += new_stack


def level_order(tree):
    # 层次遍历二叉树，使用队列保存信息
    queue = []
    if tree:
        queue.append(tree)

    while queue:
        head = queue[0]
        print(head.data, end=' ')

        if head.left is not None:
            queue.append(head.left)

        if head.right is not None:
            queue.append(head.right)
        queue.pop(0)


def transfer(tree):
    # 二叉树翻转
    if tree:
        tree.left, tree.right = tree.right, tree.left
        transfer(tree.left)
        transfer(tree.right)
    return tree


if __name__ == '__main__':
    lists = [i for i in range(10)]
    tree = create_full_binary_tree(lists)
    pre_order(tree)
    level_order(tree)
