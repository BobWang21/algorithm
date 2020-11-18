#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from binary_tree import *


class BSTNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.cnt = 1  # 节点频次
        self.left_cnt = 0  # 左节点个数


# 判断二叉树是否为二叉搜索树
# 中序遍历中判断是否递增
def is_valid_bst(tree):
    if not tree:
        return True

    def helper(tree, lo, hi):
        if not tree:
            return True
        v = tree.val
        if v <= lo or v >= hi:
            return False
        return helper(tree.left, lo, v) and helper(tree.right, v, hi)

    return helper(tree, -float('inf'), float('inf'))


# 二叉搜索树第K大的数 中序遍历
def top_k_binary_search_tree(tree, k):
    if not tree:
        return
    stack = []
    node = tree
    count = 0
    while node or stack:
        while node:
            stack.append(node)
            node = node.right
        node = stack.pop(-1)
        count += 1
        if count == k:
            return node.val
        node = node.left


# 二叉搜索树第K大的数 递归
def top_k_binary_search_tree2(tree, k):
    if not tree:
        return
    res = [k]

    def helper(tree):
        if not tree:
            return
        if res[0] <= 0:  # 提前返回
            return
        l, r = tree.left, tree.right
        if r:
            helper(r)
        res[0] -= 1
        if not res[0]:
            res.append(tree.val)
            return
        if l:
            helper(l)

    helper(tree)
    return res[1]


# Given a Binary Search Tree (BST), convert it to a Greater Tree such that
# every key of the original BST is changed to the original key plus sum of
# all keys greater than the original key in BST.
# 538. Convert BST to Greater Tree
def convert_bst_2_greater_tree(tree):
    node = tree
    stack = []
    total = 0
    res = []
    while node or stack:
        while node:
            stack.append(node)
            node = node.right
        if stack:
            node = stack.pop(-1)
            node.val = node.val + total
            res.append(node.val)
            total = node.val
            node = node.left
    return res


# 检查一个遍历是否为二叉搜索树的后序遍历
def check_poster_order(post):
    if len(post) <= 1:
        return True

    root = post[0]
    if post[0] > root or post[-2] < root:  # 仅有左子树 或 右子树
        return check_poster_order(post[:-1])
    l, n = 0, len(post)
    # 根据左子树 < 根 < 右子树 切分左右子树
    while l < n - 1 and post[l] < root:
        l += 1
    p = l
    while l < n - 1 and post[l] > root:
        l += 1
    if l != n - 1:
        return False
    return check_poster_order(post[:p]) and check_poster_order(post[p:-1])


# trim 二叉搜索树
def trimBST(tree, L, R):
    if not tree:
        return
    if tree.val < L:
        return trimBST(tree.right, L, R)
    if tree.val > R:
        return trimBST(tree.left, L, R)
    left = trimBST(tree.left, L, R)
    right = trimBST(tree.right, L, R)
    tree.left = left
    tree.right = right
    return tree


# 450. 删除二叉搜索树中的节点
def delete_node(node, key):
    if not node:
        return None

    if node.val == key:
        if not node.left:
            return node.right
        if not node.right:
            return node.left
        # 选择左子树最大点 作为根节点
        tmp = node.left
        while tmp.right:
            tmp = tmp.right
        node.val = tmp.val
        node.left = delete_node(node.left, node.val)
        return node

    if key < node.val:
        node.left = delete_node(node.left, key)
        return node

    if key > node.val:
        node.right = delete_node(node.right, key)
        return node


# 315. 计算右侧小于当前元素的个数
# 时间复杂度为 log(1) + log(2) + log(n) = O(nlog(n))
def counter_smaller(nums):
    if not nums:
        return []

    def insert(tree, v, cnt=0):  # cnt 从上到下记录
        if not tree:
            node = BSTNode(v)
            return node, cnt
        if tree.val == v:
            tree.cnt += 1
            return tree, cnt + tree.left_cnt
        if v > tree.val:
            right, cnt = insert(tree.right, v, cnt + tree.left_cnt + tree.cnt)
            tree.right = right
            return tree, cnt
        if v < tree.val:
            tree.left_cnt += 1
            left, cnt = insert(tree.left, v, cnt)
            tree.left = left
            return tree, cnt

    tree = BSTNode(float('inf'))
    n = len(nums)
    res = [0] * n
    for i in range(n - 1, -1, -1):
        res[i] = insert(tree, nums[i])[1]
    return res


if __name__ == '__main__':
    tree = create_full_binary_tree([5, 3, 7])
    print('\nbst 转换成 greater_tree')
    print(convert_bst_2_greater_tree(tree))

    print('\n检查二叉搜索树后续遍历是否合法')
    print(check_poster_order([1, 6, 3, 2, 5]))

    print('\n二叉搜索树的第k大节点')
    tree = create_full_binary_tree([5, 3, 7, 2, 4, 6, 8])
    print(top_k_binary_search_tree2(tree, 1))

    print('\ntrimBST')
    tree = create_full_binary_tree([10, 1, 15, 3, 4, 12, 17])
    print(level_traversal(trimBST(tree, 4, 12)))

    print('\n逆向个数')
    print(counter_smaller([2, 0, 1]))
