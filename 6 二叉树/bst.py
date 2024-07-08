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


# 98 判断二叉树是否为二叉搜索树
# 中序遍历中判断是否递增
def is_valid_bst(tree):
    if not tree:
        return True

    def helper(tree, lo, hi):
        if not tree:
            return True
        v = tree.val
        return lo < v < hi and helper(tree.left, lo, v) and helper(tree.right, v, hi)

    return helper(tree, -float('inf'), float('inf'))


# 二叉搜索树第K大的数 非递归
def top_k_binary_search_tree(node, k):
    if not node:
        return

    node, stack = node, []
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
def top_k_binary_search_tree2(node, k):
    if not node:
        return
    res = [k]

    def helper(tree):
        if not tree:
            return
        if res[0] == 0:  # 提前返回
            return
        # 第k大, 先访问右子树
        helper(tree.right)
        res[0] -= 1  # 理解递归
        if not res[0]:
            res.append(tree.val)
            return
        helper(tree.left)

    helper(node)
    return res[1]


# 538. Convert BST to Greater Tree
def convertBST1(root):
    total = 0
    node, stack = root, []
    while node or stack:
        while node:
            stack.append(node)
            node = node.right
        node = stack.pop(-1)
        total += node.val
        node.val = total
        node = node.left
    return root


def convertBST2(root):
    total = [0]

    def helper(node):
        if not node:
            return
        helper(node.right)
        total[0] += node.val
        node.val = total[0]
        helper(node.left)
        return node

    return helper(root)


# 剑指Offer33 检查一个遍历是否为二叉搜索树的后序遍历
def verify_postorder(post):
    if len(post) <= 1:  # 只有一侧子树
        return True
    root = post[-1]
    i, n = 0, len(post)
    # 根据左子树 < 根 < 右子树 切分左右子树
    while i < n - 1 and post[i] < root:
        i += 1
    p = i
    while i < n - 1 and post[i] > root:
        i += 1
    # 只有一侧子树也可
    return i == n - 1 and verify_postorder(post[:p]) and verify_postorder(post[p:n - 1])


# 669 trim 二叉搜索树
def trimBST(root, low, high):
    if not root:
        return

    if low <= root.val <= high:
        root.left = trimBST(root.left, low, high)
        root.right = trimBST(root.right, low, high)
        return root

    if root.val < low:
        return trimBST(root.right, low, high)

    if root.val > high:
        return trimBST(root.left, low, high)


# 450. 删除二叉搜索树中的节点
def delete_node(root, key):
    if not root:
        return None

    if root.val == key:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        # 选择左子树最大点 作为根节点
        cur_node = root.left
        while cur_node.right:
            cur_node = cur_node.right
        val = cur_node.val
        root.val = val
        root.left = delete_node(root.left, val)
        return root

    if key < root.val:
        root.left = delete_node(root.left, key)
        return root

    if key > root.val:
        root.right = delete_node(root.right, key)
        return root


# 315. 计算右侧小于当前元素的个数
# 时间复杂度为 log(1) + log(2) + log(n) = O(nlog(n))
def counter_smaller(nums):
    if not nums:
        return []

    def insert(root, val, cnt=0):  # cnt 从上到下记录
        if not root:
            node = BSTNode(val)
            return node, cnt
        if root.val == val:
            root.cnt += 1
            return root, cnt + root.left_cnt
        if val > root.val:
            right, cnt = insert(root.right, val, cnt + root.left_cnt + root.cnt)
            root.right = right
            return root, cnt
        if val < root.val:
            root.left_cnt += 1
            left, cnt = insert(root.left, val, cnt)
            root.left = left
            return root, cnt

    tree = BSTNode(float('inf'))
    n = len(nums)
    res = [0] * n
    for i in range(n - 1, -1, -1):
        res[i] = insert(tree, nums[i])[1]
    return res


if __name__ == '__main__':
    tree = create_full_binary_tree([5, 3, 7])
    print('\nbst 转换成 greater_tree')
    print(convertBST1(tree))

    print('\n检查二叉搜索树后续遍历是否合法')
    print(verify_postorder([1, 6, 3, 2, 5]))

    print('\n二叉搜索树的第k大节点')
    tree = create_full_binary_tree([5, 3, 7, 2, 4, 6, 8])
    print(top_k_binary_search_tree2(tree, 1))

    print('\ntrimBST')
    tree = create_full_binary_tree([10, 1, 15, 3, 4, 12, 17])
    print(level_traversal(trimBST(tree, 4, 12)))

    print('\n逆向个数')
    print(counter_smaller([2, 0, 1]))
