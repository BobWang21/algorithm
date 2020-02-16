#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:02:41 2017

@author: wangbao
"""

from binary_tree import *


def level(tree):
    if not tree:
        return
    current_level = [tree]
    next_level = []
    while current_level:
        node = current_level.pop(0)
        print(node.val, end=' ')
        left, right = node.left, node.right
        if left:
            next_level.append(left)
        if right:
            next_level.append(right)
        if not current_level:
            current_level = next_level
            next_level = []
    print()


def level_num(tree):
    if not tree:
        return 0
    return max(level_num(tree.left), level_num(tree.right)) + 1


# 最近公共祖先
def lowest_common_ancestor(tree, p, q):
    if not tree:
        return None
    if tree.val == p or tree.val == q:
        return tree.val
    left = lowest_common_ancestor(tree.left, p, q)
    right = lowest_common_ancestor(tree.right, p, q)
    if left is not None and right is not None:
        return tree.val
    if left is not None:
        return left
    if right is not None:
        return right


# 判断树是否平衡
# 如果平衡返回树的高度, 不平衡返回-1
def is_balanced(tree):
    def helper(tree):
        if not tree:
            return 0
        left = helper(tree.left)
        right = helper(tree.right)
        if left == -1 or right == -1:
            return -1
        if abs(right - left) > 1:
            return -1
        return max(left, right) + 1

    return helper(tree) != -1


# 递归!!!
# 打印所有路径
def binary_tree_paths(tree):
    if not tree:
        return []
    left = binary_tree_paths(tree.left)
    right = binary_tree_paths(tree.right)
    paths = left + right
    if not paths:
        return [str(tree.val)]
    res = []
    for path in paths:
        res.append(str(tree.val) + '->' + path)
    return res


# 借鉴递归思想
def binary_tree_paths2(tree):
    def helper(tree, path, res):
        if not tree.left and not tree.right:
            res.append(path + [tree.val])
            return
        if tree.left:
            helper(tree.left, path + [tree.val], res)
        if tree.right:
            helper(tree.right, path + [tree.val], res)

    res = []
    helper(tree, [], res)
    return res


# 和为某个数的路径
def is_has_path_sum(root, num):
    if not root:
        return False
    if root.val == num and not root.left and not root.right:  # 叶节点
        return True
    return is_has_path_sum(root.left, num - root.val) or is_has_path_sum(root.right, num - root.val)


# 打印所有路径 从根节点到叶节点和为某个数的路径
def has_path_sum(tree, target):
    def helper(tree, target, path, res):
        if not tree.left and not tree.right and target == tree.val:
            res.append(path + [tree.val])
            return
        if not tree.left and not tree.right:
            return
        if tree.left:
            helper(tree.left, target - tree.val, path + [tree.val], res)
        if tree.right:
            helper(tree.right, target - tree.val, path + [tree.val], res)

    res = []
    helper(tree, target, [], res)
    return res


# 右侧看到的节点
# 层次遍历
def right_side_view(tree):
    if not tree:
        return
    current_level = [tree]
    res = []
    while current_level:
        next_level = []
        for node in current_level:
            left, right = node.left, node.right
            if left:
                next_level.append(left)
            if right:
                next_level.append(right)
        res.append(node.val)
        current_level = next_level
    return res


# 锯齿形层次遍历。
# 即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行
def zigzag_level_order(tree):
    if not tree:
        return
    current_level = [tree]
    res = []
    direction = 1
    while current_level:
        next_level = []
        level_value = []
        for node in current_level:
            level_value.append(node.val)
            left, right = node.left, node.right
            if right:
                next_level.append(right)
            if left:
                next_level.append(left)

        res.append(level_value[::direction])
        direction *= -1
        current_level = next_level
    return res


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


# 508. Most Frequent Subtree Sum
def most_frequent_subtree_sum(tree):
    dic = {}

    def helper(tree):
        total = 0
        if not tree:
            return total

        left_sum = helper(tree.left)
        right_sum = helper(tree.right)
        total = tree.val + left_sum + right_sum
        dic.setdefault(total, 0)
        dic[total] += 1
        return total

    helper(tree)
    max_num = -1
    max_key = None
    for key, val in dic.items():
        if val > max_num:
            max_num = val
            max_key = key
    return max_key


# 213 21 + 23 = 46
def sum_numbers(tree):
    # top_down 记录父节点的值
    def helper(tree, val):
        if not tree:
            return 0
        if not tree.left and not tree.right:
            return val * 10 + tree.val
        return helper(tree.left, val * 10 + tree.val) + helper(tree.right, val * 10 + tree.val)

    if not tree:
        return 0
    return helper(tree, 0)


# 最大路径和 有点动态规划的意思
def max_path_sum(tree):
    def helper(tree):
        if not tree:
            return 0, 0
        current_left, max_left = helper(tree.left)
        current_right, max_right = helper(tree.right)
        val = tree.val
        current = max(current_left + val, current_right + val, val)
        max_ = max(max_left, max_right, current, current_left + current_right + val)
        return current, max_

    return helper(tree)[1]


# 根据先序遍历和中序遍历构建二叉树
def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root = TreeNode()
    val = preorder[0]
    root.val = val
    idx = inorder.index(val)
    left_inorder = inorder[:idx]
    right_inorder = inorder[idx + 1:]
    left_preorder = preorder[1: len(left_inorder) + 1]  # 重点
    right_preorder = preorder[len(left_inorder) + 1:]
    left = build_tree(left_preorder, left_inorder)
    right = build_tree(right_preorder, right_inorder)
    root.left = left
    root.right = right
    return root


# 先序访问序列化
def serialize(tree):
    if not tree:
        return
    res = []
    stack = [tree]
    while stack:
        node = stack.pop(-1)
        if not node:
            res.append('$')
        else:
            res.append(node.val)
            l, r = node.left, node.right
            stack.append(r)
            stack.append(l)
    return res


# 反序列化 递归
def deserialize(string):
    vals = [i for i in string]

    def helper():
        val = vals.pop(0)
        if val == '$':
            return
        left = helper()
        right = helper()
        root = TreeNode()
        root.val = val
        root.left = left
        root.right = right
        return root

    if string == '':
        return None
    return helper()


if __name__ == '__main__':
    tree = create_full_binary_tree([i for i in range(7)])
    print(level(tree))
    print(level_num(tree))

    print('根据先序遍历和中序遍历构建数')
    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]
    tree2 = build_tree(preorder, inorder)
    print(level(tree2))

    print('最近公共祖先')
    print(lowest_common_ancestor(tree, 1, 5))

    print('判断树是否平衡')
    mytree = TreeNode()
    mytree.left = TreeNode()
    print(is_balanced(mytree))

    print('右侧观察树')
    print(right_side_view(tree))

    print('锯齿层次遍历')
    print(zigzag_level_order(tree))

    print('bst 转换成 greater_tree')
    print(convert_bst_2_greater_tree(tree))

    tree = create_full_binary_tree([-5, 2, 5])
    print('子树和出现次数最多的数字')
    print(most_frequent_subtree_sum(tree))

    print('打印根节点到叶节点路径')
    tree = create_full_binary_tree([2, 1, 3, 2])
    print(binary_tree_paths(tree))
    print(binary_tree_paths2(tree))

    print('打印根节点到叶节和为某个数的路径')
    print(has_path_sum(tree, 5))

    print('路径组成数字之和')
    print(sum_numbers(tree))

    print('最大路径之和')
    tree = create_full_binary_tree([-10, 9, 20, 0, 0, 15, 7])
    print(max_path_sum(tree))

    print('二叉树序列化')
    tree = create_full_binary_tree([1, 2, 3])
    string = serialize(tree)
    print(string)

    print('二叉树反序列化')
    tree = deserialize(string)
    print(level(tree))
