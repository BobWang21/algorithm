#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1 层次遍历
2 路径
"""

from binary_tree import *


# 最近公共祖先 关注一个节点就是根节点
def lowest_common_ancestor(tree, p, q):
    if not tree:
        return None
    if tree.val == p or tree.val == q:
        return tree.val
    left = lowest_common_ancestor(tree.left, p, q)
    right = lowest_common_ancestor(tree.right, p, q)
    if left is not None and right is not None:
        return tree.val

    if left is None:
        return right

    if right is None:
        return left


# 返回是否平衡 已经树的高度
def is_balanced(tree):
    def helper(root):
        if not root:
            return True, 0
        l = helper(root.left)
        r = helper(root.right)
        if not l[0] or not r[0]:
            return False, max(l[1], r[1]) + 1
        h = max(l[1], r[1]) + 1
        if abs(l[1] - r[1]) > 1:
            return False, h
        return True, h

    return helper(tree)[0]


# 不平衡就退出栈
def is_balanced1(tree):
    res = [True]

    def helper(tree):  # 返回树的高度
        if not tree:
            return 0
        if not res[0]:
            return 0
        left = helper(tree.left)
        right = helper(tree.right)
        if abs(right - left) > 1:
            res[0] = False
        return max(left, right) + 1

    helper(tree)
    return res[0]


# 判断树是否平衡
def is_balanced2(tree):
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


# 543. 二叉树的直径
def diameter_of_binary_tree(root):
    res = [-1]

    def helper(root):  # 包含根节点树的最大深度
        if not root:
            return 0
        l, r = helper(root.left), helper(root.right)
        res[0] = max(res[0], l + r)
        return max(l, r) + 1

    helper(root)
    return res[0]


# 508. Most Frequent Subtree Sum
def most_frequent_subtree_sum(tree):
    dic = {}

    def helper(tree):
        total = 0
        if not tree:
            return total

        left = helper(tree.left)
        right = helper(tree.right)
        total = tree.val + left + right
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


# 101. 对称二叉树 二叉树是否对称
def is_symmetric(tree):
    """
         5
    3        3
       4        4
    """

    def helper(l, r):
        if not l and not r:
            return True
        if not l or not r:
            return False
        if l.val != r.val:  # 只判断当前
            return False
        return helper(l.left, r.right) and helper(l.right, r.left)

    if not tree:
        return True
    return helper(tree.left, tree.right)


# 572. 另一个树的子树
# DFS 暴力匹配 时间复杂度为O(M) * O(N)
# 可使用KMP算法进行优化
def is_subtree(s, t):
    def helper(s, t):  # 已知两颗树 两棵树是否完全相同
        if not s and not t:
            return True
        if not s or not t:  # 不能少 也不能多
            return False
        if s.val == t.val:
            return helper(s.left, t.left) and helper(s.right, t.right)
        return False

    # 也需要判断 因为语句2
    if not s and not t:
        return True
    if not s or not t:
        return False

    if helper(s, t):  # 只有True的时候返回 1
        return True
    return is_subtree(s.left, t) or is_subtree(s.right, t)  # 2


# 子结构
def substructure(s, t):
    def helper(s, t):  # 已知两颗树 一颗树是否包另一颗
        if not t:
            return True
        if not s:
            return False
        if s.val == t.val:
            return helper(s.left, t.left) and helper(s.right, t.right)
        return False

    if not t:
        return True
    if not s:
        return False

    if helper(s, t):  # helper返回True时才能结束, False不能返回
        return True
    return is_subtree(s.left, t) or is_subtree(s.right, t)


# 257. 二叉树的所有路径
def binary_tree_paths(tree):
    if not tree:
        return []
    left = binary_tree_paths(tree.left)
    right = binary_tree_paths(tree.right)
    paths = left + right
    if not paths:  # 叶节点 没有这个会输出空
        return [str(tree.val)]
    res = []
    for path in paths:
        res.append(str(tree.val) + '->' + path)
    return res


# 借鉴回溯思想
def binary_tree_paths2(tree):
    res = []

    def helper(tree, path):
        if not tree.left and not tree.right:
            res.append(path + [tree.val])
            return
        if tree.left:
            helper(tree.left, path + [tree.val])
        if tree.right:
            helper(tree.right, path + [tree.val])

    helper(tree, [])
    return res


# 213 21 + 23 = 46
def sum_numbers(tree):  # 先序遍历!!!
    # top_down 记录父节点的值
    def helper(tree, val):
        if not tree:
            return 0
        if not tree.left and not tree.right:  # 叶节点!!!
            return val * 10 + tree.val
        return helper(tree.left, val * 10 + tree.val) + helper(tree.right, val * 10 + tree.val)

    if not tree:
        return 0
    return helper(tree, 0)


# 是否存在路径和为某个target
def have_path_sum(tree, target):
    if not tree and target == 0:  # 递归基
        return True
    if not tree:  # 递归基
        return False
    left = have_path_sum(tree.left, target - tree.val)
    right = have_path_sum(tree.right, target - tree.val)
    return left or right


# 113 打印所有路径
# 路径的数目为 O(N)，并且每一条路径的节点个数也为 O(N)，
# 因此要将这些路径全部添加进答案中，时间复杂度为 O(N^2)
def path_sum2(root, target):
    if not root:
        return []

    res = []
    path = []

    def dfs(root, val):
        if not root:
            return
        if not root.left and not root.right and root.val == val:
            res.append(path + [root.val])
            return
        path.append(root.val)
        dfs(root.left, val + root.val)
        dfs(root.right, val + root.val)
        path.pop(-1)

    dfs(root, target)
    return res


# 垂直遍历
def vertical_order(root):
    if not root:
        return []
    dic = {}
    queue = [(root, 0)]
    while queue:
        node, i = queue.pop(0)
        dic.setdefault(i, [])
        dic[i].append(node.val)
        if node.left:
            queue.append((node.left, i - 1))
        if node.right:
            queue.append((node.right, i + 1))

    res = []
    for k, v in dic.items():
        res.append((k, v))
    res.sort()
    return [v[1] for v in res]


# 958. 判断二叉树是否为完全二叉树
def is_complete_tree(root):
    if not root:
        return True
    queue = [(root, 1)]
    i = 0
    while queue:
        i += 1
        node, idx = queue.pop(0)
        if i != idx:
            return False
        l, r = node.left, node.right
        if not l and not r:
            continue
        if not l and r:
            return False
        if l and r:
            queue.append((l, 2 * idx))
            queue.append((r, 2 * idx + 1))
        else:
            queue.append((l, 2 * idx))
    return True


# 右侧看到的节点 层次遍历
def right_side_view(tree):
    if not tree:
        return
    curr_level = [tree]
    res = []
    while curr_level:
        next_level = []
        for node in curr_level:
            left, right = node.left, node.right
            if left:
                next_level.append(left)
            if right:
                next_level.append(right)
        res.append(node.val)
        curr_level = next_level
    return res


# 锯齿形层次遍历。
# 即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行
def zigzag_level_order(tree):
    if not tree:
        return
    curr_level = [tree]
    res = []
    direction = 1
    while curr_level:
        next_level = []
        level_value = []
        for node in curr_level:
            level_value.append(node.val)
            left, right = node.left, node.right
            if right:
                next_level.append(right)
            if left:
                next_level.append(left)

        res.append(level_value[::direction])
        direction *= -1
        curr_level = next_level
    return res


# 根据先序遍历 和 中序遍历 构造树 找到根节点!!
def build_tree(preorder, inorder):
    if not preorder and not inorder:
        return None
    val = preorder[0]
    tree = TreeNode(val)
    idx = inorder.index(val)
    left_inorder = inorder[:idx]
    right_inorder = inorder[idx + 1:]
    left_preorder = preorder[1: len(left_inorder) + 1]
    right_preorder = preorder[len(left_inorder) + 1:]
    tree.left = build_tree(left_preorder, left_inorder)
    tree.right = build_tree(right_preorder, right_inorder)
    return tree


def serialize(tree):
    if not tree:
        return '$ '
    return str(tree.val) + ' ' + serialize(tree.left) + serialize(tree.right)


# 反序列化 递归
def deserialize(s):
    if not s:
        return
    lists = [c for c in s]

    def helper():
        if not lists:
            return
        val = lists.pop(0)
        if val == '$':
            return
        root = TreeNode(val)
        root.left = helper()
        root.right = helper()
        return root

    return helper()


def construct_parent():
    a = TreeNode(10)
    b = TreeNode(5)
    c = TreeNode(4)
    d = TreeNode(6)
    e = TreeNode(11)

    a.left = b
    a.right = e
    b.left = c
    b.right = d

    a.parent = None
    b.parent = a
    e.parent = a
    c.parent = b
    d.parent = b
    return [a, b, c, d, e]


# 中序遍历 节点的下一个节点
def inorder_tra_next_node(node):
    if not node:
        return
    if node.right:
        node = node.right
        while node.left:
            node = node.left
        return node.val
    if not node.parent:
        return
    if node.parent.left == node:
        return node.parent.val

    while node.parent:
        if node == node.parent.left:
            return node.parent.val
        node = node.parent
    return


# 抢钱问题 节点之间不能存在父子关系
# DFS + Memoization Solution
def rob(tree):
    dic = {}

    def helper(root):
        if not root:
            return 0, 0  # inc, max
        if root in dic:
            return dic[root]
        left = helper(root.left)
        right = helper(root.right)
        include = root.val
        if root.left:
            include += helper(root.left.left)[1]
            include += helper(root.left.right)[1]
        if root.right:
            include += helper(root.right.left)[1]
            include += helper(root.right.right)[1]
        exclude = left[1] + right[1]
        max_v = max(include, exclude)
        dic.setdefault(root, (include, max_v))
        return include, max_v

    return helper(tree)[1]


# include 和 exclude
def rob2(tree):
    def helper(tree):
        if not tree:
            return 0, 0  # exc, inc

        l = helper(tree.left)
        r = helper(tree.right)
        include = tree.val + l[0] + r[0]
        exclude = max(l) + max(r)
        return exclude, include

    return max(helper(tree))


# 124 可以不经过根节点 不含有负数 参考动态规划
# 包含根节点的最大值 和 不包含根节点的最大值
def max_sum_path(tree):
    res = [-float('inf')]

    def helper(tree):
        if not tree:
            return -float('inf'), -float('inf')  # 包含当前节点路径!! 及 最大值 因为含有负数 这里不能用0
        left = helper(tree.left)
        right = helper(tree.right)
        inc = max(tree.val, left[0] + tree.val, right[0] + tree.val)  # 路径
        exc = max(max(left), max(right))
        res[0] = max(res[0], inc, exc, left[1] + right[1] + tree.val)
        return inc, exc

    helper(tree)
    return res[0]


def max_sum_path2(tree):
    res = [-float('inf')]  # 可能存在负数

    def helper(tree):  # 包含根节点的最大值
        if not tree:
            return 0
        left, right = helper(tree.left), helper(tree.right)
        res[0] = max(res[0], tree.val, left + tree.val, right + tree.val, left + right + tree.val)

        return max(tree.val, left + tree.val, right + tree.val)

    helper(tree)
    return res[0]


# 二叉树原地改为链表 先序遍历
def flatten(tree):
    if not tree:
        return
    stack = [tree]
    dummy = pre = TreeNode(None)
    while stack:
        node = stack.pop(-1)
        l, r = node.left, node.right
        node.left = None  # 需要置空
        node.right = None
        pre.right = node
        pre = node
        if r:
            stack.append(r)
        if l:
            stack.append(l)
    return dummy.right


def width_of_tree(tree):
    if not tree:
        return 0
    current_level = [(tree, 0)]
    res = 1
    while current_level:
        next_level = []
        res = max(res, current_level[-1][1] - current_level[0][1] + 1)
        for node, idx in current_level:
            left, right = node.left, node.right
            if left:
                next_level.append((left, idx * 2))
            if right:
                next_level.append((right, idx * 2 + 1))
        current_level = next_level

    return res


# 双重递归
def path_sum(root, sum):
    res = [0]

    def from_root(tree, total):  # 包含根节点
        if not tree:
            return
        if tree.val + total == sum:
            res[0] += 1
        from_root(tree.left, tree.val + total)
        from_root(tree.right, tree.val + total)

    def helper(tree):
        if not tree:
            return
        from_root(tree, 0)
        helper(tree.left)
        helper(tree.right)

    helper(root)
    return res[0]


# 前缀和
def path_sum2(root, sum):
    res = [0]

    def from_root(tree, total):  # 包含根节点
        if not tree:
            return
        if tree.val + total == sum:
            res[0] += 1
        from_root(tree.left, tree.val + total)
        from_root(tree.right, tree.val + total)

    def helper(tree):
        if not tree:
            return
        from_root(tree, 0)
        helper(tree.left)
        helper(tree.right)

    helper(root)
    return res[0]


if __name__ == '__main__':
    print('\n根据先序遍历和中序遍历构建数')
    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]
    tree2 = build_tree(preorder, inorder)
    print(level_traversal(tree2))

    print('\n最近公共祖先')
    tree = create_full_binary_tree([i for i in range(7)])
    print(lowest_common_ancestor(tree, 1, 5))

    print('\n判断树是否平衡')
    unbalanced_tree = TreeNode(3)
    a = TreeNode(10)
    b = TreeNode(5)
    unbalanced_tree.left = a
    a.left = b
    print(is_balanced(unbalanced_tree))
    print(is_balanced2(unbalanced_tree))

    print('\n右侧观察树')
    print(right_side_view(tree))

    print('\n锯齿层次遍历')
    print(zigzag_level_order(tree))

    tree = create_full_binary_tree([-5, 2, 5])
    print('\n子树和出现次数最多的数字')
    print(most_frequent_subtree_sum(tree))

    print('\n打印根节点到叶节点路径')
    tree = create_full_binary_tree([2, 1, 3, 2])
    print(binary_tree_paths(tree))
    print(binary_tree_paths2(tree))

    print('\n是否存在和为某个数的路径')
    tree = create_full_binary_tree([i for i in range(7)])
    print(have_path_sum(tree, 4))
    print(have_path_sum(tree, 9))

    print('\n路径和为某个数的路径')
    print(path_sum2(tree, 8))

    print('\n路径组成数字之和')
    print(sum_numbers(tree))

    print('\n子树最大路径之和')
    tree = create_full_binary_tree([-10, 9, 20, 0, 0, 15, 7])
    print(max_sum_path(tree))

    print('\n二叉树序列化')
    tree = create_full_binary_tree([1, 2, 3])
    string = serialize(tree)
    print(string)

    print('\n二叉树反序列化')
    tree = deserialize(string)
    print(level_traversal(tree))

    print('\n中序遍历的下一个节点')
    tree = construct_parent()[3]
    print(inorder_tra_next_node(tree))

    print('\n二叉树转链表')
    tree = create_full_binary_tree([1, 2, 5, 3, 4])
    head = flatten(tree)
    res = []
    while head:
        res.append(head.val)
        head = head.right
    print(res)
