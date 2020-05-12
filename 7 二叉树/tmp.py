#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1 层次遍历
2 路径
3 二叉搜索树
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


# 二叉树的叶节点之间的最大路径边数 可能不经过根节点
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


# 二叉树是否对称
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


# 子树
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

    if not s and not t:
        return True
    if not s or not t:
        return False

    if helper(s, t):  # helper返回True时才能结束, False不能返回
        return True
    return is_subtree(s.left, t) or is_subtree(s.right, t)


# 判断二叉树是否为二叉搜索树
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
        if res[0] == 0:
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


# 打印所有路径
def binary_tree_paths(tree):
    if not tree:
        return []
    left = binary_tree_paths(tree.left)
    right = binary_tree_paths(tree.right)
    paths = left + right
    if not paths:  # 叶节点
        return [str(tree.val)]
    res = []
    for path in paths:
        res.append(str(tree.val) + '->' + path)
    return res


# 借鉴回溯思想
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


# 打印所有路径 从根节点到叶节点和为某个数的路径
def sum_target_path(tree, target):
    res = []

    def helper(tree, target, path):
        if not tree.left and not tree.right and target == tree.val:  # 叶节点等于target
            res.append(path + [tree.val])
            return
        if not tree.left and not tree.right:  # 叶节点不等于target
            return
        if tree.left:
            helper(tree.left, target - tree.val, path + [tree.val])
        if tree.right:
            helper(tree.right, target - tree.val, path + [tree.val])

    helper(tree, target, [])
    return res


# 右侧看到的节点 层次遍历
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
def deserialize(s1):
    if s1 == '':
        return None
    vals = s1.split()

    def helper():
        val = vals.pop(0)
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
    elif not node.parent:
        return
    elif node.parent.left == node:
        return node.parent.val
    else:
        while node.parent:
            if node == node.parent.left:
                return node.parent.val
            node = node.parent
        return


# 抢钱问题 节点之间不能存在父子关系
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

    return helper(root)[1]


# include 和 exclude
def rob2(tree):
    def helper(tree):
        if not tree:
            return 0, 0

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
        exc = max(max(left[1]), max(right[1]))
        res[0] = max(res[0], inc, exc, left[1] + right[1] + root.val)
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


# 二叉树原地改为链表
def flatten(tree):
    if not tree:
        return None
    head = TreeNode(None)
    pre = head
    stack = [tree]
    while stack:
        node = stack.pop(-1)
        pre.right = node
        pre.left = None
        pre = pre.right
        l, r = node.left, node.right
        if r:
            stack.append(r)
        if l:
            stack.append(l)
    return head.right


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


# 删除二叉搜索树中的节点
def delete_node(root, key):
    def find_max(node):
        while node.left:
            node = node.left
        return node

    if not root:
        return

    if key < root.val:
        root.left = delete_node(root.left, key)
    elif key > root.val:
        root.right = delete_node(root.right, key)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        max_node = find_max(root.right)
        root.val = max_node.val
        root.left = delete_node(root.left, root.val)

    return root


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

    print('\nbst 转换成 greater_tree')
    print(convert_bst_2_greater_tree(tree))

    tree = create_full_binary_tree([-5, 2, 5])
    print('\n子树和出现次数最多的数字')
    print(most_frequent_subtree_sum(tree))

    print('\n打印根节点到叶节点路径')
    tree = create_full_binary_tree([2, 1, 3, 2])
    print(binary_tree_paths(tree))
    print(binary_tree_paths2(tree))

    print('\n打印根节点到叶节和为某个数的路径')
    print(sum_target_path(tree, 5))

    print('是否存在和为某个数的路径')
    tree = create_full_binary_tree([i for i in range(7)])
    print(have_path_sum(tree, 4))
    print(have_path_sum(tree, 9))

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

    print('检查二叉搜索树后续遍历是否合法')
    print(check_poster_order([1, 6, 3, 2, 5]))

    print('\n二叉搜索树的第k大节点')
    tree = create_full_binary_tree([5, 3, 7, 2, 4, 6, 8])
    print(top_k_binary_search_tree2(tree, 1))

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

    print('\ntrimBST')
    tree = create_full_binary_tree([10, 1, 15, 3, 4, 12, 17])
    print(level_traversal(trimBST(tree, 4, 12)))
