#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class TreeNode():
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 创建完全二叉树
def create_full_binary_tree(nums):
    if not nums:
        return
    tree = TreeNode(nums.pop(0))
    queue = [tree]
    while nums:
        val = nums.pop(0)
        parent = queue[0]
        if not parent.left:
            node = TreeNode(val)
            parent.left = node
            queue.append(node)
        elif not parent.right:
            node = TreeNode(val)
            parent.right = node
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


# 中序遍历 左子树一直向下 访问到头就出栈
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


# 判断二叉树是否为完全二叉树
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
        elif not l and r:
            return False
        elif l and r:
            queue.append((l, 2 * idx))
            queue.append((r, 2 * idx + 1))
        else:
            queue.append((l, 2 * idx))
    return True


def level_num(tree):
    if not tree:
        return 0
    return max(level_num(tree.left), level_num(tree.right)) + 1


# 最小深度 根节点到叶节点! 也可以使用bfs
def min_depth(root):
    if not root:
        return 0
    l, r = root.left, root.right
    if not l:
        return min_depth(r) + 1
    if not r:
        return min_depth(l) + 1
    return min(min_depth(l), min_depth(r)) + 1


# 层次遍历
def serialize(tree):
    if not tree:
        return
    queue = [tree]
    res = []
    while queue:
        node = queue.pop(0)
        if not node:
            res.append('#')
        else:
            res.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
    return ' '.join(res)


def deserialize(s):
    if not s:
        return
    nums = s.split()
    root = TreeNode(eval(nums.pop(0)))
    queue = [root]
    flag = 0
    while nums:
        node = queue[0]
        val = nums.pop(0)
        if val == '#':
            if not flag:
                flag += 1
            else:
                queue.pop(0)
                flag = 0
        else:
            child = TreeNode(eval(val))
            queue.append(child)
            if not flag:
                node.left = child
                flag += 1
            else:
                node.right = child
                queue.pop(0)
                flag = 0
    return root


class BSTNode():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.count = 0


if __name__ == '__main__':
    print('\n先序遍历')
    tree = create_full_binary_tree([i for i in range(7)])
    print(level_traversal(tree))

    print('\n中序遍历')
    print(inorder_traversal(tree))

    print('\n层次遍历')
    print(level_traversal(tree))

    print('\n数的最小深度')
    root = TreeNode(1)
    a = TreeNode(2)
    b = TreeNode(3)
    root.left = a
    a.left = b
    print(min_depth(root))

    print('\n序列化反序列化')
    tree = create_full_binary_tree([i for i in range(7)])
    s = serialize(tree)
    tree = deserialize(s)
    print(level_traversal(tree))
