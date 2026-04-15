from collections import deque


class TreeNode():
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def create_full_binary_tree(nums):
    if not nums:
        return
    root = TreeNode(nums[0])
    queue = deque([root])
    i, n = 1, len(nums)
    while i < n:
        parent = queue[0]
        node = TreeNode(nums[i])
        i += 1
        if not parent.left:
            parent.left = node
            queue.append(node)  # 进队列
        else:
            parent.right = node
            queue.append(node)  # 进队列
            queue.popleft()  # 出队列

    return root


def level_traversal(tree):
    if not tree:
        return
    queue, res = deque([tree]), []
    while queue:
        node = queue.popleft()
        res.append(node.val)
        left, right = node.left, node.right
        if left:
            queue.append(left)
        if right:
            queue.append(right)
    return res
