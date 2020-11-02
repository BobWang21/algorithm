class Node():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def max_path_sum(root):
    if not root:
        return 0

    res = [-float('inf')]

    def helper(root):
        if not root:
            return 0
        left = helper(root.left)
        right = helper(root.right)
        res[0] = max(res[0], root.val, root.val + left, root.val + right, root.val + left + right)
        return max(root.val, root.val + left, root.val + right)

    helper(root)
    return res[0]


if __name__ == '__main__':
    print()
