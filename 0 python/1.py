class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Codec:

    def serialize(self, tree):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
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

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return
        lists = data.split()
        head = TreeNode(lists.pop(0))
        curr_level = [head]
        while curr_level:
            next_level = []
            for node in curr_level:
                l, r = lists.pop(0), lists.pop(0)
                if l == '#':
                    node.left = None
                else:
                    left = TreeNode(l)
                    node.left = left
                    next_level.append(left)
                if r == '#':
                    node.right = None
                else:
                    right = TreeNode(r)
                    node.right = right
                    next_level.append(right)

            curr_level = next_level
        return head

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
