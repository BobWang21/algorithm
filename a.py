def cyclic_sort(nums):
    if not nums:
        return []
    n = len(nums)

    for i in range(n):
        while nums[i] != i + 1:
            j = nums[i] - 1
            if nums[j] == j + 1:
                break
            nums[i], nums[j] = nums[j], nums[i]

    return nums


def find_first_missing(nums):
    if not nums:
        return 1
    n = len(nums)
    for i in range(n):
        while 0 < nums[i] <= n and nums[i] != i + 1:
            j = nums[i] - 1
            if nums[j] == j + 1:
                break
            nums[i], nums[j] = nums[j], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


def merge(nums1, nums2):
    if not nums1:
        return nums2

    if not nums2:
        return nums1

    i = j = 0
    m, n = len(nums1), len(nums2)

    res = []
    while i < m or j < n:
        v1 = nums1[i] if i < m else float('inf')
        v2 = nums2[j] if j < n else float('inf')
        if v1 < v2:
            res.append(v1)
            i += 1
        else:
            res.append(v2)
            j += 1
    return res


def next_permutation(nums):
    if not nums or len(nums) == 1:
        return nums
    n = len(nums)
    i = n - 1
    while i >= 1 and nums[i] <= nums[i - 1]:
        i -= 1

    if not i:
        nums.reverse()
        return nums
    k = i - 1
    j = k
    while j < n and nums[k] < nums[j]:
        j += 1
    m = nums[j - 1]
    nums[k], nums[m] = nums[m], nums[k]
    l, r = k, n - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums


class Node():
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val


def inorder(tree):
    if not tree:
        return []
    node, stack = tree, []
    res = []
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop(-1)
        res.append(node.val)
        node = node.right
    return res


def level(tree):
    if not tree:
        return []
    queue, res = [tree], []
    while queue:
        node = queue.pop(0)
        res.append(node.val)
        left, right = node.left, node.right
        if left:
            queue.append(left)
        if right:
            queue.append(right)
    return res


def right_side(tree):
    if not tree:
        return []

    curr, res = [tree], []

    while curr:
        nxt = []
        for node in curr:
            l, r = node.left, node.right
            if l:
                nxt.append(l)
            if r:
                nxt.append(r)
        res.append(curr[-1].val)
        curr = nxt
    return res


# 判断二叉树是否平衡
def balance(tree):
    if not tree:
        return True

    def helper(tree):
        if not tree:
            return 0
        left, right = helper(tree.left), helper(tree.right)
        if left == -1 or right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return max(left, right) + 1

    return helper(tree) != -1


def reverse(head):
    if not head:
        return
    pre, tail = None, head
    while head:
        nxt = head.next
        head.next = pre
        pre = head
        head = nxt
    return pre, head


def find_circle_num(m):
    n = len(m)
    parent = [i for i in range(n)]
    rank = [1] * n

    res = [n]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i = find(i)
        j = find(j)
        if i == j:
            return
        if rank[i] > rank[j]:
            parent[j] = i
        elif rank[i] < rank[j]:
            parent[i] = j
        else:
            parent[j] = i
            rank[i] += 1
        res[0] -= 1

    for i in range(n):
        for j in range(n):
            if m[i][j] == 1:
                union(i, j)
    return res[0]


from collections import defaultdict


def can_finish(course, pre):
    if not course:
        return True

    dic = defaultdict(set)
    degree = defaultdict(int)
    for start, end in pre:
        degree[end] += 1
        dic[start].add(end)

    queue = []
    for i in range(course):
        if not degree[i]:
            queue.append(i)

    while queue:
        i = queue.pop(0)
        for e in dic[i]:
            degree[e] -= 1
            if not degree[e]:
                queue.append(e)
        del degree[i]

    return True if not degree else False


def diameter_of_binary_tree(tree):
    if not tree:
        return 0

    res = [0]

    def helper(tree):
        if not tree:
            return 0
        l = helper(tree.left)
        r = helper(tree.right)
        res[0] = max(res[0], l + r)
        return max(l + r) + 1

    helper(tree)
    return res[0]


def is_subtree(s, t):
    if not s and not t:
        return True
    if not s:
        return False
    if not t:
        return True

    def helper(s, t):
        if not s and not t:
            return True
        if not s:
            return False
        if not t:
            return True
        if s.val == t.val:
            return helper(s.left, t.left) and helper(s.right, t.right)
        else:
            return False

    if helper(s, t):
        return True
    return is_subtree(s.left, t) or is_subtree(s.right, t)


def remove(s):
    if not s:
        return ['']
    l = r = 0
    for ch in s:
        if ch == '(':
            l += 1
        elif l:
            l -= 1
        else:
            r += 1
    return l, r


class BSTNode():
    def __init__(self, val):
        self.val = val
        self.cnt = 1
        self.less = 0
        self.left = None
        self.right = None


def insert(tree, v, less):
    if not tree:
        tree = BSTNode(v)
        tree.less = less
        return tree
    if tree.val == v:
        tree.cnt += 1
        return tree
    if tree.val > v:
        tree.less += 1
        tree.left = insert(tree.left, v, less)
    tree.right = insert(tree.right, v, tree.cnt + tree.less)
    return tree


def search(tree, v):
    if not tree:
        return 0
    if tree.val == v:
        return tree.less
    if tree.val > v:
        return search(tree.left, v)
    return search(tree.right, v)


def counter_smaller(nums):
    if not nums:
        return
    nums.reverse()
    tree = BSTNode(float('inf'))
    res = []
    for num in nums:
        insert(tree, num, 0)
        res.append(search(tree, num))

    res.reverse()
    return res


if __name__ == '__main__':
    print(cyclic_sort([5, 3, 1, 2, 4]))

    print(find_first_missing([3, -1, 1, 2, 5]))

    print(merge([1, 3], [2, 4, 6]))

    print(next_permutation([1, 2, 3]))

    print('\nfriend circle')
    m = [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 1],
         [1, 0, 1, 1]]
    print(find_circle_num(m))

    print(remove("()())()"))

    print(counter_smaller([5, 2, 6, 1]))
