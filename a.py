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


if __name__ == '__main__':
    print(cyclic_sort([5, 3, 1, 2, 4]))

    print(find_first_missing([3, -1, 1, 2, 5]))

    print(merge([1, 3], [2, 4, 6]))

    print(next_permutation([1, 2, 3]))
