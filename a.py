from pprint import pprint


# m个人 n个车 每车最多k个人 需要对结果去重
def permute(m, n, k):
    visited = [False] * m
    capacity = [k] * n
    res, path = [], [[] for _ in range(n)]

    def check():
        for v in visited:
            if not v:
                return False
        return True

    def copy():
        return [vs[:] for vs in path]

    def helper():
        # print(path)
        if check():
            # print(path)
            res.append(copy())
            return
        for i in range(m):
            if visited[i]:
                continue

            visited[i] = True
            for j in range(n):
                if not capacity[j]:
                    continue
                capacity[j] -= 1
                path[j].append(i)

                helper()

                capacity[j] += 1
                path[j].pop(-1)

            visited[i] = False

    helper()
    return res


def permute2(n):
    if not n:
        return []
    res, path = [], []
    status = [2] * n

    def helper():
        if len(path) == 2 * n:
            res.append(path[:])
            return

        for i in range(n):
            if not status[i]:
                continue

            if status[i] == 2:
                path.append(str(i) + '_B')
            else:
                path.append(str(i) + '_C')
            status[i] -= 1

            helper()

            path.pop(-1)
            status[i] += 1

    helper()
    return res


import heapq as hq


def merge(intervals):
    if not intervals or not intervals[0] or len(intervals) == 1:
        return intervals

    intervals.sort()

    res = [intervals[0]]
    hq.heapify(res)
    for i in range(1, len(intervals)):
        interval = intervals[i]
        if res[0][1] < interval[0]:
            hq.heappush(res, interval)
        else:
            top = hq.heappop(res)
            left = min(top[0], interval[0])
            right = max(top[1], interval[1])
            hq.heappush(res, [left, right])

    return res


def jump(start, values):
    if start == 0:
        return False, 0
    n = len(values)

    values = [start] + values

    res = [0] * (n + 1)
    res[0] = start

    def helper(i):
        step = res[i]
        for j in range(i + 1, i + step + 1):
            if j > n:
                return
            new_v = res[i] - (j - i) + values[j]
            if new_v > res[j]:
                res[j] = new_v
                helper(j)

    helper(0)
    print(res)
    return res[-1]


def merge2(num1, num2):
    i, j = 0, 0
    res = []
    m, n = len(num1), len(num2)
    while True:
        if i == len(num1) and j == len(num2):
            return res
        a = num1[i] if i < m else float('inf')
        b = num2[j] if j < n else float('inf')
        if a < b:
            v = a
            i += 1
        else:
            v = b
            j += 1
        if res and res[-1] == v:
            continue
        res.append(v)

    return res


def binary_search(nums, target):
    if not nums:
        return -1
    n = len(nums)
    l, r = 0, n - 1
    while l < r:
        mid = l + (r - l + 1) // 2
        if nums[mid] <= target:
            l = mid
        else:
            r = mid - 1
    return l if nums[l] == target else -1


'''
给定一个未排序的整数数组，找出最长连续序列的长度。
要求算法的时间复杂度为 O(n)。

示例:
输入: [100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
'''


def find_long(nums):
    if not nums:
        return 0
    s = set(nums)
    res = 0
    tmp = set()
    for v in s:
        if v in tmp:
            continue
        tmp = {v}
        while v + 1 in s:
            v += 1
            tmp.add(v + 1)
        res = max(len(tmp), res)
        tmp = set()
    return res


def find_magic_num(nums):
    if not nums:
        return -1

    n = len(nums) - 1

    def helper(l, r):
        if l > r:
            return -1

        mid = l + (r - l) // 2
        left = helper(l, mid - 1)
        if left != -1:
            return left
        if nums[mid] == mid:
            return mid
        right = helper(mid + 1, r)
        if right != -1:
            return right
        return -1

    return helper(0, n)


def partition(nums, l, r):
    pivot = nums[l]
    while l < r:
        while l < r and nums[r] >= pivot:
            r -= 1
        nums[l] = nums[r]
        while l < r and nums[l] <= pivot:
            l += 1
        nums[r] = nums[l]
    nums[l] = pivot
    return l


def quick_sort(nums, l, r):
    if l < r:
        pivot = partition(nums, l, r)
        quick_sort(nums, l, pivot - 1)
        quick_sort(nums, pivot + 1, r)


# 26 原地删除升序数组中的重复数字 并返回非重复数组的长度
# nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
def remove_duplicates1(nums):
    if not nums:
        return 0
    idx = 0
    n = len(nums)
    for j in range(1, n):
        if nums[j] != nums[idx]:
            nums[idx + 1] = nums[j]
            idx += 1
    return idx + 1


def cyclic_sort(nums):
    if not nums:
        return -1

    n = len(nums)
    for i in range(n):
        while nums[i] != i:
            j = nums[i]
            if nums[j] == j:
                return nums[j]
            nums[i], nums[j] = nums[j], nums[i]
    return -1


def diagnose_traverse(matrix):
    if not matrix or not matrix[0]:
        return []
    rows, cols = len(matrix), len(matrix[0])
    res = []
    for j in range(cols):
        i = 0
        while j > -1:
            res.append((matrix[i][j]))
            j -= 1
            i += 1

    for i in range(1, rows):
        j = cols - 1
        while i < rows:
            res.append((matrix[i][j]))
            i += 1
            j -= 1
    return res


def p():
    for j in range(10):
        j += 2
        print(j)


def search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return True
        if nums[mid] > nums[0]:
            if nums[mid] < target:
                l = mid + 1
            else:
                if nums[0] < target:
                    r = mid - 1
                else:
                    l = mid + 1
        else:
            if nums[mid] > target:
                r = mid - 1
            else:
                if nums[-1] < target:
                    r = mid - 1
                else:
                    l = mid + 1
    return -1


def knapsack(costs, values, capacity):
    rows, cols = len(costs) + 1, capacity + 1
    matrix = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost, val = costs[i - 1], values[i - 1]
        for j in range(1, cols):
            matrix[i][j] = matrix[i - 1][j]
            if j >= cost:
                matrix[i][j] = max(matrix[i][j], matrix[i - 1][j - cost] + val)
    return matrix[-1][-1]


# 排序数组 解不唯一 O(N)
def two_sum2(nums, target):
    res = []
    if not nums:
        return res
    n = len(nums)
    l, r = 0, n - 1
    while l < r:
        total = nums[l] + nums[r]
        if total < target:
            l += 1
        elif total > target:
            r -= 1
        else:
            res.append([nums[l], nums[r]])
            while l < r and nums[l] == nums[l + 1]:
                l += 1
            l += 1
    return res


def min_sub_array_len(nums, s):
    return


# 差值小于等于某个数的pair数
def nmt(nums, target):
    j, n = 1, len(nums)
    res = 0
    for i in range(n):
        while j < n and nums[j] - nums[i] <= target:
            j += 1
        res += j - i - 1
    return res


def longestIncreasingPath(matrix):
    """
    :type matrix: List[List[int]]
    :rtype: int
    """
    rows, cols = len(matrix), len(matrix[0])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    visited = [[0] * cols for _ in range(rows)]

    res = 0

    def dfs(i, j, pre):
        if i < 0 or j < 0 or i == rows or j == cols:
            return 0

        # 访问过且大于之前的值
        if visited[i][j] != 0 and pre < matrix[i][j]:
            return visited[i][j]

        # 访问过且不大于之前的值
        if visited[i][j] != 0:
            return 0

        # 未访问过
        visited[i][j] = 1
        tmp = 0
        for di, dj in directions:
            tmp = max(dfs(i + di, j + dj, matrix[i][j]), tmp)
        visited[i][j] += tmp
        return visited[i][j]

    for i in range(rows):
        for j in range(cols):
            res = max(res, dfs(i, j, -float('inf')))

    return res


def preorder2(tree):
    if not tree:
        return []
    res = []
    stack = [tree]
    while stack:
        node = stack.pop(-1)
        res.append(node.val)
        left, right = node.left, node.right
        if right:
            stack.append(right)
        if left:
            stack.append(left)
    return res


def inorder_traversal(tree):
    if not tree:
        return []
    node, stack = None, [tree]
    res = []
    while node or stack:
        if not node:
            node = stack.pop(-1)
        left, right = node.left, node.right
        if right:
            stack.append(node.right)
        if not left:
            res.append(node.val)
        node = node.left
    return res


if __name__ == '__main__':
    pprint(permute(2, 2, 2))
    pprint(permute2(2))
    #
    # print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))
    # print(jump(4, [-10, -10, 3, 10, -1, -1]))
    # print(merge2([1, 2, 3, 4, 5, 5, 6, 6, 7], [2, 3, 4, 4, 5, 6, 8, 8, 9, 10, 11]))
    # print(binary_search([1, 2, 2, 3], 3))
    # print(find_long([100, 4, 200, 1, 3, 2]))
    print(find_magic_num([2, 3, 4, 4, 5, 5, 5]))
    nums = [100, 4, 200, 1, 3, 2]
    quick_sort(nums, 0, 5)
    print(nums)

    print(remove_duplicates1([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))

    print(cyclic_sort([0, 1, 3, 2, 4, 3]))

    print(diagnose_traverse([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]))
    print(two_sum2([1, 2, 2, 3, 3, 4], 5))

    matrix = [[3, 4, 5], [3, 2, 6], [2, 2, 1]]
    print(longestIncreasingPath(matrix))
