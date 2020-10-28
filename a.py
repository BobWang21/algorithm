from collections import defaultdict


def max_area(height):
    n = len(height)
    if n <= 2:
        return 0
    stack = []
    res = 0
    for i, v in enumerate(height):
        while stack and height[stack[-1]] < v:
            j = stack.pop(-1)
            if stack:
                h = min(height[stack[-1]], v) - height[j]
                w = i - stack[-1] - 1
                res += h * w
        stack.append(i)
    return res


def find_median(nums1, nums2):
    m, n = len(nums1), len(nums2)
    if m < n:
        return find_median(nums2, nums1)

    k = (m + n + 1) // 2
    l, r = 0, m
    while l < r:
        mid = l + (r - l) // 2
        if nums1[mid - 1] <= nums2[k - mid]:
            l = mid + 1
        else:
            r = mid
    k1, k2 = l - 1, k - l - 1
    v1 = max(nums1[k1] if 0 <= k1 < m else -float('inf'),
             nums2[k2] if 0 <= k2 < n else -float('inf'))
    if (m + n) % 2:
        return v1

    v2 = min(nums1[k1 + 1] if 0 <= k1 + 1 < m else float('inf'),
             nums2[k2 + 1] if 0 <= k2 + 1 < n else float('inf'))
    return (v1 + v2) / 2.0


def findCircleNum(M):
    if not M or not M[0]:
        return 0
    n = len(M)
    parent = [range(n)]
    rank = [0] * n
    res = [n]

    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x, y = find(x), find(y)
        if x == y:
            return
        if rank[x] < rank[y]:
            parent[x] = parent[y]
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1
        res[0] -= 1

    for i in range(n):
        for j in range(n):
            if M[i][j]:
                union(i, j)

    return res[0]


def can_finish(numCourses, prerequisites):
    if not numCourses:
        return True

    dic = defaultdict(set)
    indegree = [0] * numCourses
    for u, v in prerequisites:
        dic[v].add(u)
        indegree[u] += 1

    queue = []
    for i in range(numCourses):
        if not indegree[i]:
            queue.append(i)

    while queue:
        u = queue.pop(0)
        for v in dic[u]:
            indegree[v] -= 1
            if not indegree[v]:
                queue.append(v)
        del dic[u]
    return False if dic else True


def possible_bipartition(n, dislikes):
    if not dislikes:
        return True
    colors = [0] * n

    dic = defaultdict(set)

    for u, v in dislikes:
        dic[u - 1].add(v - 1)
        dic[v - 1].add(u - 1)

    def dfs(i, col):
        if colors[i] == -col:
            return False
        if colors[i] == col:
            return True
        colors[i] = col
        for j in dic[i]:
            if colors[j] == col:
                return False
            if not colors[j] and not dfs(j, -col):
                return False
        return True

    for i in range(n):
        if not colors[i]:
            if not dfs(i, 1):
                return False
    return True


class Node(object):
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return
        dic = {}

        def dfs(node):
            if node in dic:
                return dic[node]
            new_node = Node(node.val)
            neighbors = []
            dic[node] = new_node #
            for n in node.neighbors:
                new_n = dfs(n)
                neighbors.append(new_n)
            new_node.neighbors = neighbors
            return new_node

        return dfs(node)


if __name__ == '__main__':
    print(max_area([4, 2, 0, 3, 2, 5]))
    print(find_median([1, 3], [2, 7]))

    print(possible_bipartition(5, [[1, 2], [2, 3], [3, 4], [4, 5], [1, 5]]))
