#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import deque


# 547 省份数 dfs 时间复杂度：O(edge)
def find_circle_num1(M):
    n = len(M)
    seen = set()

    def helper(i):
        if i in seen:
            return

        seen.add(i)
        for j in range(n):
            if M[i][j] and i != j:
                helper(j)

    res = 0
    for i in range(n):
        if i not in seen:
            res += 1
            helper(i)
    return res


# 547 省份数 | bfs
def find_circle_num2(is_connected):
    n = len(is_connected)
    visited = [False] * n
    province_count = 0

    def bfs(start):
        queue = deque([start])
        visited[start] = True
        while queue:
            city = queue.popleft()
            for i in range(n):
                if is_connected[city][i] == 1 and not visited[i]:
                    visited[i] = True
                    queue.append(i)

    # 遍历每个城市
    for i in range(n):
        if not visited[i]:  # 如果城市没有被访问过，说明是一个新的省份
            bfs(i)
            province_count += 1

    return province_count


# 547 省份数  o(n^3)
# 访问矩阵一次，并查集操作需要最坏O(n)的时间。
# 连通图数 = 顶点数 - 边数
def find_circle_num3(M):
    if not M or not M[0]:
        return 0
    n = len(M)
    parent = list(range(n))  # 初始状态，父节点为自己
    rank = [0] * n
    res = [n]  # 连通图数 = 顶点数

    # 路径减半
    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]  # 路径压缩 变为一半
            x = parent[x]
        return x

    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:
            return True  # 已经在同一个集合中 再加一条边 形成环
        # 减少计算 合并时rank小的树合并到rank大的树上 合并后的rank不变
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:  # 相等时 合并后的rank+1
            parent[y] = x
            rank[x] += 1
        res[0] -= 1
        return False

    for i in range(n):
        for j in range(i + 1, n):
            if M[i][j]:
                union(i, j)
    return res[0]


# 是否存在环
def check_circle(M):
    if not M or not M[0]:
        return 0
    n = len(M)
    parent = list(range(n))
    rank = [0] * n

    # 找到所属集合
    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]  # 路径减半
            x = parent[x]
        return x

    # 合并两个集合
    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:  # 有公共祖先等价于有环
            return True
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1
        return False

    for i in range(n):
        for j in range(i + 1, n):
            if M[i][j] and union(i, j):
                return True
    return False


# 684. 冗余连接
def find_redundant_connection(edges):
    if not edges:
        return
    n = len(edges)
    par = list(range(n + 1))  # 0 no use
    rank = [0] * (n + 1)

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def union(x, y):
        x, y = find(x), find(y)
        if x == y:
            return True
        if rank[x] > rank[y]:
            par[y] = x
        elif rank[x] < rank[y]:
            par[x] = y
        else:
            par[y] = x
            rank[x] += 1
        return False

    for u, v in edges:
        if union(u, v):
            return u, v


# 737 相似单词 🔐
def are_sentences_similar_two(words1, words2, pairs):
    if len(words1) != len(words2):
        return False

    word_set = set()
    for u, v in pairs:  # 去重
        word_set.add(u)
        word_set.add(v)

    n = len(word_set)

    word_dic = {}  # 词典中单词编号
    i = 0
    for v in word_set:
        word_dic[v] = i
        i += 1

    rank = [0] * n
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:
            return
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1
        return

    for u, v in pairs:
        x, y = word_dic[u], word_dic[v]
        union(x, y)

    for w1, w2 in zip(words1, words2):
        if w1 == w2:
            continue
        elif w1 in word_set and w2 in word_set:
            i, j = word_dic[w1], word_dic[w2]
            if find(i) == find(j):
                continue
            else:
                return False
        else:
            return False

    return True


# 128. 最长连续序列
def longest_consecutive(nums):
    if not nums:
        return 0

    parent = {}  # 可能存在连续数字
    rank = {}
    for num in nums:
        parent[num] = num
        rank[num] = 1  # 连通图最大顶点数

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:
            return rank[x]
        parent[x] = y
        rank[y] += rank[x]
        return rank[y]

    res = 1  # 如果为[2]
    for num in nums:
        if num + 1 in parent:
            res = max(res, union(num, num + 1))
    return res


def longest_consecutive2(nums):
    nums = set(nums)
    res = 0
    for num in nums:
        if num - 1 in nums:
            continue
        count = 1
        while num + 1 in nums:
            num = num + 1
            count += 1
        res = max(count, res)
    return res


# 399 除法求值
class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        equations_size = len(equations)
        # 并查集最多需要 2 * equations_size 个节点
        union_find = UnionFind(2 * equations_size)

        # 第 1 步：预处理，将变量的值与 id 进行映射
        hash_map = {}
        var_id = 0
        for i in range(equations_size):
            var1 = equations[i][0]
            var2 = equations[i][1]

            if var1 not in hash_map:
                hash_map[var1] = var_id
                var_id += 1
            if var2 not in hash_map:
                hash_map[var2] = var_id
                var_id += 1

            # 合并两个变量所在的集合，并记录它们之间的比值
            union_find.union(hash_map[var1], hash_map[var2], values[i])

        # 第 2 步：做查询
        queries_size = len(queries)
        res = [-1.0] * queries_size
        for i in range(queries_size):
            var1 = queries[i][0]
            var2 = queries[i][1]

            id1 = hash_map.get(var1)
            id2 = hash_map.get(var2)

            if id1 is None or id2 is None:
                res[i] = -1.0
            else:
                res[i] = union_find.is_connected(id1, id2)

        return res


class UnionFind(object):
    def __init__(self, n):
        # parent[i] 表示节点 i 的父节点
        self.parent = [i for i in xrange(n)]
        # weight[i] 表示节点 i 到其父节点的权值（即比值）
        self.weight = [1.0] * n

    def find(self, x):
        """
        查找节点 x 的根节点，并进行路径完全压缩。
        同时更新路径上所有节点的权重。
        """
        if x != self.parent[x]:
            origin = self.parent[x]
            # 递归查找根节点
            self.parent[x] = self.find(origin)
            # 更新权重：x 到根节点的权值 = x 到原父节点的权值 * 原父节点到根节点的权值
            self.weight[x] *= self.weight[origin]
        return self.parent[x]

    def union(self, x, y, value):
        """
        合并 x 和 y 所在的集合。
        已知 x / y = value，即 x = value * y。
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # 让 root_x 的父节点指向 root_y
        self.parent[root_x] = root_y
        # 计算 root_x / root_y 的值
        # 关系推导：weight[x] = x / root_x, weight[y] = y / root_y
        # 已知 x / y = value
        # 则 (root_x / root_y) = (y / root_y) * (x / y) / (x / root_x) = weight[y] * value / weight[x]
        self.weight[root_x] = self.weight[y] * value / self.weight[x]

    def is_connected(self, x, y):
        """
        判断 x 和 y 是否连通，如果连通则返回 x / y 的值，否则返回 -1.0。
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            # x / y = (x / root) / (y / root) = weight[x] / weight[y]
            return self.weight[x] / self.weight[y]
        else:
            return -1.0


if __name__ == '__main__':
    print('\ncheck graph circle')
    m = [[1, 1, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 1],
         [1, 0, 1, 1]]
    print(check_circle(m))

    print('\nfriend circle')
    m = [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 1],
         [1, 0, 1, 1]]
    print(find_circle_num1(m))

    print('\n句子相似性')
    a = ["great", "acting", "skills"]
    b = ["fine", "drama", "talent"]
    c = [["great", "good"], ["fine", "good"], ["drama", "acting"], ["skills", "talent"]]
    print(are_sentences_similar_two(a, b, c))

    print('\n重复的边')
    print(find_redundant_connection([[1, 2], [1, 3], [2, 3]]))

    print('\n最长连续序列')
    print(longest_consecutive2([100, 4, 200, 1, 3, 2]))
