#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 547 朋友圈总数 o(n^3)
# 访问整个矩阵一次，并查集操作需要最坏O(n)的时间。
def find_circle_num(M):
    if not M or not M[0]:
        return 0
    n = len(M)
    parent = list(range(n))
    rank = [0] * n
    res = [n]  # 连通图数

    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]  # 路径压缩 变为一半
            x = parent[x]
        return x

    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:
            return True  # 已经在同一个集合中 再加一条边 说明存在环
        # 减少计算 合并时rank小的树合并到rank大的树上 合并后的rank不变
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:  # 相等时 合并后的rank+1
            parent[y] = x
            rank[x] += 1
        res[0] -= 1  # 顶点数 = 边数 + 连通图数
        return False

    for i in range(n):
        for j in range(i + 1, n):
            if M[i][j]:
                union(i, j)
    return res[0]


# dfs 时间复杂度：O(edge)
def find_circle_num2(M):
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


# 是否存在环
def check_circle(M):
    if not M or not M[0]:
        return 0
    n = len(M)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径减半
            x = parent[x]
        return x

    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:
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


# 684. 冗余连接
def find_redundant_connection(edges):
    if not edges:
        return
    max_idx = 1
    for u, v in edges:
        max_idx = max(max_idx, u, v)
    par = list(range(max_idx + 1))  # 0 no use
    rank = [0] * (max_idx + 1)

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
        while parent[i] != i:
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


if __name__ == '__main__':
    print('\nfriend circle')
    m = [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 1],
         [1, 0, 1, 1]]
    print(find_circle_num(m))

    print('\ncheck graph circle')
    m = [[1, 1, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 1],
         [1, 0, 1, 1]]
    print(check_circle(m))

    print('\n句子相似性')
    a = ["great", "acting", "skills"]
    b = ["fine", "drama", "talent"]
    c = [["great", "good"], ["fine", "good"], ["drama", "acting"], ["skills", "talent"]]
    print(are_sentences_similar_two(a, b, c))

    print('\n重复的边')
    print(find_redundant_connection([[1, 2], [1, 3], [2, 3]]))

    print('\n最长连续序列')
    print(longest_consecutive2([100, 4, 200, 1, 3, 2]))
