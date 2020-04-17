def find_circle_num(M):
    if not M or not M[0]:
        return 0
    n = len(M)
    par = list(range(n))
    rank = [0] * n
    res = [n]

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]  # 路径压缩 变为一半
            x = par[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return True  # 已经在同一个集合中 再加一条边 说明存在环
        # 减少计算 合并时rank小的树合并到rank大的树上 合并后的rank不变
        if rank[root_x] < rank[root_y]:
            par[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            par[root_y] = root_x
        else:  # 相等时 合并后的rank+1
            par[root_y] = root_x
            rank[root_x] += 1
        res[0] -= 1
        return False

    for i in range(n):
        for j in range(i + 1, n):
            if M[i][j]:
                union(i, j)
    return res[0]


# dfs 版
def find_circle_num2(M):
    n = len(M)
    seen = set()

    def dfs(node):
        for i, v in enumerate(M[node]):
            if v and i not in seen:
                seen.add(i)
                dfs(i)

    res = 0
    for i in range(n):
        if i not in seen:
            dfs(i)
            res += 1
    return res


def check_circle(M):
    if not M or not M[0]:
        return 0
    n = len(M)
    par = list(range(n))
    rank = [0] * n

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]  # 路径减半
            x = par[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return True  # 在同一个集合中 再加一条边 说明存在环
        # 减少计算
        if rank[root_x] < rank[root_y]:
            par[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            par[root_y] = root_x
        else:
            par[root_y] = root_x
            rank[root_x] += 1
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
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return
        if rank[root_x] < rank[root_y]:
            par[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            par[root_y] = root_x
        else:
            par[root_y] = root_x
            rank[root_x] += 1
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
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return True
        if rank[root_x] > rank[root_y]:
            par[root_y] = root_x
        elif rank[root_x] < rank[root_y]:
            par[root_x] = root_y
        else:
            par[root_y] = root_x
            rank[root_x] += 1
        return False

    for u, v in edges:
        if union(u, v):
            return u, v


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
