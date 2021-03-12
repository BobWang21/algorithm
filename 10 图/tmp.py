from collections import defaultdict


class Node(object):
    def __init__(self, val=0, neighbors=None):
        if neighbors is None:
            neighbors = []
        self.val = val
        self.neighbors = neighbors


# 133 复制图
def clone_graph(node):
    dic = {}

    def dfs(node):
        if not node:
            return
        if node in dic:
            return dic[node]

        new_node = Node(node.val)
        dic[node] = new_node  # hash的是地址

        neighbors = []
        for n in node.neighbors:
            neighbors.append(dfs(n))

        new_node.neighbors = neighbors
        return new_node

    return dfs(node)


# 1192 tarjan强连通分量算法 用于寻找bridge
def critical_connections(n, connections):
    dic = defaultdict(list)
    dfs = [-1] * n
    low = [10001] * n
    for u, v in connections:
        dic[u].append(v)
        dic[v].append(u)

    def tarjan(prev, curr, depth):
        if dfs[curr] != -1:
            return low[curr]
        low[curr] = dfs[curr] = depth

        for nxt in dic[curr]:
            if nxt == prev:  # 无向图 会导致最终的low都相等
                continue
            low[curr] = min(low[curr], tarjan(curr, nxt, depth + 1))
        return low[curr]

    tarjan(-1, connections[0][0], 0)

    res = []
    for connection in connections:
        u, v = connection
        if low[v] > dfs[u] or low[u] > dfs[v]:  # 无向图 环中的点 low[u] <= rank[v]!
            res.append(connection)
    return res


def criticalConnections(n, connections):
    """
    :type n: int
    :type connections: List[List[int]]
    :rtype: List[List[int]]
    """
    rank, less = [0] * n, [float('inf')] * n

    dic = defaultdict(list)

    for u, v in connections:
        dic[u].append(v)
        dic[v].append(u)

    def dfs(pre, curr, r):
        if not rank[curr]:
            return less[curr]

        rank[curr] = less[curr] = r
        for nxt in dic[curr]:
            if nxt == pre:
                continue
            less[curr] = min(less[curr], dfs(curr, nxt, r + 1))
        return less[curr]

    dfs(-1, connections[0][0], 1)

    res = []
    for u, v in connections:
        if less[v] > rank[u] or less[u] > rank[v]:
            res.append([u, v])
    return res


def bipartite_graphs(l2r):
    r2l = dict()
    for left, adj in l2r.items():
        for right, score in adj.items():
            r2l.setdefault(right, dict())
            r2l[right][left] = score

    visit_left = dict()
    visit_right = dict()
    res = []

    def helper(i, path):
        if visit_left.get(i, False):
            return
        visit_left[i] = True
        path[i] = l2r[i]
        for j in l2r[i]:
            if visit_right.get(i, False):
                continue
            visit_right[j] = True
            for k in r2l[j]:
                helper(k, path)

        return

    for i in l2r:
        if visit_left.get(i, False):
            continue
        path = dict()
        helper(i, path)
        res.append(path)

    return res


def hungarian(graph, m, n):
    match = [False] * n
    visit_y = [False] * n

    def helper(i):
        # 遍历y顶点
        for j, val in graph[i].items():
            # 每一轮匹配, 每个y顶点只尝试一次
            if visit_y[j]:
                continue
            visit_y[j] = True
            if match[j] == -1 or helper(match[j]):
                match[j] = i
                return True

        return True

    res = 0
    for i in range(m):
        visit_y = [False] * n
        if helper(i):
            res += 1
    return res


if __name__ == '__main__':
    print('\n强连通分量')
    print(critical_connections(4, [[0, 1], [1, 2], [2, 0], [1, 3]]))

    print('\n 二分图拆分连通图')
    left2right = {
        0: {0: 0.5},
        1: {0: 1},
        2: {1: 0.5, 2: 0.6},
        3: {4: 0},
        4: {3: 0},
    }
    print(bipartite_graphs(left2right))

    graph = {
        0: {0: 0, 2: 0},
        1: {0: 0, 1: 0},
        2: {1: 0}
    }
    print(hungarian(graph, 3, 3))
