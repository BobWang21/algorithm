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
        dic[node] = new_node  # 防止循环必须在此数标记状态

        new_node.neighbors = [dfs(n) for n in node.neighbors]
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




if __name__ == '__main__':
    print('\n强连通分量')
    print(critical_connections(4, [[0, 1], [1, 2], [2, 0], [1, 3]]))


