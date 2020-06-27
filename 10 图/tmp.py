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
        dic[node] = new_node

        res = []
        for n in node.neighbors:
            res.append(dfs(n))

        new_node.neighbors = res
        return new_node

    return dfs(node)


# 1192 tarjan强连通分量算法 用于寻找bridge
def critical_connections(n, connections):
    graph = defaultdict(list)
    rank = [-1] * n
    low = [10001] * n
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    def dfs(prev, curr, depth):
        if rank[curr] != -1:
            return low[curr]
        low[curr] = rank[curr] = depth
        for nxt in graph[curr]:
            if nxt == prev:  # 无向图 会导致最终的low都相等
                continue
            low[curr] = min(low[curr], dfs(curr, nxt, depth + 1))
        return low[curr]

    dfs(-1, connections[0][0], 0)

    res = []
    for connection in connections:
        u, v = connection
        if low[v] > rank[u] or low[u] > rank[v]:  # 无向图 环中的点 low[u] <= rank[v]!
            res.append(connection)
    return res


if __name__ == '__main__':
    print('\n强连通分量')
    print(critical_connections(4, [[0, 1], [1, 2], [2, 0], [1, 3]]))