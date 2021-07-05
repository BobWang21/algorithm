class Edge():
    def __init__(self, u, v, w=0):
        self.u = u
        self.v = v
        self.w = w


# graph:{a:{b:w}}, s: start, n: point num
def dijkstra(graph, s, n):
    p = [False] * n
    p[0] = True
    res = [float('inf')] * n
    res[0] = 0

    for dest, dist in graph[s].items():
        res[dest] = dist

    # 从非永久集合中选择距离最小的顶点进入永久集合
    def get_vertex_from_q():
        min_vertex = -1
        min_w = float("inf")
        for i, w in enumerate(res):
            if p[i]:
                continue
            if w <= min_w:
                min_w = w
                min_vertex = i
        return min_vertex

    while len(p) < n:
        u = get_vertex_from_q()
        p[u] = True
        if u not in graph:
            continue
        for v in graph[u]:
            if v in p:
                continue
            # ac < ab + bc
            new_dist = res[u] + graph[u][v]
            if res[v] > new_dist:
                res[v] = new_dist
    return res


# 从边出发 O(VE) 
def bellman_ford(edges, s, node_num):
    # 初始化dist
    dist = [float('inf')] * node_num
    dist[0] = 0
    # ac < ab + bc
    for edge in edges:
        if edge.u == s:
            dist[edge.v] = edge.w

    # 松弛函数
    def relax(u, v, weight):
        if dist[v] > dist[u] + weight:
            dist[v] = dist[u] + weight

    # |v-1|次松弛, 每次有1个点得到松弛
    edges_num = len(edges)
    for i in range(node_num - 1):
        for j in range(edges_num):
            relax(edges[j].u, edges[j].v, edges[j].w)

    # 判断是否有负权环
    flag = False
    for i in range(edges_num):
        if dist[edges[i].v] > dist[edges[i].u] + edges[i].w:
            flag = True
            break

    return dist, flag


# shortest path faster algorithm
def spfa(graph, s, node_num):
    queue = [s]
    enqueue = [False] * node_num
    enqueue[0] = True
    visited_cnt = {s: 1}  # 进入队列次数

    # 初始化dist
    dist = [float('inf')] * node_num
    dist[0] = 0

    # ac < ab + bc
    def relax(u, v):
        if dist[v] > dist[u] + graph[u][v]:
            dist[v] = dist[u] + graph[u][v]

    flag = False
    while queue:
        u = queue.pop(0)
        enqueue[u] = False
        if u not in graph:
            continue
        for v in graph[u]:
            tmp = dist[v]
            relax(u, v)
            if tmp != dist[v] and not enqueue[v]:
                queue.append(v)
                enqueue[v] = True
                visited_cnt[v] = visited_cnt.get(v, 0) + 1
                if visited_cnt[v] > node_num:  # 不存在负权环 最多访问node num次
                    flag = True
                    return dist, flag

    return dist, flag


def get_edges(graph):
    res = []
    for u, dic in graph.items():
        for v, w in dic.items():
            res.append(Edge(u, v, w))
    return res


if __name__ == '__main__':
    graph = {0: {5: 100, 4: 30, 2: 10},
             1: {2: 5},
             2: {3: 50},
             3: {5: 10},
             4: {3: 20, 5: 50}}
    print(dijkstra(graph, s=0, n=6))
    edges = get_edges(graph)
    print(bellman_ford(edges, s=0, node_num=6))
    print(spfa(graph, s=0, node_num=6))
