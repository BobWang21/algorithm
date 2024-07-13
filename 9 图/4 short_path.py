from collections import defaultdict
from heapq import *


class Edge():
    def __init__(self, u, v, w=0):
        self.u = u
        self.v = v
        self.w = w


def dijkstra(graph, f, t):
    heap = [(0, f, ())]  # [(cost1, v1, path)]
    p_set = set()  # 永久标记
    res = {f: 0}
    while heap:
        cost1, v1, path = heappop(heap)
        if v1 in p_set:
            continue
        p_set.add(v1)
        path = (path, v1)  # 记录路径
        if v1 == t:
            return cost1, path

        for v2, cost2 in graph[v1].items():
            if v2 in p_set:
                continue
            # 松弛
            prev = res.get(v2, float('inf'))
            curr = cost1 + cost2
            if curr < prev:
                res[v2] = curr
                heappush(heap, (curr, v2, path))

    return


# 从边出发 O(VE) 
def bellman_ford(edges, s, node_num):
    # 初始化dist
    dist = [float('inf')] * node_num
    dist[0] = 0
    edges_num = len(edges)
    # ac < ab + bc
    for edge in edges:
        if edge.u == s:
            dist[edge.v] = edge.w

    # 松弛函数
    def relax(u, v, weight):
        dist[v] = min(dist[u] + weight, dist[v])

    # v-1次松弛, 每次有1个点得到松弛
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


# 带有队列优化的Bellman-Ford算法
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


# 752
import heapq


class AStar:
    # 计算启发函数
    @staticmethod
    def getH(status, target):
        ret = 0
        for i in range(4):
            dist = abs(int(status[i]) - int(target[i]))
            ret += min(dist, 10 - dist)
        return ret

    def __init__(self, status, target, g):
        self.status = status
        self.g = g
        self.h = AStar.getH(status, target)
        self.f = self.g + self.h

    def __lt__(self, other):
        return self.f < other.f


# A* 算法
class Solution:
    def openLock(self, deadends, target):
        if target == "0000":
            return 0

        dead = set(deadends)
        if "0000" in dead:
            return -1

        def num_prev(x):
            return "9" if x == "0" else str(int(x) - 1)

        def num_succ(x):
            return "0" if x == "9" else str(int(x) + 1)

        def get(status):
            s = list(status)
            for i in range(4):
                num = s[i]
                s[i] = num_prev(num)
                yield "".join(s)
                s[i] = num_succ(num)
                yield "".join(s)
                s[i] = num

        q = [AStar("0000", target, 0)]
        seen = set(["0000"])
        while q:
            node = heapq.heappop(q)
            for next_status in get(node.status):
                if next_status not in seen and next_status not in dead:
                    if next_status == target:
                        return node.g + 1
                    heapq.heappush(q, AStar(next_status, target, node.g + 1))
                    seen.add(next_status)

        return -1


if __name__ == '__main__':
    graph = {0: {5: 100, 4: 30, 2: 10},
             1: {2: 5},
             2: {3: 50},
             3: {5: 10},
             4: {3: 20, 5: 50}}
    print(dijkstra(graph, f=0, t=3))
    edges = get_edges(graph)
    print(bellman_ford(edges, s=0, node_num=6))
    print(spfa(graph, s=0, node_num=6))
