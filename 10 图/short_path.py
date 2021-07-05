import heapq as hq
import math


# 从非永久集合中选择点进如永久集合
def get_vertex_from_q(distance, p):
    min_vertex = -1
    min_dist = float("inf")
    for i, dist in enumerate(distance):
        if i in p:
            continue
        if dist <= min_dist:
            min_dist = dist
            min_vertex = i
    return min_vertex


# graph = {a:{b:dist}} s: start, n: point num
def dijkstra(graph, s, n):
    p = {s}  # 永久集合
    res = [float('inf')] * n
    res[0] = 0

    for dest, dist in graph[s].items():
        res[dest] = dist

    while len(p) < n:
        vertex = get_vertex_from_q(res, p)
        p.add(vertex)

        if vertex not in graph:
            continue
        for slack_vertex in graph[vertex]:
            if slack_vertex in p:
                # print('in')
                continue
            # ac < ab + bc
            new_dist = res[vertex] + graph[vertex][slack_vertex]  # dist(s, point dist) + dist(point, slack_point)
            if res[slack_vertex] > new_dist:
                res[slack_vertex] = new_dist
    return res


if __name__ == '__main__':
    graph = {0: {5: 100, 4: 30, 2: 10},
             1: {2: 5},
             2: {3: 50},
             3: {5: 10},
             4: {3: 20, 5: 50}}
    print(dijkstra(graph, s=0, n=6))
