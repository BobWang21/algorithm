from collections import defaultdict


# 图是否二分 染色
def possible_bipartition(N, dislikes):
    dic = defaultdict(set)
    for u, v in dislikes:
        dic[u - 1].add(v - 1)
        dic[v - 1].add(u - 1)

    colors = [0] * N

    def dfs(node, color):
        if not colors[node]:  # 未染色
            colors[node] = color
            for child in dic[node]:
                if not dfs(child, -color):  # 不能染色
                    return False
            return True
        if colors[node] != color:
            return False
        return True

    for i in range(N):
        if not colors[i] and not dfs(i, 1):
            return False

    return True


# 拓扑排序
def can_finish(num_courses, prerequisites):
    dic = defaultdict(set)  # v -> u
    indegree = [0] * num_courses  # 入度
    for u, v in prerequisites:
        dic[v].add(u)
        indegree[u] += 1

    queue = []
    for i in range(num_courses):
        if not indegree[i]:
            queue.append(i)

    while queue:
        u = queue.pop(-1)
        for v in dic[u]:
            indegree[v] -= 1
            if not indegree[v]:
                queue.append(v)
        del dic[u]
    return False if dic else True


def find_order(num_courses, prerequisites):
    dic = defaultdict(set)  # v -> u
    indegree = [0] * num_courses  # 入度
    for u, v in prerequisites:
        dic[v].add(u)
        indegree[u] += 1

    queue = []
    for i in range(num_courses):
        if not indegree[i]:
            queue.append(i)
    res = []
    while queue:
        u = queue.pop(-1)
        res.append(u)
        for v in dic[u]:
            indegree[v] -= 1
            if not indegree[v]:
                queue.append(v)
        del dic[u]
    return res if len(res) == num_courses else []


def find_min_height_trees(n, edges):
    dic = defaultdict(set)  # v -> u
    for u, v in edges:
        dic[v].add(u)
        dic[u].add(v)

    def helper(u, seen):
        height = 0
        for v in dic[u]:
            if v not in seen:
                seen.add(v)
                height = max(height, helper(u, seen))
        return height + 1

    res = []
    min_height = float('inf')
    for i in range(n):
        height = helper(i, set())
        if height == min_height:
            res.append(i)
        elif height < min_height:
            min_height = height
            res = [i]
    return res


if __name__ == '__main__':
    print('\n是否可以完课')
    print(can_finish(2, [[1, 0]]))
