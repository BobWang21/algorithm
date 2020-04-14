from collections import defaultdict


# 是否可以二分
def possible_bipartition(N, dislikes):
    dic = defaultdict(set)
    for u, v in dislikes:
        dic[u - 1].add(v - 1)
        dic[v - 1].add(u - 1)

    colors = [0] * N

    def dfs(node, color):
        if not colors[node]:
            colors[node] = color
            for child in dic[node]:
                if not dfs(child, -color):
                    return False
            return True
        if colors[node] != color:
            return False
        return True

    for i in range(N):
        if not colors[i] and not dfs(i, 1):
            return False

    return True
