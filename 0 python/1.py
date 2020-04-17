from collections import defaultdict


def has_path(maze, start, destination):
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # 定义上下左右四个方向
    m = len(maze)
    n = len(maze[0])

    def dfs(x, y):
        maze[x][y] = -1  # -1表示该点已经过遍历
        if x == destination[0] and y == destination[1]:
            return True

        res = False
        i, j = x, y  # 保存坐标值
        for dx, dy in directions:
            x, y = i, j
            while 0 <= x + dx < m and 0 <= y + dy < n and (maze[x + dx][y + dy] == 0 or maze[x + dx][y + dy] == -1):
                x = x + dx  # 继续前进，模拟小球的滚动过程
                y = y + dy  # 其中0为空地，而-1为之前遍历过的空地

            if maze[x][y] != -1:  # 如果该点的值不为-1，即未遍历过
                if dfs(x, y):
                    return True

        return res

    return dfs(start[0], start[1])


def possibleBipartition(N, dislikes):
    if not dislikes or not dislikes[0]:
        return True

    dic = defaultdict(set)
    for u, v in dislikes:
        dic[u].add(v)
        dic[v].add(u)

    colors = [0] * N

    def dfs(i, color):
        if not colors[i]:
            colors[i] = color
            for node in dic[i]:
                if not dfs(node, -color):
                    return False
            return True
        if colors[i] == color:
            return True
        return False

    for i in range(N):
        if not colors[i] and not dfs(i, 1):
            return False
    return True


import random as rd


def rand7():
    return rd.randint(1, 7)


from collections import Counter


def seven2ten():
    while True:
        v = (rand7() - 1) * 7
        if v > 10:
            continue
        v += rand7() - 1
        if v < 10:
            return v + 1


def uiform2norm():
    total = 0
    n = 30
    for i in range(n):
        total += rd.random()
    u = n / 2
    mse = n / 12
    return (total - u) / (mse ** 0.5)


def longestPalindromeSubseq(s):
    if not s:
        return 0
    n = len(s)
    dic = dict()

    def helper(l, r):
        if l == r:
            dic[(l, r)] = 1
            return 1
        if r < l:
            return 0
        if (l, r) in dic:
            return dic[(l, r)]
        if s[l] == s[r]:
            dic[(l, r)] = helper(l + 1, r - 1) + 2
            return dic[(l, r)]
        dic[(l, r)] = max(helper(l + 1, r), helper(l, r - 1))
        return dic[(l, r)]

    return helper(0, n - 1)


if __name__ == '__main__':
    m = [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0]]

    s, e = [0, 4], [4, 4]

    print(has_path(m, s, e))

    res = [0] * 1000
    for i in range(1000):
        res[i] = seven2ten()
    print(Counter(res))

    print('最长回文串')
    print(longestPalindromeSubseq('bbbab'))
