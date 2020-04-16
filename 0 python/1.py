def has_path(maze, start, destination):
    if not maze or not maze[0]:
        return False

    m, n = len(maze), len(maze[0])
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def safe(i, j):
        if m > i >= 0 and n > j >= 0:
            return True
        return False

    def dfs(i, j, d):
        if i == destination[0] and j == destination[1]:
            new_i, new_j = i + d[0], j + d[1]
            if not safe(new_i, new_j) or maze[new_i][new_j] == 1:
                return True
            else:
                return False

        if not safe(i, j):
            return False

        if maze[i][j] == -1:
            return False

        if maze[i + d[0]][j + d[1]] == 1:  # 墙
            for d1 in directions:
                if d1 == d:
                    continue
                if dfs(i, j, d1):
                    return True
            return False

        maze[i][j] = -1
        return dfs(i + d[0], j + d[0], d)

    for d in directions:
        if dfs(start[0], start[1], d):
            return True
    return False


def hasPath(maze, start, destination) -> bool:
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # 定义上下左右四个方向
    m = len(maze)  # 获取矩阵大小
    n = len(maze[0])

    # 构造dfs函数，其返回值为bool值
    def dfs(m, n, maze, x, y, directions, destination):
        maze[x][y] = -1  # -1表示该点已经过遍历，防止循环
        # 如果坐标为终点坐标，返回True
        if x == destination[0] and y == destination[1]:
            return True

        res = False
        i, j = x, y  # 保存坐标值
        for dx, dy in directions:  # 对四个方向进行遍历
            x, y = i, j
            while 0 <= x + dx < m and 0 <= y + dy < n and (maze[x + dx][y + dy] == 0 or maze[x + dx][y + dy] == -1):
                # 当x,y坐标合法，并且对应值为0或-1时
                x = x + dx  # 继续前进，模拟小球的滚动过程
                y = y + dy  # 其中0为空地，而-1为之前遍历过的空地

            if maze[x][y] != -1:  # 如果该点的值不为-1，即未遍历过
                # 进行遍历，并对res和遍历结果取或
                # 有True即为True
                res = res or dfs(m, n, maze, x, y, directions, destination)

        return res  # 返回res

    return dfs(m, n, maze, start[0], start[1], directions, destination)


# 286
# -1 -墙壁或障碍物 0 -大门  INF -无限意味着一个空房间 2147483647 来表示。
def walls_and_gates(rooms):
    if not rooms or not rooms[0]:
        return
    rows, cols = len(rooms), len(rooms[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def safe(i, j):
        if i < 0 or j < 0 or i == rows or j == cols:
            return False
        return True

    def helper(i, j):
        queue = [(i, j, 0)]
        seen = {(i, j)}
        while queue:
            i, j, k = queue.pop(0)
            for d in directions:
                new_i, new_j = i + d[0], j + d[1]
                if safe(new_i, new_j) and (new_i, new_j) not in seen:
                    if rooms[new_i, new_j] == 0:
                        return k + 1
                    if rooms[new_i, new_j] > 0:
                        queue.append((new_i, new_j, k + 1))
                        seen.add((new_i, new_j))

        return float('inf')

    for i in range(rows):
        for j in range(cols):
            if rooms[i][j] > 0:
                rooms[i][j] = helper(i, j)

    return rooms


if __name__ == '__main__':
    m = [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0]]
#
# s, e = [0, 4], [4, 4]
#
# print(has_path(m, s, e))
