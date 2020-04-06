# 26 升序数组中的重复数字
# Given nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
# Your function should return length = 5,
# with the first five elements of nums being modified to 0, 1, 2, 3, and 4 respectively.
# It doesn't matter what values are set beyond the returned length.
import random as rd


def num_islands(grid):
    if not grid:
        return 0
    visited = set()
    rows, cols = len(grid), len(grid[0])

    def helper(i, j):
        queue = [(i, j)]
        visited.add((i, j))
        while queue:
            i, j = queue.pop(0)
            if i - 1 >= 0 and grid[i - 1][j] == '1' and (i - 1, j) not in visited:
                visited.add((i - 1, j))
                queue.append((i - 1, j))
            if i + 1 <= rows - 1 and grid[i + 1][j] == '1' and (i + 1, j) not in visited:
                visited.add((i + 1, j))
                queue.append((i + 1, j))
            if j - 1 >= 0 and grid[i][j - 1] == '1' and (i, j - 1) not in visited:
                visited.add((i, j - 1))
                queue.append((i, j - 1))
            if j + 1 <= cols - 1 and grid[i][j + 1] == '1' and (i, j + 1) not in visited:
                visited.add((i, j + 1))
                queue.append((i, j + 1))

    num = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1' and (i, j) not in visited:
                helper(i, j)
                num += 1
    return num


def max_area_of_island(grid):
    if not grid:
        return 0
    visited = set()
    rows, cols = len(grid), len(grid[0])

    def helper(i, j):
        queue = [(i, j)]
        visited.add((i, j))
        area = 1
        while queue:
            i, j = queue.pop(0)
            if i - 1 >= 0 and grid[i - 1][j] == '1' and (i - 1, j) not in visited:
                visited.add((i - 1, j))
                queue.append((i - 1, j))
                area += 1
            if i + 1 <= rows - 1 and grid[i + 1][j] == '1' and (i + 1, j) not in visited:
                visited.add((i + 1, j))
                queue.append((i + 1, j))
                area += 1
            if j - 1 >= 0 and grid[i][j - 1] == '1' and (i, j - 1) not in visited:
                visited.add((i, j - 1))
                queue.append((i, j - 1))
                area += 1
            if j + 1 <= cols - 1 and grid[i][j + 1] == '1' and (i, j + 1) not in visited:
                visited.add((i, j + 1))
                queue.append((i, j + 1))
                area += 1
        return area

    res = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1' and (i, j) not in visited:
                res = max(helper(i, j), res)
    return res


# 130
def surrounded_regions(board):
    if not board:
        return
    rows, cols = len(board), len(board[0])

    def dfs(i, j):
        if i < 0 or j < 0 or i == rows or j == cols or board[i][j] == 'X':
            return
        if board[i][j] == 'A':
            return
        board[i][j] = 'A'
        dfs(i - 1, j)
        dfs(i + 1, j)
        dfs(i, j - 1)
        dfs(i, j + 1)

    for i in range(rows):
        dfs(i, 0)
        dfs(i, cols - 1)
    for j in range(cols):
        dfs(0, j)
        dfs(rows - 1, j)
    dic = {'A': 'O', 'O': 'X', 'X': 'X'}
    for i in range(rows):
        for j in range(cols):
            board[i][j] = dic[board[i][j]]
    return board


def rand2to5():
    while True:
        v = rd.randint(0, 1) * 2 ** 2 + rd.randint(0, 1) * 2 + rd.randint(0, 1)
        if v <= 4:
            return v + 1


def sorted_squares(nums):
    if not nums:
        return
    n = len(nums)
    res = [0] * n
    l, r = 0, n - 1
    i = 0
    while l < r:
        if abs(nums[l]) < abs(nums[r]):
            res[i] = nums[r] ** 2
            r -= 1
        else:
            res[i] = nums[l] ** 2
            l += 1
        i += 1
    return res


def two_sum(nums, target):
    if not nums:
        return
    nums.sort()
    if nums[0] > target:
        return
    res = []
    n = len(nums)
    l, r = 0, n - 1
    while l < r:
        s = nums[l] + nums[r]
        if s > target:
            r -= 1
        elif s < target:
            l += 1
        else:
            res.append([nums[l], nums[r]])
            while l < r and nums[l] == nums[l + 1]:
                l += 1
            l += 1
    return res


def can_partition_k_subsets(nums, k):
    if not nums or len(nums) < k:
        return []
    total = sum(nums)
    if total % k:
        return []
    target = total / k
    candidates = []
    for v in nums:
        if v < target:
            candidates.append(v)
        elif v == target:
            k -= 1
        else:
            return []
    n = len(candidates)
    status = [False] * n

    def helper(i, target, k):
        if k == 0:
            return True
        if target == 0:
            return helper(0, target, k - 1)
        for i in range(i, n):
            if status[i]:
                continue
            status[i] = True
            if target - nums[i] >= 0 and helper(i + 1, target - nums[i], k):
                return True
            status[i] = False
        return False

    return helper(0, target, k)


# 130 将围住的o变成 x 逆向思维
def surrounded_regions(board):
    if not board:
        return
    rows, cols = len(board), len(board[0])

    def back(path):
        for i, j in path:
            board[i][j] = 'O'

    def dfs(i, j, path):
        if i < 0 or j < 0 or i == rows or j == cols:
            return False
        if board[i][j] == 'X':
            return True
        board[i][j] = 'X'
        path.add((i, j))
        if dfs(i - 1, j, path) and dfs(i + 1, j, path) and dfs(i, j - 1, path) and dfs(i, j + 1, path):
            return True
        back(path)
        return False

    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                dfs(i, j, set())
    return board


if __name__ == '__main__':
    # print('\n删除排查数组中的重复数值')
    #
    # grid = [['1', '1', '0', '0', '0'],
    #         ['1', '1', '0', '0', '0'],
    #         ['0', '0', '1', '0', '0'],
    #         ['0', '0', '0', '1', '1']]
    # print(num_islands(grid))
    #
    # board = [["O", "X", "X", "O", "X"],
    #          ["X", "O", "O", "X", "O"],
    #          ["X", "O", "X", "O", "X"], ["O", "X", "O", "O", "O"],
    #          ["X", "X", "O", "X", "O"]]
    # print(surrounded_regions(board))
    #
    # print(rand2to5())
    #
    # res = []
    # for i in range(500000):
    #     res.append(rand2to5())
    # print(Counter(res))

    print(sorted_squares([-7, -3, 2, 3, 11]))
    print(two_sum([1, 2, 2, 2, 7, 7, 7, 8, 11, 15], 9))
    nums = [114, 96, 18, 190, 207, 111, 73, 471, 99, 20, 1037, 700, 295, 101, 39, 649]
    print(can_partition_k_subsets(nums, 4))

    print(surrounded_regions([["X", "X", "X", "X"], ["X", "O", "O", "X"], ["X", "X", "O", "X"], ["X", "O", "X", "X"]]))
    print([["X", "X", "X", "X"], ["X", "X", "X", "X"], ["X", "X", "X", "X"], ["X", "O", "X", "X"]])
