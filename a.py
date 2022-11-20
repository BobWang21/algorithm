def permute(m, n, k):
    # m个车 n个人
    if n < m:
        return

    block = [k] * m
    seen = [False] * n
    res, path = set(), [[] for _ in range(m)]

    def check():
        for v in block:
            if v == k:
                return False
        return True

    def sort_str(lists):
        res = []
        for v in lists:
            v1 = v[:]
            v1.sort()
            res.append(v1)
        return str(res)

    def helper(x):
        if x == n and check():
            res.add(sort_str(path))
            return

        if x == n:
            return
        for i in range(n):
            if seen[i]:
                continue
            seen[i] = True
            for j in range(m):
                if not block[j]:
                    continue
                block[j] -= 1
                path[j].append(i)

                helper(x + 1)

                block[j] += 1
                path[j].pop(-1)

            seen[i] = False

    helper(0)
    res = list(res)
    res.sort()
    return res


def permute2(n):
    if not n:
        return []
    res, path = [], []
    status = [2] * n

    def helper():
        if len(path) == 2 * n:
            res.append(path[:])
            return
        for i in range(n):
            if not status[i]:
                continue
            status[i] -= 1
            if status[i] == 1:
                path.append(chr(ord('A') + i) + '_0')
            else:
                path.append(chr(ord('A') + i) + '_1')

            helper()
            path.pop(-1)
            status[i] += 1

    helper()
    return res


import heapq as hq


def merge(intervals):
    if not intervals or not intervals[0] or len(intervals) == 1:
        return intervals

    intervals.sort()

    res = [intervals[0]]
    hq.heapify(res)
    for i in range(1, len(intervals)):
        interval = intervals[i]
        if res[0][1] < interval[0]:
            hq.heappush(res, interval)
        else:
            top = hq.heappop(res)
            left = min(top[0], interval[0])
            right = max(top[1], interval[1])
            hq.heappush(res, [left, right])

    return res


def jump(start, values):
    if start == 0:
        return False, 0
    n = len(values)

    values = [start] + values

    res = [0] * (n + 1)
    res[0] = start

    def helper(i):
        step = res[i]
        for j in range(i + 1, i + step + 1):
            if j > n:
                return
            new_v = res[i] - (j - i) + values[j]
            if new_v > res[j]:
                res[j] = new_v
                helper(j)

    helper(0)
    print(res)
    return res[-1]


def merge2(num1, num2):
    i, j = 0, 0
    res = []
    m, n = len(num1), len(num2)
    while True:
        if i == len(num1) and j == len(num2):
            return res
        a = num1[i] if i < m else float('inf')
        b = num2[j] if j < n else float('inf')
        if a < b:
            v = a
            i += 1
        else:
            v = b
            j += 1
        if res and res[-1] == v:
            continue
        res.append(v)

    return res


def binary_search(nums, target):
    if not nums:
        return -1
    n = len(nums)
    l, r = 0, n - 1
    while l < r:
        mid = l + (r - l + 1) // 2
        if nums[mid] <= target:
            l = mid
        else:
            r = mid - 1
    return l if nums[l] == target else -1


if __name__ == '__main__':
    print(permute(3, 5, 2))
    print(permute2(3))

    print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))
    print(jump(4, [-10, -10, 3, 10, -1, -1]))
    print(merge2([1, 2, 3, 4, 5, 5, 6, 6, 7], [2, 3, 4, 4, 5, 6, 8, 8, 9, 10, 11]))
    print(binary_search([1, 2, 2, 3], 3))
