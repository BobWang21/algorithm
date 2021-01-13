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


if __name__ == '__main__':
    print(permute(3, 5, 2))
    print(permute2(3))
