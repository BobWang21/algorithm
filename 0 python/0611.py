from collections import deque, defaultdict, Counter

import heapq as hq

a = [3, 9, 4, 7, 2]

hq.heapify(a)

while a:
    print(hq.heappop(a))

b = deque([3, 9, 4, 7, 2])
print('*' * 50)
while b:
    print(b.popleft())

c = defaultdict(int)
c[0] += 1
print(c)


# 全排列

def permute(nums):
    nums.sort()
    n = len(nums)
    visited = [False] * n
    path, res = [], []

    def helper():
        if len(path) == n:
            res.append(path[:])
            return
        for i in range(n):
            if visited[i]:
                continue
            if i > 0 and nums[i - 1] == nums[i] and not visited[i - 1]:
                continue
            visited[i] = True
            path.append(nums[i])
            helper()
            visited[i] = False
            path.pop()

    helper()
    return res


for line in permute([4, 2, 3, 3, 1]):
    print(line)

print(len(permute([4, 2, 3, 3, 1])))


def subset1(nums):
    n = len(nums)
    path, res = [], []

    def helper(i):
        res.append(path[:])
        for j in range(i, n):
            path.append(nums[j])
            helper(j + 1)
            path.pop()
        return

    helper(0)
    return res


def subset2(nums):
    n = len(nums)
    visited = [False] * n
    path, res = [], []

    def helper(i):
        res.append(path[:])
        for j in range(i, n):
            if visited[j]:
                continue
            visited[j] = True
            path.append(nums[j])
            helper(j + 1)
            visited[j] = False
            path.pop()
        return

    helper(0)
    return res


def subset3(nums):
    nums.sort()
    n = len(nums)
    path, res = [], []

    def helper(i):
        res.append(path[:])
        for j in range(i, n):
            if j > i and nums[j - 1] == nums[j]:
                continue
            path.append(nums[j])
            helper(j + 1)
            path.pop()
        return

    helper(0)
    return res


print(subset1([1, 2, 3]))
print(subset2([1, 2, 3]))
print(subset3([1, 2, 2]))

from collections import deque, defaultdict


# 269 火星词典
# 从给定的词典中提取出字母间的相对顺序信息
def alien_order(words):
    # 获取关系
    n = len(words)
    edges = defaultdict(set)
    degree = defaultdict(int)
    s = set()
    for i in range(n):
        s.add(list(words[i]))
        for j in range(i + 1, n):
            m, n = len(words[i]), len(words[j])
            w1, w2 = words[i], words[j]
            for k in range(min(m, n)):
                c1, c2 = w1[k], w2[k]
                if c1 != c2:
                    edges[c1].add(c2)  # c1 -> c2
                    degree[c2] += 1

    # 拓扑排序
    queue = deque([k for k, v in degree.items() if v == 0])

    i = 0
    res = ''
    while queue:
        chr = queue.popleft()
        i += 1
        res += chr
        for new_chr in edges[chr]:
            degree[new_chr] -= 1
            if not degree[new_chr]:
                queue.append(new_chr)

    return res if i == len(s) else ''
