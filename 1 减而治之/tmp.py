def next_permutation(nums):
    if not nums:
        return
    n = len(nums)

    r = n - 1
    while r > 0 and nums[r - 1] >= nums[r]:
        r -= 1
    if not r:
        nums.reverse()
        return nums
    k = r - 1
    while r < n and nums[r] > nums[k]:  # 可以二分
        r += 1
    r -= 1
    nums[k], nums[r] = nums[r], nums[k]

    # reverse
    l, r = k + 1, n - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums


# 超过一半的数
def most_data(data):
    n = len(data)
    if n == 0:
        raise Exception('')
    value = data[0]
    count = 1
    for i in range(1, n):
        if count == 0:
            value = data[i]
            count = 1
            continue
        if data[i] == value:
            count += 1
        else:
            count -= 1
    return value


def search_matrix(target, tar):
    if not target or not target[0]:
        return False
    row = 0
    col = len(target[0]) - 1
    while row < len(target) and col >= 0:
        num = target[row][col]
        if num == tar:
            return True
        elif num < tar:
            row += 1
        else:
            col += 1
    return False


# 选择排序
def select_sort(nums):
    if not nums:
        return
    n = len(nums)
    for i in range(n):
        idx = i
        for j in range(i + 1, n):
            if nums[j] < nums[idx]:
                idx = j
        nums[idx], nums[i] = nums[i], nums[idx]
    return nums


# 计数排序


# 347 出现次数最多的K个数 类似计数排序
def top_k_frequent(nums, k):
    dic = dict()
    for v in nums:
        dic.setdefault(v, 0)
        dic[v] += 1
    fre = dict()
    for k, v in dic.items():  # 将出现次数相同的数字放在一个列表中 类似链表
        fre.setdefault(v, [])
        fre[v].append(k)

    res = []
    for i in range(len(nums), 0, -1):  # 类似降序排列
        if i in fre:
            for v in fre[i]:
                res.append(v)
                if len(res) == k:
                    return res[:k]


# 最长连续数字 最小值的连续个数
def longest_consecutive(nums):
    def helper(s):
        i = 0
        v = min(s)
        while True:
            if i + v in s:
                s.remove(i + v)
                i += 1
            else:
                return i

    if not nums:
        return 0
    s = set(nums)
    res = 0
    while s:
        res = max(helper(s), res)
        if res >= (len(nums)) // 2:
            return res
    return res


# 区间合并
def merge(intervals):
    if not intervals or not intervals[0]:
        return []
    intervals.sort()
    res = []
    pre_end = -float('inf')
    for start, end in intervals:
        if start > pre_end:
            res.append([start, end])
            pre_end = end
        elif end <= pre_end:
            continue
        else:
            res[-1][1] = end
            pre_end = end

    return res


if __name__ == '__main__':
    print('\n下一个排列')
    print(next_permutation([1, 1, 3]))

    print('\n选择排序')
    print(select_sort([3, 1, 4, 4, 10, -1]))
    matrix = [
        [1, 3, 5, 7],
        [10, 11, 16, 20],
        [23, 30, 34, 50]
    ]

    print('\n众数')
    print(most_data([1, 3, 3, 3, 9]))

    print('出现次数最多的K个数')
    print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))

    print('\n区间合并')
    print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))
