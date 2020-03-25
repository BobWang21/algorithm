import random as rd
from collections import Counter


def shuffle(nums):
    if not nums:
        return
    n = len(nums)
    for i in range(n):
        idx = rd.randint(i, n - 1)
        nums[i], nums[idx] = nums[idx], nums[i]
    return nums


def binary_search(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return l


# 旋转数组中的最小值
# [3 4 1 2] 为 [1 2 3 4]的旋转数组
def get_min(nums):
    if not nums:
        return
    l, r = 0, len(nums) - 1
    if nums[0] <= nums[-1]:
        return nums
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] < nums[mid - 1]:
            return nums[mid]
        elif nums[mid] >= nums[0]:
            l = mid + 1
        else:
            r = mid - 1


def find_mis(nums):
    if not nums:
        return
    l, r = 0, len(nums)
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == mid:
            l = mid + 1
        else:
            r = mid - 1
    return l


def merge(a, b):
    if not a:
        return b
    if not b:
        return a
    if a[0] < b[0]:
        return [a[0]] + merge(a[1:], b)
    else:
        return [b[0]] + merge(a, b[1:])


def merge2(a, b):
    if not a:
        return b
    if not b:
        return a
    m, n = len(a), len(b)
    i = j = 0
    res = []
    while i < m and j < n:
        if a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    if i == m:
        res += b[j:]
    if j == n:
        res += a[i:]
    return res


def merge_sort(nums):
    if not nums:
        return
    if len(nums) == 1:
        return nums
    mid = len(nums) // 2
    a = merge_sort(nums[:mid])
    b = merge_sort(nums[mid:])
    return merge(a, b)


def merge_k_sort(lists):
    if not lists:
        return
    if len(lists) == 1:
        return lists[0]
    mid = len(lists) // 2
    a = merge_k_sort(lists[:mid])
    b = merge_k_sort(lists[mid:])
    return merge(a, b)


def partition(nums, lo, hi):
    l, r = lo, hi
    pivot = nums[lo]
    while l < r:
        while l < r and nums[r] >= pivot:
            r -= 1
        nums[l] = nums[r]
        while l < r and nums[l] <= pivot:
            l += 1
        nums[r] = nums[l]
    nums[l] = pivot
    return l


def quick_sort(nums, lo, hi):
    if lo < hi:
        k = partition(nums, lo, hi)
        quick_sort(nums, lo, k - 1)
        quick_sort(nums, k + 1, hi)


def get_top_k(nums, k):
    if not nums:
        return
    if len(nums) < k:
        return
    pivot = partition(nums, 0, len(nums) - 1)
    while pivot != k - 1:
        if pivot > k - 1:
            pivot = partition(nums, 0, pivot - 1)
        else:
            pivot = partition(nums, pivot + 1, len(nums) - 1)
    return nums[:k]


def get_most_k(nums, k):
    if not nums:
        return
    if len(nums) < k:
        return
    dic = dict()
    for v in nums:
        dic.setdefault(v, 0)
        dic[v] += 1
    dic2 = dict()
    for key, value in dic.items():
        dic2.setdefault(value, [])
        dic2[value].append(key)
    n = len(nums)
    res = []
    print(dic2)
    for i in range(n, 0, -1):
        if i in dic2:
            for v in dic2[i]:
                res.append(v)
                k -= 1
                if k == 0:
                    return res


def most(nums):
    if not nums:
        return 0

    def helper(queue, i):
        while queue and nums[queue[-1]] > nums[i]:
            queue.pop(-1)
        queue.append(i)

    res = [nums[0] * nums[0]]
    total = [nums[0]]
    queue = [0]
    for i in range(1, len(nums)):
        total.append(total[i - 1] + nums[i])
        helper(queue, i)
        max_value = nums[i] * nums[i]
        for v1, v2 in zip(queue, total):
            max_value = max(max_value, v1 * v2)
        res.append(max_value)
    return max(res)


def max_set(nums):
    if not nums:
        return
    dominate = {nums[0]}
    for pair1 in nums[1:]:
        flag = True
        removed = set()
        for pair2 in dominate:
            if pair2[0] < pair1[0] and pair2[1] < pair1[1]:
                removed.add(pair2)
            if pair1[0] < pair2[0] and pair1[1] < pair2[1]:
                flag = False
        if flag:
            dominate.add(pair1)
        dominate = dominate - removed
    return dominate


if __name__ == '__main__':
    cnt = Counter()
    for i in range(6000):
        nums = ['1', '2', '3']
        nums = shuffle(nums)
        l = ','.join(nums)
        cnt[l] += 1
    print(cnt)

    print('二分查找')
    print(binary_search([1, 3, 5, 7, 10], 9))

    print(get_min([3, 4, 1]))

    print(find_mis([0, 2, 3]))
    print(merge_sort([10, 1, 4, 15]))

    print(merge2([2, 4, 6], [1, 3, 5]))

    print(merge_k_sort([[2, 4, 6], [1, 3, 5], [7, 8, 9]]))

    nums = [6, 1, 7, 2, 8]
    quick_sort(nums, 0, 4)
    print(nums)

    nums = [6, 1, 7, 2, 8, -1]
    print(get_top_k(nums, 3))

    print(get_most_k([2, 2, 3, 3, 4, 4, 4, 1, 1, 1, 1, 1], 2))

    print(most([1, 2, 6]))

    print(max_set([(1, 2), (5, 3), (4, 6), (7, 5), (9, 0)]))
