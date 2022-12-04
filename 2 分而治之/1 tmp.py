import heapq as hq

'''
幂
归并排序
'''


def power(x, n):
    if n == 1:  # 递归基
        return x
    v = power(x, n // 2)
    return v * v * x ** (n % 2)


# 魔术索引 递增数组(含有重复值)
# 找到 nums[i] == i 的最小索引
def find_magic_index(nums):
    if not nums:
        return -1

    def find(l, r):
        # 终止条件
        if l > r:
            return -1
        mid = l + (r - l) // 2
        left = find(l, mid - 1)
        # 先左侧
        if left != -1:
            return left
        if nums[mid] == mid:
            return mid
        right = find(mid + 1, r)
        return right if right != -1 else -1

    return find(0, len(nums) - 1)


# 归并排序
def merge_sort(nums):
    n = len(nums)
    if n == 1:  # 递归基
        return nums
    mid = n // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge1(left, right)


def merge_k_sorted_nums1(nums):
    if not nums:
        return
    if len(nums) == 1:  # base
        return nums[0]
    mid = len(nums) // 2
    a = merge_k_sorted_nums1(nums[:mid])
    b = merge_k_sorted_nums1(nums[mid:])
    return merge3(a, b)


# 合并两个有序数组
def merge1(a, b):
    if not a:
        return b
    if not b:
        return a
    i, j = 0, 0
    res = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    # 判断循环跳出时的状态
    if i < len(a):
        res += a[i:]
    if j < len(b):
        res += b[j:]
    return res


# 时间复杂度高 但写法简单
def merge2(a, b):
    if not a:
        return b
    if not b:
        return a
    i, j = 0, 0
    res = []
    while i < len(a) or j < len(b):
        v1 = a[i] if i < len(a) else float('inf')  # 注意v的取值
        v2 = b[j] if j < len(b) else float('inf')
        if v1 < v2:
            res.append(v1)
            i += 1
        else:
            res.append(v2)
            j += 1

    return res


# 递归
def merge3(a, b):
    if not a:
        return b
    if not b:
        return a
    res = []
    if a[0] < b[0]:
        res.append(a[0])
        res += merge3(a[1:], b)
    else:
        res.append(b[0])
        res += merge3(a, b[1:])
    return res


def merge_k_sorted_nums2(nums):
    if not nums or not nums[0]:
        return []

    n = len(nums)
    heap = [(nums[i][0], i, 0) for i in range(n)]
    hq.heapify(heap)

    res = []
    while heap:
        v, i, j = hq.heappop(heap)
        res.append(v)
        if j + 1 < len(nums[i]):
            hq.heappush(heap, (nums[i][j + 1], i, j + 1))
    return res


if __name__ == '__main__':
    print(find_magic_index([2, 3, 4, 4, 5, 5, 5]))
    print('\n归并排序')
    print(merge_sort([1, 3, 2, 4]))

    print('\n合并两个有序数组')
    print(merge1([1, 3], [2, 4]))
    print(merge2([1, 3], [2, 4]))
    print(merge3([1, 3], [2, 4]))

    print('\n合并K个有序数组')
    print(merge_k_sorted_nums1([[2, 4, 5], [1, 1, 9], [6, 7, 8]]))
    print(merge_k_sorted_nums2([[2, 4, 5], [1, 1, 9], [6, 7, 8]]))
