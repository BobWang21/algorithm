def power(x, n):
    if n == 1:
        return x
    v = power(x, n // 2)
    return v * v * x ** (n % 2)


def merge_sort(nums):
    n = len(nums)
    if n == 1:  # 递归基
        return nums
    mid = n // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge1(left, right)


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


# 面试题 08.03. 魔术索引 递增数组(含有重复值)
# 找到 nums[i] == i 的最小索引
def find_magic_index(nums):
    if not nums:
        return -1

    def find(l, r):
        if l > r:
            return -1
        mid = l + (r - l)
        left = find(l, mid - 1)
        if left != -1:
            return left
        if nums[mid] == mid:
            return mid
        right = find(mid + 1, r)
        return right if right != -1 else -1

    return find(0, len(nums) - 1)


# 14 longest_common_prefix
def longest_common_prefix(strs):
    def lcp(start, end):
        if start == end:
            return strs[start]

        mid = (start + end) // 2
        lcpLeft, lcpRight = lcp(start, mid), lcp(mid + 1, end)
        minLength = min(len(lcpLeft), len(lcpRight))
        for i in range(minLength):
            if lcpLeft[i] != lcpRight[i]:
                return lcpLeft[:i]

        return lcpLeft[:minLength]

    return "" if not strs else lcp(0, len(strs) - 1)


def longest_common_prefix2(strs):
    if not strs or not strs[0]:
        return ''

    def lcp(l):
        for i in range(l):
            for s in strs[1:]:
                if len(s) < i + 1 or s[i] != strs[0][i]:
                    return False
        return True

    min_l = min([len(s) for s in strs])
    l, r = 0, min_l
    while l < r:
        mid = l + (r - l + 1) // 2
        if lcp(mid):
            l = mid
        else:
            r = mid - 1
    return strs[0][:l]
