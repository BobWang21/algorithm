#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter, defaultdict


# 未排序数组 解唯一 O(N)空间复杂度
def two_sum1(nums, target):
    dic = dict()

    for i in range(len(nums)):
        if target - nums[i] in dic:
            return [dic[target - nums[i]], i]
        dic[nums[i]] = i


# 排序数组 解不唯一 O(N)
def two_sum2(nums, target):
    n = len(nums)
    if n < 2:
        return []
    l, r = 0, n - 1
    res = []
    while l < r:
        total = nums[l] + nums[r]
        if total < target:
            l += 1
        elif total > target:
            r -= 1
        else:
            res.append([nums[l], nums[r]])
            while l < r and nums[l] == nums[l + 1]:  # 跳出相等部分
                l += 1
            l += 1
    return res


def three_sum(nums, target):
    n = len(nums)
    if n < 3:
        return
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:  # 防止第一个数重复
            continue
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < target:
                l += 1
            elif s > target:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                l += 1
    return res


# 和与target最相近的三个数
def three_sum_closet(nums, target):
    n = len(nums)
    if n < 3:
        return
    closet_sum = None
    gap = float('inf')
    nums.sort()
    for i in range(n - 2):
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == target:
                return s
            new_gap = abs(s - target)
            if new_gap < gap:
                gap = new_gap
                closet_sum = s
            if s > target:
                r -= 1
            else:
                l += 1
    return closet_sum


# 四个数组 每个数组中选一个数字 要求四个数和为0 返回和为0的数组个数
def four_sum_count(A, B, C, D):
    dic = defaultdict(int)
    n = len(A)
    for i in range(n):
        for j in range(n):
            s = A[i] + B[j]
            dic[s] += 1

    res = 0
    for i in range(n):
        for j in range(n):
            s = C[i] + D[j]
            if -s in dic:
                res += dic[-s]
    return res


# 左右指针 977
def sorted_squares(nums):
    if not nums:
        return []
    n = len(nums)
    res = [0] * n
    i = n - 1
    l, r = 0, n - 1
    while l <= r:
        if abs(nums[l]) <= abs(nums[r]):
            res[i] = nums[r] ** 2
            r -= 1
        else:
            res[i] = nums[l] ** 2
            l += 1
        i -= 1
    return res


# 三个指针 递推关系 假设之前的都满足
def partition2(nums, l, r):
    def swap(i, j):
        nums[i], nums[j] = nums[j], nums[i]

    if not nums:
        return []
    lt, mt = l, r
    i = l
    pivot = nums[l]
    while i <= mt:
        if nums[i] < pivot:
            swap(i, lt)
            i += 1
            lt += 1
        elif nums[i] == pivot:  # nums[i-1] <= pivot!
            i += 1
        else:
            swap(mt, i)
            mt -= 1  # 后边换过来的数 不知道其数值 因此不移动i
    return lt, mt


# 26 原地删除升序数组中的重复数字 并返回非重复数组的长度
# nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
def remove_duplicates1(nums):
    if not nums:
        return 0
    n = len(nums)
    i = 1  # 第一个指针 下一个不重复数字放置的位置 类似 三指针的lt
    for j in range(1, n):  # 第二个指针
        if nums[j - 1] != nums[j]:  # 非重复数字
            nums[i] = nums[j]
            i += 1
    return i


# 数组中每个数字最多出现2次 返回删除重复数字后的长度
def remove_duplicates2(nums):
    if len(nums) <= 2:
        return len(nums)
    # 假设 nums[0...i] 符合要求
    i = 1  # 记录最后一个满足要求的位置
    for j in range(2, len(nums)):
        if nums[j] != nums[i] or nums[j] != nums[i - 1]:
            i += 1
            nums[i] = nums[j]

    return i + 1


def remove_duplicates3(nums):
    if len(nums) <= 2:
        return nums
    cnt = 1
    n = len(nums)
    j = 1
    for i in range(1, n):
        if nums[i] == nums[i - 1]:
            if cnt == 1:
                nums[j] = nums[i]
                j += 1
                cnt += 1
        else:
            nums[j] = nums[i]
            j += 1
            cnt = 1
    return nums[:j]


# 不改变顺序 把0移到数组尾部
def move_zeros(nums):
    if not nums:
        return
    k, n = 0, len(nums)
    for i in range(n):
        if nums[i]:
            nums[k] = nums[i]
            k += 1
    for i in range(k, n):
        nums[i] = 0
    return nums


# 相差为K的pair
def find_pairs(nums, k):
    if len(nums) < 2:
        return 0

    nums.sort()

    if nums[-1] - nums[0] < k:
        return 0

    n = len(nums)
    l, r = 0, 1
    num = 0
    while r < n and l < n:
        if l == r:
            r += 1
            continue
        s = nums[r] - nums[l]
        if s < k:
            r += 1
        elif s > k:
            l += 1
        else:
            num += 1
            while l + 1 < n and nums[l] == nums[l + 1]:
                l += 1
            l += 1
            r = max(l + 1, r + 1)
    return num


def find_pairs2(nums, k):
    res = 0
    c = Counter(nums)
    for i in c:
        if (k > 0 and i + k in c) or (k == 0 and c[i] > 1):
            res += 1
    return res


# 11. 盛最多水的容器
# s = min(hl, hr) * (r-l)
# 关注移动后的短板会不会变长
# 向内移动长板小于等于当前面积 向内移动短板大于等于当前面积
# 因此我们选择向内移动短板
def max_area(height):
    res = 0
    l, r = 0, len(height) - 1
    while l < r:
        if height[l] < height[r]:
            res = max(res, (r - l) * height[l])
            l += 1
        else:
            res = max(res, (r - l) * height[r])
            r -= 1
    return res


if __name__ == '__main__':
    print('\n2 sum')
    print(two_sum1([7, 8, 2, 3], 9))
    print(two_sum2([1, 2, 7, 8, 11, 15], 9))

    print('\n3 sum')
    print(three_sum([2, 7, 7, 11, 15, 15, 20, 24, 24], 33))

    print('\n3 sum closet')
    print(three_sum_closet([-1, 2, 1, -4], 1))

    print('\n平方排序')
    print(sorted_squares([-7, -3, 3, 11]))

    print('\n删除排查数组中的重复数值')
    print(remove_duplicates1([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))
    print(remove_duplicates2([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))
    print(remove_duplicates3([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))

    print('\n移动0至数组尾部')
    print(move_zeros([0, -7, 0, 2, 3, 11]))

    print('\n相差为K的pair数目')
    print(find_pairs([1, 3, 1, 5, 4], 0))
    print(find_pairs2([1, 3, 1, 5, 4], 0))

    print('\n差值小于等于target的对数')
    nums = [1, 3, 5, 7]
