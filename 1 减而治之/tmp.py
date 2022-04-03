#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict

'''
TopK 问题的几种解法
1 桶排序
2 堆排序
3 二分查找 需要构造递增序列
4 对于未排序的 使用partition
'''


def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    i, j = 0, cols - 1
    while i < rows and 0 <= j:
        num = matrix[i][j]
        if num == target:
            return True
        elif num < target:
            i += 1
        else:
            j -= 1
    return False


# 计数排序 给定一个包含红色、白色和蓝色，n个元素的数组，
# 原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列
# 使用整数 0, 1 和 2 分别表示红色、白色和蓝色。
def sort_colors(nums):
    if not nums:
        return []
    count = [0] * 3
    for color in nums:
        count[color] += 1
    start = 0
    for i in range(3):
        for j in range(count[i]):
            nums[start] = i
            start += 1
    return nums


# 347 出现次数最多的K个数 类似计数排序
def top_k_frequent(nums, k):
    dic = defaultdict(int)
    for v in nums:
        dic[v] += 1

    fre = defaultdict(set)
    for k, v in dic.items():  # 将出现次数相同的数字放在一个列表中 类似链表
        fre[v].add(k)

    res = []
    for i in range(len(nums), 0, -1):  # 计数次数已知
        if i not in fre:
            continue

        for v in fre[i]:
            res.append(v)
            if len(res) == k:
                return res[:k]


# 128 无序数组 最长连续区间 也可以使用并查集
# 类似字典按key 排序
def longest_consecutive(nums):
    if not nums:
        return 0
    dic = dict()
    for v in nums:
        dic[v] = False

    res = 1
    for v in dic.keys():
        if v in dic:
            continue
        cnt = 1
        while v + 1 in dic:  # 序
            cnt += 1
            v += 1
        res = max(res, cnt)

    return res


# 560. 和为K的子数组
# 前缀和 如果两个数的前缀和相同，那么这两个节点之间的元素总和为零
def subarray_sum(nums, k):
    res = 0
    total = 0
    dic = dict()
    dic[0] = 1  # 初始化 可能total = k
    for num in nums:
        total += num  # 以num为结尾的连续数组
        if total - k in dic:  # pre_sum+k=total -> total-k in dic
            res += dic[total - k]
        dic[total] = dic.get(total, 0) + 1  # 后增加1 防止total - k = total
    return res


# 523 给定一个包含 非负数 的数组和一个目标 整数k，
# 编写一个函数来判断该数组是否含有连续的子数组，其大小至少为 2，
# 且总和为 k 的倍数，即总和为 n*k，其中 n 也是一个整数。
def check_subarray_sum(nums, k):
    if len(nums) < 2:
        return False
    dic, total = {0: -1}, 0
    for i, num in enumerate(nums):
        total += num
        if k:
            total %= k
        j = dic.setdefault(total, i)
        if i - j >= 2:
            return True
    return False


# 加油站 有点前缀和的意思
def can_complete_circuit(gas, cost):
    n = len(gas)
    if sum(gas) - sum(cost) < 0:
        return -1
    total = 0
    start = 0
    for i in range(n):
        if total + gas[i] - cost[i] >= 0:
            total += gas[i] - cost[i]
        else:
            total = 0
            start = i + 1
    if start == n:
        return -1
    return start


# 众数 超过一半的数
# 摩尔投票法的核心就是一一抵消
def most_data(nums):
    if not nums:
        return
    n = len(nums)
    value = nums[0]
    cnt = 1
    for i in range(1, n):
        if cnt == 0:
            value = nums[i]
            cnt = 1
        if nums[i] == value:
            cnt += 1
        else:
            cnt -= 1
    return value


# n // 3
def majority_element(nums):
    if not nums:
        return []
    n = len(nums)

    v1 = v2 = nums[0]
    cnt1 = cnt2 = 0
    for num in nums:
        if num == v1:
            cnt1 += 1
            continue
        if num == v2:
            cnt2 += 1
            continue

        if not cnt1:
            cnt1 += 1
            v1 = num
            continue
        if not cnt2:
            cnt2 += 1
            v2 = num
            continue

        cnt1 -= 1
        cnt2 -= 1

    cnt1 = cnt2 = 0
    for num in nums:
        if num == v1:  # 可能v1 = v2
            cnt1 += 1
            continue
        if num == v2:
            cnt2 += 1
    res = []

    if cnt1 > n // 3:
        res.append(v1)

    if cnt2 > n // 3:
        res.append(v2)

    return res


def second_largest(nums):
    size = len(nums)
    if size < 2:
        return
    first = second = -float('inf')
    for i in range(size):
        if nums[i] > first:
            second = first
            first = nums[i]
            continue
        if nums[i] > second:
            second = nums[i]
    return second


# 581. 最短无序连续子数组
# 单调性
def find_unsorted_subarray(nums):
    n = len(nums)
    max_v, min_v = -float('inf'), float('inf')
    left, right = n - 1, 0
    for i in range(n):
        if nums[i] < max_v:  # 比左侧最大值小
            right = i
        else:
            max_v = nums[i]  # 右侧递增
        #
        if min_v < nums[n - 1 - i]:  # 比右侧最小值大
            left = n - 1 - i
        else:
            min_v = nums[n - 1 - i]  # 左侧递减
    if left == n - 1 and right == 0:
        return 0
    return right - left + 1


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
    while r < n and nums[r] > nums[k]:  # 找到第一个小于等于该数的位置
        r += 1
    r -= 1  # 大于该数的最小值
    nums[k], nums[r] = nums[r], nums[k]

    # reverse
    l, r = k + 1, n - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums


# 66. 加一 [1, 2, 3] -> [1, 2, 4]
def plus_one(digits):
    if not digits:  # base
        return [1]
    if digits[-1] < 9:
        digits[-1] += 1
        return digits
    return plus_one(digits[:-1]) + [0]


# 66. 加一 [1, 2, 3] -> [1, 2, 4]
def plus_one2(digits):
    carry = 1
    n = len(digits)
    for i in range(n - 1, -1, -1):
        if digits[i] == 9 and carry == 1:
            digits[i] = 0
        else:
            digits[i] = digits[i] + carry
            carry = 0
    if carry == 1:
        digits = [1] + digits
    return digits


# 14. 最长公共前缀
def longest_common_prefix(strs):
    if not strs:
        return ""

    def lcp(str1, str2):
        length, index = min(len(str1), len(str2)), 0
        while index < length and str1[index] == str2[index]:
            index += 1
        return str1[:index]

    prefix, n = strs[0], len(strs)
    for i in range(1, n):
        prefix = lcp(prefix, strs[i])
        if not prefix:
            break

    return prefix


if __name__ == '__main__':
    print('\n矩阵查找')
    matrix = [
        [1, 3, 5, 7],
        [10, 11, 16, 20],
        [23, 30, 34, 50]
    ]
    print(search_matrix(matrix, 16))
    print('\n众数')
    print(most_data([1, 3, 3, 3, 9]))

    print('数组中第2大的数')
    print(second_largest([2, 3, 4, 10, 100]))

    print('\n计数排序')
    print(sort_colors([2, 0, 2, 1, 1, 0]))

    print('\n出现次数最多的K个数')
    print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))

    print('\n连续区间最大长度')
    print(longest_consecutive([1, 2, 3, 7, 8, 10, 11, 12, 13, 14]))

    print('\n区间合并')
    print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))

    print('\n下一个排列')
    print(next_permutation([1, 1, 3]))

    print('\n找到未排序的部分')
    print(find_unsorted_subarray([2, 6, 4, 8, 10, 9, 15]))

    print('\n连续数组和为K')
    print(subarray_sum([1, 2, 3, -3, 4], 0))

    print('\n加油站问题')
    print(can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]))
