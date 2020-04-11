#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:02:31 2017

@author: wangbao
"""


#  二分查找 递归
def binary_search(nums, lo, hi, tar):
    if lo <= hi:
        mid = (lo + hi) // 2  # lo + (hi-lo) // 2
        if nums[mid] == tar:
            return mid
        if nums[mid] > tar:
            return binary_search(nums, lo, mid - 1, tar)
        else:
            return binary_search(nums, mid + 1, hi, tar)
    return -1


#  二分查找 非递归
def binary_search2(nums, tar):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if tar == nums[mid]:
            return mid
        if tar < nums[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1


# 有重复数字的非降序排序数组 返回第一个等于target
def search_first_pos(nums, target):
    """
    l-1 严格小于target
    r大于等于target
    l = r 时为第一个等于target的坐标
    """
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    return l if nums[l] == target else -1


# 有重复数字的非降序排序数组 返回最后一个等于target
def search_last_pos(nums, target):
    """
    l 小于等于target
    r+1 大于target
    l = r 时为最后一个等于target的坐标
    """
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r + 1) // 2
        if nums[mid] <= target:
            l = mid  # l == mid 需要考虑 3, 4 这种无限循环的情况
        else:
            r = mid - 1
    return l if nums[l] == target else -1


def search_first_large(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] <= target:
            l = mid + 1
        else:
            r = mid
    return r if nums[r] > target else -1


# 旋转数组中的最小值
# [3 4 1 2] 为 [1 2 3 4]的旋转数组
def find_min(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return -1
    if nums[0] < nums[-1]:
        return nums[0]
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= nums[0]:
            l = mid + 1
        else:
            r = mid
    return nums[l]


# 递归
def find_min2(nums):
    if len(nums) <= 2:
        return min(nums)
    l, r = 0, len(nums) - 1
    if nums[0] < nums[r]:
        return nums[0]
    mid = (l + r) // 2
    return min(find_min2(nums[:mid]), find_min2(nums[mid:]))


def search(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] >= nums[l]:
            if nums[mid] > target >= nums[l]:
                r = mid - 1
            else:
                l = mid + 1
        if nums[mid] <= nums[r]:
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return -1


# 数字在排序数组中出现的次数
def get_number_of_k(nums, tar):
    def binary_search(nums, tar, lo, hi):
        if lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == tar:
                return mid
            elif nums[mid] < tar:
                return binary_search(nums, tar, mid + 1, hi)
            else:
                return binary_search(nums, tar, lo, mid - 1)
        return None

    n = len(nums) - 1
    idx = binary_search(nums, tar, 0, n)
    if idx is None:
        return
    # 左侧端点
    left_idx = binary_search(nums, tar, 0, idx - 1)
    min_left_idx = None
    while left_idx is not None:
        min_left_idx = left_idx
        left_idx = binary_search(nums, tar, 0, left_idx - 1)
    # 右侧端点
    right_idx = binary_search(nums, tar, idx + 1, n)
    max_right_idx = None
    while right_idx is not None:
        max_right_idx = right_idx
        right_idx = binary_search(nums, tar, right_idx + 1, n)

    if min_left_idx is not None and max_right_idx is not None:
        return max_right_idx - min_left_idx + 1

    if min_left_idx is not None:
        return idx - min_left_idx + 1

    if max_right_idx is not None:
        return max_right_idx - idx + 1

    return 1


def get_number_of_k2(nums, tar, lo, hi):
    def binary_search(nums, tar, lo, hi):
        if lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == tar:
                return mid
            elif nums[mid] < tar:
                return binary_search(nums, tar, mid + 1, hi)
            else:
                return binary_search(nums, tar, lo, mid - 1)
        return None

    if nums[lo] == nums[hi] == tar:  # 简化计算
        return hi - lo + 1
    idx = binary_search(nums, tar, lo, hi)
    if idx is None:
        return 0
    return get_number_of_k2(nums, tar, lo, idx - 1) + 1 + get_number_of_k2(nums, tar, idx + 1, hi)


# 0 - n-1 n 个数中 缺少一个数
def find_missed_val(nums):
    n = len(nums)
    if n == 1:
        return 1 - nums[0]
    l, r = 0, n
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == mid:
            l = mid + 1
        else:
            r = mid - 1
    return l


if __name__ == '__main__':
    print('\n二分查找')
    data = [1, 3, 5, 9, 10, 16, 17]
    print(binary_search2(data, 3))

    print('\n数值等于target的最小索引')
    print(search_first_pos([1, 2, 3, 3, 9], 3))

    print('\n数值等于target的最大索引')
    print(search_last_pos([1, 2, 3, 3, 9], 3))

    print('\n第一个大于target的数值索引')
    print(search_first_large([1, 2, 3, 3, 9], 6))

    print('\n旋转数组中的最小值')
    print(find_min([5, 1]))

    print('\n旋转数组查找')
    print(search([4, 5, 6, 7, 0, 1, 2], 3))

    print('\n数字在升序数字中出现的次数')
    nums = [1, 2, 3, 3, 3, 3, 4, 4]
    print(get_number_of_k(nums, 3))
    print(get_number_of_k2(nums, 1, 0, len(nums) - 1))

    print('\n找出0 - n之间缺少的一个数字')
    print(find_missed_val([0, 1, 3]))
