#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#  二分查找 递归
def binary_search(nums, lo, hi, target):
    if lo <= hi:
        mid = (lo + hi) // 2  # lo + (hi-lo) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] > target:
            return binary_search(nums, lo, mid - 1, target)
        else:
            return binary_search(nums, mid + 1, hi, target)
    return -1


#  二分查找 非递归
def binary_search2(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if target == nums[mid]:
            return mid
        if target < nums[mid]:
            r = mid - 1
        else:
            l = mid + 1
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
    if not nums:
        return -1
    if nums[0] < nums[-1]:
        return nums[0]
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] >= nums[0]:  # nums[l-1]递增 nums[r] < nums[0] l = r nums[l]即为最小值
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


# 旋转数组查找
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


# 递归
def search2(nums, target):
    def helper(l, r):
        if l > r:
            return -1
        if l == r:
            return l if nums[l] == target else -1
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] > nums[0]:  # 前半部分递增
            if nums[0] > target:
                return helper(mid + 1, r)
            if nums[mid] < target:
                return helper(mid + 1, r)
            return helper(l, mid - 1)
        else:
            if nums[-1] < target:
                return helper(l, mid - 1)
            if nums[mid] > target:
                return helper(l, mid - 1)
            return helper(mid + 1, r)

    return helper(0, len(nums) - 1)


# 数字在排序数组中出现的次数
def get_number_of_k(nums, target):
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
    idx = binary_search(nums, target, 0, n)
    if idx is None:
        return
    # 左侧端点
    left_idx = binary_search(nums, target, 0, idx - 1)
    min_left_idx = None
    while left_idx is not None:
        min_left_idx = left_idx
        left_idx = binary_search(nums, target, 0, left_idx - 1)
    # 右侧端点
    right_idx = binary_search(nums, target, idx + 1, n)
    max_right_idx = None
    while right_idx is not None:
        max_right_idx = right_idx
        right_idx = binary_search(nums, target, right_idx + 1, n)

    if min_left_idx is not None and max_right_idx is not None:
        return max_right_idx - min_left_idx + 1

    if min_left_idx is not None:
        return idx - min_left_idx + 1

    if max_right_idx is not None:
        return max_right_idx - idx + 1

    return 1


def get_number_of_k2(nums, target, lo, hi):
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

    if nums[lo] == nums[hi] == target:  # 简化计算
        return hi - lo + 1
    idx = binary_search(nums, target, lo, hi)
    if idx is None:
        return 0
    return get_number_of_k2(nums, target, lo, idx - 1) + 1 + get_number_of_k2(nums, target, idx + 1, hi)


# 0 - n-1 n 个数中 缺少一个数
def find_missed_value(nums):
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


# 对于无序 可以考虑partition
def find_median_sorted_arrays(nums1, nums2):
    m, n = len(nums1), len(nums2)
    if m > n:
        return find_median_sorted_arrays(nums2, nums1)  # 选择数组小的进行二分
    k = (m + n + 1) // 2  # 左中点
    l, r = 0, m
    while l < r:
        m1 = (l + r) // 2
        m2 = k - m1
        if nums1[m1] < nums2[m2 - 1]:  # nums1[l/r] >= nums2[m2-1] !!!
            l = m1 + 1
        else:
            r = m1
    m1 = l
    m2 = k - l
    c1 = max(nums1[m1 - 1] if m1 > 0 else -float('inf'), nums2[m2 - 1] if m2 > 0 else -float('inf'))

    if (m + n) % 2:
        return c1

    c2 = min(nums1[m1] if m1 < m else float('inf'), nums2[m2] if m2 < n else float('inf'))

    return (c1 + c2) / 2.0


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
    print(find_min([4, 5, 1, 3]))
    print(find_min2([4, 5, 1, 3]))

    print('\n旋转数组查找')
    print(search([4, 5, 6, 7, 0, 1, 2], 0))

    print(search2([4, 5, 6, 7, 0, 1, 2], 0))

    print('\n数字在升序数字中出现的次数')
    nums = [1, 2, 3, 3, 3, 3, 4, 4]
    print(get_number_of_k(nums, 3))
    print(get_number_of_k2(nums, 1, 0, len(nums) - 1))

    print('\n找出0 - n之间缺少的一个数字')
    print(find_missed_value([0, 1, 3]))

    print('\n中位数')
    print(find_median_sorted_arrays([1, 2], [3, 4]))
