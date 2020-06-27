#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 非递归
def binary_search1(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if target == nums[mid]:
            return mid
        if target < nums[mid]:
            r = mid - 1  # nums[r+1]>target
        else:
            l = mid + 1  # nums[l-1]<target
    # 跳出循环时l-r=1 nums[l]>target, nums[r]<target
    return -1


# 递归
def binary_search2(nums, l, r, target):
    if l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] > target:
            return binary_search2(nums, l, mid - 1, target)
        else:
            return binary_search2(nums, mid + 1, r, target)
    return -1


# 有重复数字的非降序排序数组 返回第一个等于target
def search_first_pos(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2  # 左中位数
        if nums[mid] < target:
            l = mid + 1  # nums[l-1] < target
        else:
            r = mid  # nums[r] >= target
    # return l
    return l if nums[l] == target else -1  # 配合l < r 使用 因为l < r 时 取不到 n - 1


# 有重复数字的非降序排序数组 返回最后一个等于target
def search_last_pos(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l + 1) // 2  # 右中位数
        if nums[mid] <= target:
            l = mid  # l == mid 需要考虑3, 4这种无限循环的情况 nums[l] <= target
        else:
            r = mid - 1  # nums[r+1] > target
    return l if nums[l] == target else -1


def search_first_large(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] <= target:
            l = mid + 1
        else:
            r = mid
    return l if nums[l] > target else -1


# 数字在排序数组中出现的次数
def get_number_of_k(nums, target):
    if not nums:
        return 0
    n = len(nums)

    # 第一个出现的位置
    l, r = 0, n - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    if nums[l] != target:
        return 0
    left = l

    # 最后一个出现的位置
    l, r = 0, n - 1
    while l < r:
        mid = l + (r - l + 1) // 2
        if nums[mid] <= target:
            l = mid
        else:
            r = mid - 1
    right = l

    return right - left + 1


# 旋转数组中的最小值
# [3 4 1 2] 为 [1 2 3 4]的旋转数组
def find_min(nums):
    if len(nums) <= 2:
        return min(nums)
    l, r = 0, len(nums) - 1
    if nums[0] < nums[r]:  # 递增
        return nums[0]
    mid = l + (r - l) // 2
    return min(find_min(nums[:mid]), find_min(nums[mid:]))


def find_min2(nums):
    if nums[0] <= nums[-1]:
        return nums[0]

    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] >= nums[0]:  # 左半部分递增
            l = mid + 1
        else:
            r = mid
    return nums[l]


# 旋转数组查找
def search(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] > nums[0]:  # 前部分递增
            if nums[0] > target or nums[mid] < target:  # 不在递增部分
                l = mid + 1
            else:
                r = mid - 1
        elif nums[mid] < nums[0]:  # 后部分递增
            if nums[-1] < target or nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        else:  # nums[0] == nums[mid] 并不能判断是否单调
            l += 1
    return -1


# 递归
def search2(nums, target):
    def helper(l, r):
        if l > r:
            return -1
        if l == r:
            return l if nums[l] == target else -1
        mid = l + (r - l) // 2
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


# 0 - n-1 n 个数中 缺少一个数
def find_missed_value(nums):
    n = len(nums)
    if n == 1:
        return 1 - nums[0]
    l, r = 0, n
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == mid:
            l = mid + 1
        else:
            r = mid - 1
    return l


def find_median_sorted_arrays(nums1, nums2):
    m, n = len(nums1), len(nums2)
    if m > n:
        nums1, nums2 = nums2, nums1
    k = (m + n + 1) // 2  # 个数

    l, r = 0, m  # 我们需要的时l - 1
    while l < r:
        mid = l + (r - l + 1) // 2
        m1 = k - mid
        if nums1[mid - 1] <= nums2[m1]:  # l - 1 < x,  l > x
            l = mid
        else:
            r = mid - 1
    # l为边界
    x1, x2 = l - 1, k - l - 1
    v1 = max(nums1[x1] if 0 <= x1 < m else -float('inf'),
             nums2[x2] if 0 <= x2 < n else -float('inf')
             )
    print(v1)

    if (m + n) % 2:
        return v1

    v2 = min(
        nums1[x1 + 1] if 0 <= x1 + 1 < m else float('inf'),
        nums2[x2 + 1] if 0 <= x2 + 1 < n else float('inf')
    )

    return (v1 + v2) / 2.0


# 比左、右两边数都大的数
def find_peak_element(nums):
    if len(nums) == 1:
        return 0
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] < nums[mid + 1]:  # 只有l=r=len(nums)-1, mid+1 才可能越界
            l = mid + 1  # nums[l-1] < nums[l]
        else:
            r = mid  # nums[r] > nums[r+1]
    # l=r 时  nums[l-1] < nums[l] = nums[r] > nums[r+1]
    return l


# 441 排列硬币 潜在递增函数
def arrange_coins(n):
    def total_coins(n):
        return (1 + n) * n // 2

    l, r = 1, n // 2 + 1
    while l <= r:
        mid = l + (r - l) // 2
        total = total_coins(mid)
        if total == n:
            return mid  # 相等返回当前
        if total < n:
            l = mid + 1
        else:
            r = mid - 1
    return r  # 小于target


# 乘法口诀表 第 k 大的数
# 也可以用堆 堆得时间复杂度 o(klog(k))
def find_kth_number(m, n, k):
    def no_more_than(val):  # 小于等于某个数的个数
        res = 0
        for i in range(1, min(m, val) + 1):
            res += min(val // i, n)
        return res

    l, r = 1, m * n
    while l < r:
        mid = l + (r - l) // 2
        if no_more_than(mid) < k:  # nmt(l-1) < k nmt(r) >= k
            l = mid + 1
        else:
            r = mid
    return l


# Find Kth Smallest Pair Distance
# 双指针 + 2 二分查找 差值小于等于某个数的pair数 为递增函数!!!
def smallest_distance_pair_3(nums, k):
    nums.sort()
    n = len(nums)

    def no_more_than(target):
        j = 1
        res = 0
        for i in range(n - 1):
            while j < n and nums[j] - nums[i] <= target:
                j += 1
            res += j - i - 1
        return res

    l, r = 0, nums[-1] - nums[0]
    while l < r:
        mid = (l + r) // 2
        count = no_more_than(mid)
        print(mid, count)
        if count >= k:
            r = mid
        else:
            l = mid + 1
    return l


# 字典序 第k个数字 2 二分查找
def find_kth_number(n, k):
    def prefix_num(prefix, n):
        cnt = 0
        a = prefix
        b = prefix + 1
        while a <= n:
            cnt += min(n + 1, b) - a
            a *= 10
            b *= 10
        return cnt

    k -= 1
    prefix = 1
    while k:
        c = prefix_num(prefix, n)
        if c > k:
            k -= 1
            prefix *= 10
        else:
            k -= c
            prefix += 1
    return prefix


if __name__ == '__main__':
    print('\n2 二分查找')
    nums = [1, 3, 5, 9, 10, 16, 17]
    print(binary_search1(nums, 3))

    print('\n最小索引')
    print(search_first_pos([1, 2, 3, 3, 10], 9))

    print('\n最大索引')
    print(search_last_pos([1, 2, 3, 3, 9], 3))

    print('\n第一个大于target的数值索引')
    print(search_first_large([1, 2, 3, 3, 9], 6))

    print('\n数字在升序数字中出现的次数')
    nums = [1, 2, 3, 3, 3, 3, 4, 4]
    print(get_number_of_k(nums, -1))

    print('\n旋转数组中的最小值')
    print(find_min([5, 4, 3]))

    print('\n旋转数组查找')
    print(search([4, 5, 6, 7, 0, 1, 2], 0))

    print(search2([4, 5, 6, 7, 0, 1, 2], 0))

    print('\n找出0-n之间缺少的一个数字')
    print(find_missed_value([0, 1, 3]))

    print('\n中位数')
    print(find_median_sorted_arrays([1, 2], [3, 4]))
