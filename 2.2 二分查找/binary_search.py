#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 非递归版本 标准二分查找
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
    # 跳出循环时l=r+1 nums[l]>target, nums[r]<target
    return -1


# 递归版本 二分查找
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
    return nums[l] if nums[l] == target else -1  # 取不到 l=r 补丁!


# 有重复数字的非降序排序数组 返回最后一个等于target
def search_last_pos(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l + 1) // 2  # 右中位数
        if nums[mid] <= target:
            l = mid  # l == mid 需要考虑l,r=3,4这种无限循环的情况 nums[l] <= target
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


# 34. 在排序数组中查找元素的第一个和最后一个位置
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


# 33. 搜索旋转排序数组
def search(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1

    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[0] < nums[mid]:  # 此处小于等于
            if nums[0] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1  # 若等于nums[0]=nums[mid] 则l + 1
        elif nums[mid] < nums[0]:
            if nums[mid] < target <= nums[-1]:
                l = mid + 1
            else:
                r = mid - 1
        else:
            l += 1
    return -1


# 162. 寻找峰值
def find_peak_element(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] < nums[mid + 1]:  # 只有l=r=len(nums)-1, mid+1 才可能越界
            l = mid + 1  # nums[l-1] < nums[l]
        else:
            r = mid  # nums[r] > nums[r+1]
    # l=r 时  nums[l-1] < nums[l] = nums[r] > nums[r+1]
    return l


# 旋转数组中的最小值
# [3 4 1 2] 为 [1 2 3 4]的旋转数组
def find_min(nums):
    if nums[0] <= nums[-1]:
        return nums[0]

    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] >= nums[0]:  # nums[l-1] >= nums[0]
            l = mid + 1
        else:
            r = mid  # nums[r] < nums[0]
    return nums[l]


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


# 719 Find Kth Smallest Pair Distance
# 双指针 + 二分查找 差值小于等于某个数的pair数 为递增函数!!!
def smallest_distance_pair_3(nums, k):
    nums.sort()
    n = len(nums)

    def nmt(target):
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
        count = nmt(mid)
        if count >= k:
            r = mid
        else:
            l = mid + 1
    return l


# 668. 乘法表中第k小的数 也可以用堆 堆得时间复杂度 o(klog(k))
def find_kth_number1(m, n, k):
    def no_more_than(val):  # 小于等于某个数的个数
        res = 0
        for i in range(1, min(m, val) + 1):
            res += min(val // i, n)
        return res

    l, r = 1, m * n
    while l < r:  # 这里的隐含信息为 找到第一个等于K的值 这个值肯定在乘法表中 first_loc
        mid = l + (r - l) // 2
        if no_more_than(mid) < k:
            l = mid + 1
        else:
            r = mid
    return l  # 一定满足条件 所以不用补丁


# 378. 有序矩阵中第K小的元素
# 给定一n x n矩阵，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
# 示例：matrix = [[ 1,  5,  9],
#               [10, 11, 13],
#               [12, 13, 15]],
# k = 8, 返回 13。
def kth_smallest(matrix, k):
    n = len(matrix)

    def count(v):
        res = 0
        i, j = n - 1, 0
        while i >= 0 and j < n:
            if matrix[i][j] <= v:
                res += i + 1
                j += 1
            else:
                i -= 1
        return res

    l, r = matrix[0][0], matrix[-1][-1]
    while l < r:
        mid = l + (r - l) // 2
        cnt = count(mid)
        if cnt < k:
            l = mid + 1
        else:
            r = mid
    return l


# 440 字典序 第k个数字 2.2 二分查找 十叉树
def find_kth_number2(n, k):
    def prefix_num(prefix):
        cnt = 0
        curr = prefix
        nxt = prefix + 1
        while curr <= n:
            cnt += min(n + 1, nxt) - curr
            curr *= 10
            nxt *= 10
        return cnt - 1  # 此节点的子节点

    prefix = 1
    rank = 1  # 当前的节点的序
    while rank < k:
        cnt = prefix_num(prefix)
        if rank + cnt > k:  # 如何移动节点
            rank += 1
            prefix *= 10
        else:
            rank += cnt
            prefix += 1
    return prefix


# 14. 最长公共前缀
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


# 4. 寻找两个正序数组的中位数
def find_median_sorted_arrays(nums1, nums2):
    m, n = len(nums1), len(nums2)
    if m > n:
        return find_median_sorted_arrays(nums2, nums1)

    k = (m + n + 1) // 2  # 左中位数的个数
    l, r = 0, m - 1  # 取每个数字
    while l < r:
        mid = l + (r - l) // 2
        if nums1[mid] <= nums2[k - mid - 1]:  # nums1[l-1] <= nums2[k-l] 此处必须是mid 不能是mid-1
            l = mid + 1
        else:
            r = mid  # nums1[l] > nums2[k-l-1]
    l = l + 1 if nums1 and nums1[l] <= nums2[k - l - 1] else l  # 需要补丁

    x1, x2 = l - 1, k - l - 1  # nums1[l-1] <= nums2[k-l]
    v1 = max(nums1[x1] if 0 <= x1 < m else -float('inf'),
             nums2[x2] if 0 <= x2 < n else -float('inf')
             )

    if (m + n) % 2:
        return v1

    v2 = min(
        nums1[x1 + 1] if 0 <= x1 + 1 < m else float('inf'),
        nums2[x2 + 1] if 0 <= x2 + 1 < n else float('inf')
    )

    return (v1 + v2) / 2


if __name__ == '__main__':
    print('\n2.2 二分查找')
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

    print('\n找出0-n之间缺少的一个数字')
    print(find_missed_value([0, 1, 3]))

    print('\n乘法表中第k小的数')
    print(find_kth_number1(3, 2, 6))

    print('\n字典序')
    print(find_kth_number2(13, 2))

    print('\n中位数')
    print(find_median_sorted_arrays([1, 2], [3, 4]))
