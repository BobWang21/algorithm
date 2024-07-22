#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter, defaultdict


##############################前后指针-关注点####################################
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


# 80
# 数组中每个数字最多出现2次 返回删除重复数字后的长度
def remove_duplicates2(nums):
    if len(nums) < 3:
        return len(nums)

    l, r = 2, 2
    while r < len(nums):
        if nums[r] != nums[l - 2]:
            nums[l] = nums[r]
            l += 1
        r += 1
    return l


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


#############################首尾指针-关注点##################################
def partition(nums, left, right):
    def swap(i, j):
        nums[i], nums[j] = nums[j], nums[i]

    l, r = left, right
    pivot = nums[l]
    while l < r:
        # 找到
        while l < r and nums[r] >= pivot:
            r -= 1
        while l < r and nums[l] < pivot:
            l += 1
        if l < r:
            swap(l, r)  # 保证 nums[l] < pivot <= nums[r]
    return nums


# 奇数在左边 偶数在右边
def sort_array_by_parity(nums):
    n = len(nums)
    if n < 2:
        return nums
    l, r = 0, n - 1
    while l < r:
        while l < r and not nums[r] % 2:
            r -= 1
        while l < r and nums[l] % 2:
            l += 1
        if l < r:
            nums[l], nums[r] = nums[r], nums[l]
    return nums


# 922.按奇偶排序数组
def sort_array_by_parity2(nums):
    n = len(nums)

    left, right = 0, 1
    while left < n and right < n:
        while left < n and nums[left] % 2 == 0:
            left += 2
        while right < n and nums[right] % 2 == 1:
            right += 2
        if right < n and left < n:
            nums[left], nums[right] = nums[right], nums[left]
    return nums


#########################首尾指针-端点#########################################
# 未排序数组 解唯一
# has map O(N)空间复杂度
def two_sum1(nums, target):
    dic = dict()

    for i in range(len(nums)):
        if target - nums[i] in dic:
            return [dic[target - nums[i]], i]
        dic[nums[i]] = i


# 排序数组 解唯一 O(N)
def two_sum2(nums, target):
    n = len(nums)
    if n < 2:
        return []
    l, r = 0, n - 1
    while l < r:
        total = nums[l] + nums[r]
        if total < target:
            l += 1
        elif total > target:
            r -= 1
        else:
            res = [nums[l], nums[r]]
            return res


# 排序数组 数组中有重复数字 解不唯一 O(N)
def two_sum3(nums, target):
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
            while l < r and nums[l] == nums[l + 1]:  # 跳出重复值
                l += 1
            l += 1  # l 先跳出，自然不满足条件
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


if __name__ == '__main__':
    print('\n2 sum')
    print(two_sum1([7, 8, 2, 3], 9))
    print(two_sum3([1, 2, 7, 8, 11, 15], 9))

    print('\n3 sum')
    print(three_sum([2, 7, 7, 11, 15, 15, 20, 24, 24], 33))

    print('\n3 sum closet')
    print(three_sum_closet([-1, 2, 1, -4], 1))

    print('\n删除排查数组中的重复数值')
    print(remove_duplicates1([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))
    print(remove_duplicates2([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))
    print(remove_duplicates3([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))

    print('\n移动0至数组尾部')
    print(move_zeros([0, -7, 0, 2, 3, 11]))

    print('\n差值小于等于target的对数')
    nums = [1, 3, 5, 7]
