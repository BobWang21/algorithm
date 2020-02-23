import math


# 二分查找
def binary_search(nums, tar, lo, hi):
    if lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == tar:
            return mid
        if nums[mid] < tar:
            return binary_search(nums, tar, mid + 1, hi)
        if nums[mid] > tar:
            return binary_search(nums, tar, lo, mid - 1)
    return hi  # 返回小于target最大数所在索引


# 双指标的法
def partition(nums, lo, hi):
    pivot = nums[lo]
    while lo < hi:
        while lo < hi and nums[hi] >= pivot:
            hi -= 1
        nums[lo] = nums[hi]
        while lo < hi and nums[lo] <= pivot:
            lo += 1
        nums[hi] = nums[lo]
    nums[lo] = pivot
    return lo


# 快排
def quick_sort(nums, lo, hi):
    if lo < hi:
        mid = partition(nums, lo, hi)
        quick_sort(nums, lo, mid - 1)
        quick_sort(nums, mid + 1, hi)


# 快速幂
def power(x, n):
    if n < 0:
        return 1 / power(x, -n)
    if n == 0:  # base
        return 1
    if n == 1:  # base
        return x
    b = power(x, n // 2)
    if n % 2 == 0:
        return b * b
    else:
        return x * b * b


#  二分查找 递归
def binary_search(nums, lo, hi, tar):
    if lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == tar:
            return mid
        if nums[mid] > tar:
            return binary_search(nums, lo, mid - 1, tar)
        else:
            return binary_search(nums, mid + 1, hi, tar)
    return -1


#  二分查找 非递归
def binary_search2(arr, tar):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if tar == arr[mid]:
            return mid
        if tar < arr[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1


def find_max(a, b):
    return max(a, b), min(a, b)


# 第二大的数
def second_largest(nums):
    first = second = -math.inf
    size = len(nums)
    if size < 2:
        raise Exception('array length must be more than 2')
    for i in range(size):
        if nums[i] > first:
            second = first
            first = nums[i]
        elif nums[i] > second:
            second = nums[i]
    return second


# 第K大的数
def k_largest(arr, k):
    '''
    可以用冒泡排序 也可以使用快排的思想
    判断pivot的坐标
    '''
    size = len(arr)
    if size < k:
        raise Exception('>= K')
    mid = partition(arr, 0, size - 1)
    while True:
        if mid == size - k:  # 升序
            return arr[mid]
        if mid > size - k:
            mid = partition(arr, 0, mid - 1)
        else:
            mid = partition(arr, mid + 1, size - 1)


# 归并排序
def merge_sort(nums):
    n = len(nums)
    # 递归基
    if n <= 1:
        return nums
    mid = n // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge(left, right)


# 合并两个有序数组
def merge(a, b):
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return b
    if len_b == 0:
        return a
    i, j = 0, 0
    res = []
    while i < len_a and j < len_b:
        if a[i] <= b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    # 判断循环跳出的状态
    if i < len_a:
        res += a[i:]
    if j < len_b:
        res += b[j:]
    return res


if __name__ == '__main__':
    print('快排')
    arr = [4, 7, 5, 7, 9]
    quick_sort(arr, 0, 4)
    print(arr)
    print('Kth largest num')
    print(second_largest([2, 3, 4, 10, 100]))
    print(k_largest([2, 3, 4, 10, 100], 2))

    print('merge sorted array')
    print(merge([1, 3, 5], [2, 4, 6]))

    print('快速幂')
    print(power(2, 11))

    print('二分查找')
    print(binary_search([1, 3, 5], 0, 2, 2))
    print(binary_search2([1, 3, 5, 9, 10, 16, 17], 3))
