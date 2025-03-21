#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 适用于排序或部分排序的数组
# 递推思想，可使用循环不变性证明其正确性!

# 标准-非递归版本
def binary_search1(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if target == nums[mid]:
            return mid
        if nums[mid] < target:
            l = mid + 1  # nums[l-1] < target
        else:
            r = mid - 1  # nums[r+1] > target
    # 跳出循环时, l=r+1 nums[l-1] < target < nums[l]
    return -1


# 69 非负整数求平方根
# 查找第一个小于等于 target 的下标
def sqrt1(x):
    l, r = 0, x
    while l <= r:
        mid = (l + r) // 2
        if mid * mid <= x:
            l = mid + 1  # (l-1) ** 2 <= x
        else:
            r = mid - 1  # (r+1) ** 2 <= x
    # 跳出循环时, r=l-1, nums[r] <= target < nums[l]
    return r


# 带精度的求平方根
def sqrt2(x, precision):
    l, r = (0, x) if x > 1 else (x, 1)
    while l <= r:
        mid = l + (r - l) / 2  # 小数
        s = mid * mid
        if abs(s - x) <= precision:
            return mid
        if s < x:
            l = mid  # 浮点数 不会出现相等的情况
        else:
            r = mid
    # 跳出循环时, l > r, nums[l] < target | nums[r] > target


# 35 插入位置 nums = [1,3,5,6], target = 2
# 查找第一个大于等于 target 的下标
def search_insert(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] < target:
            l = mid + 1  # nums[l-1] < target
        else:
            r = mid - 1  # nums[r+1] >= target
        # nums[l-1] < target <= nums[l]
    return l  # 跳出时可取到len(nums)


# 标准-递归版本
def binary_search2(nums, target, l, r):
    if l > r:
        return -1
    # l <= r
    mid = l + (r - l) // 2
    if nums[mid] == target:
        return mid

    if nums[mid] < target:
        return binary_search2(nums, target, mid + 1, r)
    return binary_search2(nums, target, l, mid - 1)


# 非标准版二分查找。有重复数字, 返回第一个等于target
def search_first_pos(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:  # 取不到l==r, 需要补丁!
        mid = l + (r - l) // 2  # 左中位数
        if nums[mid] < target:
            l = mid + 1  # nums[l-1] < target
        else:
            r = mid  # nums[r] >= target
    # 在l < r时， nums[l-1] < target <= nums[r]
    return l if nums[l] == target else -1


# 有重复数字, 返回最后一个等于target
def search_last_pos(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l + 1) // 2  # 右中位数
        if nums[mid] <= target:
            l = mid  # l==mid & mid取左节点时, l,r=0,1会无限循环
        else:
            r = mid - 1  # nums[r+1] > target
    # 在l < r时， nums[l] <= target < nums[l+1]
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


# 0 ~n-1的n个有序数组中，缺少一个数
def find_missed_value(nums):
    n = len(nums) - 1
    if n == 1:
        return 1 - nums[0]
    l, r = 0, n
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == mid:
            l = mid + 1  # nums[l-1]==mid
        else:
            r = mid - 1  # nums[r+1]>mid
    return l


# 34. 在排序数组中查找元素出现位置
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
    left = l if nums[l] == target else -1

    if left == -1:
        return [-1, -1]

    # 最后一个出现的位置
    l, r = 0, n - 1
    while l < r:
        mid = l + (r - l + 1) // 2
        if nums[mid] <= target:
            l = mid
        else:
            r = mid - 1
    right = l if nums[l] == target else -1
    return [left, right]


# 153 旋转数组中的最小值
# [1 2 3 4]的旋转数组[3 4 1 2], [1 2 3 4]
def find_min1(nums):
    if nums[0] <= nums[-1]:
        return nums[0]

    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] >= nums[0]:  # nums[l-1] >= nums[0]
            l = mid + 1
        else:
            r = mid  # nums[l] < nums[0]
    return nums[l]


# 153 和队尾比
# num[l-1]>=nums[-1] && nums[l]<nums[-1]
# [1 2 3 4]的旋转数组[3 4 1 2], [1 2 3 4]
def find_min2(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] >= nums[-1]:  # nums[l-1] >= nums[-1]; l == 0时, nums[-1] == nums[-1]
            l = mid + 1
        else:
            r = mid  # nums[l] < nums[-1]
    return nums[l]


# 33.旋转排序数组查找target
# [1 2 3 4]的旋转数组[3 4 1 2]
# [nums[k], nums[k+1]...nums[n-1], nums[0], nums[1], ..., nums[k-1]]
def search(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1

    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        # 二分思想核心为减枝
        if nums[mid] >= nums[0]:  # mid左侧有序
            if nums[0] <= target < nums[mid]:  # 位于有序序列中
                r = mid - 1
            else:
                l = mid + 1  # 若等于nums[0]=nums[mid] 则l + 1
        else:  # mid右侧有序
            if nums[mid] < target <= nums[-1]:  # 位于有序序列中
                l = mid + 1
            else:
                r = mid - 1
    return -1


# 162.寻找峰值 [1,2,1,3,5,6,4]
# 对于所有有效的 i 都有 nums[i] != nums[i + 1]
def find_peak_element2(nums):
    n = len(nums)

    def get(i):
        return nums[i] if 0 <= i <= n else -float('inf')

    l, r = 0, n - 1
    while l < r:
        mid = l + (r - l) // 2
        if get(mid) < get(mid + 1):  # get(l-1) < get(l)
            l = mid + 1
        else:  # get(r) > get(r+1)
            r = mid
    return l


# 4. 寻找两个正序数组的中位数
# [1| 3 5 7]
# [2 4 6 10| 11]
def find_median_sorted_arrays(nums1, nums2):
    m, n = len(nums1), len(nums2)
    if m > n:
        return find_median_sorted_arrays(nums2, nums1)

    k = (m + n + 1) // 2  # 取个数的左中位数

    # 二分查找取第一个数组中数字的个数
    l, r = 0, m
    while l < r:
        mid = l + (r - l + 1) // 2  # mid可取到m, 取不到0
        if nums1[mid - 1] <= nums2[k - mid]:  # nums1[l-1] <= nums2[k-l]
            l = mid
        else:
            r = mid - 1  # nums1[l] > nums2[k-l-1]

    x1, x2 = l - 1, k - l - 1  # nums1[l-1] <= nums2[k-l] and nums1[l] > nums2[k-l-1]
    # 分割线左侧的最大值
    v1 = max(nums1[x1] if 0 <= x1 < m else -float('inf'),
             nums2[x2] if 0 <= x2 < n else -float('inf'))

    if (m + n) % 2:
        return v1

    # 分割线右侧的最小值
    v2 = min(nums1[x1 + 1] if 0 <= x1 + 1 < m else float('inf'),
             nums2[x2 + 1] if 0 <= x2 + 1 < n else float('inf'))

    return (v1 + v2) / 2


# 有序数组合并后的第K个数
# [1 3 5| 7]
# [2| 4 6 8]
def find_kth_num(nums1, nums2, k):
    if len(nums1) > len(nums2):
        return find_kth_num(nums2, nums1, k)

    l, r = 0, min(k, len(nums1))
    while l < r:
        mid = l + (r - l + 1) // 2
        j = k - mid
        value = nums2[j] if j < len(nums2) else float('inf')  # 如果数组2越界 假设越界数无穷大
        if nums1[mid - 1] <= value:  # nums1[l-1] <= nums2[k-l]
            l = mid
        else:
            r = mid - 1
    # 取了l个数
    value = max(
        nums1[l - 1] if 0 <= l - 1 < len(nums1) else -float('inf'),
        nums2[k - l - 1] if 0 <= k - l - 1 < len(nums2) else -float('inf')
    )
    return value


# 441 排列硬币
# 潜在递增函数 数组中小于等于某个值的最大值
# 可给出解析解
def arrange_coins(n):
    def total_coins(n):
        return (1 + n) * n / 2

    l, r = 1, n // 2 + 1
    while l < r:
        mid = l + (r - l + 1) // 2
        if total_coins(mid) <= n:
            l = mid  # nums[l] <= n
        else:
            r = mid - 1  # nums[l+1] > n
    return r


# 719 Find Kth Smallest Pair Distance
# 双指针 + 二分查找 差值小于等于某个数的pair数 为递增函数!!!
# 找到第一个cnt(x)等于K的值, x可能有多个取值
def smallest_distance_pair_3(nums, k):
    def no_more_than(gap):
        j = 1
        cnt = 0
        n = len(nums)
        for i in range(n - 1):
            while j < n and nums[j] - nums[i] <= gap:
                j += 1
            cnt += j - i - 1
        return cnt

    nums.sort()

    l, r = 0, nums[-1] - nums[0]  # 边界已知 类似topK频次
    while l < r:
        mid = (l + r) // 2
        if no_more_than(mid) < k:
            l = mid + 1
        else:
            r = mid
    return l


# 668. 乘法表中第k小的数 也可以用堆 堆得时间复杂度 o(klog(k))
# 区别于二维数组查找 数组展开没有升序
def find_kth_number1(m, n, k):
    def no_more_than(val):  # 小于等于某个数的个数
        res = 0
        for i in range(1, min(m, val) + 1):
            res += min(val // i, n)
        return res

    l, r = 1, m * n
    while l < r:  # 这里的隐含信息为 找到第一个等于K的值 这个值肯定在乘法表中
        mid = l + (r - l) // 2
        if no_more_than(mid) < k:
            l = mid + 1
        else:
            r = mid
    return l  # 一定满足条件 所以不用补丁


# 矩阵具有如下特性：
# 每行中的整数从左到右按升序排列
# 每行的第一个整数大于前一行的最后一个整数
def search_matrix(matrix, target):
    if not matrix or not len(matrix[0]):
        return False

    rows, cols = len(matrix), len(matrix[0])
    l, r = 0, rows * cols - 1
    while l <= r:
        mid = l + (r - l) // 2
        val = matrix[mid // cols][mid % cols]
        if val == target:
            return True
        if val < target:
            l = mid + 1
            continue
        if val > target:
            r = mid - 1
    return -1


# 378. 有序矩阵中第K小的元素
# 给定一n x n矩阵，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
# 示例：matrix = [[ 1,  5,  9],
#               [10, 11, 13],
#               [12, 13, 15]],
# k = 8, 返回 13。
def kth_smallest(matrix, k):
    n = len(matrix)

    def count(mid):
        num = 0
        i, j = n - 1, 0  # 初始位置即左下角
        while i >= 0 and j < n:
            if matrix[i][j] <= mid:
                num += i + 1  # 当前列不大于mid的值
                j += 1  # 并向右移动，否则向上移动；
            else:
                i -= 1
        return num

    l, r = matrix[0][0], matrix[-1][-1]
    while l < r:
        mid = l + (r - l) // 2
        cnt = count(mid)
        if cnt < k:
            l = mid + 1
        else:
            r = mid
    return l


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


# 440 字典序 第k个数字 o(k) 先序遍历
def find_kth_number2(n, k):
    res = [k + 1]  # 多加一个0元素

    def dfs(i):
        if i > n:
            return
        if not res[0]:  # 已经找到了
            return
        res[0] -= 1
        if not res[0]:
            res.append(i)
            return

        if i == 0:  # 第一层 和 其他层不一样
            for j in range(1, 10):  # 1, 2, ..9
                dfs(j)
        else:
            for j in range(10):
                dfs(i * 10 + j)

    dfs(0)
    return res[1]


# 十叉树的先序遍历
def find_kth_number3(n, k):
    def prefix_num(val):
        cnt = 0
        cur, nxt = val, val + 1
        while cur <= n:
            cnt += min(n + 1, nxt) - cur
            cur *= 10
            nxt *= 10
        return cnt  # 包含该节点的子节点数

    prefix = 1
    i = 1  # 当前的节点的序
    while i < k:
        num = prefix_num(prefix)
        if i + num > k:
            i += 1
            prefix *= 10
        else:
            i += num
            prefix += 1
    return prefix


if __name__ == '__main__':
    print('\n二分查找')
    nums = [1, 3, 5, 9, 10, 16, 17]
    print(binary_search1(nums, 3))

    print('\nsqrt1')
    print(sqrt1(15))

    print('\nsqrt2')
    print(sqrt2(0.04, 0.0001))

    print('\n最小索引')
    print(search_first_pos([1, 2, 3, 3, 10], 3))

    print('\n最大索引')
    print(search_last_pos([1, 2, 3, 3, 9], 3))

    print('\n第一个大于target的数值索引')
    print(search_first_large([1, 2, 3, 3, 9], 6))

    print('\n矩阵查找')
    matrix = [
        [1, 3, 5, 7],
        [10, 11, 16, 20],
        [23, 30, 34, 50]
    ]
    print(search_matrix(matrix, 16))

    print('\n数字在升序数字中出现的次数')
    nums = [1, 2, 3, 3, 3, 3, 4, 4]
    print(get_number_of_k(nums, -1))

    print('\n旋转数组中的最小值')
    print(find_min1([5, 4, 3]))

    print('\n旋转数组查找')
    print(search([3, 1], 1))

    print('\n找出0-n之间缺少的一个数字')
    print(find_missed_value([0, 1, 3]))

    print('\n乘法表中第k小的数')
    print(find_kth_number1(3, 2, 6))

    print('\n字典序')
    print(find_kth_number2(13, 2))

    print('\n中位数')
    print(find_median_sorted_arrays([1, 2], [3, 4, 5]))
    print(find_kth_num([1, 2], [3, 4, 5], 3))

    print('\n最小距离')
    print(smallest_distance_pair_3([1, 6, 1], 3))
