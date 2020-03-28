def lis(nums):
    def binary_search(nums, tar):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == tar:
                return mid
            elif nums[mid] > tar:
                right -= 1
            else:
                left += 1
        return left

    if not nums:
        return 0
    res = [nums[0]]
    for v in nums[1:]:
        if v > res[-1]:
            res.append(v)
        elif v < res[-1]:
            idx = binary_search(res, v)
            res[idx] = v
    return len(res)


# 有重复数字的非降序排序数组 返回第一个等于target
def search_first_pos(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    return l if nums[l] == target else -1


# 等于target
def search_last_pos(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r + 1) // 2
        if nums[mid] <= target:
            l = mid  # l == mid 需要考虑 3, 4 这种无限循环的情况
        else:
            r = mid - 1  # r+1 严格大于target  l <= target l = r 最后一个大于该数的
    return l if nums[l] == target else -1


def search_bigger_pos(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r + 1) // 2
        if nums[mid] > target:
            l = mid + 1
        else:
            r = mid
    return l - 1


def search_smaller_pos(nums, target):
    n = len(nums)
    l, r = 0, n - 1
    while l < r:
        mid = (l + r + 1) // 2
        if nums[mid] >= target:
            l = mid
        else:
            r = mid - 1
    return r + 1 if r + 1 < n else -1


def quick_sort(nums, lo, hi):
    if lo < hi:  # 长度为1不用排序
        mid = partition(nums, lo, hi)
        quick_sort(nums, lo, mid - 1)
        quick_sort(nums, mid + 1, hi)


def partition(nums, lo, hi):
    pivot = nums[lo]
    while lo < hi:
        while lo < hi and pivot <= nums[hi]:
            hi -= 1
        nums[lo] = nums[hi]  # 替换已保存的数据
        while lo < hi and nums[lo] <= pivot:
            lo += 1
        nums[hi] = nums[lo]  # 替换已保存的数据
    nums[lo] = pivot
    return lo




def nextPermutation(nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
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
        while r < n and nums[r] > nums[k]:
            r += 1
        r -= 1
        nums[k], nums[r] = nums[r], nums[k]

        # reverse
        l, r = k + 1, n - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
        return nums


def first(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    return l if nums[l] == target else -1


def last(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r + 1) // 2
        if nums[mid] <= target:
            l = mid
        else:
            r = mid - 1
    return l if nums[l] == target else -1


if __name__ == '__main__':
    print(lis([10, 9, 2, 5, 3, 7, 101, 18]))
    print('first')
    print(first([1, 2, 3, 3, 4], 3))

    print('last')
    print(last([1, 2, 3, 3, 3, 4], 3))

    print(search_last_pos([1, 2, 3, 3, 4], 3))
    print(search_bigger_pos([1, 2, 3, 3, 4, 4, 4, 4, 4, 4], 3))
    print(search_smaller_pos([1, 2, 3, 3, 4, 4, 4, 4, 4, 4], 3))

    print(nextPermutation([1, 1, 3]))
