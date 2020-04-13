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
    while r < n and nums[r] > nums[k]:  # 可以二分
        r += 1
    r -= 1
    nums[k], nums[r] = nums[r], nums[k]

    # reverse
    l, r = k + 1, n - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums


matrix = [
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 50]
]


def search_matrix(matrix, tar):
    if not matrix or not matrix[0]:
        return False
    row = 0
    col = len(matrix[0]) - 1
    while row < len(matrix) and col >= 0:
        num = matrix[row][col]
        if num == tar:
            return True
        elif num < tar:
            row += 1
        else:
            col += 1
    return False


# 选择排序
def select_sort(nums):
    if not nums:
        return
    n = len(nums)
    for i in range(n):
        idx = i
        for j in range(i + 1, n):
            if nums[j] < nums[idx]:
                idx = j
        nums[idx], nums[i] = nums[i], nums[idx]
    return nums


# 最长连续数字 最小值的连续个数
def longest_consecutive(nums):
    def helper(s):
        i = 0
        v = min(s)
        while True:
            if i + v in s:
                s.remove(i + v)
                i += 1
            else:
                return i

    if not nums:
        return 0
    s = set(nums)
    res = 0
    while s:
        res = max(helper(s), res)
        if res >= (len(nums)) // 2:
            return res
    return res


if __name__ == '__main__':
    print('下一个排列')
    print(next_permutation([1, 1, 3]))

    print('选择排序')
    print(select_sort([3, 1, 4, 4, 10, -1]))
