def sort(nums):
    if not nums:
        return
    n = len(nums)
    odd_num = 0
    for i in range(n):
        if nums[i] % 2:
            odd_num += 1
    if not odd_num or odd_num == n:
        nums.sort()
        return nums
    
    l, r = 0, n - 1
    pivot = nums[0]
    while l < r:
        while l < r and not nums[r] % 2:
            r -= 1
        nums[l] = nums[r]

        while l < r and nums[l] % 2:
            l += 1
        nums[r] = nums[l]
    nums[l] = pivot

    a = nums[:odd_num]
    a.sort()
    b = nums[odd_num:]
    b.sort()
    return a + b


if __name__ == '__main__':
    nums = [4, 5, 1, 3, 7, 6]
    print(sort(nums))
