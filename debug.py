def remove_duplicates(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    if n <= 2:
        return n
    l, r = 0, 1
    flag = 1
    while r < n:
        if nums[l] == nums[r]:
            if flag:
                flag = 0
                l += 1
                nums[l] = nums[r]
                r += 1
            else:
                r += 1
        else:
            flag = 1
            l += 1
            nums[l] = nums[r]
            r += 1
    return l + 1


if __name__ == '__main__':
    s = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    l = remove_duplicates(s)
    print(s[:l])
