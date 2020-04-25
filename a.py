def find_unsorted_subarray(nums):
    if not nums or len(nums) == 1:
        return 0
    nums1 = nums[:]
    nums1.sort()
    n = len(nums)
    l1 = 0
    for i in range(n):
        if nums[i] == nums1[i]:
            l1 += 1
            continue
        else:
            break
    if l1 == n:
        return 0

    l2 = 0
    for i in range(n - 1, -1, -1):
        if nums[i] == nums1[i]:
            l2 += 1
            continue
        else:
            break
    return n - (l1 + l2)


def find_unsorted_subarray2(nums):
    if not nums or len(nums) == 1:
        return 0
    stack = []
    n = len(nums)
    for i in range(n):
        while stack and nums[stack[-1]] > nums[i]:
            stack.pop(-1)
        stack.append(i)
    l1 = 0
    for i, v in enumerate(stack):
        if i == v:
            l1 += 1
        else:
            break
    if l1 == n:
        return 0

    stack = []
    for i in range(n - 1, -1, -1):
        while stack and nums[stack[-1]] < nums[i]:
            stack.pop(-1)
        stack.append(i)

    l2 = 0
    for i, v in enumerate(stack):
        if nums[v] == nums[n - i - 1]:
            l2 += 1
        else:
            break
    print(l1, l2)
    return n - (l1 + l2)


if __name__ == '__main__':
    print(find_unsorted_subarray2([2, 6, 4, 8, 10, 9, 15]))
