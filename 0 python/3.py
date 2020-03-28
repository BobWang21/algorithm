def max_product(nums):
    if not nums:
        return 0
    nums.append(-1)  # 为了使剩余的元素出栈
    n = len(nums)
    stack = []
    total = [0] * n
    max_v = 0

    for i, v in enumerate(nums):
        v = v if v >= 0 else 0
        total[i] = total[i - 1] + v

        while stack and nums[stack[-1]] > v:
            j = stack.pop(-1)
            pre_total = 0
            if stack:
                pre_total = total[stack[-1]]
            max_v = max(max_v, (total[i - 1] - pre_total) * nums[j])
        stack.append(i)
    print(total)
    return max_v


def next_greater(nums):
    if not nums:
        return
    stack = []
    res = [-1] * len(nums)
    for i, v in enumerate(nums):
        while stack and nums[stack[-1]] < v:
            j = stack.pop(-1)
            res[j] = i - j
        stack.append(i)
    return res


if __name__ == '__main__':
    print(max_product([81, 87, 47, 59, 81, 18, 25, 40, 56, 0]))
    print(next_greater([5, 3, 1, 2, 4]))
