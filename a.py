def max_sum(nums):
    if not nums:
        return 0
    res = -float('inf')
    inc = exc = -float('inf')
    n = len(nums)
    for i in range(n):
        inc, exc = max(nums[i], exc + nums[i]), max(inc, exc)
        res = max(res, inc, exc)

    return res


def permutate(nums, v):
    if not nums:
        return []
    cnt = 0
    n = len(nums)
    for num in nums:
        if num == v:
            cnt += 1
    if cnt % 2:
        return []

    nums.sort()
    res = []
    seen = [False] * n

    def dfs(path, v_cnt):
        if len(path) == n:
            res.append(path[:])
            return

        for i in range(n):
            if seen[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and nums[i] != v and not seen[i - 1]:
                continue
            if nums[i] == v and v_cnt < cnt:
                path.append(nums[i])
                path.append(nums[i])

                dfs(path, v_cnt + 2)
            else:
                seen[i] = True
                path.append(nums[i])
                dfs(path, v_cnt)
                seen[i] = False
                path.pop(-1)

    dfs([], 0)
    return res


if __name__ == '__main__':
    nums = [1, 1, 1, 1, 2]
    print(permutate(nums, 1))
