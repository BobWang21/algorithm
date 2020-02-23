# Given a set of candidate numbers (candidates) (without duplicates)
# and a target number (target), find all unique combinations in candidates
# where the candidate numbers sums to target.
# The same repeated number may be chosen from candidates unlimited number of times.


def combination_sum(nums, tar):
    nums.sort()
    res = []
    if tar == 0:
        return res

    def dfs(nums, tar, index, path, res):
        if tar == 0:
            res.append(path)
        if tar < 0:
            return
        for i in range(index, len(nums)):
            dfs(nums, tar - nums[i], i, path + [nums[i]], res)

    dfs(nums, tar, 0, [], res)
    return res


# 输入中有重复数字, 每个数字只能用一次
# 不建议使用
def combination_sum_(nums, tar):
    nums.sort()
    res = []

    def dfs(nums, tar, index, path, res):
        if len(path) > 0 and tar == 0:
            if path not in res:
                res.append(path)
            return
        for i in range(index, len(nums)):
            dfs(nums, tar - nums[i], i + 1, path + [nums[i]], res)

    dfs(nums, tar, 0, [], res)
    return res


# 输入中有重复数字, 每个数字只能用一次
def combination_sum2(nums, tar):
    nums.sort()
    res = []

    def dfs(nums, tar, index, path, res):
        if len(path) > 0 and tar == 0:
            res.append(path)
            return
        for i in range(index, len(nums)):
            if i > index and nums[i] == nums[i - 1]:
                continue
            dfs(nums, tar - nums[i], i + 1, path + [nums[i]], res)

    dfs(nums, tar, 0, [], res)
    return res


# 全排列 输入数组中不含重复数字
def permutations(nums):
    res = []

    def dfs(nums, path, res):
        if len(nums) == len(path):
            res.append(path)
        candidate = nums - set(path)  # 从未访问过的集合中选取元素
        for val in candidate:
            dfs(nums, path + [val], res)

    dfs(set(nums), [], res)
    return res


# 全排列
# 输入数组中含重复数字
def permutations2(nums):
    res = []

    def dfs(nums, path, res):
        if len(nums) == 0:
            res.append(path)
        for val in set(nums):
            # 候选集大小-1
            index = nums.index(val)
            dfs(nums[:index] + nums[index + 1:], path + [val], res)

    dfs(nums, [], res)
    return res


# 子集问题
def subset(nums):
    res = []

    def dfs(nums, index, path, res, k):
        if k >= 0:
            res.append(path)  # 把路径全部记录下来
        for i in range(index, len(nums)):
            dfs(nums, i + 1, path + [nums[i]], res, k - 1)

    dfs(nums, 0, [], res, len(nums))
    return res


# 背包问题
def knapsack_problem(cost, val, cap):
    res = []

    def dfs(cost, val, cap, index, amount, res):
        if cap == 0:
            res.append(amount)
            return
        for i in range(index, len(cost)):
            if cap - cost[i] < 0:
                continue
            dfs(cost, val, cap - cost[i], i + 1, amount + val[i], res)

    dfs(cost, val, cap, 0, 0, res)
    return max(res)


# W=<90, 80, 40, 30, 20, 12, 10> c1 =152, c2 =130
# 有n个集装箱，需要装上两艘载重分别为 c1 和 c2 的轮船。
# wi 为第i个集装箱的重量，且 w1+w2+...+wn ≤ c1+c2。
# 问是否存在一种合理的装载方 案把这n个集装箱装上船? 如果有，给出一种方案。
# 算法思想: 令第一艘船的载入量为W1
# 1. 用回溯法求使得c1 -W1 达到最小的装载方案
# 2. 若满足 w1+w2+...+wn -W1 ≤ c2
def pack(nums, c1, c2):
    nums.sort()
    res = []

    def dfs(nums, c1, c2, path, index, res):
        if c1 < 0 and sum(nums) - sum(path[:-1]) <= c2:
            res.append(path[:-1])
            return
        if c1 < 0:
            return
        for i in range(index, len(nums)):
            dfs(nums, c1 - nums[i], c2, path + [nums[i]], i + 1, res)

    dfs(nums, c1, c2, [], 0, res)
    return res


if __name__ == '__main__':
    print('和为指定数的组合问题')
    print(combination_sum([2, 3, 5], 8))
    print(combination_sum2([1, 1, 2, 3], 4))

    print('排列问题')
    print(permutations([1, 2, 3]))
    print(permutations2([1, 2, 1]))

    print('全集')
    print(subset([1, 2, 3]))

    print('背包问题')
    print(knapsack_problem([1, 2, 3, 4], [1, 3, 5, 8], 5))
    print(pack([90, 80, 40, 30, 20, 12, 10], 152, 130))
