def combination_sum_dfs(nums, tar, index, path, res):
    if tar == 0:
        res.append(path)
    if tar < 0:
        return
    for i in range(index, len(nums)):
        combination_sum_dfs(nums, tar - nums[i], i, path + [nums[i]], res)


# 数字可以用多次
def combination_sum(nums, tar):
    nums.sort()
    res = []
    if tar == 0:
        return res
    combination_sum_dfs(nums, tar, 0, [], res)
    return res


def n_sum_dfs(nums, tar, k, index, path, res):
    if tar == 0 and k == 0:
        res.append(path)
        return
    if k == 0:
        return
    for i in range(index, len(nums)):
        n_sum_dfs(nums, tar - nums[i], k - 1, i + 1, path + [nums[i]], res)


def n_sum(nums, tar, k):
    nums.sort()
    res = []
    n_sum_dfs(nums, tar, k, 0, [], res)
    return res


# 全排列 数组中不含重复数字
def permutations(nums):
    res = []
    permutations_dfs(set(nums), [], res)
    return res


def permutations_dfs(nums, path, res):
    if len(nums) == len(path):
        res.append(path)
    candidate = nums - set(path)  # 从未访问过的集合中选取元素
    for val in candidate:
        permutations_dfs(nums, path + [val], res)


# 包含重复数子
def permutations2(nums):
    res = []
    permutations2_dfs(nums, [], res)
    return res


# 数组中包含重复元素, 返回所有可能的排列
def permutations2_dfs(nums, path, res):
    if len(nums) == 0:
        res.append(path)
    for val in set(nums):
        # 候选集大小-1
        index = nums.index(val)
        permutations2_dfs(nums[:index] + nums[index + 1:], path + [val], res)


# 子集问题
def subset(nums):
    res = []
    subset_dfs(nums, 0, [], res, len(nums))
    return res


def subset_dfs(nums, index, path, res, k):
    if k >= 0:
        res.append(path)  # 把路径全部记录下来
    for i in range(index, len(nums)):
        subset_dfs(nums, i + 1, path + [nums[i]], res, k - 1)


# 背包问题
def knapsack_problem(cost, val, cap):
    res = []
    knapsack_problem_dfs(cost, val, cap, 0, 0, res)
    return max(res)


def knapsack_problem_dfs(cost, val, cap, index, amount, res):
    if cap == 0:
        res.append(amount)
        return
    for i in range(index, len(cost)):
        if cap - cost[i] < 0:
            continue
        knapsack_problem_dfs(cost, val, cap - cost[i], i + 1, amount + val[i], res)


def stair(n, arr):
    if len(arr) - 1 >= n:
        return arr[n]
    return stair(n - 1, arr) + stair(n - 2, arr)


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
    pack_dfs(nums, c1, c2, [], 0, res)
    return res


def pack_dfs(nums, c1, c2, path, index, res):
    if c1 < 0 and sum(nums) - sum(path[:-1]) <= c2:
        res.append(path[:-1])
        return
    if c1 < 0:
        return
    for i in range(index, len(nums)):
        pack_dfs(nums, c1 - nums[i], c2, path + [nums[i]], i + 1, res)


if __name__ == '__main__':
    print(combination_sum([2, 3, 5], 8))
    print(permutations([1, 2, 3]))
    print(permutations2([1, 2, 1]))
    print(n_sum([2, 4, 3, 5], 7, 2))
    print(subset([1, 2, 3]))
    print(knapsack_problem([1, 2, 3, 4], [1, 3, 5, 8], 5))
    print(stair(3, [1, 1]))
    print(pack([90, 80, 40, 30, 20, 12, 10], 152, 130))
