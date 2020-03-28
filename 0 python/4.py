def knap(cost, weight, capacity):
    if len(cost) != len(weight):
        return False
    rows, cols = len(cost) + 1, capacity + 1
    matrix = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        w, c = weight[i - 1], cost[i - 1]
        for j in range(1, cols):
            matrix[i][j] = matrix[i - 1][j]
            if j >= c:
                matrix[i][j] = max(matrix[i][j], matrix[i - 1][j - c] + w)
    return matrix[-1][-1]


def knap(cost, weight, nums, capacity):
    if len(cost) != len(weight):
        return False
    rows, cols = len(cost) + 1, capacity + 1
    matrix = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        w, c, n = weight[i - 1], cost[i - 1], nums[i - 1]
        for j in range(1, cols):
            matrix[i][j] = matrix[i - 1][j]
            for i in range(1, n + 1):
                if j < c * i:
                    break
                matrix[i][j] = max(matrix[i][j], matrix[i - 1][j - c] + w)
    return matrix[-1][-1]


def knap3(cost, weight, capacity):
    if len(cost) != len(weight):
        return False
    rows, cols = len(cost) + 1, capacity + 1
    matrix = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        w, c = weight[i - 1], cost[i - 1]
        for j in range(1, cols):
            matrix[i][j] = matrix[i - 1][j]
            if j >= c:
                matrix[i][j] = max(matrix[i][j], matrix[i][j - c] + w)
    return matrix[-1][-1]


def knap3(cost, weight, capacity):
    if len(cost) != len(weight):
        return False
    dp = [0] * (capacity + 1)
    for i in range(capacity + 1):
        for j, c in enumerate(cost):
            if i >= c:
                dp[i] = max(dp[i], dp[i - c] + weight[i])


def coin_change(coins, amount):
    if not coins or min(coins) > amount:
        return False
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for v in range(1, amount + 1):
        for coin in coins:
            if v >= coin:
                dp[v] = min(dp[v], dp[v - coin] + 1)
    return dp[-1] if dp[-1] < float('inf') else -1


# Input: [-2,1,-3,4,-1,2,1,-5,4],
def max_sub_array(nums):
    if not nums:
        return 0
    include = nums[0]
    res = nums[0]
    for i in range(1, len(nums)):
        include = max(nums[i], include + nums[i])
        res = max(include, res)
    return res


def max_product(nums):
    if not nums:
        return 0
    res = pre_min = pre_max = nums[0]
    for v in nums[1:]:
        new_pre_min = min(v, v * pre_min, v * pre_max)
        new_pre_max = max(v, v * pre_min, v * pre_max)
        pre_min = new_pre_min
        pre_max = new_pre_max
        res = max(res, new_pre_max, new_pre_min)
    return res


def num_subarray_product_less_than_k(nums, k):
    if not nums:
        return 0
    res = 0
    prod = dict()
    for i, v in enumerate(nums):
        if v < k:
            res = 1
            prod.setdefault(v, 0)
            prod[v] += 1
            break
    if not res:
        return 0

    for v in nums[i + 1:]:
        if v >= k:
            prod = dict()
        else:
            total = 1
            new_pro = {v: 1}
            for key, val in prod.items():
                new_p = key * v
                if new_p < k:
                    new_pro.setdefault(new_p, 0)
                    new_pro[new_p] += val
                    total += val
            res += total
            prod = new_pro
    return res


def lis(nums):
    if not nums:
        return 0
    stack = []
    n = len(nums)
    res = [-1] * n
    for i, v in enumerate(nums):
        while stack and nums[stack[-1]] > v:
            j = stack.pop(-1)
            res[j] = i - j
        stack.append(i)
    return res


if __name__ == '__main__':
    print('coin')
    print(coin_change([1, 5, 10, 20, 50], 100))

    print(max_sub_array([-2, 1, -3, 4, -1, 2, 1, -5, 4]))

    print(max_product([-2, 0, -1]))

    print(num_subarray_product_less_than_k([1, 1, 1], 2))

    print(lis([1, 5, 4, 3, 2, 1]))
