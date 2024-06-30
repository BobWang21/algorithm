from functools import lru_cache


@lru_cache(maxsize=None)  # maxsize=None 表示缓存可以无限制地增长
def fib1(n):
    if n <= 2:
        return 1  # 注意：斐波那契数列的前两个数字通常定义为1，但根据具体定义，可能需要稍作调整
    return fib1(n - 1) + fib1(n - 2)


print(fib1(300))


def coin_change1(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 初始状态 当coin = amount 时使用
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


def mckp2(groups, capacity):
    # groups 是一个二维列表，其中每个子列表代表一个组，子列表中的元素是 (volume, value) 对
    # capacity 是背包的容量
    # dp[i] 表示容量为 i 的背包能装下的最大价值
    dp = [0] * (capacity + 1)

    # 遍历每个组
    for group in groups:
        # 逆序遍历 保证每个人容量引用状态是上一分组
        for i in range(capacity, -1, -1):
            for volume, value in group:
                if i >= volume:
                    dp[i] = max(dp[i], dp[i - volume] + value)

            # 更新 dp 数组为当前组的最大价值
    return dp[-1]


groups2 = [
    [(10, 100), (5, 60), (2, 40)],  # 第一组
    [(4, 100), (3, 70)],  # 第二组
    [(9, 90), (8, 80)]  # 第三组
]
capacity2 = 22
print(mckp2(groups2, capacity2))


def get_total(groups):
    dic = []

    def helpper(i, volume, total):
        dic.append((volume, total))
        if i == len(groups):
            return
        for vol, val in groups[i]:
            helpper(i + 1, volume + vol, total + val)

    helpper(0, 0, 0)
    return dic


s = get_total(groups2)
sorted_lst = sorted(s, key=lambda x: (x[0], -x[1]))
print(sorted_lst)
