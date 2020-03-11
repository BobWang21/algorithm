# 使用回溯法解决N sum
def n_sum(nums, k, target):
    def helper(nums, idx, k, target, path, res):
        if k == 0 and target == 0:
            res.append(path)
            return
        if k == 0:
            return
        for i in range(idx, len(nums)):
            if target - nums[i] < 0:
                break
            helper(nums, i + 1, k - 1, target - nums[i], path + [nums[i]], res)

    if not nums or len(nums) < k or target < 0:
        return
    res = []
    nums.sort()
    helper(nums, 0, k, target, [], res)
    return res


def combination_sum(candidates, target):
    """
    Given a set of candidate numbers (candidates) (without duplicates)
    and a target number (target), find all unique combinations in candidates
    where the candidate numbers sums to target.
    The same repeated number may be chosen from candidates unlimited number of times.
    Input: candidates = [2,3,6,7], target = 7,
    A solution set is:
    [
      [7],
      [2,2,3]
    ]
    """

    def dfs(candidates, target, idx, path, res):
        if target == 0:
            res.append(path)
        if target < 0:
            return
        for i in range(idx, len(candidates)):
            dfs(candidates, target - candidates[i], i, path + [candidates[i]], res)

    res = []
    if target == 0:
        return res
    candidates.sort()
    dfs(candidates, target, 0, [], res)
    return res


def combination_sum2(candidates, target):
    """
    # Input: candidates = [10,1,2,7,6,1,5], target = 8,
    A solution set is:
    [
    [1, 1, 6]
    [1, 2, 5],
    [1, 7],
    [2, 6]
    ]
    """

    def dfs(candidates, idx, target, path, res):
        if target == 0:
            res.append(path)
            return
        if target < 0:
            return
        for i in range(idx, len(candidates)):  # 保证顺序
            if i > idx and candidates[i] == candidates[i - 1]:  # 排除相同的数字出现在同一层
                continue
            # 当前迭代索引为i 下一个迭代的索引为i+1
            dfs(candidates, i + 1, target - candidates[i], path + [candidates[i]], res)

    res = []
    if not candidates or target < 0:
        return
    candidates.sort()
    dfs(candidates, 0, target, [], res)
    return res


# 全排列 输入数组中不含重复数字
def permutations(candidates):
    """
    Given a collection of distinct integers, return all possible permutations.
    Example:
    Input: [1,2,3]
    Output:
    [
      [1, 2, 3],
      [1, 3, 2],
      [2, 1, 3],
      [2, 3, 1],
      [3, 1, 2],
      [3, 2, 1]
    ]
    """

    def dfs(candidates, path, res):
        if len(candidates) == len(path):
            res.append(path)
            return
        candidate = candidates - set(path)  # 从未访问过的集合中选取元素
        for val in candidate:
            dfs(candidates, path + [val], res)

    res = []
    dfs(set(candidates), [], res)
    return res

    # 全排列
    # 输入数组中含重复数字


def permutations2(candidates):
    def dfs(candidates, path, res):
        if not candidates:
            res.append(path)
        for i, val in enumerate(candidates):
            if i > 0 and candidates[i - 1] == val:  # 不可出现在同一层
                continue
            # 候选集中去除访问过的元素
            dfs(candidates[:i] + candidates[i + 1:], path + [val], res)

    res = []
    if not candidates:
        return res
    candidates.sort()
    dfs(candidates, [], res)
    return res


# 子集问题
def subset(candidates):
    res = []

    def dfs(candidates, index, path, res, k):
        if k >= 0:
            res.append(path)  # 把路径全部记录下来
        for i in range(index, len(candidates)):
            dfs(candidates, i + 1, path + [candidates[i]], res, k - 1)

    dfs(candidates, 0, [], res, len(candidates))
    return res


# 背包问题
def knapsack(cost, val, cap):
    def dfs(cap, idx, amount, res):
        for i in range(idx, len(cost)):
            if cap - cost[i] < 0:  # base 1
                res[0] = max(res[0], amount)
                continue
            elif cap - cost[i] == 0:  # base 2
                res[0] = max(res[0], amount + val[i])
                continue
            else:
                dfs(cap - cost[i], i + 1, amount + val[i], res)

    res = [-1]
    dfs(cap, 0, 0, res)
    return res[0]


def can_partition_k_subsets(nums, k):
    if len(nums) < k:
        return False
    total = sum(nums)
    if total % k != 0:
        return False
    target = int(total / k)
    candidates = []
    for v in nums:
        if v < target:
            candidates.append(v)
        elif v == target:
            k -= 1
        else:
            return False
    if not k:
        return True

    def helper(num, k, tar):
        if k == 1:
            return True
        if not tar:
            return helper(num, k - 1, target)
        for i, v in enumerate(num):
            if v <= tar and helper(num[:i] + num[i + 1:], k, tar - v):
                return True
        return False

    candidates.sort()
    return helper(candidates, k, target)


def pack(candidates, c1, c2):
    """
    # W=<90, 80, 40, 30, 20, 12, 10> c1 =152, c2 =130
    # 有n个集装箱，需要装上两艘载重分别为 c1 和 c2 的轮船。
    # wi 为第i个集装箱的重量，且 w1+w2+...+wn ≤ c1+c2。
    # 问是否存在一种合理的装载方 案把这n个集装箱装上船? 如果有，给出一种方案。
    # 算法思想: 令第一艘船的载入量为W1
    # 1. 用回溯法求使得c1 -W1 达到最小的装载方案
    # 2. 若满足 w1+w2+...+wn -W1 ≤ c2
    """
    candidates.sort()
    res = []

    def dfs(candidates, index, c1, path, res):
        if sum(candidates) - sum(path) <= c2:  # 加了新的后 开始小于0
            res.append(path)
            return
        for i in range(index, len(candidates)):
            if c1 - candidates[i] < 0:
                break
            dfs(candidates, i + 1, c1 - candidates[i], path + [candidates[i]], res)

    dfs(candidates, 0, c1, [], res)
    return res


def letter_case_permutation(s):
    if not s:
        return
    l = list(s)
    res = []

    def helper(l, idx, path):
        if len(path) == len(l):
            res.append(''.join(path))
            return
        for i in range(idx, len(l)):
            c = l[i]
            if c.isalpha():
                helper(l, i + 1, path + [c.lower()])
                helper(l, i + 1, path + [c.upper()])
            else:
                helper(l, i + 1, path + [c])

    helper(l, 0, [])
    return res


# Input:
# beginWord = "hit",
# endWord = "cog",
# wordList = ["hot","dot","dog","lot","log","cog"] BFS
def word_ladder(begin_word, end_word, word_list):
    if not word_list:
        return 0
    queue = [(begin_word, 0)]  # 记录层数
    seen_set = {begin_word}  # 保存已经加入过队列的字符串
    word_set = set(word_list)
    while queue:
        word, l = queue.pop(0)
        if word == end_word:
            return l + 1
        for i in range(len(word)):
            for j in range(26):  # 访问每个字符串的近邻 如果近邻满足则返回
                c = chr(ord('a') + j)
                if word[i] == c:
                    continue
                new_word = word[:i] + c + word[i + 1:]
                if new_word in word_set and new_word not in seen_set:
                    seen_set.add(new_word)
                    queue.append((new_word, l + 1))

    return 0


def find_cheapest_price(n, flights, src, dst, K):
    """
    :type n: int
    :type flights: List[List[int]]
    :type src: int
    :type dst: int
    :type K: int
    :rtype: int
    """
    dic = dict()  # {s: {d:p}}

    for s, d, p in flights:
        dic.setdefault(s, dict())
        dic[s][d] = p
    print(dic)
    queue = [(src, -1, 0, {src})]
    amount = float('inf')
    while queue:
        node, stop, price, path = queue.pop(0)
        if stop > K:
            break
        if price >= amount:
            continue
        if node == dst and stop <= K:
            amount = min(amount, price)
        if node in dic:
            for new_node, new_price in dic[node].items():
                if new_node not in path:
                    new_path = path | {new_node}
                    queue.append((new_node, stop + 1, price + new_price, new_path))

    return amount if amount < float('inf') else -1


def find_cheapest_price2(n, flights, src, dst, K):
    """
    :type n: int
    :type flights: List[List[int]]
    :type src: int
    :type dst: int
    :type K: int
    :rtype: int
    """
    dic = dict()  # {s: {d:p}}

    for s, d, p in flights:
        dic.setdefault(s, dict())
        dic[s][d] = p

    res = [float('inf')]

    def helper(node, path, amount):
        if len(path) > K + 2:
            return
        if node in dic:
            for new_node, price in dic[node].items():
                if new_node not in path:
                    if new_node == dst:
                        res[0] = min(res[0], amount + price)
                        return
                    else:
                        new_path = path | {new_node}
                        helper(new_node, new_path, amount + price)

    helper(src, set(), 0)
    return res[0]


if __name__ == '__main__':
    print('n sum 回溯版')
    print(n_sum([1, 2, 3, 4], 3, 6))

    print('数组中不包含重复数字 一个数字可以用无数次')
    print(combination_sum([2, 3, 5], 8))

    print('\n数组中包含重复数字 一个数字只能用一次')
    print(combination_sum2([1, 1, 2, 3, 4], 4))

    print('\n排列问题')
    print(permutations([1, 2, 3]))
    print(permutations2([1, 2, 1]))

    print('\n全集')
    print(subset([1, 2, 3]))

    print('\n背包问题')
    print(knapsack([1, 2, 3, 4], [1, 3, 5, 8], 5))

    print('\n两个轮船分集装箱')
    print(pack([90, 80, 40, 30, 20, 12, 10], 152, 130))

    print('Letter Case Permutation')
    print(letter_case_permutation("a1b2"))

    print('word_ladder')
    a = "qa"
    b = "sq"
    c = ["si", "go", "se", "cm", "so", "ph", "mt", "db", "mb", "sb", "kr", "ln", "tm", "le", "av", "sm", "ar", "ci",
         "ca", "br", "ti", "ba", "to", "ra", "fa", "yo", "ow", "sn", "ya", "cr", "po", "fe", "ho", "ma", "re", "or",
         "rn", "au", "ur", "rh", "sr", "tc", "lt", "lo", "as", "fr", "nb", "yb", "if", "pb", "ge", "th", "pm", "rb",
         "sh", "co", "ga", "li", "ha", "hz", "no", "bi", "di", "hi", "qa", "pi", "os", "uh", "wm", "an", "me", "mo",
         "na", "la", "st", "er", "sc", "ne", "mn", "mi", "am", "ex", "pt", "io", "be", "fm", "ta", "tb", "ni", "mr",
         "pa", "he", "lr", "sq", "ye"]
    print(word_ladder(a, b, c))

    print('find_cheapest_price')
    print(find_cheapest_price2(3, [[0, 1, 100], [1, 2, 100], [0, 2, 500]], 0, 2, 1))

    print(
        can_partition_k_subsets(
            [114, 96, 18, 190, 207, 111, 73, 471, 99, 20, 1037, 700, 295, 101, 39, 649],
            4))
