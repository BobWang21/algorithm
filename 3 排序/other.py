from collections import defaultdict


# 原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列
# 使用整数 0, 1 和 2 分别表示红色、白色和蓝色。
def sort_colors(nums):
    if not nums:
        return []
    count = [0] * 3
    for color in nums:
        count[color] += 1
    start = 0
    for i in range(3):
        for j in range(count[i]):
            nums[start] = i
            start += 1
    return nums


# 347 出现次数最多的K个数 类似计数排序
def top_k_frequent(nums, k):
    dic = defaultdict(int)
    for v in nums:
        dic[v] += 1

    fre = defaultdict(set)
    for k, v in dic.items():  # 将出现次数相同的数字放在一个列表中 类似链表
        fre[v].add(k)

    res = []
    for i in range(len(nums), 0, -1):  # 类似降序排列
        if i in fre:
            for v in fre[i]:
                res.append(v)
                if len(res) == k:
                    return res[:k]


def cyclic_sort(nums):
    if not nums:
        return []
    n = len(nums)
    for i in range(n):
        while nums[i] != i + 1:
            j = nums[i] - 1
            if nums[j] == j + 1:  # 该位置已经排好
                break
            nums[i], nums[j] = nums[j], nums[i]
    return nums
