from collections import defaultdict
import heapq as hq


# 计数排序 适合大数据小范围的数据
# 再统计个数 然后
def count_sort(nums):
    max_value = -1
    # 元素取值范围
    for v in nums:
        max_value = max(max_value, v)

    # 计数
    cnt = [0 for _ in range(max_value + 1)]
    for v in nums:
        cnt[v] += 1

    res = []
    i = 0
    for v, cnt in enumerate(cnt):
        for _ in range(cnt):
            res[i] = v
            i += 1
    return res


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

    # 将出现次数相同的数字放在一个列表中 类似链表
    fre = defaultdict(set)
    for k, v in dic.items():
        fre[v].add(k)

    res = []
    for i in range(len(nums), 0, -1):  # 类似降序排列
        if i not in fre:
            continue
        for v in fre[i]:
            res.append(v)
            if len(res) == k:
                return res[:k]


def cyclic_sort(nums):
    n = len(nums)
    for i in range(n):
        while nums[i] != i + 1:
            j = nums[i] - 1
            nums[i], nums[j] = nums[j], nums[i]
    return nums
