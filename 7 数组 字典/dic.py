# 连续数组和为K 前缀和
def subarray_sum(nums, k):
    dic = dict()
    res = 0
    total = 0
    dic[0] = 1  # 初始化
    for v in nums:
        total += v
        if total - k in dic:
            res += dic[total - k]
        dic[total] = dic.get(total, 0) + 1
    return res


if __name__ == '__main__':
    print('\n连续数组和为K')
    print(subarray_sum([1, 1, -1], 1))
