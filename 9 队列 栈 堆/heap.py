import heapq as hq
from collections import defaultdict


# 合并K个有序数组[[1, 1], [2, 3]]
def merge(nums):
    if not nums or not nums[0]:
        return
    n = len(nums)
    if n == 1:
        return nums[0]
    heap = [(nums[i][0], i, 0) for i in range(n)]  # 小顶堆 v, row, col
    hq.heapify(heap)

    res = []
    while heap:
        v, row, col = hq.heappop(heap)
        res.append(v)
        if col + 1 < n:
            hq.heappush(heap, (nums[row][col + 1], row, col + 1))
    return res


# 二维数组中找到第K大的数 O(Klog(k))
def kth_smallest(matrix, k):
    if not matrix or not matrix[0]:
        return
    n = len(matrix)
    heap = [(matrix[i][0], i, 0) for i in range(n)]  # value, row, col
    hq.heapify(heap)
    for i in range(k):
        v, row, col = hq.heappop(heap)
        if col + 1 < n:
            hq.heappush(heap, (matrix[row][col + 1], row, col + 1))
    return v


# 373. Find K Pairs with Smallest Sums
# 也可以使用二分
def k_smallest_pairs(nums1, nums2, k):
    heap = []
    for v1 in nums1:
        for v2 in nums2:
            v = v1 + v2
            if len(heap) < k:
                hq.heappush(heap, (-v, (v1, v2)))  # 大的数先出堆 因此使用大顶堆
            elif v < -heap[0][0]:
                hq.heappop(heap)
                hq.heappush(heap, (-v, (v1, v2)))
            else:  # 如果v1 + v2 大于栈顶元素 则断开
                break
    res = [pair[1] for pair in heap]
    return res


# 692 给一非空的单词列表，返回前 k 个出现次数最多的单词。
# 返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率，按字母顺序排序。
def top_k_frequent(words, k):
    if not words:
        return
    dic = defaultdict(int)

    for word in words:
        dic[word] += 1

    heap = [(-cnt, word) for word, cnt in dic.items()]
    hq.heapify(heap)

    res = []
    for i in range(k):
        if heap:
            res.append(hq.heappop(heap)[1])
    return res


# 295 partition
class MedianFinder(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.big = []
        self.small = []

    def add_num(self, num):  # 大顶堆多一个
        if not self.big or num <= -self.big[0]:
            hq.heappush(self.big, -num)
        else:
            hq.heappush(self.small, num)

        m = len(self.big)
        n = len(self.small)
        # 0 <= m - n <= 1
        if m - n > 1:
            hq.heappush(self.small, -hq.heappop(self.big))
        if n - m > 0:
            hq.heappush(self.big, -hq.heappop(self.small))

    def find_median(self):
        m = len(self.big)
        n = len(self.small)
        if m > n:
            return -self.big[0]
        return (-self.big[0] + self.small[0]) / 2.0


def check():
    nums = [-10, 10, 1, 5, 2]
    hq.heapify(nums)
    res = []
    while nums:
        res.append(hq.heappop(nums))
    print(res)

    nums = [-10, 10, 1, 5, 2]
    n = len(nums)
    k = 3
    heap = [-nums[i] for i in range(k)]
    hq.heapify(heap)

    for i in range(k, n):
        if nums[i] < -heap[0]:  # 最大的数出堆
            hq.heappop(heap)
            hq.heappush(heap, -nums[i])

    res = []
    while heap:
        res.append(-hq.heappop(heap))
    print(res)  # 返回顺序为[2, 1, -10] 不是 [-10, 1, 2]


# 253. 会议室 II
def min_meeting_rooms(intervals):
    if not intervals:
        return 0

    intervals.sort()
    heap = [intervals[0][1]]

    for s, e in intervals[1:]:
        if heap[0] <= s:
            hq.heappop(heap)
            hq.heappush(heap, e)
        else:
            hq.heappush(heap, e)

    return len(heap)


if __name__ == '__main__':
    print('\n合并K个有序数组')
    print(merge([[1, 1, 1, 1], [2, 2, 2, 2], [3, 4, 5, 6]]))
    heap = []

    print('\n最小的k个pair')
    print(k_smallest_pairs([1, 7, 11], [2, 4, 6], 3))

    print('\n单词频率topK')
    print(top_k_frequent(["i", "love", "leetcode", "i", "love", "coding"], 2))

    print('\n数据流中位数')
    obj = MedianFinder()
    obj.add_num(1)
    obj.add_num(2)
    print(obj.find_median())

    check()
