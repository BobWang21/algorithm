import heapq as hq
from collections import defaultdict


def merge(nums):
    if not nums or not nums[0]:
        return
    n = len(nums)
    if n == 1:
        return nums[0]
    heap = [(nums[i].pop(0), i) for i in range(n)]
    hq.heapify(heap)

    res = []
    while heap:
        v, i = hq.heappop(heap)
        res.append(v)
        if nums[i]:
            v = nums[i].pop(0)
            hq.heappush(heap, (v, i))
    return res


def kth_smallest(matrix, k):
    if not matrix:
        return
    heap = []
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if len(heap) < k:
                hq.heappush(heap, -matrix[i][j])
            else:
                if matrix[i][j] < -heap[0]:  # 比最大的要小的话
                    hq.heappop(heap)
                    hq.heappush(heap, -matrix[i][j])

    return -heap[0]


# 373. Find K Pairs with Smallest Sums
def k_smallest_pairs(nums1, nums2, k):
    heap = []
    dic = defaultdict(list)
    for i, v1 in enumerate(nums1):
        for j, v2 in enumerate(nums2):
            new_value = v1 + v2
            if len(heap) < k:
                hq.heappush(heap, -new_value)
                dic[-new_value].append([v1, v2])
            elif new_value < -heap[0]:
                v = hq.heappop(heap)
                dic[v].pop(0)  # 可能有重复值
                hq.heappush(heap, -new_value)
                dic[-new_value].append([v1, v2])
    res = []
    for k, v in dic.items():
        res.extend(v)
    return res


# 也可以使用 partition
class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.large = []
        self.small = []

    def addNum(self, target):
        """
        :type num: int
        :rtype: None
        """
        if not self.large:
            hq.heappush(self.large, -target)
            return
        if target <= -self.large[0]:
            hq.heappush(self.large, -target)
        else:
            hq.heappush(self.small, target)

        m = len(self.large)
        n = len(self.small)
        if m - n > 1:
            v = -hq.heappop(self.large)
            hq.heappush(self.small, v)
        if n - m > 1:
            v = hq.heappop(self.small)
            hq.heappush(self.large, -v)

    def findMedian(self):
        """
        :rtype: float
        """
        m = len(self.large)
        n = len(self.small)
        if m + n == 1:
            return -self.large[0]
        print(m, n)
        if (m + n) % 2:
            print('v')
            if m > n:
                return -self.large[0]
            else:
                return self.small[0]
        else:
            return (-self.large[0] + self.small[0]) / 2.0


if __name__ == '__main__':
    print('\n合并K个有序数组')
    print(merge([[1, 1, 1, 1], [2, 2, 2, 2], [3, 4, 5, 6]]))
    heap = []

    print('\n最小的k个pair')
    print(k_smallest_pairs([1, 7, 11], [2, 4, 6], 3))

    print('\n数据流中位数')
    obj = MedianFinder()
    obj.addNum(1)
    obj.addNum(2)
    print(obj.findMedian())
