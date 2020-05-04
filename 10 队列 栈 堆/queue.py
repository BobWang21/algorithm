# 可以返回最大值的队列
class MaxQueue():
    def __init__(self):
        self.queue1 = []
        self.queue2 = []

    def enqueue(self, v):
        self.queue1.append(v)
        while self.queue2 and self.queue2[-1] < v:
            self.queue2.pop(-1)
        self.queue2.append(v)

    def dequeue(self):
        v = self.queue1.pop(0)
        if self.queue2[0] == v:
            self.queue2.pop(0)

    def get_max(self):
        return self.queue2[0]


# 滑动窗口的最大值 最大队列和最小栈类似
def max_sliding_window(nums, k):
    def enqueue(queue, i):  #
        # 防止第一个划出窗口
        if queue and i - queue[0] == k:
            queue.pop(0)
        # 比当前小的数字 都不可能是窗口中的最大值
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop(-1)
        queue.append(i)

    n = len(nums)
    if n * k == 0:
        return nums
    res = []
    max_idx = 0
    queue = []
    for i in range(1, k):
        enqueue(queue, i)
        if nums[i] > nums[max_idx]:
            max_idx = i
    res.append(nums[max_idx])

    for i in range(k, n):
        enqueue(queue, i)
        res.append(nums[queue[0]])
    return res


# Input: nums1 = [4,1,2], nums2 = [1,3,4,2].
# Output: [-1,3,-1]
# Explanation:
#     For number 4 in the first array, you cannot find the next greater number for it in the second array, so output -1.
#     For number 1 in the first array, the next greater number for it in the second array is 3.
#     For number 2 in the first array, there is no next greater number for it in the second array, so output -1.
def next_greater_element(nums1, nums2):
    if not nums1 or not nums2:
        return
    dic = dict()

    for i, v in enumerate(nums1):
        dic[v] = i
    res = [-1] * len(nums1)
    stack = []
    for v in nums2:
        while stack and stack[-1] < v:
            tail = stack.pop(-1)
            if tail in dic:
                res[dic[tail]] = v
        stack.append(v)
    return res


# Given a list of daily temperatures T,
# return a list such that, for each day in the input,
# tells you how many days you would have to wait until a warmer temperature.
# If there is no future day for which this is possible, put 0 instead.
# For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73],
# your output should be [1, 1, 4, 2, 1, 1, 0, 0].
def daily_temperatures(t):
    if not t:
        return
    stack = []
    res = [0] * len(t)
    for i, v in enumerate(t):
        while stack and t[stack[-1]] < v:
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res


# 暴力法O(n^2)
def largest_rectangle_area1(heights):
    max_rec = 0
    n = len(heights)
    for i, h in enumerate(heights):
        l = r = i
        while l > 0 and heights[l - 1] >= h:  # 前面第一个小于该数
            l -= 1
        while r < n - 1 and heights[r + 1] >= h:  # 后面第一个小于该数
            r += 1
        max_rec = max(max_rec, (r - l - 1) * h)
    return max_rec


def largest_rectangle_area2(height):
    height.append(0)  # 为了让剩余元素出栈
    stack = []
    ans = 0
    n = len(height)
    for i in range(n):
        while stack and height[stack[-1]] > height[i]:
            h = height[stack.pop()]
            w = i - stack[-1] - 1 if stack else i
            ans = max(ans, h * w)
        stack.append(i)
    return ans


def max_area_min_sum_product(nums):
    if not nums:
        return 0
    nums.append(-1)  # 为了使栈中剩余元素出栈
    n = len(nums)
    stack = []
    total = [0] * n
    max_v = 0

    for i, v in enumerate(nums):
        v = v if v >= 0 else 0
        total[i] = total[i - 1] + v

        while stack and v < nums[stack[-1]]:
            j = stack.pop(-1)
            pre_total = 0
            if stack:
                pre_total = total[stack[-1]]
            max_v = max(max_v, (total[i - 1] - pre_total) * nums[j])
        stack.append(i)
    return max_v


def trap(height):
    res = 0
    stack = []
    for i in range(len(height)):
        while stack and height[i] > height[stack[-1]]:
            j = stack.pop(-1)
            if not stack:
                break
            distance = i - stack[-1] - 1
            bounded_height = min(height[i], height[stack[-1]]) - height[j]
            res += distance * bounded_height  # 可以多加的水
        stack.append(i)

    return res


if __name__ == '__main__':
    print('最大值值队列')
    maxQueue = MaxQueue()
    for i in range(10):
        maxQueue.enqueue(i)
    maxQueue.dequeue()
    maxQueue.enqueue(11)
    print(maxQueue.get_max())

    print('\n滑动窗口的最大值')
    print(max_sliding_window([9, 10, 9, -7, -4, -8, 2, -6], 5))

    print('\n 下一个比其大数值')
    print(next_greater_element([2, 4], [1, 2, 3, 4]))

    print('\n下一个天气比当前热')
    print(daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]))

    print('\n柱状图最大矩形')
    print((largest_rectangle_area2([2, 1, 5, 6, 2, 3])))

    print('\n区间数字和与区间最小值乘积最大')
    print(max_area_min_sum_product([81, 87, 47, 59, 81, 18, 25, 40, 56, 0]))

    print('\n接雨水')
    print(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
