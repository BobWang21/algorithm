# 滑动窗口的最大值 最大队列和最小栈类似
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
