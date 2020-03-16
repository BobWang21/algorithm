# 滑动窗口的最大值
def max_sliding_window(nums, k):
    def clear_queue(queue, i):
        # 防止第一个划出窗口
        if queue and queue[0] == i - k:
            queue.pop(0)
        # 比当前小的数字 都不可能是窗口中的最大值
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop(-1)

    n = len(nums)
    if n * k == 0:
        return nums
    res = []
    max_idx = 0
    queue = []
    for i in range(1, k):
        clear_queue(queue, i)
        queue.append(i)
        if nums[i] > nums[max_idx]:
            max_idx = i
    res.append(nums[max_idx])

    for i in range(k, n):
        clear_queue(queue, i)
        queue.append(i)
        res.append(nums[queue[0]])
    return res


if __name__ == '__main__':
    print('滑动窗口的最大值')
    print(max_sliding_window([9, 10, 9, -7, -4, -8, 2, -6], 5))
