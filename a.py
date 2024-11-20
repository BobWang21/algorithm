#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, deque

import heapq as hq

num = [1, 4, 10]

hq.heapify(num)

while num:
    print(hq.heappop(num))

queue = deque()
queue.append(1)
queue.append(10)
queue.append(20)

queue.pop()
queue.popleft()

print(queue)


def c_sort(nums):
    n = len(nums)
    for i in range(n):
        while nums[i] != i:
            j = nums[i]
            if nums[j] == j:
                break
            nums[i], nums[j] = nums[j], nums[i]
    return nums


nums = [3, 1, 0, 2, 2]
print(c_sort(nums))


def find_missing_value(nums):
    n = len(nums)

    for i in range(n):
        while 1 <= nums[i] <= n and nums[i] != i + 1:
            j = nums[i] - 1
            if nums[j] == j + 1:
                break
            nums[i], nums[j] = nums[j], nums[i]
    print(nums)
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


nums = [3, 1, 2, 4]
print(find_missing_value(nums))


def subarray_sum(nums, k):
    dic = defaultdict(int)
    dic[0] = 1

    total = 0
    res = 0
    for v in nums:
        total += v
        if total - k in dic:
            res += dic[total - k]
        dic[total] += 1
    return res


print(subarray_sum([2, 3, 1, 4], 5))


def us(nums):
    left = -1
    n = len(nums)
    # left 小于右侧的最小值的最后值的索引
    min_v = float('inf')
    for i in range(n - 1, -1, -1):
        if nums[i] < min_v:
            min_v = nums[i]
        else:
            left = i

    right = -1
    max_v = -float('inf')
    # 小于左侧的最大值的最后值的索引
    for i in range(n):
        if nums[i] > max_v:
            max_v = nums[i]
        else:
            right = i

    return right - left + 1 if left != right else 0


print(us([1]))


def find_magic_index(nums):
    def find_magic_index_(l, r):
        if l >= r:
            return -1
        mid = l + (r - l) // 2
        idx = find_magic_index_(l, mid - 1)
        if idx != -1:
            return idx
        if nums[mid] == mid:
            return mid
        return find_magic_index_(mid + 1, r)

    return find_magic_index_(0, len(nums) - 1)


print(find_magic_index([2, 3, 4, 4, 5, 5, 5]))

from collections import Counter
import heapq as hq


def top_k_frequent(nums, k):
    counter = Counter(nums)
    min_hq = []
    for value, cnt in counter.items():
        hq.heappush(min_hq, (cnt, value))
        if len(min_hq) > k:
            hq.heappop(min_hq)
    return [value for cnt, value in min_hq]


print('\n出现次数最多的K个数')
print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))


def next_permute(nums):
    n = len(nums)
    i = n - 1
    while i > 0 and nums[i] <= nums[i - 1]:
        i -= 1
    if i == 0:
        nums.reverse()
        return nums
    print(i)

    k = i - 1
    j = i
    while j < n and nums[j] > nums[k]:
        j += 1
    nums[k], nums[j - 1] = nums[j - 1], nums[k]

    l, r = k + 1, n - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums


print(next_permute([3, 2, 1]))


def diagnose_traverse(matrix):
    rows, cols = len(matrix), len(matrix[0])
    res = []
    for j in range(cols):
        i = 0
        while j >= 0:
            res.append(matrix[i][j])
            i += 1
            j -= 1

    for i in range(1, rows):
        j = cols - 1
        while i < rows:
            res.append(matrix[i][j])
            i += 1
            j -= 1
    return res


matrix = [[1, 2, 3, -1],
          [4, 5, 6, -2],
          [7, 8, 9, -3]]


def spiral_order(matrix):
    res = []
    left, right, up, down = 0, len(matrix[0]) - 1, 0, len(matrix) - 1

    while True:
        # 左->右
        for j in range(left, right + 1):
            res.append(matrix[up][j])
        up += 1
        if up > down:
            break

        # 上->下
        for i in range(up, down + 1):
            res.append(matrix[i][right])
        right -= 1
        if left > right:
            break

        # 右到左
        for j in range(right, left - 1, -1):
            res.append(matrix[down][j])
        down -= 1
        if up > down:
            break

        # 下到上
        for i in range(down, up - 1, -1):
            res.append(matrix[i][left])

        left += 1
        if left > right:
            break

    return res


print(spiral_order(matrix))


def binary_search1(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1


def binary_search2(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return l if nums[l] == target else -1


nums = [1, 3, 5, 7, 8]
for v in nums:
    print(binary_search2(nums, v))


def get_number_of_k(nums, target):
    if not nums:
        return [-1, -1]
    # 左侧
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    left = l if nums[l] == target else -1
    if left == -1:
        return -1, -1

    # 右侧
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l + 1) // 2
        if nums[mid] <= target:
            l = mid
        else:
            r = mid - 1
    right = l if nums[l] == target else -1
    return left, right


print(get_number_of_k([], 6))


def sqrt(value, measure):
    l, r = 0, value
    while l < r:
        mid = l + (r - l) / 2.0
        square = mid ** 2
        if abs(square - value) < measure:
            print(value)
            return square
        if square < value:
            l = mid
        else:
            r = mid


print('sqrt', sqrt(1, 0.001))


# 输入：nums = [3,4,5,1,2]
# 输出：1
# 解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
def findMin(nums):
    if nums[0] < nums[-1]:
        return nums[0]

    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[0]:  # nums[l - 1] > nums[l] < nums[l+1]
            l = mid + 1
        else:
            r = mid
    return nums[l]


print(findMin([3, 4, 5, 1, 2]))


# 输入：nums = [4,5,6,7,0,1,2], target = 0
# 输出：4


def search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if target == nums[mid]:
            return mid
        if nums[mid] >= nums[0]:
            if nums[0] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:
            if nums[mid] < target <= nums[-1]:
                l = mid + 1
            else:
                r = mid - 1
    return -1


for v in [4, 5, 6, 7, 0, 1, 2]:
    print('x', search([4], v + 10))


def find_median(nums1, nums2):
    if len(nums1) > len(nums2):
        return find_median(nums2, nums1)

    m, n = len(nums1), len(nums2)
    k = (m + n + 1) // 2  #
    l, r = 0, m - 1
    while l < r:
        mid = l + (r - l + 1) // 2
        if nums1[mid] <= nums2[k - mid - 1]:  # nums1[l] <= nums2[k - l - 1]
            l = mid
        else:
            r = mid - 1  # nums1[l + 1] > nums2[k - mid -2]
    l = -1 if nums1[l] > nums2[k - l - 1] else l

    left = max(nums1[l] if m > l >= 0 else -float('inf'),
               nums2[k - l - 2] if n > k - l - 2 >= 0 else -float('inf'))

    if (m + n) % 2:
        print('xx')
        return left

    right = min(nums1[l + 1] if m > l + 1 >= 0 else float('inf'),
                nums2[k - l - 1] if n > k - l - 1 >= 0 else float('inf'))
    return (left + right) / 2.0


def find_median2(nums1, nums2):
    if len(nums1) > len(nums2):
        return find_median(nums2, nums1)

    m, n = len(nums1), len(nums2)
    k = (m + n + 1) // 2  #
    l, r = 0, m
    while l < r:
        mid = l + (r - l + 1) // 2
        if nums1[mid - 1] <= nums2[k - mid]:  # nums1[l-1] <= nums2[k - l]
            l = mid
        else:
            r = mid - 1  # nums1[l] > nums2[k-mid-1]
    l = -1 if nums1[l] > nums2[k - l - 1] else l

    left = max(nums1[l] if m > l >= 0 else -float('inf'),
               nums2[k - l - 2] if n > k - l - 2 >= 0 else -float('inf'))

    if (m + n) % 2:
        print('xx')
        return left

    right = min(nums1[l + 1] if m > l + 1 >= 0 else float('inf'),
                nums2[k - l - 1] if n > k - l - 1 >= 0 else float('inf'))
    return (left + right) / 2.0


print(find_median([1, 3], [5]))


def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge(left, right)


def merge(nums1, nums2):
    res = []
    i = j = 0
    m, n = len(nums1), len(nums2)
    while i < m or j < n:
        v1 = nums1[i] if i < m else float('inf')
        v2 = nums2[j] if j < n else float('inf')
        if v1 <= v2:
            res.append(v1)
            i += 1
        else:
            res.append(v2)
            j += 1

    return res


print(merge_sort([1, 3, 5, 2, 4, 6]))


def quick_sort(nums, l, r):
    if l >= r:
        return nums
    pivot = partition(l, r)
    quick_sort(nums, l, pivot - 1)
    quick_sort(nums, pivot + 1, r)
    return nums


def partition(l, r):
    pivot = nums[l]
    while l < r:
        while l < r and nums[r] >= pivot:
            r -= 1
        nums[l] = nums[r]

        while l < r and nums[l] < pivot:
            l += 1
        nums[r] = nums[l]
    nums[l] = pivot
    return l


nums = [1, 3, 5, 2, 4, 6]

print(quick_sort(nums, 0, 5))


def odd_partition(nums):
    l, r = 0, len(nums) - 1
    pivot = nums[l]

    def is_odd(v):
        return v % 2 == 1

    while l < r:
        while l < r and not is_odd(nums[r]):
            r -= 1

        nums[l] = nums[r]

        while l < r and is_odd(nums[l]):
            l += 1

        nums[r] = nums[l]
    nums[l] = pivot
    return nums


print(odd_partition([1, 2, 3, 4, 5]))


def s(nums, target):
    nums.sort()

    n = len(nums)
    for i in range(n - 2):
        v = nums[i]
        l, r = i + 1, n - 1
        while l < r:
            total = v + nums[l] + nums[r]
            if total == target:
                return [nums[i], nums[l], nums[r]]
            if total > target:
                r -= 1
            else:
                l += 1
    return [-1, -1, -1]


print(s([1, 2, 3, 4, 2], 5))


def lengthOfLongestSubstringKDistinct(s, k):
    if len(s) <= k:
        return len(s)

    dic = defaultdict(int)
    n = len(s)
    l = 0
    res = 0
    for i in range(n):
        c = s[i]
        dic[c] += 1
        while len(dic) > k:
            c = s[l]
            dic[c] -= 1
            if dic[c] == 0:
                del dic[c]
            l += 1
        res = max(res, i - l + 1)
    return res


print('lengthOfLongestSubstringKDistinct', lengthOfLongestSubstringKDistinct('eceba', 2))


def min_window1(s, t):
    if len(s) < len(t):
        return ''
    counter = Counter(t)
    match = 0
    l, r = 0, 0
    m, n = len(s), len(t)
    min_len = m + 1
    dic = defaultdict(int)
    res = ''
    for r in range(m):
        c = s[r]
        if c not in counter:
            continue
        dic[c] += 1
        if dic[c] == counter[c]:
            match += 1
        while match == len(counter):
            if r - l + 1 < min_len:
                min_len = r - l + 1
                res = s[l:r + 1]
            char = s[l]
            if char in dic:
                dic[char] -= 1
                if dic[char] < counter[char]:
                    match -= 1
            l += 1
    return res


s = "ADOBECODEBANC"
t = "ABC"
print(min_window1(s, t))


def knapsack2(values, weights, capacity):
    dp = [0] * (capacity + 1)
    for val, weight in zip(values, weights):
        for cap in range(capacity, weight - 1, -1):
            dp[cap] = max(dp[cap - weight] + val, dp[cap])
    print(dp)
    return dp[-1]


values = [1, 5, 4, 8]
weights = [1, 2, 3, 4]
print(knapsack2(values, weights, 7))


def coin_change1(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for amt in range(coin, amount + 1):
            dp[amt] = min(dp[amt], dp[amt - coin] + 1)
    return dp[-1]


def coin_change2(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for amt in range(coin, amount + 1):
            dp[amt] += dp[amt - coin]
    print(dp)
    return dp[-1]


print(coin_change1([1, 2, 5, 10], 11))
print(coin_change2([1, 2, 5, 10], 11))


def LIS(nums):
    dp = [nums[0]]

    def search(target):
        l, r = 0, len(dp) - 1
        while l < r:
            mid = l + (r - l) // 2
            if dp[mid] < target:
                l = mid + 1
            else:
                r = mid
        return l

    n = len(nums)
    for i in range(1, n):
        if dp[-1] < nums[i]:
            dp.append(nums[i])
            continue
        idx = search(nums[i])
        dp[idx] = nums[i]
    print(dp)
    return len(dp)


print(LIS([0, 1, 0, 3, 2, 3]))


def numSquares(n):
    dp = [n] * (n + 1)
    dp[0] = 0
    for i in range(1, n // 2 + 1):
        for j in range(i ** 2, n + 1):
            dp[j] = min(dp[j], dp[j - i ** 2] + 1)
    return dp[-1]


print(numSquares(1))


def longest_increasing_path(matrix):
    rows, cols = len(matrix), len(matrix[0])

    memory = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(i, j):
        if i < 0 or i == rows or j < 0 or j == cols:
            return 0
        if (i, j) in memory:
            return memory[(i, j)]
        res = 0
        for di, dj in directions:
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < rows and 0 <= new_j < cols and matrix[new_i][new_j] > matrix[i][j]:
                res = max(res, dfs(new_i, new_j))
        memory[(i, j)] = res + 1
        return res + 1

    res = 0

    for i in range(rows):
        for j in range(cols):
            res = max(res, dfs(i, j))
    return res


print(longest_increasing_path([[1]]))


def subsets(nums):
    n = len(nums)
    res = []
    path = []

    def helper(i):
        res.append(path[:])

        for j in range(i, n):
            path.append(nums[j])
            helper(j + 1)
            path.pop()

    helper(0)

    return res


print(subsets([1, 2, 3]))

'''
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
'''


def subsets2(nums):
    nums.sort()
    n = len(nums)
    res = []
    path = []

    def helper(i):
        res.append(path[:])

        for j in range(i, n):
            if j > i and nums[j] == nums[j - 1]:
                continue
            path.append(nums[j])
            helper(j + 1)
            path.pop()

    helper(0)

    return res


print(subsets2([1, 2, 2]))


def permute(nums):
    nums.sort()

    res = []
    path = []
    n = len(nums)
    visited = [False] * n

    def helper():
        if len(path) == n:
            res.append(path[:])

        for i in range(n):
            if visited[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]:
                continue
            visited[i] = True
            path.append(nums[i])
            helper()
            visited[i] = False
            path.pop()

    helper()
    return res


print(permute([1, 2, 2]))

'''
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
'''


def combinationSum(candidates, target):
    path = []
    res = []
    n = len(candidates)

    def helper(total, i):
        if total > target:
            return

        if total == target:
            res.append(path[:])
            return

        for j in range(i, n):
            path.append(candidates[j])
            helper(total + candidates[j], j)
            path.pop()

        return

    candidates.sort()

    helper(0, 0)
    return res


# , target = 7
print('combinationSum', combinationSum([2, 3, 5], 8))


def preorder_traverse(root):
    if not root:
        return []
    stack = [root]
    res = []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return res


def inorder_traverse(root):
    node = root
    stack = []
    res = []
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        res.append(node.val)
        node = node.right
    return res


def post_order(root):
    if not root:
        return []

    res = []

    pre_node = None

    stack, node = [], root

    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack[-1]
        if not node.right or pre_node == node.left:
            node = stack.pop()
            pre_node = node
            res.append(node.val)
            node = None
        else:
            node = node.right


def lca(root, p, q):
    def helper(node):
        if not node:
            return None
        if node.val == p or node.val == q:
            return node
        left = helper(node.left)
        right = helper(node.right)
        if left and right:
            return node
        return left or right

    return helper(root)


def is_balanced(root):
    def helper(node):
        if not node:
            return 0
        left = helper(node.left)
        if left == -1:
            return -1
        right = helper(node.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return max(left, right) + 1

    return helper(root) != -1


def have_path_sum(root, target):
    def helper(total, node):
        if not root:
            return False
        total += node.val
        if not node.left and not node.right:
            return total == target
        return helper(total, node.left) or helper(total, node.right)

    return helper(0, root)


def path_sum2(root, target):
    dic = defaultdict(int)
    dic[0] = 1
    res = [0]

    def from_root(total, node):
        if not node:
            return
        total += node.val
        if total - target in dic:
            res[0] += dic[total - target]
        dic[total] += 1
        from_root(total, node.left)
        from_root(total, node.right)
        dic[total] -= 1

    from_root(0, root)
    return res[0]


def max_sum_path1(root):
    if not root:
        return 0

    res = [-float('inf')]

    def from_root(node):
        if not root:
            return -float('inf'), -float('inf')  # fixme
        left = from_root(node.left)
        right = from_root(node.right)
        inc = max(node.val, node.val + left[0], node.val + right[0])
        exc = max(max(left), max(right))
        res[0] = max(res[0], inc, exc, node.val + right[0] + left[0])
        return inc, exc

    from_root(root)
    return res[0]


class LinkNode():
    def __init__(self, key=None, val=0):
        self.key = key
        self.val = val
        self.pre = None
        self.next = None


class LRUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.dic = dict()
        self.head = LinkNode()
        self.tail = LinkNode()
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.dic:
            return -1
        node = self.dic[key]
        value = node.val
        self.remove(key)
        self.add(key, value)
        return value

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """

        if len(self.dic) == self.capacity:
            self.remove(self.tail.pre.key)

        if key in self.dic:
            self.remove(key)
        self.add(key, value)

    def add(self, key, value):
        succ = self.head.next
        node = LinkNode(key, value)
        node.next = succ
        node.pre = self.head
        self.head.next = node
        succ.pre = node
        self.dic[key] = self.dic.get(key, node)
        return

    def remove(self, key):
        node = self.dic[key]
        pre = node.pre
        succ = node.next
        pre.next = succ
        succ.pre = node
        del self.dic[key]
        return

        # Your LRUCache object will be instantiated and called as such:


# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


def trap(height):
    if len(height) == 1:
        return 0
    res = 0
    stack = []
    for i, h in enumerate(height):
        while stack and h > height[stack[-1]]:
            j = stack.pop()
            if stack:
                left = stack[-1]
                res += (min(height[left], h) - height[j]) * (i - left - 1)
        stack.append(i)

    return res


print(trap([4, 2, 0, 3, 2, 5]))


def max_area(heights):
    stack = []
    res = 0
    for i, height in enumerate(heights):
        while stack and height < heights[stack[-1]]:
            j = stack.pop()
            w = i - stack[-1] + 1 if stack else i
            res = max(res, heights * w)

    return res


from collections import deque, defaultdict


def slide_windows(nums, k):
    s = set()
    queue = deque()
    res = []
    for i, v1 in enumerate(nums):
        if len(queue) == k:
            v2 = queue.popleft()
            res.append(v2)
            s.remove(v2)
        if v1 not in s:
            queue.append(v1)
            s.add(v1)

    for v in queue:
        res.append(v)
    return res


print(slide_windows([1, 2, 3, 1, 2, 3, 4, 5, 1], 4))


# set(i,val) get(i) set_all(val)
# 这三个方法的时间复杂度要求都是O(1)基于数组或者hashmap都可以。
class MyDict():
    def __init__(self):
        self.default_value = -1
        self.dic_v = dict()
        self.dic_t = dict()
        self.i = 0
        self.j = 0

    def get(self, index):
        if index not in self.dic_v:
            return -1
        if self.j > self.dic_t[index]:
            return self.default_value
        return self.dic_v[index]

    def set(self, index, value):
        self.i += 1
        self.dic_v[index] = value
        self.dic_t[index] = self.i

    def set_all(self, value):
        self.j = self.i + 1
        self.default_value = value


dic = MyDict()

print(dic.get(1))

dic.set(1, 2)
dic.set(1, 3)
dic.set(1, 4)

dic.set_all(0)
dic.set(2, 10)
print(dic.get(2))


# 合并数组topK
# [1 3 5| 7]
# [2| 4 6 8]

def find_k(nums1, nums2, k):
    if len(nums1) > len(nums2):
        return find_k(nums2, nums1, k)

    l, r = 0, min(k, len(nums1))
    while l < r:
        mid = l + (r - l + 1) // 2
        j = k - mid
        value = nums2[j] if j < len(nums2) else float('inf')
        if nums1[mid - 1] <= value:  # nums1[l-1] <= nums2[k-l]
            l = mid
        else:
            r = mid - 1
    print(l)
    # 取了l个数
    value = max(
        nums1[l - 1] if 0 <= l - 1 < len(nums1) else -float('inf'),
        nums2[k - l - 1] if 0 <= k - l - 1 < len(nums2) else -float('inf')
    )
    return value


print('*' * 10)
nums1 = [1, 3, 5, 7]
nums2 = [2, 4, 6, 8]
print(find_k(nums1, nums2, k=3))


# 120 triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
def minimum_total(triangle):
    n = len(triangle[-1])
    res = [float('inf')] * n
    res[0] = triangle[0][0]
    for j in range(1, len(triangle)):
        nums = triangle[j]
        n = len(nums)
        # 逆序
        for i in range(n - 1, -1, -1):
            res[i] = min(res[i] if n - 1 > i > -1 else float('inf'),
                         res[i - 1] if n - 1 > i - 1 > -1 else float('inf')) + nums[i]
    return min(res)
