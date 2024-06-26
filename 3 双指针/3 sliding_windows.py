from collections import Counter, defaultdict


# 3 最长非重复子串
def length_of_longest_substring(s):
    if not s:
        return 0
    dic = dict()  # 更新l和r窗口之间的信息
    l, r = 0, 0
    res = 0
    for r in range(len(s)):
        c = s[r]
        if c not in dic:
            dic[c] = r
        # left滑出
        elif dic[c] >= l:
            l = dic[c] + 1
            dic[c] = r
        else:
            # 不在窗口内 但需要更新value
            dic[c] = r
        res = max(res, r - l + 1)
    return res


def length_of_longest_substring2(s):
    res = 0
    dic = {}
    l, r = 0, 0  # 记录非重复子串的起始位置
    while r < len(s):
        c = s[r]
        # left滑出窗口
        if c in dic and dic[c] >= l:
            l = dic[c] + 1
        dic[c] = r
        res = max(res, r - l + 1)
        r += 1

    return res


# 340 包含k个不同字符的最大长度
def length_of_longest_substring_k_distinct(s, k):
    if not s or len(s) <= k:
        return len(s)

    dic = defaultdict(int)
    res, n = 0, len(s)
    l, r = 0, 0
    for r in range(len(s)):
        c = s[r]
        dic[c] += 1
        # left滑出窗口
        while len(dic) > k:
            c = s[l]
            dic[c] -= 1
            if not dic[c]:
                del dic[c]  # 用到len 所以需要删除
            l += 1
        # len(dic) == k
        res = max(res, r - l + 1)

    return res


# 76 最小覆盖子串 滑动窗口
def min_window1(s, t):
    dic1 = Counter(t)
    dic2 = defaultdict(int)

    n = len(s)
    min_len = n + 1
    res = ""
    match = 0
    l = 0
    for r in range(n):
        c = s[r]
        if c in dic1:
            dic2[c] += 1
            if dic2[c] == dic1[c]:
                match += 1  # 字母匹配数
        # left滑出窗口
        while match == len(dic1):
            if r - l + 1 < min_len:
                min_len = r - l + 1
                res = s[l:r + 1]
            c = s[l]
            if c in dic1:
                dic2[c] -= 1
                if dic2[c] < dic1[c]:
                    match -= 1
            l += 1
    return res


def min_window2(s, t):
    target = Counter(t)
    cnt = defaultdict(int)

    def check():
        return all(cnt[c] >= target[c] for c in target)  # 替代check函数

    l, r = 0, -1
    len_min = float('inf')
    ans = ""

    while r < len(s) - 1:
        r += 1
        if s[r] in target:
            cnt[s[r]] += 1

        while check() and l <= r:
            if r - l + 1 < len_min:
                len_min = r - l + 1
                ans = s[l:r + 1]
            if s[l] in target:
                cnt[s[l]] -= 1
            l += 1

    return ans


# 567 s1的全排列之一是s2的子串
def check_inclusion(s1, s2):
    if not s1 or not s2 or len(s1) > len(s2):
        return False

    d1 = Counter(s1)

    match = 0
    d2 = defaultdict(int)
    m, n = len(s1), len(s2)
    for i in range(n):
        # 右侧先进
        c = s2[i]
        if c in d1:
            d2[c] += 1
            if d2[c] == d1[c]:
                match += 1

        # 左侧后出 维持长度
        if i >= m:
            c = s2[i - m]
            if c in d1:
                if d1[c] == d2[c]:  # 只有以前match 现在不match 才减去1
                    match -= 1
                d2[c] -= 1
        # 保证长度为m
        if len(d1) == match:
            return True

    return False


# 排序数组中距离不大于target的pair数 O(N)
def no_more_than(nums, target):
    n = len(nums)
    right = 1
    res = 0
    for left in range(n - 1):
        # 一直移动
        while right < n and nums[right] - nums[left] <= target:
            right += 1
        res += right - left - 1
    return res


# 532 相差为K的pair
def find_pairs(nums, k):
    if len(nums) < 2:
        return 0

    nums.sort()

    if nums[-1] - nums[0] < k:
        return 0

    n = len(nums)
    l, r = 0, 1
    res = 0
    while r < n:
        if l == r:  # 可能相遇
            r += 1
            continue
        gap = nums[r] - nums[l]
        # 只移动一步
        if gap < k:
            r += 1
        elif gap > k:
            l += 1  # l 移动了 但是r未移动
        else:
            res += 1
            while r + 1 < n and nums[r] == nums[r + 1]:
                r += 1
            r += 1
    return res


def find_pairs2(nums, k):
    res = 0
    c = Counter(nums)
    for i in c:
        # k>0, 另一个key | k=0 自身
        if (k > 0 and i + k in c) or (k == 0 and c[i] > 1):
            res += 1
    return res


# 和为s的连续正数序列 至少两个数
def find_continuous_sequence(target):
    if target < 3:
        return
    l, r = 1, 2
    total = l + r
    res = []
    while r <= (1 + target) / 2:  # 缩减计算
        if total == target:
            res.append(list(range(l, r + 1)))
            r += 1
            total += r
        elif total < target:
            r += 1
            total += r
        else:
            total -= l
            l += 1
    return res


# 取值为正数的数组 和大于等于s的最短数组
def min_sub_array_len(nums, s):
    if not nums or min(nums) > s:
        return 0
    if max(nums) >= s:
        return 1
    n = len(nums)
    i = 0
    total = 0
    res = n + 1
    for j in range(n):
        total += nums[j]
        while total >= s:
            res = min(res, j - i + 1)  # 注意这个位置!!!
            total -= nums[i]
            i += 1
    return res if res == n + 1 else 0


# 11. 盛最多水的容器
# s = min(h[l], h[r]) * (r-l)
# 关注移动后的短板会不会变长
# 向内移动短板大于等于当前面积;
# 向内移动长板小于当前面积 --这部分可以减枝
# 因此我们选择向内移动短板
def max_area(height):
    res = 0
    l, r = 0, len(height) - 1
    while l < r:
        res = max(res, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return res


# 适合判断多个s是否在t中
def is_subsequence2(s, t):
    dic = defaultdict(list)
    for i, c in enumerate(t):
        dic[c].append(i)

    # print(dic)
    def binary_search(nums, target):
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] <= target:
                l = mid + 1
            else:
                r = mid
        return nums[l] if nums[l] > target else -1

    t = -1
    for c in s:
        if c in dic:
            idx = binary_search(dic[c], t)
            # print(dic[c], c, idx)
            if idx == -1:
                return False
            t = idx
        else:
            return False
    return True


if __name__ == '__main__':
    print('\n最长非重复子串')
    print(length_of_longest_substring("abca"))

    print('\n长度为K的最长不重复子串')
    print(length_of_longest_substring_k_distinct('eceebaaaa', 2))

    print('\n最小覆盖子串')
    print(min_window1('aaabbbbbcdd', 'abcdd'))

    print('\n一个字符串是否包含另外一个字符串的任一全排列')
    s1 = 'trinitrophenylmethylnitramine'
    s2 = 'dinitrophenylhydrazinetrinitrophenylmethylnitramine'
    print(check_inclusion(s1, s2))

    print('\n差不大于k')
    nums = [1, 7, 8, 9, 12]
    print(no_more_than(nums, 1))

    print('\n相差为K的pair数目')
    print(find_pairs([1, 3, 1, 5, 4], 0))
    print(find_pairs2([1, 3, 1, 5, 4], 0))

    print('\n和为S的连续子序列')
    print(find_continuous_sequence(15))
