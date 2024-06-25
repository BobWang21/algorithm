from collections import defaultdict


# 3 最长非重复子串的长度
def length_of_longest_substring(s):
    if not s:
        return 0
    dic = dict()
    l, r = 0, 0
    res = 0
    for r in range(len(s)):
        c = s[r]
        if c not in dic:
            dic[c] = r
        elif dic[c] >= l:
            l = dic[c] + 1
            dic[c] = r
        else:
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
                del dic[c]
            l += 1
        # len(dic) <= k
        res = max(res, r - l + 1)

    return res


# 76 最小覆盖子串 滑动窗口
def min_window(s, t):
    dic1 = defaultdict(int)
    dic2 = defaultdict(int)
    for c in t:
        dic1[c] += 1

    l = 0
    start, end = 0, len(s)
    match = 0
    for r in range(len(s)):
        c = s[r]
        if c in dic1:
            dic2[c] += 1
            if dic2[c] == dic1[c]:
                match += 1  # 字母匹配数
        # left滑出窗口
        while match == len(dic1):
            if r - l < end - start:
                start, end = l, r
            c = s[l]
            if c in dic1:
                dic2[c] -= 1
                if dic2[c] < dic1[c]:
                    match -= 1
            l += 1
    return '' if end == len(s) else s[start:end + 1]


# 567 s1的全排列之一是s2的子串
def check_inclusion(s1, s2):
    if not s1 or not s2 or len(s1) > len(s2):
        return False

    d1 = defaultdict(int)
    for c in s1:
        d1[c] += 1

    match = 0
    d2 = defaultdict(int)
    m, n = len(s1), len(s2)
    for i in range(n):
        c = s2[i]
        if c in d1:
            d2[c] += 1
            if d2[c] == d1[c]:
                match += 1
        # 左侧滑出窗口, 保持长度
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


# 前后指针
# 排序数组中距离不大于target的pair数 O(N)
def no_more_than(nums, target):
    n = len(nums)
    right = 1
    res = 0
    for left in range(n - 1):
        while right < n and nums[right] - nums[left] <= target:
            right += 1
        res += right - left - 1
    return res


# 和为s的连续正数序列 至少两个数
def find_continuous_sequence(target):
    if target < 3:
        return
    l, r = 1, 2
    s = l + r
    res = []
    while r <= (1 + target) / 2:  # 缩减计算
        if s == target:
            res.append(list(range(l, r + 1)))
            r += 1
            s += r
        elif s < target:
            r += 1
            s += r
        else:
            s -= l
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


if __name__ == '__main__':
    print('\n长度为K的最长不重复子串')
    print(length_of_longest_substring_k_distinct('eceebaaaa', 2))

    print('\n最小覆盖子串')
    print(min_window('aaaaaaaaaaaabbbbbcdd', 'abcdd'))

    print('\n最小非重复子串')
    print(length_of_longest_substring("abca"))

    print('\n和为S的连续子序列')
    print(find_continuous_sequence(15))

    print('\n差不大于k')
    nums = [1, 7, 8, 9, 12]
    print(no_more_than(nums, 1))

    print('\n一个字符串是否包含另外一个字符串的任一全排列')
    s1 = 'trinitrophenylmethylnitramine'
    s2 = 'dinitrophenylhydrazinetrinitrophenylmethylnitramine'
    print(check_inclusion(s1, s2))
