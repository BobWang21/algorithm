def four_sum(nums, target):
    if not nums or len(nums) < 4:
        return False
    n = len(nums)
    res = []
    nums.sort()
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            new_target = target - nums[i] - nums[j]
            l, r = j + 1, n - 1
            while l < r:
                total = nums[l] + nums[r]
                if total > new_target:
                    r -= 1
                elif total == new_target:
                    res.append([nums[i], nums[j], nums[l], nums[r]])
                    r -= 1
                    l += 1
                else:
                    l += 1

    return res


def two_sum(nums, target):
    if not nums or len(nums) < 2:
        return []
    nums.sort()
    n = len(nums)
    l, r = 0, n - 1
    res = []
    while l < r:
        total = nums[l] + nums[r]
        if total > target:
            r -= 1
        elif total == target:
            res.append([nums[l], nums[r]])
            r -= 1
            l += 1
        else:
            l += 1

    return res


def three_sum_close(nums, target):
    if not nums or len(nums) < 3:
        return []
    gap = float('inf')
    n = len(nums)
    nums.sort()
    res = []
    for i in range(n - 2):
        l, r = i + 1, n - 1
        while l < r:
            total = nums[i] + nums[l] + nums[r]
            new_gap = abs(total - target)
            if total == target:
                res = [nums[i], nums[l], nums[r]]
                return res
            if new_gap < gap:
                gap = new_gap
                res = [nums[i], nums[l], nums[r]]
            if total > target:
                r -= 1
            else:

                l += 1
    return res


# 和为s的(连续)正数序列 至少两个数 滑动窗口
def find_continuous_sequence(target):
    if target < 0:
        return
    n = target // 2 + 1
    res = []
    l, r = 1, 2
    total = 3
    while r <= n:
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


# 最小覆盖子串 滑动窗口
def min_window(s, t):
    if not s or not t:
        return ''
    dic1 = dict()
    for chr in t:
        dic1.setdefault(chr, 0)
        dic1[chr] += 1
    match = 0
    l, r = 0, 0
    dic2 = dict()
    min_len = float('inf')
    res = ''
    while r < len(s):
        c = s[r]
        if c in dic1:
            dic2.setdefault(c, 0)
            dic2[c] += 1
            if dic2[c] == dic1[c]:
                match += 1

            while match == len(dic1):
                if r - l + 1 < min_len:
                    res = s[l:r + 1]
                    min_len = r - l + 1
                c = s[l]
                if c in dic1:
                    if dic1[c] == dic2[c]:
                        match -= 1
                    dic2[c] -= 1
                l += 1
        r += 1
    return res


def check_inclusion(s1, s2):
    if not s1 or not s2 or len(s1) > len(s2):
        return False
    dic1 = dict()
    for c in s1:
        dic1.setdefault(c, 0)
        dic1[c] += 1
    dic2 = dict()
    match = 0
    r = 0
    while r < len(s2):
        if r >= len(s1):  # pop
            c = s2[r - len(s1)]
            if c in dic2:
                if dic2[c] == dic1[c]:
                    match -= 1
                dic2[c] -= 1
        c = s2[r]
        if c in dic1:
            dic2.setdefault(c, 0)
            dic2[c] += 1
            if dic2[c] == dic1[c]:
                match += 1
        if match == len(dic1):
            return True
        r += 1
    return False


def length_of_longest_substring(s):
    if not s:
        return 0
    dic = dict()
    l, r = 0, 0
    res = 0
    while r < len(s):
        c = s[r]
        if c in dic and dic[c] >= l:
            l = dic[c] + 1
        res = max(res, r - l + 1)
        dic[c] = r
        r += 1
    return res


def max_k_char(s, k):
    if not s:
        return 0
    n = len(s)
    dic = {}
    l, r = 0, 0
    match = 0
    res = 0
    while r < n:
        c = s[r]
        if c in dic:
            dic[c] += 1
            res = max(res, r - l + 1)
        else:
            dic[c] = 1
            match += 1
        while match > k:
            c = s[l]
            if c in dic:
                dic[c] -= 1
                if dic[c] == 0:
                    match -= 1
                    del dic[c]
            l += 1

        r += 1
    return res


if __name__ == '__main__':
    print(four_sum([1, 0, -1, 0, -2, 2], 0))
    print(two_sum([1, 0, -1, 0, -2, 2], 0))
    print(three_sum_close([-1, 2, 1, -4], 1))

    print(find_continuous_sequence(15))

    print(min_window('cabwefgewcwaefgcf', 'cae'))

    print('\n一个字符串是否包含另外一个字符串的任一全排列')
    s1 = "ab"
    s2 = "eidbaooo"
    print(check_inclusion(s1, s2))

    print(length_of_longest_substring("abcabcbb"))

    print(max_k_char('eceebaaaa', 2))
