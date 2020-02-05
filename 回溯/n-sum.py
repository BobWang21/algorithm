#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:57:10 2019

@author: wangbao
"""


# dfs
class Solution(object):
    def combinationSum(self, candidates, target):
        candidates.sort()
        path = []
        self.dfs(candidates, target, 0, path, [])

        return path

    def dfs(self, candidates, target, i, res, path):
        if target < 0:
            return
        if target == 0:
            res.append(path)

        size = len(candidates)
        for i in range(i, size):
            if candidates[i] > target:
                return
            self.dfs(candidates, target - candidates[i], i, res, path + [candidates[i]])


class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, res, [])
        return res

    def dfs(self, nums, target, index, res, path):
        if target < 0:  # base 1
            return
        elif target == 0:  # base2
            res.append(path)
            return
        for i in range(index, len(nums)):
            if nums[index] > target:  # base3
                return
            self.dfs(nums, target - nums[i], i, res, path + [nums[i]])


# 回溯
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, res, [])
        return res

    def dfs(self, nums, target, index, res, path):
        if target < 0:  # base 1
            return
        elif target == 0:  # base2
            res.append(path)
            return
        for i in range(index, len(nums)):
            if nums[index] > target:  # 最小值大于 base3
                return
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            new_path = path[:]
            self.dfs(nums, target - nums[i], i, res, new_path)
            path.pop()


def combinationSum2(candidates, target):
    # Sorting is really helpful, se we can avoid over counting easily
    candidates.sort()
    result = []
    combine_sum_2(candidates, 0, [], result, target)
    return result


def combine_sum_2(nums, start, path, result, target):
    # Base case: if the sum of the path satisfies the target, we will consider 
    # it as a solution, and stop there
    if not target:
        result.append(path)
        return

    for i in range(start, len(nums)):
        # Very important here! We don't use `i > 0` because we always want 
        # to count the first element in this recursive step even if it is the same 
        # as one before. To avoid overcounting, we just ignore the duplicates
        # after the first element.
        if i > start and nums[i] == nums[i - 1]:
            continue

        # If the current element is bigger than the assigned target, there is 
        # no need to keep searching, since all the numbers are positive
        if nums[i] > target:
            break

        # We change the start to `i + 1` because one element only could
        # be used once
        combine_sum_2(nums, i + 1, path + [nums[i]],
                      result, target - nums[i])



# Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.
# S = [1, 0, -1, 0, -2, 2], and target = 0.
def sum_4(data, target):
    res = []
    size = len(data)
    data.sort()
    if size < 4:
        return res
    dfs(data, target, res, 0, [], 4)

    return res


def dfs(data, target, res, index, path, k):
    if k == 0 and target == 0:
        res.append(path)
        return
    if k == 0 and target != 0:
        return
    for i in range(index, len(data)):
        dfs(data, target - data[i], res, i + 1, path + [data[i]], k - 1)
