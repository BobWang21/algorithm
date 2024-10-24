#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import heapq as hq
from collections import defaultdict, deque, Counter


# 列表作为函数参数，按地址传递。
# 可以修改列表引用的内容, 不可以更改引用的对象
def change1(nums):
    nums[0] = 100

    nums = [1, 2]


if __name__ == '__main__':
    print('\n计算操作')
    print(4 / 2)  # 2.0,  2.0 = 2, 列表索引必须是整数
    nums = float('inf')
    print(math.isinf(nums))
    print(round(3.1, 1))  # 四舍五入
    print(divmod(6, 4))

    print('\n随机数')
    print(random.random())  # 0-1
    print(random.randint(0, 10))  # [0, 10]
    nums = [1, 2, 3]
    random.shuffle(nums)
    print(nums)
    print(random.choice([1, 2, 3]))  # 随机抽取一个点

    # 进制转换
    print('\n位操作')
    print(bin(8)[2:])  # bin(8) = '0b1000'
    print((2 << 2) - 1)  # 位操作的优先级低于 + -

    print('\n字符串')
    s = 'abc'
    s.isalpha()  # 判断是否为字母
    s.isdigit()  # 判断是否为数字
    s.isalnum()  # 是否为数字或字母
    s.isspace()  # 判断空格
    ''.join(['a', 'b', 'c'])  # 列表转换成字符串 列表元素为字符
    print(ord('a'))  # ordinary字母对应的数字序
    print(chr(97))  # 数字转字符
    print([chr(ord('a') + i) for i in range(26)])
    lists = [c for c in s]  # split 是针对分隔符分割 默认空格
    print(lists)
    print('/a'.split('/'))  # ['', 'a']
    print(int('001'))  # ✔️
    print(eval('1'))  # eval('001') ×

    print('\n列表')
    nums = [2, 3, 4, 5]
    nums.remove(2)  # 移除数值
    nums.append(2)  # 增加一个数
    print('list pop: ', nums.pop())  # 按位置移除, pop(0) -> o(n)
    print('list.index', nums.index(3))  # 查找数据在列表中索引, 返回第一个
    nums.reverse()  # 反转
    nums = [[1, 3], [2, 2], [3, 6]]
    nums.sort(key=lambda x: x[1] - x[0])  # 排序
    print(nums + [4])  # 生成一个新的变量
    print(nums[1:])  # 切片和[:] 生成一个新列表
    print(nums[3:10:2])  # [star, end, step]

    print('\n字典')
    dic = {1: 3}
    print(dic.get(3, -1))  # -1 default value
    for k, v in dic.items():
        print(k, v)
    print(dic.setdefault(1, 0))  # 如果字典中包含key，返回对应value，否则返回为设置value。

    print('\n集合')
    nums = {1, 2, 3}  # set(v) v 需要可迭代
    nums.add(2)
    nums.remove(2)  # 移除元素
    print(nums)
    A = {1, 2, 3}
    B = {3, 4, 5}
    C = A | B  # A与B的并集
    D = A & B  # A与B的交集
    E = A ^ B  # A与B的差集 {1, 2, 4, 5}
    F = A - B  # B在A的补集 {1, 2}
    G = B - A  # A在B的补集 {4, 5}
    print(E, F, G)

    print(isinstance([], list))  # 判断类型
    print(type({}))  # {}表示字典

    print('\ndefaultdict')
    dic = defaultdict(set)
    dic[1].add(2)
    dic = defaultdict(int)
    dic[1] += 5

    print('\nCounter')
    # Counter 是一个字典子类，用于计数可哈希对象
    nums = [2, 3, 3, 5]
    dic = Counter(nums)
    print(dic)

    # 双端队列
    # Python’s deque is a low-level and highly optimized double-ended queue
    # list-like container with fast appends and pops on either end
    print('\n双端队列')
    queue = deque()
    queue.append(1)
    queue.append(3)
    print(queue)
    queue.popleft()  # o(1) vs list o(n)
    queue.pop()  # o(1)
    print(queue)

    print('\nheapq')
    heap = [3, 4, 1, 2]
    hq.heapify(heap)  # 原地修改成堆, 默认小顶堆
    hq.heappush(heap, 5)
    while heap:
        print(hq.heappop(heap))
