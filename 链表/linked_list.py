#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 21:50:40 2017

@author: wangbao
"""


# 翻转/合并/环

class Node(object):
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkList(object):
    def __init__(self, value=None):
        if value:
            self.head = Node(value)
            self.tail = self.head
        else:
            self.head = None
            self.tail = None

    def append(self, value):
        if self.tail:
            new = Node(value)
            self.tail.next = new
            self.tail = new
        else:
            self.head = Node(value)
            self.tail = self.head

    def __iter__(self):
        cursor = self.head
        while cursor:
            yield cursor.value
            cursor = cursor.next

    def last_k(self, k):
        if k == 0 or self.head is None:
            return False

        first_cursor = self.head
        i = 1
        while first_cursor and i < k:
            first_cursor = first_cursor.next
            i += 1
        if i < k:
            return False

        second_cursor = self.head
        # 若next为空, 则为末节点
        while first_cursor.next:
            first_cursor = first_cursor.next
            second_cursor = second_cursor.next
        return second_cursor.value

    def mid(self):
        first_cursor = self.head
        second_cursor = self.head

        # 可能有奇数个, 也可能偶数个
        while first_cursor.next and first_cursor.next.next:
            first_cursor = first_cursor.next.next
            second_cursor = second_cursor.next
        return second_cursor.value

    def reverse(self):
        if self.head is None:
            return False

        now = self.head
        self.head, self.tail = self.tail, self.head
        after = now.next
        if after is None:
            return
        now.next = None
        feature = after.next
        after.next = now
        now = after
        while feature:
            after = feature
            feature = after.next
            after.next = now
            # now.next = None
            if feature:
                now = after
            else:
                break


def merge_linked_list(a, b):
    if a.head is None:
        return b
    else:
        if b.head is None:
            return a
    new = LinkList()
    # 头
    if a.head.value < b.head.value:
        new.head = a.head
        a.head = a.head.next
    else:
        new.head = b.head
        b.head = b.head.next
    cursor = new.head

    while a.head and b.head:
        if a.head.value < b.head.value:
            cursor.next = a.head
            a.head = a.head.next
        else:
            cursor.next = b.head
            b.head = b.head.next
        cursor = cursor.next

    if a.head is None:
        cursor.next = b.head
        new.tail = b.tail
    if b.head is None:
        cursor.next = a.head
        new.tail = a.tail
    return new





if __name__ == '__main__':
    a = [1, 4, 19, 30]
    link_a = LinkList()
    for value in a:
        link_a.append(value)
    for value in link_a:
        print(value, end=' ')
    print()
    #    link.reverse()
    #    for value in link:
    #        print(value, end=' ')
    print()
    print(link_a.mid())

    b = [2, 3, 21]
    link_b = LinkList()
    for value in b:
        link_b.append(value)
    for value in link_b:
        print(value, end=' ')
    print()
    merge = merge_linked_list(link_a, link_b)
    for value in merge:
        print(value, end=' ')
