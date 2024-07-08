import heapq as hq
import random as rd


class ListNode():
    def __init__(self, val):
        self.val = val
        self.next = None


class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random


class NodeX:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random

    def __lt__(self, other):
        return self.val < other.val


def construct_list_node(nums):
    head = ListNode(nums[0])
    cur = head
    for i in range(1, len(nums)):
        cur.next = ListNode(nums[i])
        cur = cur.next
    return head


def print_list_node(head):
    res = []
    node = head
    while node:
        res.append(node.val)
        node = node.next
    print(res)


# 206 链表翻转 递推思想(假设之前的满足要求)
def reverse1(head):
    pre = None
    while head:
        nxt = head.next
        head.next = pre
        pre = head
        head = nxt
    return pre


# 递归1
def reverse2(head):
    if not head or not head.next:
        return head
    new_head = reverse2(head.next)
    head.next.next = head  # head指向new_head的尾结点
    head.next = None
    return new_head


# 递归2
def reverse3(head):
    if not head or not head.next:
        return head, head
    node = head.next
    head.next = None
    new_head, new_tail = reverse3(node)
    # if not new_tail:
    #     new_head.next = head
    # else:
    #     new_tail.next = head
    new_tail.next = head
    return new_head, head


# 24 1->2->3->4 转换成 2->1->4->3
def swap_pairs1(head):
    if not head or head.next:
        return head
    dummy = pre = ListNode(None)
    while head and head.next:
        nxt = head.next.next  # 每次移动两步
        pre.next = head.next
        head.next.next = head
        pre = pre.next.next
        head = nxt
    pre.next = head  # head可能为空或尾结点
    return dummy.next


# 递归
def swap_pairs2(head):
    if not head or not head.next:
        return head
    new_head = head.next
    head.next = swap_pairs2(new_head.next)
    new_head.next = head
    return new_head


# 25 每K个元素反转
def reverse_k_group(head, k):
    def reverse(head):
        pre = None
        while head:
            nxt = head.next
            head.next = pre
            pre = head
            head = nxt
        return pre

    i = 1
    cur = head
    while i < k and cur:  # i = 1 包含k个节点
        i += 1
        cur = cur.next

    if not cur:  # 下面用到cur.next 需要判断cur是否为空
        return head

    nxt = cur.next  #
    cur.next = None  # 切断
    new_head = reverse(head)
    head.next = reverse_k_group(nxt, k)
    return new_head


# 删除节点 可以修改链表val
# 非第一个节点 和 最后一个节点
def delete_node(node):
    node.val = node.next.val
    node.next = node.next.next


# 83 例如链表1->2->3->3->4->4->5 处理后为 1->2-3->5
def remove_duplicates(head):
    if not head:
        return
    dummy = pre = ListNode(None)
    while head:
        if head.val != pre.val:  # 向前看
            pre.next = head
            pre = pre.next
        head = head.next
    pre.next = None  # 截尾
    return dummy.next


# 82 例如链表1->2->3->3->4->4->5 处理后为1->2->5
def remove_duplicates2(head):
    if not head or not head.next:
        return head

    dummy = pre = ListNode(-101)

    while head and head.next:
        if head.val != head.next.val:  # 向后看
            pre.next = head
            pre = pre.next
        else:  # 类似数组去重 移动到下一个节点
            while head and head.next and head.val == head.next.val:
                head = head.next
        head = head.next
    pre.next = head  # head 可能为空或不相同的尾结点
    return dummy.next


# 链表倒数第N个元素
# 前后指针
def tail(head, k):
    if not head:
        return
    first = head
    while k > 0 and first:
        first = first.next
        k -= 1
    if k > 0:
        return
    second = head
    while first:
        first = first.next
        second = second.next
    return second.val


# 19 前后指针
def remove_nth_from_end(head, n):
    if not head:
        return
    first = second = head
    while n > 0:
        first = first.next
        n -= 1

    pre = None
    while first:
        first = first.next
        pre = second
        second = second.next

    if not pre:
        return head.next
    pre.next = second.next  # pre赋值前需要判断
    return head


# 链表中间的元素 快慢指针
def middle_node(head):
    if not head:
        return
    first = head
    second = head
    while first and first.next:
        first = first.next.next
        second = second.next
    return second.val


# 148 链表排序
def sort_list(head):
    def merge(l1, l2):
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val < l2.val:
            head = l1
            l1 = l1.next
        else:
            head = l2
            l2 = l2.next
        head.next = merge(l1, l2)
        return head

    if not head or not head.next:  # 单节点
        return head

    fast = slow = head
    pre_slow = None
    while fast and fast.next:
        fast = fast.next.next
        pre_slow = slow
        slow = slow.next

    pre_slow.next = None  # 一定不为空

    l1 = sort_list(head)
    l2 = sort_list(slow)

    return merge(l1, l2)


# 合并连个排序链表 归并 o(min(m, n))
def merge_two_sorted_lists1(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    dummy = curr = ListNode(-1)
    while l1 and l2:
        if l1.val < l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    if not l1:
        curr.next = l2
    if not l2:
        curr.next = l1
    return dummy.next


# o(m+n)
def merge_two_sorted_lists2(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    if l1.val < l2.val:
        head = l1
        l1 = l1.next
    else:
        head = l2
        l2 = l2.next
    head.next = merge_two_sorted_lists2(l1, l2)
    return head


def merge_two_sorted_lists3(l1, l2):
    dummy = curr = ListNode(None)
    while l1 or l2:
        a = l1.val if l1 else float('inf')
        b = l2.val if l2 else float('inf')
        if a < b:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    return dummy.next


# O(max(m, n)) 最高位位于链表尾
def add_two_nums(l1, l2):
    dummy = pre = ListNode(-1)
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        v = v1 + v2 + carry
        carry = v // 10
        node = ListNode(v % 10)
        pre.next = node
        pre = pre.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next


# 最高位位于链表头
def add_two_nums2(l1, l2):
    def to_stack(l):
        stack = []
        while l:
            stack.append(l)
            l = l.next
        return stack

    stack1 = to_stack(l1)
    stack2 = to_stack(l2)
    nxt = None
    carry = 0
    while stack1 or stack2 or carry:
        v1 = stack1.pop(-1).val if stack1 else 0
        v2 = stack2.pop(-1).val if stack2 else 0
        v = v1 + v2 + carry
        carry = v // 10
        node = ListNode(v % 10)
        node.next = nxt
        nxt = node
    return nxt


# 23 合并连个排序链表 归并
def merge_k_sorted_lists1(lists):
    if not lists:
        return
    if len(lists) == 1:
        return lists[0]
    m = len(lists) // 2
    a = merge_k_sorted_lists1(lists[:m])
    b = merge_k_sorted_lists1(lists[m:])

    return merge_two_sorted_lists1(a, b)


# 23 或者定义node 的 __lt__(m, n) 函数
def merge_k_sorted_lists2(lists):
    if not lists:
        return
    if len(lists) == 1:
        return lists[0]
    heap = []
    for l in lists:
        if l:
            hq.heappush(heap, (l.val, l))  # 加入value为了排序
    dummy = curr = ListNode(-1)
    while heap:
        v, node = hq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            hq.heappush(heap, (node.next.val, node.next))
    return dummy.next


# 23 或者定义node 的 __lt__(m, n) 函数
def merge_k_sorted_lists3(lists):
    def __lt__(self, other):
        return self.val < other.val

    ListNode.__lt__ = __lt__

    if not lists:
        return
    if len(lists) == 1:
        return lists[0]
    heap = []
    for l in lists:
        if l:
            hq.heappush(heap, l)  # 加入value为了排序
    dummy = curr = ListNode(-1)
    while heap:
        node = hq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            hq.heappush(heap, node.next)
    return dummy.next


# a + l + b = b + l + a
# 寻找公共点 和 环 差不多
def get_intersection_node(headA, headB):
    s1, s2 = headA, headB
    while s1 != s2:
        if not s1:
            s1 = headB
        else:
            s1 = s1.next

        if not s2:
            s2 = headA
        else:
            s2 = s2.next
    return s1


# 判断链表存在环以及环的入口
# 不改变数组
def detect_cycle2(head):
    if not head or head.next:
        return False
    fast = slow = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if fast == slow:  # 存在循环
            break
    if fast != slow:  # fast 或 fast.next 为空
        return False
    slow = head
    while slow != fast:
        fast = fast.next
        slow = slow.next
    return slow.val


# 链表是否为回文
def is_palindrome(head):
    if not head or not head.next:
        return True

    slow = fast = head
    pre_slow = None
    while fast and fast.next:
        fast = fast.next.next
        pre_slow = slow
        slow = slow.next

    pre_slow.next = None  # 置空
    if fast:  # 奇数
        slow = slow.next

    slow = reverse1(slow)
    while head and slow:
        if slow.val != head.val:
            return False
        slow = slow.next
        head = head.next
    return True


# 1->2->3->4, 重新排列为 1->4->2->3.
def reorder_list(head):
    if not head or not head.next:
        return head

    fast = slow = pre = head
    while fast and fast.next:
        fast = fast.next.next
        pre = slow
        slow = slow.next
    pre.next = None

    second = reverse1(slow)
    first = head
    dummy = pre = ListNode(-1)
    while first:
        pre.next = first
        curr = first.next
        first.next = second
        pre = pre.next.next
        first = curr
        second = second.next
    return dummy.next


# 86. 分隔链表
def partition(head, x):
    if not head:
        return
    dummy1 = small = ListNode(-1)
    dummy2 = big = ListNode(-1)
    while head:
        if head.val < x:
            small.next = head
            small = small.next
        else:
            big.next = head
            big = big.next
        head = head.next
    if big:
        big.next = None  # 截尾
    small.next = dummy2.next

    return dummy1.next


# 328 链表奇偶分离
def odd_even_list(head):
    if not head:
        return head

    odd = head
    even_head = even = head.next
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = even_head
    return head


# 更加通用
def odd_even_list2(head):
    if not head or not head.next:
        return head

    head1 = odd = ListNode(None)
    head2 = even = ListNode(None)
    i = 1
    while head:
        if i % 2:
            odd.next = head
            odd = odd.next
        else:
            even.next = head
            even = even.next
        head = head.next
        i += 1
    if even:  # 截尾
        even.next = None
    odd.next = head2.next
    return head1.next


# 重复删除链表中连续和为0的节点 head = [1,2,3,-3,-2] -> [1]
# 前缀和
def remove_zero_sum_sublists(head):
    dummy = ListNode(0)
    dummy.next = head
    dic = {}
    total = 0
    node = dummy
    while node:
        total += node.val
        dic[total] = node  # 记录前缀和的最后一个节点
        node = node.next
    # [1 2 3 -3]
    # [1:0, 6:2, 3:3]
    node = dummy
    total = 0
    while node:
        total += node.val
        node.next = dic[total].next
        node = node.next

    return dummy.next


# 随机
def random_node(head):
    if not head:
        return

    i = 1
    while head:
        if rd.randint(1, i) == 1:
            res = head.val
        i += 1
        head = head.next

    return res


# 加1
def plus_one(head):
    if not head:
        return
    dummy = ListNode(0)
    dummy.next = head

    node = dummy
    while node:
        if node.val != 9:
            not_nine = node
        node = node.next
    not_nine.val += 1

    not_nine = not_nine.next
    while not_nine:
        not_nine.val = 0
        not_nine = not_nine.next

    return dummy if dummy.val else dummy.next


# 138. 复制带随机指针的链表
def copy_random_list(head):
    if not head:
        return
    node = head
    while node:  # 老新交替链表 代替dic
        new_node = Node(node.val)
        new_node.next = node.next
        node.next = new_node
        node = node.next.next

    node = head
    while node:
        node.next.random = node.random.next if node.random else None
        node = node.next.next

    dummy = pre = Node(-1)
    node = head
    while node:
        nxt = node.next
        node.next = node.next.next if node.next else None
        pre.next = nxt
        pre = pre.next
        node = node.next

    return dummy.next


def rotate_right(head, k):
    if not head:
        return
    n = 0
    node = head
    tail = head
    while node:
        tail = node
        node = node.next
        n += 1
    tail.next = head
    k = n - k % n - 1  # 新头部为 n - k % n
    new_tail = head
    for _ in range(k):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    return new_head


# 92. 反转链表 II 反转从位置 m 到 n 的链表
def reverse_between(head, m, n):
    if not head or m == n:
        return head
    i = 1
    first_tail = None
    node = head
    while i < m:
        i += 1
        first_tail = node
        node = node.next

    second_tail = node
    pre = None
    while i <= n:
        i += 1
        nxt = node.next
        node.next = pre
        pre = node
        node = nxt
    if first_tail:  # m == 1
        first_tail.next = pre
    else:
        head = pre
    second_tail.next = node
    return head


if __name__ == '__main__':
    print('\n链表翻转')
    head = construct_list_node([1, 3, 5, 7])
    print_list_node(reverse1(head))

    head = construct_list_node([1, 3, 5])
    print_list_node(reverse2(head))

    head = construct_list_node([1, 3, 5, 7])
    print_list_node(reverse3(head)[0])

    print('\n链表倒数第K个节点')
    head = construct_list_node([1, 3, 5, 7])
    print(tail(head, 1))

    head = construct_list_node([1, 3, 5, 7, 9])
    print('\n链表中间节点')
    print(middle_node(head))

    print('\n链表对翻转')
    head = construct_list_node([1, 3, 5, 7])
    print_list_node(swap_pairs1(head))

    print('\nreverse k group')
    head = construct_list_node(list(range(10)))
    print_list_node(reverse_k_group(head, 5))

    print('\npartition')
    head = construct_list_node([1, 4, 3, 2, 5, 2])
    print_list_node(partition(head, 4))

    print('\n链表排序')
    l1 = construct_list_node([8, 1, 3, 7])
    print_list_node(sort_list(l1))

    print('\n合并两个排序链表')
    l1 = construct_list_node([1, 3, 5, 7])
    l2 = construct_list_node([2, 4, 6, 8])
    print_list_node(merge_two_sorted_lists2(l1, l2))

    print('\n合并K个排序链表')
    l1 = construct_list_node([1, 3, 5, 7])
    l2 = construct_list_node([2, 4, 6, 8])
    l3 = construct_list_node([10, 11, 12, 13])
    print_list_node(merge_k_sorted_lists1([l1, l2, l3]))

    l1 = construct_list_node([1, 3, 5, 7])
    l2 = construct_list_node([2, 4, 6, 8])
    l3 = construct_list_node([10, 11, 12, 13])
    print_list_node(merge_k_sorted_lists2([l1, l2, l3]))

    l1 = construct_list_node([1, 3, 5, 7])
    l2 = construct_list_node([2, 4, 6, 8])
    l3 = construct_list_node([10, 11, 12, 13])
    print_list_node(merge_k_sorted_lists3([l1, l2, l3]))

    print('\n删除链表中重复元素')
    head = construct_list_node([1, 1, 2, 3, 3])
    print_list_node(remove_duplicates(head))

    print('\n删除链表中重复元素')
    head = construct_list_node([1, 1, 2, 3, 3])
    print_list_node(remove_duplicates2(head))

    print('\n链表是否存在环')
    a = ListNode(1)
    b = ListNode(2)
    c = ListNode(3)
    a.next = b
    b.next = c
    c.next = b
    head = a
    print(detect_cycle2(head))

    l1 = construct_list_node([2, 4])
    l2 = construct_list_node([5, 6, 9, 9])
    print_list_node(add_two_nums(l1, l2))

    print('\n判断是否为回文')
    l = construct_list_node([1, 2])
    print(is_palindrome(l))

    print('\n链表排序')
    l = construct_list_node([5, 6, 7, 8])
    print_list_node(reorder_list(l))

    print('\n链表奇偶分离')
    l = construct_list_node([5, 6, 7, 8])
    print_list_node(odd_even_list(l))

    head = construct_list_node([1, 3, 5, 7])
    print(random_node(head))

    print('\n链表右移动K位')
    l = construct_list_node([5, 6, 7, 8, 9, 10])
    print_list_node(rotate_right(l, 2))
