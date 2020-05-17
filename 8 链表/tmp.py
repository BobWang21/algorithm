import heapq as hq


class ListNode():
    def __init__(self, val):
        self.val = val
        self.next = None


def construct_list_node(nums):
    head = ListNode(nums[0])
    current = head
    for i in range(1, len(nums)):
        current.next = ListNode(nums[i])
        current = current.next
    return head


def print_list_node(head):
    res = []
    node = head
    while node:
        res.append(node.val)
        node = node.next
    print(res)


# 链表翻转
def reverse(head):
    if not head:
        return
    pre = None
    while head:
        cur = head.next
        head.next = pre
        pre = head
        head = cur
    return pre


# 1->2->3->4 转换成 2->1->4->3
def swap_pairs(head):
    if not head:
        return
    if not head.next:
        return head
    dummy = pre = ListNode(None)
    while head and head.next:
        curr = head.next.next  # 每次移动两步
        pre.next = head.next
        head.next.next = head
        pre = pre.next.next
        # pre.next = None
        head = curr
    pre.next = head  # head 可能为空 也可能为最后一个节点 最后被覆盖 所以1可以省略
    return dummy.next


# 25
def reverse_k_group(head, k):
    def reverse(head):
        pre = None
        tail = head
        while head:
            nxt = head.next
            head.next = pre
            pre = head
            head = nxt
        return pre, tail

    i = 1
    fast = head
    while fast and i < k:  # i = 1 包含k个节点
        fast = fast.next
        i += 1

    if not fast:
        return head

    nxt = fast.next
    fast.next = None  # 切断
    new_head, tail = reverse(head)
    tail.next = reverse_k_group(nxt, k)
    return new_head


# 例如链表1->2->3->3->4->4->5 处理后为 1->2-3->5
def remove_duplicates(head):
    if not head:
        return
    dummy = pre = ListNode(None)
    while head:
        if head.val == pre.val:
            head = head.next
        else:
            pre.next = head
            head = head.next
            pre = pre.next
            pre.next = None  # 防止尾部有重复
    return dummy.next


# 82 例如链表1->2->3->3->4->4->5 处理后为 2->5
def remove_duplicates2(head):
    if not head:
        return
    if not head.next:
        return head

    dummy = pre = ListNode(-1)

    while head and head.next:
        if head.val != head.next.val:
            pre.next = head
            pre = pre.next
            head = head.next
        else:
            while head and head.next and head.val == head.next.val:
                head = head.next
            head = head.next  # 每次移动一步 出口为head.next为空 或者 head.val != head.next.val
    pre.next = head  # head 可能为空 也可能为尾结点
    return dummy.next


# 链表倒数第N个元素
def tail(head, k):
    if not head:
        return
    fast = head
    while k > 0 and fast:
        fast = fast.next
        k -= 1
    if k > 0:
        return
    slow = head
    while fast:
        fast = fast.next
        slow = slow.next
    return slow.val


# 链表中间的元素
def middle_node(head):
    if not head:
        return
    fast = head
    slow = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    return slow.val


def remove_nth_from_end(head, n):
    if not head:
        return
    fast = head
    while n > 0:
        fast = fast.next
        n -= 1
    slow = head
    pre = None
    while fast:
        fast = fast.next
        pre = slow
        slow = slow.next
    if not pre:
        return head.next
    pre.next = slow.next  # 赋值前需要判断
    return head


# 链表排序 148
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

    if not head:
        return
    if not head.next:
        return head
    fast = slow = head
    pre_slow = None
    while fast and fast.next:
        fast = fast.next.next
        pre_slow = slow
        slow = slow.next

    l1 = head
    l2 = slow
    pre_slow.next = None

    l1 = sort_list(l1)
    l2 = sort_list(l2)

    return merge(l1, l2)


# 合并连个排序链表 归并
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
            curr.next = ListNode(a)
            l1 = l1.next
        else:
            curr.next = ListNode(b)
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


# 合并连个排序链表 归并
def merge_k_sorted_lists1(lists):
    if not lists:
        return
    if len(lists) == 1:
        return lists[0]
    m = len(lists) // 2
    a = merge_k_sorted_lists1(lists[:m])
    b = merge_k_sorted_lists1(lists[m:])

    return merge_two_sorted_lists1(a, b)


def merge_k_sorted_lists2(lists):
    if not lists:
        return
    if len(lists) == 1:
        return lists[0]
    heap = []
    for l in lists:
        if l:
            hq.heappush(heap, (l.val, l))

    dummy = curr = ListNode(-1)
    while heap:
        v, node = hq.heappop(heap)
        curr.next = node
        curr = curr.next
        node = node.next
        curr.next = None
        if node:
            hq.heappush(heap, (node.val, node))
    return dummy.next


def get_intersection_node(headA, headB):
    if not headA or not headB:
        return
    s1, s2 = [], []
    while headA:
        s1.append(headA)
        headA = headA.next
    while headB:
        s2.append(headB)
        headB = headB.next
    res = None
    while s1 and s2:
        v1 = s1.pop(-1)
        v2 = s2.pop(-1)
        if v1 == v2:
            res = v1
        else:
            break
    return res


# a + l + b = b + l + a
# 寻找公共点 和 环 差不多
def get_intersection_node2(headA, headB):
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


# 判断链表存在环以及环的入口 也可以使用set保存已经访问过的节点
def detect_cycle2(head):
    if not head:
        return
    if not head.next:
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


def find_duplicate_num(nums):
    fast = slow = nums[0]
    # 证明有环 快慢两个指针
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if fast == slow:
            break
    # 入口
    ptr1 = nums[0]
    ptr2 = fast
    while ptr1 != ptr2:
        ptr1 = nums[ptr1]
        ptr2 = nums[ptr2]
    return ptr1


def is_palindrome(head):
    if not head:
        return True
    if not head.next:
        return True
    pre = None
    node = head
    while node:
        node.pre = pre
        pre = node
        node = node.next
    while head != pre:
        if head.val != pre.val:
            return False
        head = head.next
        pre = pre.pre
    return True


def is_palindrome2(head):
    if not head:
        return True
    if not head.next:
        return True

    fast = slow = head
    pre_slow = None
    while fast and fast.next:
        fast = fast.next.next
        pre_slow = slow
        slow = slow.next
    pre_slow.next = None
    if fast:
        slow = slow.next

    slow = reverse(slow)
    while head and slow:
        if slow.val != head.val:
            return False
        slow = slow.next
        head = head.next
    return True


def partition(head, x):
    if not head:
        return
    head1 = pre1 = ListNode(-1)
    head2 = pre2 = ListNode(-1)
    while head:
        if head.val < x:
            pre1.next = head
            pre1 = pre1.next
            head = head.next
            pre1.next = None  # 可以省略
        else:
            pre2.next = head
            pre2 = pre2.next
            head = head.next
            pre2.next = None  # 不可以省略
    pre1.next = head2.next
    return head1.next


# 链表奇偶分离
def odd_even_list(head):
    if not head:
        return head

    odd = head
    even = even_head = head.next
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
        curr = head.next
        head.next = None
        if i % 2:
            odd.next = head
            odd = odd.next
        else:
            even.next = head
            even = even.next
        head = curr
        i += 1
    odd.next = head2.next
    return head1.next


# 重复删除链表中连续和为0的节点 head = [1,2,3,-3,-2] -> [1]
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


if __name__ == '__main__':
    print('\n链表翻转')
    head = construct_list_node([1, 3, 5, 7])
    print_list_node(reverse(head))

    print('\n链表倒数第K个节点')
    head = construct_list_node([1, 3, 5, 7])
    print(tail(head, 1))

    head = construct_list_node([1, 3, 5, 7, 9])
    print('\n链表中间节点')
    print(middle_node(head))

    print('\n链表对翻转')
    head = construct_list_node([1, 3, 5, 7])
    print_list_node(swap_pairs(head))

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

    l1 = construct_list_node([1, 3, 5, 7])
    l2 = construct_list_node([2, 4, 6, 8])
    print_list_node(merge_two_sorted_lists3(l1, l2))

    print('\n合并K个排序链表')
    l1 = construct_list_node([1, 3, 5, 7])
    l2 = construct_list_node([2, 4, 6, 8])
    l3 = construct_list_node([10, 11, 12, 13])
    print_list_node(merge_k_sorted_lists1([l1, l2, l3]))

    l1 = construct_list_node([1, 3, 5, 7])
    l2 = construct_list_node([2, 4, 6, 8])
    l3 = construct_list_node([10, 11, 12, 13])
    print_list_node(merge_k_sorted_lists2([l1, l2, l3]))

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
    print(is_palindrome2(l))

    print('\n链表奇偶分离')
    l = construct_list_node([5, 6, 7, 8])
    print_list_node(odd_even_list(l))
