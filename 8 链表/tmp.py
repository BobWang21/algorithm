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


def reverse_k_group(head, k):
    def reverse(head):
        if not head:
            return
        pre = None
        tail = head
        while head:
            nxt = head.next
            head.next = pre
            pre = head
            head = nxt
        return pre, tail

    i = k
    fast = head
    while fast and i > 1:  # i = 1 包含k个节点
        fast = fast.next
        i -= 1
    if not fast:
        return head

    nxt = fast.next
    fast.next = None
    new_head, tail = reverse(head)
    tail.next = reverse_k_group(nxt, k)
    return new_head


# 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
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


# 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
# 例如链表1->2->3->3->4->4->5 处理后为 2->5
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
                head = head.next  # 每次移动一步
            head = head.next  # 出口为head.next 为空 或者 head.val!=head.next.val
    pre.next = head
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


def partition(head, x):
    if not head:
        return
    s_pre = small = ListNode(-1)
    b_pre = big = ListNode(-1)
    while head:
        if head.val < x:
            s_pre.next = head
            s_pre = s_pre.next
            head = head.next
            s_pre.next = None
        else:
            b_pre.next = head
            b_pre = b_pre.next
            head = head.next
            b_pre.next = None
    if not small.next:
        return big.next
    if not big.next:
        return small.next
    s_pre.next = big.next
    return small.next


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


def add_two_numbers1(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    add_one = False
    head = l1
    pre = None
    while l1 and l2:
        if add_one:
            l1.val += l2.val + 1
        else:
            l1.val += l2.val
        if l1.val >= 10:
            add_one = True
            l1.val -= 10
        else:
            add_one = False
        pre = l1
        l1 = l1.next
        l2 = l2.next

    if not l1:
        pre.next = l2

    if not add_one:
        return head

    if add_one and not pre.next:
        node = ListNode(1)
        pre.next = node
        return head

    l1 = pre.next
    pre = None
    while add_one and l1:
        l1.val += 1
        if l1.val >= 10:
            add_one = True
            l1.val -= 10
            pre = l1
            l1 = l1.next
        else:
            add_one = False

    if not add_one:
        return head

    if not l1:
        node = ListNode(1)
        pre.next = node
        return head


def add_two_nums2(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    dummy = curr = ListNode(-1)
    carry = 0
    while l1 or l2 or carry:
        if l1:
            carry += l1.val
            l1 = l1.next
        if l2:
            carry += l2.val
            l2 = l2.next
        curr.next = ListNode(carry % 10)
        curr = curr.next
        carry = carry // 10

    return dummy.next


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


# 判断链表存在环以及环的入口
# 也可以使用set保存已经访问过的节点
def detect_cycle2(head):
    if not head:
        return
    if not head.next:
        return False
    fast = slow = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if fast == slow:
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


if __name__ == '__main__':
    head = construct_list_node([1, 3, 5, 7])
    print('链表翻转')
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
    print_list_node(add_two_numbers1(l1, l2))

    print('\n判断是否为回文')
    l = construct_list_node([1, 2])
    print(is_palindrome2(l))
