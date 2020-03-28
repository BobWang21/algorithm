class ListNode():
    def __init__(self, val):
        self.val = val
        self.next = None


def construct_listnode(nums):
    head = ListNode(nums[0])
    current = head
    for i in range(1, len(nums)):
        current.next = ListNode(nums[i])
        current = current.next
    return head


def print_listnode(head):
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
    new_head = ListNode(-1)
    tail = new_head
    while head and head.next:
        cur = head.next.next
        tail.next = head.next
        tail.next.next = head
        head.next = None
        tail = tail.next.next
        head = cur

    tail.next = head
    return new_head.next


# 链表倒数第N个元素
def tail(head, k):
    if not head:
        return ()
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


def remove_nth_from_end(head, n):
    if not head:
        return head
    fast = head
    while fast and n > 0:
        fast = fast.next
        n -= 1
    if n > 0:
        return
    pre = None
    slow = head
    while fast:
        fast = fast.next
        pre = slow
        slow = slow.next
    if not pre:  # 恰好为头节点
        return head.next
    else:
        pre.next = slow.next
        return head


# 合并连个排序链表 归并
def merge_two_sorted_lists(l1, l2):
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
    head.next = merge_two_sorted_lists(l1, l2)
    return head


# 合并连个排序链表 归并
def merge_two_sorted_lists2(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    head = ListNode(-1)
    cur = head
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    if not l1:
        cur.next = l2
    if not l2:
        cur.next = l1
    return head.next


def merge(a, b):
    if not a:
        return b
    if not b:
        return a
    node = ListNode(0)
    if a.val < b.val:
        node.val = a.val
        node.next = merge(a.next, b)
    else:
        node.val = b.val
        node.next = merge(a, b.next)
    return node


def merge_k_sorted_lists2(lists):
    if not lists:
        return
    if len(lists) == 1:
        return lists[0]
    m = len(lists) // 2
    a = merge_k_sorted_lists2(lists[:m])
    b = merge_k_sorted_lists2(lists[m:])

    return merge(a, b)


# 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
# 例如链表1->2->3->3->4->4->5 处理后为 1->2->5
def remove_duplicates(head):
    if not head:
        return
    current = head
    while current and current.next:  # 滑动窗口 窗口的大小为2
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next
    return head


# 最近公共节点
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


# 判断链表存在环
def detect_cycle(head):
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
    if fast != slow:
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


if __name__ == '__main__':
    head = construct_listnode([1, 3, 5, 7])
    print('链表翻转')
    print_listnode(reverse(head))

    print('\n链表倒数第K个节点')
    print(tail(head, 1))

    print('\n链表对翻转')
    head = construct_listnode([1, 3, 5, 7])
    print_listnode(swap_pairs(head))

    print('\n合并两个排序链表')
    l1 = construct_listnode([1, 3, 5, 7])
    l2 = construct_listnode([2, 4, 6, 8])
    print_listnode(merge_two_sorted_lists(l1, l2))

    print('\n合并K个排序链表')
    l1 = construct_listnode([1, 3, 5, 7])
    l2 = construct_listnode([2, 4, 6, 8])
    l3 = construct_listnode([10, 11, 12, 13])
    print_listnode(merge_k_sorted_lists2([l1, l2, l3]))

    print('\n删除链表中重复元素')
    head = construct_listnode([1, 1, 2, 3, 3])
    print_listnode(remove_duplicates(head))

    print('\n链表是否存在环')
    a = ListNode(1)
    b = ListNode(2)
    c = ListNode(3)
    a.next = b
    b.next = c
    c.next = b
    head = a
    print(detect_cycle(head))
