import random as rd


class Node():
    def __init__(self, key=None, val=None, pre=None, next=None):
        self.key = key
        self.val = val
        self.pre = pre
        self.next = next


# 707 双向链表
class ListNode:

    def __init__(self, x):
        self.val = x
        self.next = None
        self.prev = None


class MyLinkedList:

    def __init__(self):
        self.size = 0
        self.head = ListNode(0)
        self.tail = ListNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, index):
        if index < 0 or index >= self.size:
            return -1
        # 从前向后扫描
        if index + 1 < self.size - index:
            curr = self.head
            for _ in range(index + 1):
                curr = curr.next
        # 从后向前扫描
        else:
            curr = self.tail
            for _ in range(self.size - index):
                curr = curr.prev
        return curr.val

    def addAtHead(self, val):
        self.addAtIndex(0, val)

    def addAtTail(self, val):
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index, val):
        if index > self.size:
            return
        index = max(0, index)
        if index < self.size - index:
            pred = self.head
            for _ in range(index):
                pred = pred.next
            succ = pred.next
        else:
            succ = self.tail
            for _ in range(self.size - index):
                succ = succ.prev
            pred = succ.prev
        self.size += 1
        to_add = ListNode(val)
        to_add.prev = pred
        to_add.next = succ
        pred.next = to_add
        succ.prev = to_add

    def deleteAtIndex(self, index):
        if index < 0 or index >= self.size:
            return
        if index < self.size - index:
            pred = self.head
            for _ in range(index):
                pred = pred.next
            succ = pred.next.next
        else:
            succ = self.tail
            for _ in range(self.size - index - 1):
                succ = succ.prev
            pred = succ.prev.prev
        self.size -= 1
        pred.next = succ
        succ.prev = pred


# 146 LRU (最近最少使用) 缓存
# 使用包括插入和查询
class LRUCache(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.dic = dict()
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.dic:
            node = self.dic[key]
            val = node.val
            self.remove(key)
            self.add(key, val)
            return val
        else:
            return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.dic:
            self.remove(key)
            self.add(key, value)
        elif len(self.dic) < self.capacity:
            self.add(key, value)
        else:
            node = self.tail.pre
            self.remove(node.key)
            self.add(key, value)

    def add(self, key, val):
        node = Node(key, val)
        succ = self.head.next
        node.pre = self.head
        node.next = succ
        self.head.next = node
        succ.pre = node
        self.dic[key] = node

    def remove(self, key):
        node = self.dic[key]
        pre = node.pre
        succ = node.next
        pre.next = succ
        succ.pre = pre
        del self.dic[key]


# o(1) 删除数组元素
class RandomizedSet(object):
    def __init__(self):
        self.array = []
        self.dic = {}

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.dic:
            return False
        n = len(self.array)
        self.array.append(val)
        self.dic[val] = n
        return True

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.dic:
            idx = self.dic[val]
            n = len(self.array)
            self.array[n - 1], self.array[idx] = self.array[idx], self.array[n - 1]
            self.dic[self.array[idx]] = idx
            del self.dic[val]
            self.array.pop(-1)
            return True
        else:
            return False

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        n = len(self.array)
        idx = rd.randint(0, n - 1)
        return self.array[idx]


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

if __name__ == '__main__':
    cache = LRUCache(2)

    cache.put(1, 1)
    cache.put(2, 2)
    print(cache.dic)

    print(cache.get(1))  # returns 1
    print(cache.put(3, 3))  # evicts key2
    print(cache.get(2))  # returns - 1(not found)
    print(cache.put(4, 4))  # evicts key 1
    print(cache.get(1))  # returns - 1(not found)
    print(cache.get(3))  # returns 3
    print(cache.get(4))  # returns 4
