class Node():
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.pre = None
        self.next = None


class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.dic = dict()
        self.head = Node(None, None)
        self.tail = Node(None, None)
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
        nxt = self.head.next
        node.pre = self.head
        node.next = nxt
        self.head.next = node
        nxt.pre = node
        self.dic[key] = node

    def remove(self, key):
        node = self.dic[key]
        pre = node.pre
        nxt = node.next
        pre.next = nxt
        nxt.pre = pre
        del self.dic[key]


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
