import random as rd


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
