class Node():
    def __init__(self, val):
        self.val = val
        self.next = None
        self.pre = None


class LRUCache(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.dic = dict()
        self.node_dic = dict()
        head = Node(None)
        tail = Node(None)
        head.next = tail
        tail.pre = head
        self.head = head
        self.tail = tail

    def get(self, key):
        if key in self.dic:
            self.update(key)
            return self.dic[key]
        else:
            return -1

    def put(self, key, value):
        if key not in self.dic:
            if len(self.dic) == self.capacity:
                self.delete()
            self.add(key)
            self.dic[key] = value
        else:
            self.dic[key] = value
            self.update(key)

    # 增加到头部
    def add(self, key):
        node = Node(key)
        nxt = self.head.next
        node.pre = self.head
        node.next = nxt
        self.head.next = node
        nxt.pre = node
        self.node_dic[key] = node

    # 删除尾部
    def delete(self):
        node = self.tail.pre
        pre = node.pre
        pre.next = self.tail
        self.tail.pre = pre
        key = node.val
        del self.node_dic[key]
        del self.dic[key]

    def update(self, key):
        # 删除原节点
        node = self.node_dic[key]
        pre = node.pre
        nxt = node.next
        pre.next = nxt
        nxt.pre = pre

        # 增加在头部
        self.add(key)


if __name__ == '__main__':
    cache = LRUCache(2)

    cache.put(1, 1)
    cache.put(2, 2)
    print(cache.dic)
    print(cache.node_dic)

    print(cache.get(1))  # returns 1
    print(cache.put(3, 3))  # evicts key2
    print(cache.get(2))  # returns - 1(not found)
    print(cache.put(4, 4))  # evicts key 1
    print(cache.get(1))  # returns - 1(not found)
    print(cache.get(3))  # returns 3
    print(cache.get(4))  # returns 4
