#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 208 前缀树
class Trie(object):
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, word):
        node = self  # 当前实例对象的引用赋值给局部变量 node
        for chr in word:
            if chr not in node.children:
                node.children[chr] = Trie()
            node = node.children[chr]
        node.is_end = True

    def search(self, word):
        node = self._find(word)
        return True if node is not None and node.is_end else False

    def startsWith(self, prefix):
        node = self._find(prefix)
        return True if node is not None else False

    def _find(self, prefix):
        node = self
        for chr in prefix:
            if chr in node.children:
                node = node.children[chr]
            else:
                return None
        return node


# 212 单词搜索II hard
# 给定一个mxn二维字符网格board和一个单词(字符串)列表words，
# 返回所有二维网格上的单词。
class Trie():
    def __init__(self):
        self.children = {}
        self.word = None

    def insert(self, word):
        node = self
        for chr in word:
            if chr not in node.children:
                node.children[chr] = Trie()
            node = node.children[chr]
        node.word = word


def find_words(board, words):
    """
    :type board: List[List[str]]
    :type words: List[str]
    :rtype: List[str]
    """

    trie = Trie()
    for word in words:
        trie.insert(word)

    rows, cols = len(board), len(board[0])
    res = []
    seen = [[False] * cols for _ in range(rows)]

    def dfs(trie, i, j):
        chr = board[i][j]
        if chr not in trie.children or seen[i][j]:
            return

        seen[i][j] = True
        trie = trie.children[chr]
        if trie.word:
            res.append(trie.word)
            trie.word = None

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < rows and 0 <= new_j < cols:
                dfs(trie, new_i, new_j)

        seen[i][j] = False

    # 生成字典
    for i in range(rows):
        for j in range(cols):
            dfs(trie, i, j)

    return res


if __name__ == '__main__':
    print('\n单词查找')
    words = ['oath', 'pea', 'eat', 'rain']
    board = [
        ['o', 'a', 'a', 'n'],
        ['e', 't', 'a', 'e'],
        ['i', 'h', 'k', 'r'],
        ['i', 'f', 'l', 'v']
    ]
    print(find_words(board, words))
