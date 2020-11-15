#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 前缀树
class Trie(object):
    def __init__(self):
        self.dic = {}  # 嵌套字典

    def insert(self, word):
        dic = self.dic
        for c in word:
            dic.setdefault(c, {})  # 如果不在字典中 新建一个key
            dic = dic[c]
        dic['#'] = '#'  # 结束标志

    # 是否在树中
    def search(self, word):
        dic = self.dic
        for c in word:
            if c in dic:
                dic = dic[c]
            else:
                return False
        return '#' in dic

    # 是否为树的某段
    def starts_with(self, prefix):
        dic = self.dic
        for c in prefix:
            if c in dic:
                dic = dic[c]
            else:
                return False
        return True


# 212
def find_words(board, words):
    if not words or not board or not board[0]:
        return []
    trie = Trie()
    for word in words:
        trie.insert(word)
    print(trie.dic)
    rows, cols = len(board), len(board[0])

    res = set()

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # ← → ↑ ↓

    def helper(i, j, word):
        if i < 0 or j < 0 or i == rows or j == cols:
            return

        if board[i][j] == '.':  # 访问过
            return

        new_word = word + board[i][j]
        if not trie.starts_with(new_word):
            return

        if trie.search(new_word):
            res.add(new_word)

        char = board[i][j]
        board[i][j] = '.'
        for direction in directions:  # 四个方向独立
            helper(i + direction[0], j + direction[1], new_word)
        board[i][j] = char  # 无论如何都回溯

    for i in range(rows):
        for j in range(cols):
            helper(i, j, '')

    return list(res)


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
