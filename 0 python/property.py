# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:14:26 2017

@author: wangbao
"""


class Student(object):
    __slots__ = ('age')  # 限制属性

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
