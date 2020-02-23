# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:06:16 2016

@author: wangbao
"""


# property 将方法变为属性
# @property 相当于属性访问 @birth.setter 相当于属性访问
class Student(object):
    @property  # 访问
    def birth(self):
        return self._birth

    @birth.setter  # 设置
    def birth(self, value):
        if value <= 2017:
            self._birth = value
        if value > 2017:
            raise ValueError('birth must below 2017')

    @property
    def age(self):  # 只可访问
        return 2017 - self._birth

    def __str__(self):
        return 'Student object (name: %s)' % self.birth


'''
    @age.setter
    def age(self,value):
        self._age=value
'''


class add():
    def __init__(self, a):
        self.a = a

    def __call__(self, b):
        return self.a + b
