#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:20:35 2017

@author: wangbao
"""

from collections import deque

data = [i for i in range(99999999)]

# data.insert(0, 4585)

deq = deque(data)

deq.appendleft(4585)
