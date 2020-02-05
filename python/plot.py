#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:17:13 2015

@author: Eddy_zheng
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
x1 = np.arange(-4, 4, 0.25)
y1 = [4 for i in range(32)]
x1, y1 = np.meshgrid(x1, y1)
z1 = 3 / 5 * np.cos(5) * x1 + 5 - 9 / 5 * np.cos(5)
# k1 =
ax.plot_surface(X, Y, Z, rstride=1, cstride=0.5, cmap='rainbow')
# ax.scatter(x1, y1, z1, c='b')
ax.scatter(3, 4, np.sin(5), c='b')
# ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='rainbow')
ax.figure(figuresize=(10, 10, 10))
plt.show()
