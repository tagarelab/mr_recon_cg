# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:05:24 2024

@author: Yonghyun
"""

import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler
from scipy.io import loadmat

data = loadmat("B1_51.mat")
B1 = data['B1']

n = 51
x = np.linspace(-0.120, 0.120, n)
z = np.linspace(-0.120, 0.120, n)

mask = np.zeros_like(B1)
R = 0.06
height = 0.100

for i in range(n):
    if x[i] < 0:
        r = np.sqrt(R ** 2 * (1 - (z[i] / height) ** 2)) - 0.005
        for j in range(n):
            for k in range(n):
                if x[j] ** 2 + z[k] ** 2 < r ** 2:
                    mask[:, i, j, k] = np.ones(3)

B1_masked = B1 * mask

fig = plt.figure(figsize=(11, 6))
ax1 = fig.add_subplot(131)

nslice = 25

B_map = ax1.imshow(np.flipud(B1[2, :, nslice, :]), cmap='jet')
ax1.axis('off')

ax2 = fig.add_subplot(132)

mask = ax2.imshow(np.flipud(mask[2, :, nslice, :]), cmap='gray')
ax2.axis('off')

ax3 = fig.add_subplot(133)

mask = ax3.imshow(np.flipud(B1_masked[2, :, nslice, :]), cmap='jet')
ax3.axis('off')
