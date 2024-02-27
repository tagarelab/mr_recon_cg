"""
   Name: expr_log_240227.py
   Purpose:
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import gen_op
from sim import acquisition as acq
import mr_io
import numpy as np
import matplotlib.pyplot as plt
import visualization as vis
import algebra as algb

# %% parameters
x = np.array([-3, -4, -2])
y = np.array([5, 7, -1])
z = np.array([2, 5, 8])

# %% initialize single B0 and B1
# B0
B0 = -z
# B1
B1 = x

# %% simulate the effective B1
B1_eff = acq.B1_effective(B1, B0)

# visualize
ax = plt.figure().add_subplot(projection='3d')
ax.quiver(0, 0, 0, B1_eff[0], B1_eff[1], B1_eff[2], color='r', label='Effective B1')
ax.quiver(0, 0, 0, B0[0], B0[1], B0[2], color='b', label='B0')
ax.quiver(0, 0, 0, B1[0], B1[1], B1[2], color='g', label='B1')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])
ax.legend()
plt.show()
