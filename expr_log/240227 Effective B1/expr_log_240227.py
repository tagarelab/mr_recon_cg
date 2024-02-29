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

# # %% parameters
# x = np.array([-3, -4, -2])
# y = np.array([5, 7, -1])
# z = np.array([2, 5, 8])
#
# # %% initialize single B0 and B1
# # B0
# B0 = -z
# # B1
# B1 = x
#
# # %% simulate the effective B1
# B1_eff = acq.B1_effective(B1, B0)
#
# # visualize
# ax = plt.figure().add_subplot(projection='3d')
# ax.quiver(0, 0, 0, B1_eff[0], B1_eff[1], B1_eff[2], color='r', label='Effective B1')
# ax.quiver(0, 0, 0, B0[0], B0[1], B0[2], color='b', label='B0')
# ax.quiver(0, 0, 0, B1[0], B1[1], B1[2], color='g', label='B1')
# ax.set_xlim([-10, 10])
# ax.set_ylim([-10, 10])
# ax.set_zlim([-10, 10])
# ax.legend()
# plt.show()

# %% read B0
path = 'sim_inputs/magnetData.csv'
nubo_b0_raw, b0_X, b0_Y, b0_Z = mr_io.read_nubo_b0(path=path, intrp_x=30, intrp_y=30, intrp_z=30)

vis.scatter3d(b0_X, b0_Y, b0_Z, nubo_b0_raw)

# %% slice selection
# Constants
gamma = 42.58e6  # Hz/T
read_mag = 0.046  # T
DC = 0.1  # percent DC when polarizing
ctr_mag = read_mag  # slice selection gradient
slc_tkns_frq = 100e3 * 2  # Hz
slc_tkns_mag = slc_tkns_frq / gamma  # T

# Adjust nubo_b0
nubo_b0 = nubo_b0_raw * DC  # slice strength

# Call slice_select function
id = acq.slice_select(nubo_b0, ctr_mag, slc_tkns_mag)

vis.scatter3d(b0_X, b0_Y, b0_Z, nubo_b0_raw, mask=id)

# %% read B1
path = 'sim_inputs/'
filename = 'B1_ROI_240mm_240mm_240mm'
intrp_x = 30
intrp_y = 30
intrp_z = 30
FOV = 0.24
scale = 1

# nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp = read_b1(filename=filename, path=path, intrp_x=30, intrp_y=30,
#                                                             intrp_z=30, FOV=0.24)
B1_data = mr_io.load_single_mat(name=filename, path=path)['B1']

# Calculate the magnitude of the magnetic field and apply a scaling factor
nubo_b1 = B1_data
nubo_b1 = scale * nubo_b1

# Create a 3D grid for the magnetic field data
x_M = np.linspace(-0.12, +0.12, 31)
y_M = np.linspace(-0.12, +0.12, 31)
z_M = np.linspace(-0.12, +0.12, 31)

nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp = algb.interp_3dmat(nubo_b1, x_M, y_M, z_M, intrp_x,
                                                                      intrp_y, intrp_z)

# %% plot B1
vis.scatter3d(b1_X_intrp, b1_Y_intrp, b1_Z_intrp, nubo_B1_intrp)
vis.scatter3d(b1_X_intrp, b1_Y_intrp, b1_Z_intrp, nubo_B1_intrp, mask=id)

# %% effective B1
B1_eff = acq.B1_effective(nubo_B1_intrp, nubo_b0)
