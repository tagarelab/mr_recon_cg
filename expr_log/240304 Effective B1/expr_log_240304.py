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
import pandas as pd

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
intrp_x = 31
intrp_y = 31
intrp_z = 31

# b0_X = np.linspace(-0.2, 0.2, intrp_x)
# b0_Y = np.linspace(-0.2, 0.2, intrp_y)
# b0_Z = np.linspace(-0.2, 0.2, intrp_z)
b0_X = np.linspace(-120, 120, intrp_x)
b0_Y = np.linspace(-120, 120, intrp_y)
b0_Z = np.linspace(-120, 120, intrp_z)

# Read the data
B0_data = pd.read_csv(path, header=None)

FOV = 0.4
scale_b0 = 2.104

# Extract coordinates and magnetic field components
# B0_X_coord = B0_data.iloc[:, 0].values * 0.0002
# B0_Y_coord = B0_data.iloc[:, 1].values * 0.0002
# B0_Z_coord = B0_data.iloc[:, 2].values * 0.0002
B0_X_coord = B0_data.iloc[:, 0].values * 2  # mm
B0_Y_coord = B0_data.iloc[:, 1].values * 2  # mm
B0_Z_coord = B0_data.iloc[:, 2].values * 2  # mm
# B0_X_coord = B0_data.iloc[:, 0].values
# B0_Y_coord = B0_data.iloc[:, 1].values
# B0_Z_coord = B0_data.iloc[:, 2].values
# X_data = B0_data.iloc[:, 3].values
# Y_data = B0_data.iloc[:, 4].values
# Z_data = B0_data.iloc[:, 5].values

B0_intrp = np.zeros((3, intrp_x, intrp_y, intrp_z))
b0_X_intrp = np.zeros((intrp_x, intrp_y, intrp_z))
b0_Y_intrp = np.zeros((intrp_x, intrp_y, intrp_z))
b0_Z_intrp = np.zeros((intrp_x, intrp_y, intrp_z))
intrp_pts = algb.gen_interp_pts(b0_X, b0_Y, b0_Z)

for i in range(3):
    nubo_b0 = B0_data.iloc[:, 3 + i].values
    nubo_b0 = scale_b0 * nubo_b0
    nubo_b0_mesh, x_M, y_M, z_M = algb.vec2mesh(nubo_b0, B0_X_coord, B0_Y_coord, B0_Z_coord, 11, 11, 11)
    # vis.scatter3d(x_M, y_M, z_M, nubo_b0_mesh)
    nubo_b0_mesh = nubo_b0_mesh.T
    B0_intrp[i, :, :, :] = algb.interp_by_pts(nubo_b0_mesh, x_M, y_M, z_M, intrp_pts, method='linear')

nubo_b0_raw = B0_intrp
# nubo_b0_raw, b0_X, b0_Y, b0_Z = mr_io.read_nubo_b0(path=path, intrp_x=intrp_x, intrp_y=intrp_y, intrp_z=intrp_z)
nubo_b0_amp = np.linalg.norm(nubo_b0_raw, axis=0)
vis.scatter3d(b0_X, b0_Y, b0_Z, nubo_b0_amp)

# %% slice selection
# Constants
gamma = 42.58e6  # Hz/T
read_mag = 0.046  # T
DC = 0.1  # percent DC when polarizing
ctr_mag = read_mag  # slice selection gradient
slc_tkns_frq = 10e3 * 2  # Hz
slc_tkns_mag = slc_tkns_frq / gamma  # T

# Adjust nubo_b0
nubo_b0 = nubo_b0_raw * DC  # slice strength

# Call slice_select function
id = acq.slice_select(nubo_b0, ctr_mag, slc_tkns_mag)

vis.scatter3d(b0_X, b0_Y, b0_Z, nubo_b0_amp, mask=id, title='B0 (T)')

# %% read B1
path = 'sim_inputs/'
filename = 'B1_ROI_240mm_240mm_240mm'
FOV = 0.24
scale = 1
x_shfit = 60

# Create a 3D grid for the magnetic field data
x_b1_raw = np.linspace(-120, +120, 31) + x_shfit
y_b1_raw = np.linspace(-120, +120, 31)
z_b1_raw = np.linspace(-120, +120, 31)

B1_data = mr_io.load_single_mat(name=filename, path=path)['B1']
B1_intrp = np.zeros((3, intrp_x, intrp_y, intrp_z))

for i in [0, 1, 2]:
    # Calculate the magnitude of the magnetic field and apply a scaling factor
    nubo_b1 = B1_data[i, :, :, :]
    nubo_b1 = scale * nubo_b1

    B1_intrp[i, :, :, :] = algb.interp_by_pts(nubo_b1, x_b1_raw, y_b1_raw, z_b1_raw, intrp_pts, method='linear')

# %% plot B1
B1_intrp_amp = np.linalg.norm(B1_intrp, axis=0)
B1_mask = (B1_intrp_amp > 0.002)

vis.scatter3d(b0_X, b0_Y, b0_Z, B1_intrp_amp, title='B1 (T)')
vis.scatter3d(b0_X, b0_Y, b0_Z, B1_intrp_amp, mask=id & B1_mask, title='B1 (T)')
vis.scatter3d(b0_X, b0_Y, b0_Z, B1_intrp_amp, mask=B1_mask, title='B1 (T)')

# %% effective B1
B1_eff = np.zeros((3, intrp_x, intrp_y, intrp_z))
for i in range(intrp_x):
    for j in range(intrp_y):
        for k in range(intrp_z):
            B1_eff[:, i, j, k] = acq.B1_effective(B1_intrp[:, i, j, k], B0_intrp[:, i, j, k])
B1_eff_amp = np.linalg.norm(B1_eff, axis=0)
vis.scatter3d(b0_X, b0_Y, b0_Z, B1_eff_amp)
vis.scatter3d(b0_X, b0_Y, b0_Z, B1_eff_amp, mask=id & B1_mask)
vis.scatter3d(b0_X, b0_Y, b0_Z, B1_eff_amp, mask=B1_mask)
ratio_perc = B1_eff_amp / B1_intrp_amp * 100
vis.scatter3d(b0_X, b0_Y, b0_Z, ratio_perc, title='Effective B1 / B1 (%)')
vis.scatter3d(b0_X, b0_Y, b0_Z, ratio_perc, mask=id & B1_mask, title='Effective B1 / B1 (%)')
vis.scatter3d(b0_X, b0_Y, b0_Z, ratio_perc, mask=B1_mask, title='Effective B1 / B1 (%)')

# %% flip angle
flip_angle_deg = B1_eff_amp / np.max(B1_eff_amp[id & B1_mask]) * 90
vis.scatter3d(b0_X, b0_Y, b0_Z, flip_angle_deg, title='Flip Angle (degree)')
vis.scatter3d(b0_X, b0_Y, b0_Z, flip_angle_deg, mask=id & B1_mask, title='Flip Angle (degree)')
vis.scatter3d(b0_X, b0_Y, b0_Z, flip_angle_deg, mask=B1_mask, title='Flip Angle (degree)')
