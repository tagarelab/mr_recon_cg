"""
   Name: expr_log_240311_2.py
   Purpose:
   Created on: 3/12/2024
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
from optlib import operators as ops
from sim import masks as mk

# %% read B0
path = 'sim_inputs/magnetData.csv'
intrp_x = 101
intrp_y = 101
intrp_z = 101

xlim = [-120, 120]
ylim = [-120, 120]
zlim = [-120, 120]

X_axis = np.linspace(xlim[0], xlim[1], intrp_x)
Y_axis = np.linspace(ylim[0], ylim[1], intrp_y)
Z_axis = np.linspace(zlim[0], zlim[1], intrp_z)
intrp_pts = algb.gen_interp_pts(X_axis, Y_axis, Z_axis)

B0_raw = mr_io.read_nubo_b0(path=path, intrp_pts=intrp_pts, scale_b0=2.104)
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_raw, axis=0), xlim=xlim, ylim=ylim, zlim=zlim, title='B0 (T)')

# %% read B1
path = 'sim_inputs/'
filename = 'B1_51'
FOV = 0.24
scale = 1

B1_data = mr_io.load_single_mat(name=filename, path=path)['B1']

# Create a 3D grid for the magnetic field data
x_b1_raw = np.linspace(-120, +120, B1_data.shape[1])
y_b1_raw = np.linspace(-120, +120, B1_data.shape[2])
z_b1_raw = np.linspace(-120, +120, B1_data.shape[3])

B1_x = algb.interp_by_pts(scale * B1_data[0, :, :, :], x_b1_raw, y_b1_raw, z_b1_raw, intrp_pts, method='linear')
B1_y = algb.interp_by_pts(scale * B1_data[1, :, :, :], x_b1_raw, y_b1_raw, z_b1_raw, intrp_pts, method='linear')
B1_z = algb.interp_by_pts(scale * B1_data[2, :, :, :], x_b1_raw, y_b1_raw, z_b1_raw, intrp_pts, method='linear')
B1_raw = np.array([B1_x, B1_y, B1_z])

# %% Generate Mask and Phantom
# Breast Mask
R = 60
height = 100
loc = [0, 0, Z_axis[0]]
chest_dim = [200, 200, 30]
coil_tkns = 5
breast_mask = mk.gen_breast_mask(X_axis, Y_axis, Z_axis, R=R, height=height, loc=loc, tkns=coil_tkns,
                                 chest_dim=chest_dim)

# Sphere phantom
phantom = mk.gen_sphere(X_axis, Y_axis, Z_axis, loc=loc + [0, 0, 20], rad=60)
vis.scatter3d(X_axis, Y_axis, Z_axis, phantom, title='Sphere Phantom', mask=phantom, xlim=xlim, ylim=ylim,
              zlim=zlim)
vis.scatter3d(X_axis, Y_axis, Z_axis, phantom, title='Breast Mask', mask=breast_mask, xlim=xlim, ylim=ylim,
              zlim=zlim)

# %% slice selection
# Constants
gamma = 42.58e6  # Hz/T
read_mag = 0.046  # T
DC = 0.09  # percent DC when polarizing
ctr_mag = read_mag  # slice selection gradient
slc_tkns_frq = 10e3 * 2  # Hz
slc_tkns_mag = slc_tkns_frq / gamma  # T

# Polarize
B0_polar = B0_raw * DC  # slice strength

# Slice Selection
slice = acq.slice_select(np.linalg.norm(B0_polar, axis=0), ctr_mag, slc_tkns_mag)

# Cut in SI direction
X_M, Y_M, Z_M = np.meshgrid(X_axis, Y_axis, Z_axis, indexing='ij')
SI_cut = (Y_M > -3) & (Y_M < 3)
LR_cut = (X_M > -3) & (X_M < 3)

# Visualize
vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_polar, axis=0), xlim=xlim, ylim=ylim, zlim=zlim, mask=slice,
              title='B0 (T)')
vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_polar, axis=0), xlim=xlim, ylim=ylim, zlim=zlim, mask=SI_cut,
              title='B0 (T)')

# %% effective B1
B1_eff = np.zeros((3, intrp_x, intrp_y, intrp_z))
for i in range(intrp_x):
    for j in range(intrp_y):
        for k in range(intrp_z):
            B1_eff[:, i, j, k] = acq.B1_effective(B1_raw[:, i, j, k], B0_raw[:, i, j, k])
ratio_perc = np.linalg.norm(B1_eff, axis=0) / np.linalg.norm(B1_raw, axis=0) * 100
vis.scatter3d(X_axis, Y_axis, Z_axis, ratio_perc, xlim=xlim, ylim=ylim, zlim=zlim, mask=breast_mask & SI_cut,
              title='Effective B1 / B1 (%)')
vis.scatter3d(X_axis, Y_axis, Z_axis, ratio_perc, xlim=xlim, ylim=ylim, zlim=zlim, mask=breast_mask & LR_cut,
              title='Effective B1 / B1 (%)')

# %% flip angle
clim = [40, 130]
# B1_eff_amp = np.linalg.norm(B1_eff, axis=0)
# flip_angle_deg = B1_eff_amp / np.mean(B1_eff_amp[slice & breast_mask]) * 90
# vis.scatter3d(X_axis, Y_axis, Z_axis, flip_angle_deg, xlim=xlim, ylim=ylim, zlim=zlim, clim=clim, mask=breast_mask &slice,
#               title='Flip Angle (degree)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, flip_angle_deg, xlim=xlim, ylim=ylim, zlim=zlim, clim=clim, mask=breast_mask &SI_cut,
#               title='Flip Angle (degree)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, flip_angle_deg, xlim=xlim, ylim=ylim, zlim=zlim, clim=clim, mask=breast_mask &LR_cut,
#               title='Flip Angle (degree)')

# %% Select voxels of interest
VOI = breast_mask & slice

# %% Object
phantom_VOI = phantom[VOI]
O = ops.hadamard_op(phantom_VOI)

# %% Polarization
B0_VOI = B0_raw[:, VOI]
P = ops.hadamard_op(B0_VOI)

# sanity check
# B0_VOI_2mat = mk.mask2matrix(B0_VOI, VOI, X_axis, Y_axis, Z_axis)
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_VOI_2mat, axis=0), xlim=xlim, ylim=ylim, zlim=zlim,
#               title='B0 (T)', mask = np.linalg.norm(B0_VOI_2mat, axis=0)>0)
# print(np.array_equal(B0_VOI, B0_VOI_2mat[:, VOI]))

# %% Excitation
t = np.linspace(0, 5, 100)  # ms
B1_VOI = B1_raw[:, VOI]
omega_0 = gamma * np.linalg.norm(B0_VOI, axis=0)

# Relaxation
# flip_angle_deg = B1_eff_amp / np.mean(B1_eff_amp[slice & breast_mask]) * 90

# %%
t = np.linspace(0, 5, 100)  # ms
