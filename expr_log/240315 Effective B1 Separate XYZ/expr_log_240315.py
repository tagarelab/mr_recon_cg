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
import pandas as pd
from sim import masks as mk


# %% read B0
path = 'sim_inputs/magnetData.csv'
intrp_x = 51
intrp_y = 51
intrp_z = 51

xlim = [-120, 120]
ylim = [-120, 120]
zlim = [-120, 120]

X_axis = np.linspace(xlim[0], xlim[1], intrp_x)
Y_axis = np.linspace(ylim[0], ylim[1], intrp_y)
Z_axis = np.linspace(zlim[0], zlim[1], intrp_z)

# Read the data
B0_data = pd.read_csv(path, header=None)
FOV = 0.4
scale_b0 = 2.104

# Extract coordinates and magnetic field components
X_axis_coord = B0_data.iloc[:, 0].values * 2  # mm
Y_axis_coord = B0_data.iloc[:, 1].values * 2  # mm
Z_axis_coord = B0_data.iloc[:, 2].values * 2  # mm

B0_intrp = np.zeros((intrp_x, intrp_y, intrp_z, 3))
B0_x = np.zeros((intrp_x, intrp_y, intrp_z))
B0_y = np.zeros((intrp_x, intrp_y, intrp_z))
B0_z = np.zeros((intrp_x, intrp_y, intrp_z))
intrp_pts = algb.gen_interp_pts(X_axis, Y_axis, Z_axis)

nubo_b0 = B0_data.iloc[:, 3].values
nubo_b0 = scale_b0 * nubo_b0
nubo_b0_mesh, x_M, y_M, z_M = algb.vec2mesh(nubo_b0, X_axis_coord, Y_axis_coord, Z_axis_coord, 11, 11, 11)
nubo_b0_mesh = nubo_b0_mesh.T
B0_x = algb.interp_by_pts(nubo_b0_mesh, x_M, y_M, z_M, intrp_pts, method='linear')

nubo_b0 = B0_data.iloc[:, 4].values
nubo_b0 = scale_b0 * nubo_b0
nubo_b0_mesh, x_M, y_M, z_M = algb.vec2mesh(nubo_b0, X_axis_coord, Y_axis_coord, Z_axis_coord, 11, 11, 11)
nubo_b0_mesh = nubo_b0_mesh.T
B0_y = algb.interp_by_pts(nubo_b0_mesh, x_M, y_M, z_M, intrp_pts, method='linear')

nubo_b0 = B0_data.iloc[:, 5].values
nubo_b0 = scale_b0 * nubo_b0
nubo_b0_mesh, x_M, y_M, z_M = algb.vec2mesh(nubo_b0, X_axis_coord, Y_axis_coord, Z_axis_coord, 11, 11, 11)
nubo_b0_mesh = nubo_b0_mesh.T
B0_z = algb.interp_by_pts(nubo_b0_mesh, x_M, y_M, z_M, intrp_pts, method='linear')

nubo_b0_amp = np.linalg.norm([B0_x, B0_y, B0_z], axis=0)
vis.scatter3d(X_axis, Y_axis, Z_axis, nubo_b0_amp, xlim=xlim, ylim=ylim, zlim=zlim, title='B0 (T)')

# %% Generate Mask and Phantom
# Breast Mask
R = 60
height = 100
loc = [0, 0, Z_axis[0]]
chest_dim = [120, 120, 30]
coil_tkns = 5
breast_mask = mk.gen_breast_mask(X_axis, Y_axis, Z_axis, R=R, height=height, loc=loc, tkns=coil_tkns,
                                 chest_dim=chest_dim)

# Sphere phantom
phantom = mk.gen_sphere(X_axis, Y_axis, Z_axis, loc=loc + [0, 0, 2], rad=20)
vis.scatter3d(X_axis, Y_axis, Z_axis, phantom, title='Sphere Phantom', mask=phantom > 0, xlim=xlim, ylim=ylim,
              zlim=zlim)
vis.scatter3d(X_axis, Y_axis, Z_axis, phantom, title='Breast Mask', mask=breast_mask > 0, xlim=xlim, ylim=ylim,
              zlim=zlim)

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

# Slice Selection
slice = acq.slice_select(nubo_b0_amp, ctr_mag, slc_tkns_mag)

# Cut in SI direction
X_M, Y_M, Z_M = np.meshgrid(X_axis, Y_axis, Z_axis, indexing='ij')
SI_cut = (Y_M > -3) & (Y_M < 3)
LR_cut = (X_M > -3) & (X_M < 3)

# Visualize
vis.scatter3d(X_axis, Y_axis, Z_axis, nubo_b0_amp, xlim=xlim, ylim=ylim, zlim=zlim, mask=slice, title='B0 (T)')
vis.scatter3d(X_axis, Y_axis, Z_axis, nubo_b0_amp, xlim=xlim, ylim=ylim, zlim=zlim, mask=SI_cut, title='B0 (T)')

# %% read B1
path = 'sim_inputs/'
filename = 'B1_51'
FOV = 0.24
scale = 1
x_shfit = 0

B1_data = mr_io.load_single_mat(name=filename, path=path)['B1']
B1_intrp = np.zeros((intrp_x, intrp_y, intrp_z, 3))

# Create a 3D grid for the magnetic field data
x_b1_raw = np.linspace(-120, +120, B1_data.shape[1]) + x_shfit
y_b1_raw = np.linspace(-120, +120, B1_data.shape[2])
z_b1_raw = np.linspace(-120, +120, B1_data.shape[3])

for i in [0, 1, 2]:
    # Calculate the magnitude of the magnetic field and apply a scaling factor
    nubo_b1 = B1_data[:, :, :, i]
    nubo_b1 = scale * nubo_b1

    B1_intrp[:, :, :, i] = algb.interp_by_pts(nubo_b1, x_b1_raw, y_b1_raw, z_b1_raw, intrp_pts, method='linear')

# B1_intrp_amp = np.linalg.norm(B1_intrp, axis=3)
# B1_mask = (B1_intrp_amp > 0.002)
# B1_mask = mk.gen_breast_mask(X_axis, Y_axis, Z_axis, R=0.06, height=0.100)

B1_masked = B1_intrp[breast_mask]

# %% plot B1
# vis.scatter3d(X_axis, Y_axis, Z_axis, B1_intrp_amp, xlim=xlim, ylim=ylim, zlim=zlim, title='B1 (T)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, B1_intrp_amp, xlim=xlim, ylim=ylim, zlim=zlim, mask=slice & B1_mask, title='B1 (T)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, B1_intrp_amp, xlim=xlim, ylim=ylim, zlim=zlim, mask=SI_cut, title='B1 (T)')

# %% effective B1
B1_eff = np.zeros((3, intrp_x, intrp_y, intrp_z))
for i in range(intrp_x):
    for j in range(intrp_y):
        for k in range(intrp_z):
            B1_eff[:, i, j, k] = acq.B1_effective(B1_intrp[:, i, j, k], B0_intrp[:, i, j, k])
# B1_eff_amp = np.linalg.norm(B1_eff, axis=0)
# ratio_perc = B1_eff_amp / B1_intrp_amp * 100
# vis.scatter3d(X_axis, Y_axis, Z_axis, ratio_perc, xlim=xlim, ylim=ylim, zlim=zlim, mask=SI_cut,
#               title='Effective B1 / B1 (%)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, ratio_perc, xlim=xlim, ylim=ylim, zlim=zlim, mask=LR_cut,
#               title='Effective B1 / B1 (%)')

# %% flip angle
# clim = [40, 130]
# flip_angle_deg = B1_eff_amp / np.mean(B1_eff_amp[slice & B1_mask]) * 90
# vis.scatter3d(X_axis, Y_axis, Z_axis, flip_angle_deg, xlim=xlim, ylim=ylim, zlim=zlim, clim=clim, mask=slice & B1_mask,
#               title='Flip Angle (degree)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, flip_angle_deg, xlim=xlim, ylim=ylim, zlim=zlim, clim=clim, mask=SI_cut,
#               title='Flip Angle (degree)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, flip_angle_deg, xlim=xlim, ylim=ylim, zlim=zlim, clim=clim, mask=LR_cut,
#               title='Flip Angle (degree)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, flip_angle_deg, xlim=xlim, ylim=ylim, zlim=zlim, clim=clim, mask=B1_mask,
#               title='Flip Angle (degree)')
