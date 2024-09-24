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
from sim import dephasing as dp

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
current_grad = 100

# Yonghyun's B1 data
B1_data_raw = mr_io.load_single_mat(name=filename, path=path)['B1'] * 1e3
# Create a 3D grid for the magnetic field data
x_b1_raw = np.expand_dims(np.linspace(xlim[0], xlim[1], B1_data_raw.shape[1]), axis=0)
y_b1_raw = np.expand_dims(np.linspace(ylim[0], ylim[1], B1_data_raw.shape[2]), axis=0)
z_b1_raw = np.expand_dims(np.linspace(zlim[0], zlim[1], B1_data_raw.shape[3]), axis=0)
B1_coords = np.concatenate((x_b1_raw, y_b1_raw, z_b1_raw), axis=0)

# Anja's B1 data
# B1_coords = mr_io.load_single_mat(name='glr_plane_field_coords', path=path)['field_coords'] * 1e3
# B1_data_raw = mr_io.load_single_mat(name='glr_plane_field_strength', path=path)['field_strength']

B1_data, x_b1_raw, y_b1_raw, z_b1_raw = algb.vec2mesh(B1_data_raw, B1_coords[0, :], B1_coords[1, :],
                                                      B1_coords[2, :], empty_val=0.0001)

vis.scatter3d(x_b1_raw, y_b1_raw, z_b1_raw, grad=np.linalg.norm(B1_data, axis=0), title='B1 raw',
              xlim=xlim, ylim=ylim,
              zlim=zlim)

# Create a 3D grid for the magnetic field data
# x_b1_raw = np.linspace(-120, +120, B1_data.shape[1])
# y_b1_raw = np.linspace(-120, +120, B1_data.shape[2])
# z_b1_raw = np.linspace(-120, +120, B1_data.shape[3])

B1_x = algb.interp_by_pts(current_grad * B1_data[0, :, :, :], x_b1_raw, y_b1_raw, z_b1_raw,
                          intrp_pts, method='linear')
B1_y = algb.interp_by_pts(current_grad * B1_data[1, :, :, :], x_b1_raw, y_b1_raw, z_b1_raw,
                          intrp_pts, method='linear')
B1_z = algb.interp_by_pts(current_grad * B1_data[2, :, :, :], x_b1_raw, y_b1_raw, z_b1_raw,
                          intrp_pts, method='linear')
B1_raw = np.array([B1_x, B1_y, B1_z])

# %% Generate Mask and Phantom
# Breast Mask
R = 60  # mm
height = 100  # mm
breast_loc = np.array([0, 0, Z_axis[0]])  # mm
chest_dim = np.array([200, 200, 30])  # mm
coil_tkns = 5  # mm
breast_mask = mk.gen_breast_mask(X_axis, Y_axis, Z_axis, R=R, height=height,
                                 breast_loc=breast_loc.copy(),
                                 tkns=coil_tkns,
                                 chest_dim=chest_dim)

# Generate phantom
# phantom = mk.gen_sphere(X_axis, Y_axis, Z_axis, loc=loc + [0, 0, 20], rad=60)
phantom = mk.gen_breast_mask(X_axis, Y_axis, Z_axis, R=R - 10, height=height,
                             breast_loc=breast_loc.copy(),
                             tkns=coil_tkns,
                             chest_dim=chest_dim)
Y_shape = mk.generate_Y_shape(len(X_axis), len(Y_axis))
Y_shape = np.logical_not(Y_shape)
Y_shape = np.stack([Y_shape] * len(Z_axis), axis=2)
phantom = np.logical_and(phantom, Y_shape)
vis.scatter3d(X_axis, Y_axis, Z_axis, phantom, title='Phantom in Breast Mask', mask=breast_mask,
              xlim=xlim,
              ylim=ylim, zlim=zlim)

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
vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_polar, axis=0), xlim=xlim, ylim=ylim,
              zlim=zlim, mask=slice,
              title='B0 (mT)')
vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_polar, axis=0), xlim=xlim, ylim=ylim,
              zlim=zlim, mask=SI_cut,
              title='B0 (mT)')
vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B1_raw, axis=0), xlim=xlim, ylim=ylim,
              zlim=zlim, mask=slice,
              title='B1 (mT)')
vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B1_raw, axis=0), xlim=xlim, ylim=ylim,
              zlim=zlim, mask=SI_cut,
              title='B1 (mT)')

# %% effective B1
# B1_eff = np.zeros((3, intrp_x, intrp_y, intrp_z))
# for i in range(intrp_x):
#     for j in range(intrp_y):
#         for k in range(intrp_z):
#             B1_eff[:, i, j, k] = acq.B1_effective(B1_raw[:, i, j, k], B0_raw[:, i, j, k])
# ratio_perc = np.linalg.norm(B1_eff, axis=0) / np.linalg.norm(B1_raw, axis=0) * 100
# vis.scatter3d(X_axis, Y_axis, Z_axis, ratio_perc, xlim=xlim, ylim=ylim, zlim=zlim, mask=breast_mask & SI_cut,
#               title='Effective B1 / B1 (%)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, ratio_perc, xlim=xlim, ylim=ylim, zlim=zlim, mask=breast_mask & LR_cut,
#               title='Effective B1 / B1 (%)')

# %% flip angle
# clim = [40, 130]
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
T1 = 1000  # ms
T2 = 100  # ms
T1_VOI = np.ones(len(phantom_VOI)) * T1
T2_VOI = np.ones(len(phantom_VOI)) * T2

O = phantom_VOI

# %% Polarization
B0_VOI = B0_raw[:, VOI]
P = ops.hadamard_op(B0_VOI)

# sanity check
# B0_VOI_2mat = mk.mask2matrix(B0_VOI, VOI, X_axis, Y_axis, Z_axis)
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_VOI_2mat, axis=0), xlim=xlim, ylim=ylim, zlim=zlim,
#               title='B0 (T)', mask = np.linalg.norm(B0_VOI_2mat, axis=0)>0)
# print(np.array_equal(B0_VOI, B0_VOI_2mat[:, VOI]))

# %% Excitation
B1_VOI = B1_raw[:, VOI]
omega_0 = gamma * np.linalg.norm(B0_VOI, axis=0)

# Get B1_eff
B1_eff = acq.B1_effective(B1_VOI, B0_VOI)

# Get flip angle
B1_eff_amp = np.linalg.norm(B1_eff, axis=0)
flip_angle_deg = B1_eff_amp / np.mean(B1_eff_amp) * 90

vis.scatter3d(X_axis, Y_axis, Z_axis, flip_angle_deg, xlim=xlim, ylim=ylim, zlim=zlim,
              title='Flip Angle', mask=VOI)
vis.show_all_plots()

# Excite
flip_angle_rad = np.deg2rad(flip_angle_deg)
rot_mat = algb.rot_mat(B1_eff, flip_angle_rad)
E = ops.hadamard_matrix_op(rot_mat)

# %% Coil Sensitivity
sensi_mat = np.diag(B1_eff)
C = ops.hadamard_op(sensi_mat)  # TODO: what is the coil sensitivity matrix for our receiving coil?

# %% Dephasing
dt = 5e-6  # s
acq_time = dt * 100  # ms
# TODO: there is a time between the excitation and the acquisition window
t = np.arange(0, acq_time, dt)  # ms


class detection_op:
    def __init__(self, B_mat, sensi_mat, T1_mat, T2_mat):
        self.B = B_mat
        self.C = sensi_mat
        self.T1 = T1_mat
        self.T2 = T2_mat
        self.omega = gamma * np.linalg.norm(self.B, axis=0)

    def forward(self, x):
        # for each magnetization point and time point, calculate the signal

        return self.A * x

    def transpose(self, x):
        return self.forward(x)


# M_t_mat = dp.M_t_operator(T1_VOI, T2_VOI, t)
# D = ops.hadamard_op_expand(M_t_mat)

# %% Acquisition
# dM/dt for all voxels (thus considering the field inhomogeneity)
# class manual_derivative:
#     """
#        Takes the end of the
#     """
#     def __init__(self):
#
#     def forward(self,x):
#         result = np.zeros_like(x)
#         t = x.shape[2]
#         for i in range(t - 1):
#             result[:, :, i] = x[:, :, i + 1] - x[:, :, i]
#         return result
#     def transpose(self,x):
#         return self.forward(x)


# %% Visualization
# Magnetization
M = P.forward(O)

vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(M, axis=0), xlim=xlim, ylim=ylim, zlim=zlim,
              title='Magnetization',
              mask=VOI)

# Excited Magnetization
M_excited = E.forward(M)
vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(M_excited, axis=0), xlim=xlim, ylim=ylim,
              zlim=zlim,
              title='Excited Magnetization', mask=VOI)

# Dephased Magnetization over time
M_0 = np.linalg.norm(M_0[:, :, 0], axis=0)  # TODO: is this the correct M0?

M_t = D.forward(M_excited)
vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(M_t[:, :, 49], axis=0), xlim=xlim, ylim=ylim,
              zlim=zlim,
              title='Dephased Magnetization at the middle of acq window', mask=VOI)
vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(M_t[:, :, -1], axis=0), xlim=xlim, ylim=ylim,
              zlim=zlim,
              title='Dephased Magnetization at the end of acq window', mask=VOI)
# Measured Signal


# %% Reconstruction
