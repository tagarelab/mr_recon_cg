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
from sim import space as spc
from sim import gradient as grad

generate_new_data = True

if generate_new_data:
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
    # B0_raw = np.array([np.zeros((101,101,101)), np.zeros((101,101,101)), np.ones((101,101,101))])
    # TODO: just for testing
    # vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_raw, axis=0), xlim=xlim, ylim=ylim, zlim=zlim, title='B0 (T)')

    # %% read B1
    path = 'sim_inputs/'
    filename = 'B1_51'
    FOV = 0.24

    # Yonghyun's B1 data
    B1_data_raw = mr_io.load_single_mat(name=filename, path=path)[
                      'B1'] * 1e3  # TODO: check the unit
    # Create a 3D grid for the magnetic field data
    x_b1_raw = np.expand_dims(np.linspace(xlim[0], xlim[1], B1_data_raw.shape[1]), axis=0)
    y_b1_raw = np.expand_dims(np.linspace(ylim[0], ylim[1], B1_data_raw.shape[2]), axis=0)
    z_b1_raw = np.expand_dims(np.linspace(zlim[0], zlim[1], B1_data_raw.shape[3]), axis=0)
    B1_coords = np.concatenate((x_b1_raw, y_b1_raw, z_b1_raw), axis=0)

    B1_data, x_b1_raw, y_b1_raw, z_b1_raw = algb.vec2mesh(B1_data_raw, B1_coords[0, :],
                                                          B1_coords[1, :],
                                                          B1_coords[2, :], empty_val=0.0001)

    # vis.scatter3d(x_b1_raw, y_b1_raw, z_b1_raw, grad=np.linalg.norm(B1_data, axis=0), title='B1 raw',
    #               xlim=xlim, ylim=ylim,
    #               zlim=zlim)

    B1_x = algb.interp_by_pts(B1_data[0, :, :, :], x_b1_raw, y_b1_raw, z_b1_raw,
                              intrp_pts, method='linear')
    B1_y = algb.interp_by_pts(B1_data[1, :, :, :], x_b1_raw, y_b1_raw, z_b1_raw,
                              intrp_pts, method='linear')
    B1_z = algb.interp_by_pts(B1_data[2, :, :, :], x_b1_raw, y_b1_raw, z_b1_raw,
                              intrp_pts, method='linear')
    B1_raw = np.array([B1_x, B1_y, B1_z])
    # B1_raw = np.array(
    #     [np.ones(B1_x.shape), np.zeros(B1_x.shape), np.zeros(B1_x.shape)])  # TODO: just for testing

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
    # vis.scatter3d(X_axis, Y_axis, Z_axis, phantom, title='Phantom in Breast Mask', mask=breast_mask,
    #               xlim=xlim,
    #               ylim=ylim, zlim=zlim)

    # %% Read Gsi data
    # Generate a linear Gsi
    t_PE = 1e-3  # s
    gamma = 42.58e6  # Hz/T
    Gsi_str = 4 / gamma / 2 / R / t_PE  # 4 cycles in the FOV
    # Gsi_str = 0.0013e-3  # T/mm
    Gsi_raw = grad.generate_linear_gradient_3d_Bz((intrp_x, intrp_y, intrp_z), grad_dir='y',
                                                  orientation='x',
                                                  start_value=Gsi_str * ylim[0], end_value=Gsi_str
                                                                                           * ylim[
                                                                                               1])
    # visualize
    # vis.scatter3d(X_axis, Y_axis, Z_axis, Gsi_raw[0,:,:,:], xlim=xlim, ylim=ylim,
    #               zlim=zlim, title='Gsi (T) in LR direction', mask=breast_mask)

    # %% Read Glr Data
    # Glr_coords = mr_io.load_single_mat(name='glr_plane_field_coords', path=path)[
    #                  'field_coords'] * 1e3
    # Glr_data_raw = mr_io.load_single_mat(name='glr_plane_field_strength', path=path)[
    #     'field_strength']
    # Glr_data, x_glr_raw, y_glr_raw, z_glr_raw = algb.vec2mesh(Glr_data_raw, Glr_coords[0, :],
    #                                                           Glr_coords[1, :], Glr_coords[2, :],
    #                                                           empty_val=0.0001)
    # # correct the orientation to keep it consistent with the other data
    # Glr_data = spc.swapaxes(Glr_data, 1, 3)
    # Glr_data = spc.swapaxes(Glr_data, 2, 3)
    # temp = x_glr_raw
    # x_glr_raw = z_glr_raw
    # z_glr_raw = y_glr_raw
    # y_glr_raw = temp
    #
    # # vis.scatter3d(x_glr_raw, y_glr_raw, z_glr_raw, grad=np.linalg.norm(Glr_data, axis=0),
    # #               title='Glr raw', xlim=xlim, ylim=ylim, zlim=zlim)
    # Glr_x = algb.interp_by_pts(Glr_data[0, :, :, :], x_glr_raw, y_glr_raw, z_glr_raw,
    #                            intrp_pts, method='linear')
    # Glr_y = algb.interp_by_pts(Glr_data[1, :, :, :], x_glr_raw, y_glr_raw, z_glr_raw,
    #                            intrp_pts, method='linear')
    # Glr_z = algb.interp_by_pts(Glr_data[2, :, :, :], x_glr_raw, y_glr_raw, z_glr_raw,
    #                            intrp_pts, method='linear')
    # Glr_raw = np.array([Glr_x, Glr_y, Glr_z])

    # Generate a linear Glr
    Glr_str = Gsi_str  # T/mm   #TODO: just for testing
    Glr_raw = grad.generate_linear_gradient_3d_Bz((intrp_x, intrp_y, intrp_z), grad_dir='x',
                                                  orientation='x',
                                                  start_value=Glr_str * xlim[0], end_value=Glr_str
                                                                                           * xlim[
                                                                                               1])

    # visualize
    # vis.scatter3d(X_axis, Y_axis, Z_axis, Glr_raw[0,:], xlim=xlim, ylim=ylim,
    #                   zlim=zlim, title='LR component of Glr (T)',mask=breast_mask&slice)

    # %% save the data
    dict = {'B0': B0_raw, 'B1': B1_raw, 'Glr': Glr_raw, 'Gsi': Gsi_raw, 'phantom': phantom,
            'breast_mask': breast_mask, 'X_axis': X_axis, 'Y_axis': Y_axis, 'Z_axis': Z_axis,
            'xlim': xlim, 'ylim': ylim, 'zlim': zlim, 't_PE': t_PE, 'Gsi_str': Gsi_str,
            'Glr_str': Glr_str, 'gamma': gamma}
    mr_io.save_dict(dict, name='sim_inputs', path='sim_inputs/', date=True)
else:
    # read saved data
    dict = mr_io.load_single_mat(name='sim_inputs_10082024', path='sim_inputs/')
    B0_raw = dict['B0']
    B1_raw = dict['B1']
    Glr_raw = dict['Glr']
    Gsi_raw = dict['Gsi']
    phantom = dict['phantom']
    breast_mask = dict['breast_mask']
    X_axis = dict['X_axis']
    Y_axis = dict['Y_axis']
    Z_axis = dict['Z_axis']
    xlim = dict['xlim']
    ylim = dict['ylim']
    zlim = dict['zlim']
    t_PE = dict['t_PE']
    Gsi_str = dict['Gsi_str']
    Glr_str = dict['Glr_str']
    gamma = dict['gamma']

# %% slice selection
# Constants
# gamma = 42.58e6  # Hz/T
read_mag = 0.046  # T
DC_pol = 0.09  # percent DC_pol when polarizing
ctr_mag = read_mag  # slice selection gradient
slc_tkns_frq = 10e3 * 2  # Hz
slc_tkns_mag = slc_tkns_frq / gamma  # T

# Polarize
B0_polar = B0_raw * DC_pol  # slice strength

# Slice Selection
slice = acq.slice_select(np.linalg.norm(B0_polar, axis=0), ctr_mag, slc_tkns_mag)

# Cut in SI direction
X_M, Y_M, Z_M = np.meshgrid(X_axis, Y_axis, Z_axis, indexing='ij')
SI_cut = (Y_M > -3) & (Y_M < 3)
LR_cut = (X_M > -3) & (X_M < 3)

# Visualize
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_polar, axis=0), xlim=xlim, ylim=ylim,
#               zlim=zlim, mask=slice,
#               title='B0 (mT)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_polar, axis=0), xlim=xlim, ylim=ylim,
#               zlim=zlim, mask=SI_cut,
#               title='B0 (mT)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B1_raw, axis=0), xlim=xlim, ylim=ylim,
#               zlim=zlim, mask=slice,
#               title='B1 (mT)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B1_raw, axis=0), xlim=xlim, ylim=ylim,
#               zlim=zlim, mask=SI_cut,
#               title='B1 (mT)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(Glr_raw, axis=0), xlim=xlim, ylim=ylim,
#               zlim=zlim, mask=slice, title='Glr (mT)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(Glr_raw, axis=0), xlim=xlim, ylim=ylim,
#               zlim=zlim, mask=SI_cut, title='Glr (mT)')
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(Glr_raw, axis=0), xlim=xlim, ylim=ylim,
#               zlim=zlim, mask=breast_mask, title='Glr (mT)')

# %% Select voxels of interest
VOI = breast_mask & slice

# %% Back projection operator
# class back_projection_op:
#     def __init__(self, VOI):
#         self.VOI = VOI
#         data = np.ones(VOI.shape)
#         mk.mask2matrix(data, VOI, X_axis, Y_axis, Z_axis)
#
#     def forward(self, x):
#         # transform x back to the original shape
#         result = np.zeros_like(self.VOI, dtype=x.dtype)
#
#         return x[self.mask]
#
#     def transpose(self, x):
#
#         result = np.zeros_like(self.mask, dtype=x.dtype)
#         result[self.mask] = x
#         return result

# %% Object
phantom_VOI = phantom[VOI]
T1 = 1000  # ms
T2 = 100  # ms
T1_VOI = np.ones(len(phantom_VOI)) * T1
T2_VOI = np.ones(len(phantom_VOI)) * T2

O = phantom_VOI

# %% Polarization
B0_VOI = B0_raw[:, VOI]
P = ops.hadamard_op(B0_VOI * DC_pol)

# sanity check
# B0_VOI_2mat = mk.mask2matrix(B0_VOI, VOI, X_axis, Y_axis, Z_axis)
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(B0_VOI_2mat, axis=0), xlim=xlim, ylim=ylim, zlim=zlim,
#               title='B0 (T)', mask = np.linalg.norm(B0_VOI_2mat, axis=0)>0)
# print(np.array_equal(B0_VOI, B0_VOI_2mat[:, VOI]))

# %% Excitation
DC_ex_B1 = 100  # B1 current when excitation
B1_VOI = B1_raw[:, VOI] * DC_ex_B1
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
E = ops.hadamard_matrix_op(rot_mat)  # TODO: check this

# %% Coil Sensitivity
sensi_mat = np.diag(B1_eff)
C = ops.hadamard_op(sensi_mat)  # TODO: what is the coil sensitivity matrix for our receiving coil?
Sensi_VOI = np.ones(shape=B1_VOI.shape)
Sensi_VOI = np.expand_dims(Sensi_VOI, axis=2)  # in case we have multiple coils

# %% Dephasing
# dt = 5e-6  # s
# acq_time = dt * 100  # s
dt = 2.5e-4  # s
acq_time = dt * 60  # s

# TODO: there is a time between the excitation and the acquisition window
t_acq = np.arange(-acq_time / 2, acq_time / 2, dt)  # s
# TODO: add dephasing to acquisition

# %% Readout
# Reading
Glr_VOI = Glr_raw[:, VOI]
Gsi_VOI = Gsi_raw[:, VOI]
DC_RO = 0.06  # DC_pol when readout
DC_Glr = 1  # Gx when readout
DC_Gsi = 1  # Gz when readout   # TODO: should this be an array instead?
# B_PE_VOI = B0_VOI * DC_RO + Gsi_VOI * DC_Gsi  # TODO: what is the B field during PE?
B_PE_VOI = Gsi_VOI * DC_Gsi + np.repeat(np.expand_dims(np.array([1, 0, 0]), axis=1),
                                        B0_VOI.shape[1],
                                        axis=1) * 24e-3  # TODO: just for testing
# B_FE_VOI = B0_VOI * DC_RO + Glr_VOI * DC_Glr
B_FE_VOI = Glr_VOI * DC_Glr + np.repeat(np.expand_dims(np.array([1, 0, 0]), axis=1),
                                        B0_VOI.shape[1],
                                        axis=1) * 24e-3


# B_net_axes, B_net_angles = algb.get_rotation_to_vector(vectors=B_net_VOI, target_vectors=[0, 0, 1])

class rotation_op:
    def __init__(self, axes, angles):
        if len(np.array(axes).shape) == 1:
            axes = np.repeat(np.expand_dims(axes, axis=1), len(angles), axis=1)
        # Precompute all rotation matrices
        self.rot_mats = np.array(
            [algb.rot_mat(np.array(axes)[:, i], np.array(angles)[i]) for i in range(
                len(np.array(angles)))])

    def forward(self, x):
        # Assumes x of shape (n, self.len)
        return np.einsum('ijk,ki->ji', self.rot_mats, x)

    def transpose(self, x):
        # Assumes x of shape (n, self.len)
        rot_mats_T = np.transpose(self.rot_mats, (0, 2, 1))
        return np.einsum('ijk,ki->ji', rot_mats_T, x)


class phase_encoding_op:
    def __init__(self, B_net, t_PE, gyro_ratio=gamma, larmor_freq=1e6):
        # TODO: define gyro_ratio as a global variable
        axes, angles = algb.get_rotation_to_vector(vectors=B_net,
                                                   target_vectors=[0, 0, 1])
        rot_z = rotation_op(axes, angles)  # rotate B_net_VOI to z-axis

        evol_angle = (gyro_ratio * rot_z.forward(B_net)[2, :] - larmor_freq) * t_PE * np.pi * 2
        evol_rot = rotation_op(np.array([0, 0, 1]), evol_angle)
        self.pe_rot = ops.composite_op(ops.transposed_op(rot_z), evol_rot, rot_z)

    def forward(self, x):
        return self.pe_rot.forward(x)

    def transpose(self, x):
        return self.pe_rot.transpose(x)


# t_PE = 1e-4  # change this to the actual time
PE = phase_encoding_op(B_PE_VOI, t_PE)


# %% Detection
class detection_op:
    def __init__(self, B_net, t, sensi_mats=None, larmor_freq=1e6, T1_mat=None, T2_mat=None):
        axes, angles = algb.get_rotation_to_vector(vectors=B_net,
                                                   target_vectors=[0, 0, 1])  # each output is Np
        self.rot_z = rotation_op(axes, angles)  # rotate B_net_VOI to z-axis

        # Np
        delta_omega = gamma * self.rot_z.forward(B_net)[2, :] - larmor_freq

        if sensi_mats is not None:
            # initialize coil_eff
            coil_eff = self.rot_z.forward(sensi_mats)

            # transverse component of the coil sensitivity, Np*Nc
            for i in range(sensi_mats.shape[2]):
                coil_eff[:, i] = np.matmul(sensi_mats[:, :, i], np.array([1, 1j, 0]))
                # 3*Np*Nc ->Np*Nc

            # Nt * Nc * Np
            TR_encode = np.repeat(np.exp(-1j * np.matmul(np.array(t).T, np.array(delta_omega))),
                                  coil_eff.shape[1], axis=1)

            # initialize TR_encode
            self.TR_encode = np.zeros((len(t), sensi_mats.shape[2]), dtype=complex)

            # Nt * Nc * Np
            for i in sensi_mats.shape[2]:
                self.TR_encode[:, i] = np.dot(np.repeat(coil_eff[:, i], len(t), axis=1),
                                              TR_encode)  #
                # TODO: fix this!!! test with a dummy example

            C = np.repeat(coil_eff, len(t), axis=1)
            self.TR_encode = np.dot(C, self.TR_encode)
            # np.repeat(self.TR_encode, sensi_mat.shape[1], axis=0)
            # TR_encode is now Nt * Nc * Np
            self.Nc = sensi_mats.shape[2]

        else:
            # Nt * Np
            TR_encode = np.exp(-1j * np.matmul(np.expand_dims(np.array(t), axis=1), np.expand_dims(
                np.array(delta_omega), axis=0)) * 2 * np.pi)

            # vis.imshow(np.real(TR_encode), name='TR_encode real')

            # initialize TR_encode
            self.Nc = 1
            # Nt * Nc * Np
            self.TR_encode = np.expand_dims(TR_encode, axis=1)
        self.Nt = len(t)
        self.Np = B_net.shape[1]

    def forward(self, x):
        x = np.complex64(self.rot_z.forward(x))
        x = np.matmul(np.array([1, 1j, 0]), x)
        # initialize y
        y = np.zeros((self.Nt, self.Nc), dtype=complex)
        for c in range(self.Nc):
            y[:, c] = np.matmul(self.TR_encode[:, c, :], x)

        return y

    # for each magnetization point and time point, calculate the signal
    # return acq.detect_signal(x, self.B, self.C, self.t, T1=self.T1, T2=self.T2)

    # Change coordinates to net B point to z-axis for M and C
    def transpose(self, y):
        # initialize x
        x = np.zeros((self.Np,), dtype=complex)
        for c in range(self.Nc):
            x += np.matmul(self.TR_encode[:, c, :].T, y)  # x should be Np
        x = np.concatenate(np.real(x), np.imag(x), np.zeros(x.shape), axis=0)  # 3*Np
        return self.rot_z.transpose(x)


D = detection_op(B_FE_VOI, t_acq)

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
vis.scatter3d(X_axis, Y_axis, Z_axis, M_excited[0, :], xlim=xlim, ylim=ylim,
              zlim=zlim,
              title='Excited Mx', mask=VOI)
#
# # %% Phase Encoding
M_PE = PE.forward(M_excited)

# sanity check
# M_dummy = np.repeat(np.expand_dims(np.array([0, 0, 1]), axis=1), O.shape[0], axis=1)
# M_PE = PE.forward(M_dummy)

# vis.scatter3d(X_axis, Y_axis, Z_axis, M_dummy[0, :], xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Dummy M_LR', mask=VOI)
# vis.scatter3d(X_axis, Y_axis, Z_axis, M_dummy[1, :], xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Dummy M_SI', mask=VOI)
# vis.scatter3d(X_axis, Y_axis, Z_axis, M_dummy[2, :], xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Dummy M_AP', mask=VOI)
vis.scatter3d(X_axis, Y_axis, Z_axis, M_PE[0, :], xlim=xlim,
              ylim=ylim, zlim=zlim, title='Phase Encoded M_LR', mask=VOI)
# vis.scatter3d(X_axis, Y_axis, Z_axis, M_PE[1, :], xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Phase Encoded M_SI', mask=VOI)
# vis.scatter3d(X_axis, Y_axis, Z_axis, M_PE[2, :], xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Phase Encoded M_AP', mask=VOI)

# %% Detection
# M_dummy = np.repeat(np.expand_dims(np.array([0, 0, 1]), axis=1), O.shape[0], axis=1)
# Signal = D.forward(M_dummy)
Signal = D.forward(M_PE)

vis.complex(Signal, name='Signal')
vis.plot_against_frequency(Signal, frag_len=len(Signal), dt=dt, name='Image')

# # Dephased Magnetization over time
# M_0 = np.linalg.norm(M_0[:, :, 0], axis=0)  # TODO: is this the correct M0?
#
# M_t = D.forward(M_excited)
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(M_t[:, :, 49], axis=0), xlim=xlim, ylim=ylim,
#               zlim=zlim,
#               title='Dephased Magnetization at the middle of acq window', mask=VOI)
# vis.scatter3d(X_axis, Y_axis, Z_axis, np.linalg.norm(M_t[:, :, -1], axis=0), xlim=xlim, ylim=ylim,
#               zlim=zlim,
#               title='Dephased Magnetization at the end of acq window', mask=VOI)
# Measured Signal


# %% Reconstruction
A = ops.composite_op(P, E, PE, C, D)
