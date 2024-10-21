"""
   Name: expr_log_240311_2.py
   Purpose:
   Created on: 3/12/2024
   Created by: Heng Sun
   Additional Notes: 
"""

from sim import acquisition as acq
import mr_io
import numpy as np
import visualization as vis
import algebra as algb
from optlib import operators as ops, mr_op as mr_op
from sim import masks as mk
from sim import gradient as grad
from tests.tests_optlib import test_operators as test_ops

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
    # # phantom = mk.gen_sphere(X_axis, Y_axis, Z_axis, loc=loc + [0, 0, 20], rad=60)
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
    # Gsi_str = 0.01e-3  # T/mm
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

vis.scatter3d(flip_angle_deg, X_axis, Y_axis, Z_axis, xlim=xlim, ylim=ylim, zlim=zlim, mask=VOI, title='Flip Angle')
vis.show_all_plots()

# Excite
flip_angle_rad = np.deg2rad(flip_angle_deg)
rot_mat = algb.rot_mat(B1_eff, flip_angle_rad)
X = ops.hadamard_matrix_op(rot_mat)  # TODO: check this

# %% Coil Sensitivity
sensi_mat = np.diag(B1_eff)
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

# t_PE = 1e-4  # change this to the actual time
PE = mr_op.phase_encoding_op(B_PE_VOI, t_PE)

# %% Detection
D = mr_op.detection_op(B_FE_VOI, t_acq)

# %% Visualization
# Magnetization
M = P.forward(O)

vis.scatter3d(np.linalg.norm(M, axis=0), X_axis, Y_axis, Z_axis, xlim=xlim, ylim=ylim, zlim=zlim, mask=VOI,
              title='Magnetization')

# Excited Magnetization
M_excited = X.forward(M)
vis.scatter3d(np.linalg.norm(M_excited, axis=0), X_axis, Y_axis, Z_axis, xlim=xlim, ylim=ylim, zlim=zlim, mask=VOI,
              title='Excited Magnetization')
vis.scatter3d(M_excited[0, :], X_axis, Y_axis, Z_axis, xlim=xlim, ylim=ylim, zlim=zlim, mask=VOI, title='Excited Mx')
#
# # %% Phase Encoding
M_PE = PE.forward(M_excited)

# # sanity check
# M_dummy = np.repeat(np.expand_dims(np.array([0, 0, 1]), axis=1), O.shape[0], axis=1)
# M_PE = PE.forward(M_dummy)
#
# vis.scatter3d(M_dummy[0, :], X_axis, Y_axis, Z_axis, xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Dummy M_LR', mask=VOI)
# vis.scatter3d(M_dummy[1, :], X_axis, Y_axis, Z_axis, xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Dummy M_SI', mask=VOI)
# vis.scatter3d(M_dummy[2, :], X_axis, Y_axis, Z_axis, xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Dummy M_AP', mask=VOI)
vis.scatter3d(M_PE[0, :], X_axis, Y_axis, Z_axis, xlim=xlim, ylim=ylim, zlim=zlim, mask=VOI, title='Phase Encoded M_LR')
# vis.scatter3d(M_PE[1, :], X_axis, Y_axis, Z_axis, xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Phase Encoded M_SI', mask=VOI)
# vis.scatter3d(M_PE[2, :], X_axis, Y_axis, Z_axis, xlim=xlim,
#               ylim=ylim, zlim=zlim, title='Phase Encoded M_AP', mask=VOI)

# %% Detection
# M_dummy = np.repeat(np.expand_dims(np.array([0, 0, 1]), axis=1), O.shape[0], axis=1)
# Signal = D.forward(M_dummy)
concat_sig = D.forward(M_PE)
signal = acq.concatenated_to_complex(concat_sig, axis=0, mode="real&imag")

vis.complex(signal, name='Signal')
vis.plot_against_frequency(signal, frag_len=len(signal), dt=dt, name='Image')

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

# %% Combined Operator
A = ops.composite_op(D, PE, X, P)

# %% Adjoint Tests
# if test_ops.test_adjoint_property(op_instance=P):
#     print('P passed the adjoint test')
# if test_ops.test_adjoint_property(op_instance=X):
#     print('X passed the adjoint test')
# if test_ops.test_adjoint_property(op_instance=PE):
#     print('PE passed the adjoint test')
# if test_ops.test_adjoint_property(op_instance=D):
#     print('D passed the adjoint test')
# if test_ops.test_adjoint_property(op_instance=A):
#     print('A passed the adjoint test')


# %% Reconstruction
