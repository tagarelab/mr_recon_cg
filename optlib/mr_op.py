"""
   Name: sim_op.py
   Purpose: Operators for MR physics simulation/ reconstruction
   Created on: 10/10/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
from optlib import operators as ops
import algebra as algb
from sim import masks as mks

__all__ = ['rotation_op', 'phase_encoding_op', 'detection_op']

# Define global variables
gamma = 42.58e6  # Gyromagnetic ratio in Hz/T


class rotation_op(ops.operator):
    def __init__(self, axes, angles):
        if len(np.array(axes).shape) == 1:
            axes = np.repeat(np.expand_dims(axes, axis=1), len(angles), axis=1)
        # Precompute all rotation matrices
        self.rot_mats = np.array(
            [algb.rot_mat(np.array(axes)[:, i], np.array(angles)[i]) for i in range(
                len(np.array(angles)))])
        self.x_shape = (self.rot_mats.shape[2], self.rot_mats.shape[0])
        self.y_shape = (self.rot_mats.shape[1], self.rot_mats.shape[0])

    def forward(self, x):
        # Assumes x of shape (n, self.len)
        return np.einsum('ijk,ki->ji', self.rot_mats, x)

    def transpose(self, y):
        # Assumes x of shape (n, self.len)
        rot_mats_T = np.transpose(self.rot_mats, (0, 2, 1))
        return np.einsum('ijk,ki->ji', rot_mats_T, y)


class polarization_op(ops.operator):
    def __init__(self, B_net):
        self.B_net = B_net
        self.x_shape = (B_net.shape[1],)
        self.y_shape = B_net.shape

    def forward(self, x):
        return self.B_net * x

    def transpose(self, y):
        return np.sum(self.B_net * y, axis=0)



class phase_encoding_op(ops.operator):
    # The no-rotation-to-z-axis version of the phase encoding operator
    def __init__(self, B_net, t_PE, gyro_ratio=gamma, larmor_freq=1e6):
        evol_angle = (gyro_ratio * np.linalg.norm(B_net, axis=0) - larmor_freq) * t_PE * np.pi * 2
        self.pe_rot = rotation_op(B_net, evol_angle)

        self.x_shape = self.pe_rot.get_x_shape()
        self.y_shape = self.pe_rot.get_y_shape()

    def forward(self, x):
        return self.pe_rot.forward(x)

    def transpose(self, y):
        return self.pe_rot.transpose(y)


class detection_op(ops.operator):
    def __init__(self, B_net, t, sensi_mats=None, larmor_freq=1e6, T1_mat=None, T2_mat=None):
        axes, angles = algb.get_rotation_to_vector(vectors=B_net,
                                                   target_vectors=[0, 0, 1])  # each output is Np
        self.rot_z = rotation_op(axes, angles)  # rotate B_net_VOI to z-axis

        # Np
        omega_net = gamma * self.rot_z.forward(B_net)[2, :] * 2 * np.pi  # Np
        delta_omega = omega_net - larmor_freq * 2 * np.pi

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
                                              TR_encode)

            C = np.repeat(coil_eff, len(t), axis=1)
            self.TR_encode = np.dot(C, self.TR_encode)
            # np.repeat(self.TR_encode, sensi_mat.shape[1], axis=0)
            # TR_encode is now Nt * Nc * Np
            Nc = sensi_mats.shape[2]

        else:
            # Nt * Np
            TR_encode = omega_net * np.exp(-1j * np.matmul(np.expand_dims(np.array(t), axis=1), np.expand_dims(
                np.array(delta_omega), axis=0)))

            # vis.imshow(np.real(TR_encode), name='TR_encode real')

            # initialize TR_encode
            Nc = 1
            # Nt * Nc * Np
            self.TR_encode = np.expand_dims(TR_encode, axis=1)
        self.Nt = len(t)
        self.x_shape = self.rot_z.get_x_shape()  # 3, Np
        self.y_shape = (2, len(t), Nc)  # 2 * Nt * Nc   # the first dim is real and imaginary

    def forward(self, x):
        x = np.complex64(self.rot_z.forward(x))
        x = np.matmul(np.array([1, 1j, 0]), x)
        # initialize y
        y = np.zeros(self.y_shape, dtype=complex)
        for c in range(self.y_shape[2]):
            y_temp = np.matmul(self.TR_encode[:, c, :], x)
            y[0, :, c] = np.real(y_temp)
            y[1, :, c] = np.imag(y_temp)

        return y

    # for each magnetization point and time point, calculate the signal
    # return acq.detect_signal(x, self.B, self.C, self.t, T1=self.T1, T2=self.T2)

    # Change coordinates to net B point to z-axis for M and C
    def transpose(self, y):
        y = y[0, :, :] + 1j * y[1, :, :]  # TODO: bug here
        # initialize x
        x_comp = np.zeros((self.x_shape[1], self.y_shape[2]), dtype=complex)
        for c in range(self.y_shape[2]):
            x_comp += self.TR_encode[:, c, :].conj().T @ y
        # sum over coils
        x_comp = np.expand_dims(np.sum(x_comp, axis=1), axis=0)
        x = np.concatenate((np.real(x_comp), np.imag(x_comp), np.zeros(x_comp.shape)), axis=0)
        # 3*Np
        return self.rot_z.transpose(x)


class projection_op(ops.operator):
    """
    Takes an array of points in 3D space and projects them onto a 2D plane.
    """

    def __init__(self, mask_3D, projection_axis=2):
        self.mask_3D = mask_3D
        self.mask_tkns = np.sum(mask_3D, axis=projection_axis)
        self.mask_2D = self.mask_tkns > 0
        self.x_shape = (np.sum(self.mask_3D),)  # default shape of x
        self.y_shape = (np.sum(self.mask_2D),)  # default shape of y
        self.projection_axis = projection_axis

    def forward(self, x):
        # Initialize y with the same shape as mask_tkns but with floating point type to handle division
        # y = np.zeros(self.y_shape, dtype=x.dtype)

        # Get matrix representation of x
        x_matrix = mks.mask2matrix(x, self.mask_3D, matrix_shape=self.mask_3D.shape)
        y_matrix = np.sum(x_matrix, axis=self.projection_axis)
        y = y_matrix[self.mask_2D] / self.mask_tkns[self.mask_2D]
        # y = y_matrix[self.mask_2D]

        # # Populate y with the sum of points in x divided by mask_tkns
        # index_x = 0
        # index_y = 0
        # for i in range(self.mask_tkns.shape[0]):
        #     for j in range(self.mask_tkns.shape[1]):
        #         if self.mask_tkns[i, j] > 0:
        #             y[index_y] = np.sum(x[index_x:index_x + self.mask_tkns[i, j]]) / self.mask_tkns[i, j]
        #             index_x += self.mask_tkns[i, j]
        #             index_y += 1
        return y

    def transpose(self, y):
        # Initialize x
        # x = np.zeros(self.x_shape, dtype=y.dtype)

        # Get matrix representation of y
        y = y / self.mask_tkns[self.mask_2D]
        y_matrix = np.expand_dims(mks.mask2matrix(y, self.mask_2D, matrix_shape=self.mask_2D.shape),
                                  axis=self.projection_axis)
        x_matrix = np.repeat(y_matrix, self.mask_3D.shape[self.projection_axis], axis=self.projection_axis)
        x = x_matrix[self.mask_3D]

        # # Populate x
        # index_x = 0
        # index_y = 0
        # for i in range(self.mask_tkns.shape[0]):
        #     for j in range(self.mask_tkns.shape[1]):
        #         if self.mask_tkns[i, j] > 0:
        #             x[index_x:index_x + self.mask_tkns[i, j]] = y[index_y]
        #             index_x += self.mask_tkns[i, j]
        #             index_y += 1
        return x
