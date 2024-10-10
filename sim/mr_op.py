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

__all__ = ['rotation_op', 'phase_encoding_op', 'detection_op']

# Define global variables
gamma = 42.58e6  # Gyromagnetic ratio in Hz/T


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
