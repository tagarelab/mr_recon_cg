"""
   Name: test_rotation_op.py
   Purpose:
   Created on: 10/10/2024
   Created by: Heng Sun
   Additional Notes: 
"""
import unittest
import numpy as np


# Assuming algb.rot_mat is defined elsewhere
def rot_mat(axis, angle):
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(axis * angle).as_matrix()


class rotation_op_v1:
    def __init__(self, axes, angles):
        if len(np.array(axes).shape) == 1:
            axes = np.repeat(np.expand_dims(axes, axis=1), len(angles), axis=1)
        self.rot_mats = np.array([rot_mat(np.array(axes)[:, i], np.array(angles)[i]) for i in
                                  range(len(np.array(angles)))])

    def forward(self, x):
        return np.einsum('ijk,ki->ji', self.rot_mats, x)

    def transpose(self, x):
        rot_mats_T = np.transpose(self.rot_mats, (0, 2, 1))
        return np.einsum('ijk,ki->ji', rot_mats_T, x)


class rotation_op_v2:
    def __init__(self, axes, angles):
        if len(np.array(axes).shape) == 1:
            axes = np.repeat(np.expand_dims(axes, axis=1), len(angles), axis=1)
        self.axes = np.array(axes)
        self.angles = np.array(angles)
        self.len = len(self.angles)

    def forward(self, x):
        x_rot = np.zeros(x.shape)
        for i in range(self.len):
            rot_mat_i = rot_mat(self.axes[:, i], self.angles[i])
            x_rot[:, i] = rot_mat_i @ x[:, i]
        return x_rot

    def transpose(self, x):
        x_rot = np.zeros(x.shape)
        for i in range(self.len):
            rot_mat_i = rot_mat(self.axes[:, i], self.angles[i])
            x_rot[:, i] = rot_mat_i.T @ x[:, i]
        return x_rot


class TestRotationOpImplementations(unittest.TestCase):

    def setUp(self):
        self.axes = [1, 0, 0]
        self.angles = [0, np.pi / 2, np.pi]
        self.rotation1 = rotation_op_v1(self.axes, self.angles)
        self.rotation2 = rotation_op_v2(self.axes, self.angles)

    def test_forward_method(self):
        x = np.random.rand(3, 3)
        result1 = self.rotation1.forward(x)
        result2 = self.rotation2.forward(x)
        np.testing.assert_array_almost_equal(result1, result2, decimal=6,
                                             err_msg="Forward method results differ between implementations")

    def test_transpose_method(self):
        x = np.random.rand(3, 3)
        result1 = self.rotation1.transpose(x)
        result2 = self.rotation2.transpose(x)
        np.testing.assert_array_almost_equal(result1, result2, decimal=6,
                                             err_msg="Transpose method results differ between implementations")


if __name__ == '__main__':
    unittest.main()
