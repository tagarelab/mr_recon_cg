"""
   Name: test_pe_operators.py
   Purpose:
   Created on: 10/17/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
from tests.tests_optlib import test_operators as test_ops
from optlib import operators as ops, mr_op as mr_op
import algebra as algb
import visualization as vis

gamma = 42.58e6  # Gyromagnetic ratio in Hz/T


class phase_encoding_op_orig(ops.operator):
    # TODO: replace this with a simple "gen_PE_op" function in sim
    def __init__(self, B_net, t_PE, gyro_ratio=gamma, larmor_freq=1e6):
        axes, angles = algb.get_rotation_to_vector(vectors=B_net,
                                                   target_vectors=[0, 0, 1])
        rot_z = mr_op.rotation_op(axes, angles)  # rotate B_net_VOI to z-axis

        evol_angle = (gyro_ratio * rot_z.forward(B_net)[2, :] - larmor_freq) * t_PE * np.pi * 2
        evol_rot = mr_op.rotation_op(np.array([0, 0, 1]), evol_angle)
        self.pe_rot = ops.composite_op(ops.transposed_op(rot_z), evol_rot, rot_z)

        self.x_shape = self.pe_rot.get_x_shape()
        self.y_shape = self.pe_rot.get_y_shape()

    def forward(self, x):
        return self.pe_rot.forward(x)

    def transpose(self, y):
        return self.pe_rot.transpose(y)


def test_phase_encoding_operators(B_net, t_PE):
    # Instantiate objects of both classes
    orig_op = phase_encoding_op_orig(B_net, t_PE)
    new_op = mr_op.phase_encoding_op(B_net, t_PE)

    # check if the input and output has the same shape and dtype
    assert orig_op.get_x_shape() == new_op.get_x_shape()
    assert orig_op.get_y_shape() == new_op.get_y_shape()
    assert orig_op.get_x_dtype() == new_op.get_x_dtype()
    assert orig_op.get_y_dtype() == new_op.get_y_dtype()

    # Generate random input and output
    random_x = test_ops.gen_test_x(orig_op)
    random_y = test_ops.gen_test_y(orig_op)

    # Test the forward method
    forward_orig = orig_op.forward(random_x)
    forward_new = new_op.forward(random_x)

    # Test the transpose method
    transpose_orig = orig_op.transpose(random_y)
    transpose_new = new_op.transpose(random_y)

    # Check if the results are the same for forward method
    forward_same = np.allclose(forward_orig, forward_new, atol=1e-6)
    if forward_same:
        print("Forward methods produce identical results.")
    else:
        print("Forward methods produce different results.")
        print("Original:", forward_orig)
        print("New:", forward_new)

    # Check if the results are the same for transpose method
    transpose_same = np.allclose(transpose_orig, transpose_new, atol=1e-6)
    if transpose_same:
        print("Transpose methods produce identical results.")
    else:
        print("Transpose methods produce different results.")
        print("Original:", transpose_orig)
        print("New:", transpose_new)

    return forward_same and transpose_same


# Example parameters for testing
B_net = np.random.randn(3, 10)  # Example magnetic field vectors
t_PE = 1e-3  # Example phase encoding time

# Run the test
test_result = test_phase_encoding_operators(B_net, t_PE)
if test_result:
    print("Both classes behave identically.")
else:
    print("There is a discrepancy between the two classes.")
