"""
   Name: ${FILE_NAME}
   Purpose:
   Created on: 10/14/2024
   Created by: Heng Sun
   Additional Notes: 
"""
from unittest import TestCase
import numpy as np
import test_operators as test_ops
from optlib import mr_op as mr_op


class Test_rotation_op(TestCase):
    def test_adjoint_property(self):
        # gnerate a random rotation operator
        axes = np.random.rand(3, 10)
        angles = np.random.rand(10)
        op = mr_op.rotation_op(axes, angles)
        test_ops.test_adjoint_property(op_instance=op)

    def test_forward(self):
        self.fail()

    def test_transpose(self):
        self.fail()


class Test_phase_encoding_op(TestCase):

    def test_adjoint_property(self):
        # gnerate a random phase encoding operator
        B_net = np.random.rand(3, 10)
        t_PE = np.random.rand(10)
        op = mr_op.phase_encoding_op(B_net, t_PE)
        test_ops.test_adjoint_property(op_instance=op)

    def test_forward(self):
        self.fail()

    def test_transpose(self):
        self.fail()


class Test_detection_op(TestCase):
    def test_adjoint_property(self):
        # gnerate a random detection operator
        t = np.random.rand(10)
        B_net = np.random.rand(3, 10)
        op = mr_op.detection_op(B_net, t)
        test_ops.test_adjoint_property(op_instance=op)

    def test_forward(self):
        self.fail()

    def test_transpose(self):
        self.fail()
