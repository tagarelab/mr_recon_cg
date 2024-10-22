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
from optlib import mr_op as mr_op, operators as ops
import visualization as vis
import algebra as algb
from sim import masks as mks


class Test_rotation_op(TestCase):
    def test_adjoint_property(self):
        # generate a random rotation operator
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


class Test_projection_op(TestCase):

    def test_adjoint_property(self):
        # generate a random projection operator
        # generate random 3D boolean mask
        mask = np.random.rand(101, 101, 101) > 0.8
        op = mr_op.projection_op(mask_3D=mask, projection_axis=2)
        test_ops.test_adjoint_property(op)
        # test_ops.test_adjoint_property(ops.transposed_op(op))

    def test_forward(self):
        # Create sample data (3D array
        data_shape = (5, 5, 5)
        data = np.random.random(data_shape)  # Example shape (4, 3, 5)
        x_axis = np.arange(data.shape[0])
        y_axis = np.arange(data.shape[1])
        z_axis = np.arange(data.shape[2])

        # Create a sample mask (3D boolean array)
        mask_3D = (data > 0.8)  # Random boolean mask

        # Extract the elements of data that match the mask
        x = data[mask_3D]

        # Create the projection operator
        op = mr_op.projection_op(mask_3D)

        # Apply the operator to the data
        y = op.forward(x)

        # Check that the output has the correct shape
        self.assertEqual(y.shape, op.get_y_shape())

        # visually check the output
        # vis.scatter3d(x, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, mask=mask_3D)
        # vis.scatter2d(y, x_axis=x_axis, y_axis=y_axis, mask=np.nonzero(op.mask_tkns))

        # Check that the output has the correct values
        x_matrix = mks.mask2matrix(x, mask_3D, matrix_shape=data_shape)
        y_expected = np.sum(x_matrix, axis=2) / np.sum(mask_3D, axis=2)
        y_expected = y_expected[~np.isnan(y_expected)]
        np.testing.assert_allclose(y, y_expected)

    def test_transpose(self):
        pass


class Test_polarization_op(TestCase):

    def test_adjoint_property(self):
        # generate a random polarization operator
        B_net = np.random.rand(3, 10)
        op = mr_op.polarization_op(B_net)
        test_ops.test_adjoint_property(op)

    def test_forward(self):
        self.fail()

    def test_transpose(self):
        self.fail()
