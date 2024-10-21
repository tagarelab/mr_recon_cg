"""
   Name: Test_masks.py
   Purpose: Test the functions in masks.py
   Created on: 3/12/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import unittest
import numpy as np
from sim import masks as mk
import visualization as vis


class TestMasks(unittest.TestCase):

    def test_gen_sphere(self):
        """
        Test the sphere_phantom function
        This function is drafted by Github Copilot and edited & tested by the author.
        :return:
        """
        # Create a 3D phantom
        X_axis = np.linspace(-10, 10, 100)
        Y_axis = np.linspace(-10, 10, 100)
        Z_axis = np.linspace(-10, 10, 100)
        loc = np.array([0, 0, -10])
        rad = 3
        phantom_3d = mk.gen_sphere(X_axis, Y_axis, Z_axis, loc, rad)

        # Check if the output dimensions are correct
        assert phantom_3d.shape == (100, 100, 100)

        vis.scatter3d(phantom_3d, X_axis, Y_axis, Z_axis, mask=phantom_3d > 0, title='Sphere Phantom')

        # Check if the phantom is correct
        # assert np.sum(phantom_3d) == 523598

    def test_gen_breast_mask(self):
        """
        Test the gen_breast_mask function
        This function is drafted by Github Copilot and edited & tested by the author.
        :return:
        """
        # Create a 3D phantom
        X_axis = np.linspace(0, 20, 100)
        Y_axis = np.linspace(0, 20, 100)
        Z_axis = np.linspace(0, 20, 100)
        loc = [6, 6, 6]
        phantom_3d = mk.gen_breast_mask(X_axis, Y_axis, Z_axis, loc=loc, R=6, height=10)
        R = 6
        height = 10

        # Check if the output dimensions are correct
        assert phantom_3d.shape == (100, 100, 100)

        # visualization
        vis.scatter3d(phantom_3d, X_axis, Y_axis, Z_axis, mask=abs(phantom_3d) > 0, title='Breast Mask')

        # Check if the phantom is correct
        # assert np.sum(phantom_3d) == 523598

    def test_mask2matrix(self):
        """
        Test the mask2matrix function
        This function is drafted by Github Copilot and edited & tested by the author.
        :return:
        """
        # Create a 1D array
        data = np.random.rand(3, 3, 3)
        # Create a 3D mask
        mask = np.ones((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = False
        masked_data = data[mask]

        # Create x, y, z axes
        x = np.arange(3)
        y = np.arange(3)
        z = np.arange(3)

        # Use mask2matrix to convert the data to a 3D matrix
        data_matrix = mk.mask2matrix(masked_data, mask, (len(x), len(y), len(z)))

        # Check if the output dimensions are correct
        assert data_matrix.shape == (3, 3, 3)

        data[1, 1, 1] = 0
        # Check if the masked data is correct
        assert data_matrix[1, 1, 1] == 0
        assert np.array_equal(data, data_matrix)

        # unit dimension > 1 data case
        data_shape = (3, 3, 3, 4)
        data = np.random.rand(data_shape[0], data_shape[1], data_shape[2], data_shape[3])

        # Create a 3D mask
        mask = np.ones((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = False
        masked_data = data[mask]

        # Create x, y, z axes
        x = np.arange(3)
        y = np.arange(3)
        z = np.arange(3)

        # Use mask2matrix to convert the data to a 3D matrix
        data_matrix = mk.mask2matrix(masked_data, mask, (len(x), len(y), len(z)))

        # Check if the output dimensions are correct
        assert data_matrix.shape == data_shape

        data[1, 1, 1, :] = 0
        # Check if the masked data is correct
        assert (data_matrix[1, 1, 1, :] == 0).all()
        assert np.array_equal(data, data_matrix)
