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


class TestMasks(unittest.TestCase):

    def test_maskmask2matrix(self):
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
        data_matrix = mk.mask2matrix(masked_data, mask, x, y, z)

        # Check if the output dimensions are correct
        assert data_matrix.shape == (3, 3, 3)

        data[1, 1, 1] = 0
        # Check if the masked data is correct
        assert data_matrix[1, 1, 1] == 0
        assert np.array_equal(data, data_matrix)
