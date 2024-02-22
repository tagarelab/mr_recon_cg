"""
   Name: Test_acquisition.py
   Purpose: Test the acquisition.py file
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import unittest
import numpy as np
from sim import acquisition as acq


class TestAcquisition(unittest.TestCase):

    def test_slice_select(self):
        """
        Test the slice_select function
        This function is drafted by Github Copilot and edited & tested by the author.
        :return:
        """
        # Create a B0 field map
        b0 = np.arange(27).reshape((3, 3, 3))
        ctr_mag = 13
        slc_tkns = 5

        # Use slice_select to create a slice selection mask
        slice_mask = acq.slice_select(b0, ctr_mag, slc_tkns)

        # Check if the output dimensions are correct
        assert slice_mask.shape == (3, 3, 3)

        # Check if the slice selection mask is correct TODO: fix the test
        # assert np.all(slice_mask[0, 0, :])
        # assert np.all(slice_mask[2, 2, :])
        # assert not np.any(slice_mask[1, 1, :])
