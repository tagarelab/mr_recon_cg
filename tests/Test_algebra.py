"""
   Name: Test_algebra.py
   Purpose: Test the algebra.py file
   Created on: 2/15/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import unittest
import numpy as np
import algebra

if __name__ == '__main__':
    unittest.main()


class TestAlgebra(unittest.TestCase):

    def test_interp_3dmat(self):
        """
        Test the interp_3dmat function
        This function is drafted by Github Copilot and edited & tested by the author.
        :return:
        """
        # Create a 3D array of values and corresponding coordinates
        M = np.arange(27).reshape((3, 3, 3))
        x_arr = np.arange(3)
        y_arr = np.arange(3)
        z_arr = np.arange(3)

        # Use interp_3dmat to transform them into a 3D mesh with new dimensions
        interpolated_matrix, X_intrp, Y_intrp, Z_intrp = algebra.interp_3dmat(M, x_arr, y_arr, z_arr, 6, 6, 6)

        # Check if the output dimensions are correct
        assert interpolated_matrix.shape == (6, 6, 6)
        assert X_intrp.shape == (6,)
        assert Y_intrp.shape == (6,)
        assert Z_intrp.shape == (6,)

        # Check if the values at certain indices match the expected values
        assert interpolated_matrix[0, 0, 0] == M[0, 0, 0]
        assert interpolated_matrix[5, 5, 5] == M[2, 2, 2]

    def test_vec2mesh(self):
        """
        Test the vec2mesh function
        This function is drafted by Github Copilot and edited & tested by the author.
        :return:
        """
        # Create a 1D array of values and corresponding coordinates
        mag = np.arange(27)
        x_coord = np.repeat(np.arange(3), 9)
        y_coord = np.tile(np.repeat(np.arange(3), 3), 3)
        z_coord = np.tile(np.arange(3), 9)

        # Use vec2mesh to transform them into a 3D mesh
        mag_mesh, X_M, Y_M, Z_M = algebra.vec2mesh(mag, x_coord, y_coord, z_coord, 3, 3, 3)

        # Check if the output dimensions are correct
        assert mag_mesh.shape == (3, 3, 3)
        assert X_M.shape == (3, 3, 3)
        assert Y_M.shape == (3, 3, 3)
        assert Z_M.shape == (3, 3, 3)

        # Check if the values at certain indices match the expected values
        assert mag_mesh[0, 0, 0] == 0
        assert mag_mesh[2, 2, 2] == 26
        assert mag_mesh[1, 1, 1] == 13

    def test_complex2array(self):
        """
        Test the complex2array function
        This function is drafted by Github Copilot and edited & tested by the author.
        :return:
        """
        x = np.array([[1 + 1.j, 2 + 2.j, 3 + 3.j], [4 + 4.j, 5 + 5.j, 6 + 6.j]])
        y = np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]])
        np.testing.assert_array_equal(algebra.complex2array(x), y)

    def test_array2complex(self):
        """
        Test the array2complex function
        This function is drafted by Github Copilot and edited & tested by the author.
        :return:
        """
        x = np.array([[1 + 1.j, 2 + 2.j, 3 + 3.j], [4 + 4.j, 5 + 5.j, 6 + 6.j]])
        y = np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]])
        np.testing.assert_array_equal(algebra.array2complex(y), x)
