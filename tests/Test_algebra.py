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
import visualization as vis

if __name__ == '__main__':
    unittest.main()


class TestAlgebra(unittest.TestCase):

    def test_rot_mat(self):
        """
        Test the rot_mat function
        :return:
        """

        orig_vec = np.array([1, 1, 1])
        rot_axis = np.array([0, 0, 1])
        theta = 360

        # Create a rotation matrix
        rot_mat = algebra.rot_mat(rot_axis, theta * np.pi / 180)

        # Check if the output dimensions are correct
        assert rot_mat.shape == (3, 3)

        new_vec = np.matmul(rot_mat, orig_vec)

        # Check if the new vector is correct
        vis.quiver3d(np.array([orig_vec, new_vec, rot_axis]).T,
                     xlim=[-2, 2], ylim=[-2, 2], zlim=[-2, 2],
                     label=['Orig', 'Rotated', 'Rot Axis'],
                     title='Rotated ' + str(theta) + ' degrees')

        # Check if the rotation matrix is correct
        # np.testing.assert_array_equal(rot_mat, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

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
        # mag_mesh, x_M, y_M, z_M = algebra.vec2mesh(mag, x_coord, y_coord, z_coord, 3, 3, 3)
        mag_mesh, x_M, y_M, z_M = algebra.vec2mesh(mag, x_coord, y_coord, z_coord)

        # Check if the output dimensions are correct
        assert mag_mesh.shape == (3, 3, 3)
        assert x_M.shape == (3,)
        assert y_M.shape == (3,)
        assert z_M.shape == (3,)

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
