"""
   Name: Test_algebra.py
   Purpose: Test the algebra.py file
   Created on: 2/15/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import unittest
from unittest import TestCase

import numpy as np
import algebra
import visualization as vis

if __name__ == '__main__':
    unittest.main()


class TestAlgebra(unittest.TestCase):
    def test_parallel_component(self):
        self.fail()

    def test_rot_mat(self):
        """
        Test the rot_mat function
        :return:
        """

        orig_vec = np.array([1, 1, 0])
        rot_axis = np.array([0, 0, 1]) * 2
        theta = 180

        # Create a rotation matrix
        rot_mat = algebra.rot_mat(rot_axis, theta * np.pi / 180)

        # Check if the output dimensions are correct
        assert rot_mat.shape == (3, 3)

        new_vec = np.matmul(rot_mat, orig_vec)
        print("New vector: ", new_vec)

        # Check if the new vector is correct
        vis.quiver3d(np.array([orig_vec, new_vec, rot_axis]).T,
                     xlim=[-2, 2], ylim=[-2, 2], zlim=[-2, 2],
                     label=['Orig', 'Rotated', 'Rot Axis'],
                     title='Rotated ' + str(theta) + ' degrees')

        # Check if the before and after has the same length
        assert np.isclose(np.linalg.norm(orig_vec), np.linalg.norm(new_vec), atol=1e-6)

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
        interpolated_matrix, X_intrp, Y_intrp, Z_intrp = algebra.interp_3dmat(M, x_arr, y_arr,
                                                                              z_arr, 6, 6, 6)

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

    def test_normalized_cross_correlation(self):
        """
        Test the normalized_cross_correlation function
        :return:
        """
        x = np.array([1, 2, 3, 4, 5])

    def test_create_perpendicular_unit_vectors(self):
        # w = np.array([1, 2, 3])
        # u, v, unit_w = algebra.create_perpendicular_unit_vectors(w)
        #
        # # Check if the output vectors are perpendicular to each other
        tolerance = 1e-6
        # assert np.isclose(np.dot(u, v), 0, atol=tolerance), f"u . v = {np.dot(u, v)}"
        # assert np.isclose(np.dot(u, unit_w), 0, atol=tolerance), f"u . w = {np.dot(u, unit_w)}"
        # assert np.isclose(np.dot(v, unit_w), 0, atol=tolerance), f"v . w = {np.dot(v, unit_w)}"
        #
        # # Plot the vectors
        # vis.quiver3d(np.array([u, v, w]).T, xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1],
        #              label=['u', 'v', 'w'], title='Perpendicular Unit Vectors')

        # Test cases
        w_1d = np.array([1, 2, 3])
        w_2d = np.array([[1, 0], [0, 1], [0, 0]])

        # Single vector test
        u_1d, v_1d, w_1d_norm = algebra.create_perpendicular_unit_vectors(w_1d)

        # Check orthogonality for single vector
        assert np.isclose(np.dot(u_1d, v_1d),
                          0,
                          atol=tolerance), f"Vectors u and v are not perpendicular: dot(u, v) = {np.dot(u_1d, v_1d)}"
        assert np.isclose(np.dot(u_1d, w_1d_norm),
                          0,
                          atol=tolerance), f"Vectors u and w are not perpendicular: dot(u, w) = {np.dot(u_1d, w_1d_norm)}"
        assert np.isclose(np.dot(v_1d, w_1d_norm),
                          0,
                          atol=tolerance), f"Vectors v and w are not perpendicular: dot(v, w) = {np.dot(v_1d, w_1d_norm)}"

        # Multiple vectors test
        u_2d, v_2d, w_2d_norm = algebra.create_perpendicular_unit_vectors(w_2d)

        for i in range(w_2d_norm.shape[1]):
            # Check orthogonality for each vector in the array
            assert np.isclose(np.dot(u_2d[:, i], v_2d[:, i]),
                              0,
                              atol=tolerance), f"Vectors u and v are not perpendicular for vector {i}: dot(u, v) = {np.dot(u_2d[:, i], v_2d[:, i])}"
            assert np.isclose(np.dot(u_2d[:, i], w_2d_norm[:, i]),
                              0, atol=tolerance), (f"Vectors u and w are not perpendicular for "
                                                   f"vector {i}: dot(u, w) = {np.dot(u_2d[:, i], w_2d_norm[:, i])}")
            assert np.isclose(np.dot(v_2d[:, i], w_2d_norm[:, i]),
                              0,
                              atol=tolerance), f"Vectors v and w are not perpendicular for vector {i}: dot(v, w) = {np.dot(v_2d[:, i], w_2d_norm[:, i])}"

        print("All tests passed!")

    def test_get_rotation_to_vector(self):
        # random vectors
        vectors = (np.random.rand(3, 4) - 0.5) * 2
        target_vectors = (np.random.rand(3, 4) - 0.5) * 2
        target_vector = [0, 0, 1]

        # test the situation where there is only one target
        axes, angles = algebra.get_rotation_to_vector(vectors, target_vector)
        # Use rotation matrix to rotate the vectors
        rotated_vectors = np.zeros(vectors.shape)
        for i in range(vectors.shape[1]):
            rotated_vectors[:, i] = np.dot(algebra.rot_mat(axes[:, i], angles[i]), vectors[:, i])

        # Check if the vectors are rotated to the target vector
        for i in range(vectors.shape[1]):
            vis.quiver3d(np.array([vectors[:, i], rotated_vectors[:, i], target_vector,
                                   axes[:, i]]).T,
                         xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1],
                         label=['Orig', 'Rotated', 'Target', 'Rot Axis'],
                         title='Rotated to Target Vector')
            # assert same direction
            assert np.allclose(np.dot(rotated_vectors[:, i], target_vector),
                               np.linalg.norm(rotated_vectors[:, i]) *
                               np.linalg.norm(target_vector)), \
                "Vectors are not in the same direction"
            # assert same length
            assert np.isclose(np.linalg.norm(rotated_vectors[:, i]), np.linalg.norm(vectors[:, i]),
                              atol=1e-6)

        # test the situation where there are multiple targets
        axes, angles = algebra.get_rotation_to_vector(vectors, target_vectors)
        # print("Axes of rotation:\n", axes)
        # print("Angles of rotation:\n", angles)

        # Use rotation matrix to rotate the vectors
        rotated_vectors = np.zeros(vectors.shape)
        for i in range(vectors.shape[1]):
            rotated_vectors[:, i] = np.dot(algebra.rot_mat(axes[:, i], angles[i]), vectors[:, i])

        # Check if the vectors are rotated to the target vector
        for i in range(vectors.shape[1]):
            vis.quiver3d(np.array([vectors[:, i], rotated_vectors[:, i], target_vectors[:, i],
                                   axes[:, i]]).T,
                         xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1],
                         label=['Orig', 'Rotated', 'Target', 'Rot Axis'],
                         title='Rotated to Target Vector')
            # assert same direction
            assert np.allclose(np.dot(rotated_vectors[:, i], target_vectors[:, i]),
                               np.linalg.norm(rotated_vectors[:, i]) *
                               np.linalg.norm(target_vectors[:, i])), \
                "Vectors are not in the same direction"
            # assert same length
            assert np.isclose(np.linalg.norm(rotated_vectors[:, i]), np.linalg.norm(vectors[:, i]),
                              atol=1e-6)
