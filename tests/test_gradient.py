"""
   Name: ${FILE_NAME}
   Purpose:
   Created on: 10/8/2024
   Created by: Heng Sun
   Additional Notes: 
"""
from unittest import TestCase
import numpy as np
from sim import gradient as grad
import visualization as vis


class TestGradient(TestCase):

    def test_generate_linear_gradient_3d_Bz(self):
        def check_B(B, shape, direction, start_value, end_value):
            z, y, x = shape
            Bxy = np.zeros(shape)
            Bz = grad.generate_linear_gradient_3d(shape, direction, start_value, end_value)

            if not np.allclose(B[0], Bxy):
                print(f"Test failed for Bxy component in direction {direction}")
                return False
            if not np.allclose(B[1], Bxy):
                print(f"Test failed for Bxy component in direction {direction}")
                return False
            if not np.allclose(B[2], Bz):
                print(f"Test failed for Bz component in direction {direction}")
                return False
            return True

        shape = (5, 5, 5)
        start_value = 0
        end_value = 1

        # Test for gradient along 'x' direction
        B = grad.generate_linear_gradient_3d_Bz(shape, direction='x', start_value=start_value,
                                                end_value=end_value)
        assert check_B(B, shape, 'x', start_value,
                       end_value), "Gradient test failed for direction 'x'"

        # Test for gradient along 'y' direction
        B = grad.generate_linear_gradient_3d_Bz(shape, direction='y', start_value=start_value,
                                                end_value=end_value)
        assert check_B(B, shape, 'y', start_value,
                       end_value), "Gradient test failed for direction 'y'"

        # Test for gradient along 'z' direction
        B = grad.generate_linear_gradient_3d_Bz(shape, direction='z', start_value=start_value,
                                                end_value=end_value)
        assert check_B(B, shape, 'z', start_value,
                       end_value), "Gradient test failed for direction 'z'"

        print("All tests passed.")

    def test_generate_linear_gradient_3d(self):
        def check_gradient(matrix, shape, direction, start_value, end_value):
            z, y, x = shape

            if direction == 'x':
                for i in range(x):
                    expected_value = start_value + (end_value - start_value) * (i / (x - 1))
                    if not np.allclose(matrix[:, :, i], expected_value):
                        print(f"Test failed for direction {direction} at x={i}")
                        return False
            elif direction == 'y':
                for i in range(y):
                    expected_value = start_value + (end_value - start_value) * (i / (y - 1))
                    if not np.allclose(matrix[:, i, :], expected_value):
                        print(f"Test failed for direction {direction} at y={i}")
                        return False
            elif direction == 'z':
                for i in range(z):
                    expected_value = start_value + (end_value - start_value) * (i / (z - 1))
                    if not np.allclose(matrix[i, :, :], expected_value):
                        print(f"Test failed for direction {direction} at z={i}")
                        return False
            return True

        shape = (5, 5, 5)
        start_value = 0
        end_value = 1

        # Test for gradient along 'x' direction
        matrix = grad.generate_linear_gradient_3d(shape, direction='x', start_value=start_value,
                                                  end_value=end_value)
        assert check_gradient(matrix, shape, 'x', start_value,
                              end_value), "Gradient test failed for direction 'x'"

        # Test for gradient along 'y' direction
        matrix = grad.generate_linear_gradient_3d(shape, direction='y', start_value=start_value,
                                                  end_value=end_value)
        assert check_gradient(matrix, shape, 'y', start_value,
                              end_value), "Gradient test failed for direction 'y'"

        # Test for gradient along 'z' direction
        matrix = grad.generate_linear_gradient_3d(shape, direction='z', start_value=start_value,
                                                  end_value=end_value)
        assert check_gradient(matrix, shape, 'z', start_value,
                              end_value), "Gradient test failed for direction 'z'"

        print("All tests passed.")
