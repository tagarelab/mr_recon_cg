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

    def test_complex2array(self):
        x = np.array([[1 + 1.j, 2 + 2.j, 3 + 3.j], [4 + 4.j, 5 + 5.j, 6 + 6.j]])
        y = np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]])
        np.testing.assert_array_equal(algebra.complex2array(x), y)

    def test_array2complex(self):
        x = np.array([[1 + 1.j, 2 + 2.j, 3 + 3.j], [4 + 4.j, 5 + 5.j, 6 + 6.j]])
        y = np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]])
        np.testing.assert_array_equal(algebra.array2complex(y), x)
