"""
   Name: ${FILE_NAME}
   Purpose:
   Created on: 9/17/2024
   Created by: Heng Sun
   Additional Notes: 
"""
from unittest import TestCase
from sim import masks as mk
import visualization as vis


class TestMasks(TestCase):
    def test_generate_y_shape(self):
        m = 101
        n = 101
        y_shape_matrix = mk.generate_Y_shape(m, n)
        vis.imshow(y_shape_matrix)
