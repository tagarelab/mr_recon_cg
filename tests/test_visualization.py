"""
   Name: ${FILE_NAME}
   Purpose:
   Created on: 9/18/2024
   Created by: Heng Sun
   Additional Notes: 
"""
from unittest import TestCase
import visualization as vis
import numpy as np


class TestVisualization(TestCase):
    def test_compare_overlaid_signals(self):
        num_samples = 1000
        num_signals = 5
        signal_matrix = np.random.rand(num_signals, num_samples)
        signal_matrix = signal_matrix / np.max(signal_matrix)
        ytick_names = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5']
        vis.compare_overlaid_signals(signal_matrix, ytick_names=ytick_names)
