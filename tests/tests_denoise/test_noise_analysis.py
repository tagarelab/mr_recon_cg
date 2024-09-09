"""
   Name: ${FILE_NAME}
   Purpose:
   Created on: 9/9/2024
   Created by: Heng Sun
   Additional Notes: 
"""
from unittest import TestCase
from denoise import noise_analysis as na
import visualization as vis
import numpy as np


class Test_Noise_Analysis(TestCase):
    def test_snr(self):
        """
        Test the snr function
        :return:
        """
        # generate a singal with noise with known snr
        designed_snr = 10
        signal = np.ones(1000)
        noise = np.random.normal(0, 1, 1000)
        data = signal * designed_snr + noise
        noise_region = None
        signal_region = None

        # # visualize the data
        # vis.absolute(data, name="Data")
        # noise_region = (0, 500)
        # signal_region = (500, 1000)

        # test the snr function
        snr = na.snr(data, signal_region=signal_region, noise_region=noise_region, method='std')
        self.assertTrue(snr > designed_snr - 1 and snr < designed_snr + 1)
