"""
   Name: ${FILE_NAME}
   Purpose:
   Created on: 8/20/2024
   Created by: Heng Sun
   Additional Notes: 
"""
from unittest import TestCase
import numpy as np
from scipy.fftpack import fftn, fftshift
import matplotlib.pyplot as plt
from denoise import editer
from scipy.io import loadmat


class TestEDITER(TestCase):
    def test_editer_process_2d(self):
        """
        Test the editer_process_2D function.
        Translated from MATLAB to Python with ChatGPT based on the original EDITER code by Sai Abitha Srinivas.
        Edited & tested by the author.
        :return:
        """
        # Number of coils
        Nc = 5

        # Load data (replace with actual data loading)
        # Load data from .mat file
        data = loadmat('test_inputs/data_BBEMI_2D_brainslice.mat')
        datafft = data['datafft']
        datanoise_fft_list = [data[f'datanoise_fft_{i + 1}'] for i in range(Nc)]

        # Process the brain slice
        gksp = editer.editer_process_2D(datafft, datanoise_fft_list)

        corr_img_opt_toep = fftshift(fftn(fftshift(gksp)))

        # Visualization
        x_range = np.arange(149, 350)
        y_range = np.arange(0, 101)

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        uncorr = fftshift(fftn(fftshift(datafft)))
        plt.imshow(np.flipud(np.rot90(np.abs(uncorr[x_range[:, None], y_range]))), cmap='gray', aspect='equal')
        plt.axis('tight')
        plt.title('Primary uncorrected')

        plt.subplot(2, 1, 2)
        plt.imshow(np.flipud(np.rot90(np.abs(corr_img_opt_toep[x_range[:, None], y_range]))), cmap='gray',
                   aspect='equal')
        plt.axis('tight')
        plt.title('Corrected with EDITER, Number of EMI Coils = %d' % Nc)

        plt.show()
