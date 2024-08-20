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


class TestEDITER(TestCase):
    def test_editer_process_2d(self):
        # Number of coils
        Nc = 5

        # Load data (replace with actual data loading)
        datafft = np.load('test_inputs/data_BBEMI_2D_brainslice.mat')  # Example placeholder
        datanoise_fft_list = [np.load(f'datanoise_fft_{i + 1}.npy') for i in range(Nc)]

        # Process the brain slice
        corr_img_opt_toep = editer.editer_process_2D(datafft, datanoise_fft_list, Nc)

        # Visualization
        x_range = np.arange(150, 351)
        y_range = np.arange(1, 102)

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
        plt.title('Corrected with EDITER')

        plt.show()
