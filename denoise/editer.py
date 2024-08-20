"""
   Name: editer.py
   Purpose:
   Created on: 8/20/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, fftshift


def editer_process_2D(datafft, datanoise_fft_list, Nc):
    """
    Translated from MATLAB to Python with ChatGPT based on the original EDITER code by Sai Abitha Srinivas.
    Edited & tested by the author.
    Process 2D brain slice data using a broadband EMI algorithm.

    Parameters:
        datafft: 2D numpy array, Fourier-transformed brain slice data.
        datanoise_fft_list: List of 2D numpy arrays, noise data for different channels.
        Nc: int, number of channels.

    Returns:
        corr_img_opt_toep: 2D numpy array, processed image after correction.
    """

    # Image size
    ncol, nlin = datafft.shape

    # Initial pass using single PE line (Nw=1)
    ksz_col, ksz_lin = 0, 0

    # Kernels across PE lines
    kern_pe = np.zeros((Nc * (2 * ksz_col + 1) * (2 * ksz_lin + 1), nlin), dtype='complex')

    for clin in range(nlin):
        noise_mat = []

        pe_rng = [clin]

        padded_dfs = [np.pad(datanoise_fft[:, pe_rng], ((ksz_col, ksz_col), (ksz_lin, ksz_lin)), mode='constant') for
                      datanoise_fft in datanoise_fft_list]

        for col_shift in range(-ksz_col, ksz_col + 1):
            for lin_shift in range(-ksz_lin, ksz_lin + 1):
                for padded_df in padded_dfs:
                    dftmp = np.roll(np.roll(padded_df, col_shift, axis=0), lin_shift, axis=1)
                    if ksz_col > 0:
                        dftmp = dftmp[ksz_col:-ksz_col, :]
                    if ksz_lin > 0:
                        dftmp = dftmp[:, ksz_lin:-ksz_lin]
                    noise_mat.append(dftmp)

        gmat = np.reshape(np.stack(noise_mat, axis=-1), (-1, len(noise_mat)))

        init_mat_sub = datafft[:, pe_rng]
        kern = np.linalg.lstsq(gmat, init_mat_sub.flatten(), rcond=None)[0]
        kern_pe[:, clin] = kern

    # Normalize kernels
    kern_pe_normalized = kern_pe / np.linalg.norm(kern_pe, axis=0, keepdims=True)
    kcor = np.abs(np.dot(kern_pe_normalized.T, kern_pe_normalized))

    # Threshold
    kcor_thresh = kcor > 5e-1

    # Start with the full set of lines
    aval_lins = list(range(nlin))

    # Window stack
    win_stack = []

    while aval_lins:
        clin = aval_lins[0]
        pe_rng = list(range(clin, clin + np.max(np.where(kcor_thresh[clin, clin:])) + 1))
        win_stack.append(pe_rng)
        aval_lins = sorted(set(aval_lins) - set(pe_rng))

    # Final processing with the selected lines
    ksz_col, ksz_lin = 7, 0
    gksp = np.zeros((ncol, nlin), dtype='complex')

    for pe_rng in win_stack:
        noise_mat = []

        padded_dfs = [np.pad(datanoise_fft[:, pe_rng], ((ksz_col, ksz_col), (ksz_lin, ksz_lin)), mode='constant') for
                      datanoise_fft in datanoise_fft_list]

        for col_shift in range(-ksz_col, ksz_col + 1):
            for lin_shift in range(-ksz_lin, ksz_lin + 1):
                for padded_df in padded_dfs:
                    dftmp = np.roll(np.roll(padded_df, col_shift, axis=0), lin_shift, axis=1)
                    if ksz_col > 0:
                        dftmp = dftmp[ksz_col:-ksz_col, :]
                    if ksz_lin > 0:
                        dftmp = dftmp[:, ksz_lin:-ksz_lin]
                    noise_mat.append(dftmp)

        gmat = np.reshape(np.stack(noise_mat, axis=-1), (-1, len(noise_mat)))

        init_mat_sub = datafft[:, pe_rng]
        kern = np.linalg.lstsq(gmat, init_mat_sub.flatten(), rcond=None)[0]

        # Put the solution back
        tosub = np.reshape(np.dot(gmat, kern), (ncol, len(pe_rng)))
        gksp[:, pe_rng] = init_mat_sub - tosub

    corr_img_opt_toep = fftshift(fftn(fftshift(gksp)))

    return corr_img_opt_toep