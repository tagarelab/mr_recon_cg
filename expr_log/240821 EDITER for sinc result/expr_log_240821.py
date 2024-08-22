"""
   Name: expr_log_240819.py
   Purpose:
   Created on: 8/19/2024
   Created by: Heng Sun
   Additional Notes: 
"""

from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from denoise import editer
import scipy as sp
import visualization as vis

# Load data (replace with actual data loading)

# %% Load raw data
# file_name = 'Step_10_Coil_Everything_Pol_Phantom_withPol_Phant_16avg'
file_name = 'Step_10_Coil_Everything_Pol_Phantom_withPol_Phant'
# file_name = 'Step_9_Coil_Everything_Pol_withPol_noPhant_16avg'
# file_name = 'Step_9_Coil_Everything_Pol_withPol_noPhant'
data = sp.io.loadmat('sim_input/' + file_name + '.mat')

sig_ch_name = 'ch2'
emi_ch_name = ['ch1', 'ch3', 'ch4']
# comb_sig_all = np.zeros(raw_sig_all.shape, dtype='complex')

# Load data from .mat file
datafft = data[sig_ch_name]
datanoise_fft_list = [data[name] for name in emi_ch_name]
Nc = len(datanoise_fft_list)

# Process only the signal part
datafft = datafft[1020000:, :]
datanoise_fft_list = [datanoise_fft[1020000:, :] for datanoise_fft in datanoise_fft_list]

# %% Process the brain slice
editer_corr = editer.editer_process_2D(datafft, datanoise_fft_list)

# %% Visualization
ylim_temp = [0, 10000]
vis.absolute(editer_corr, name='Corrected with EDITER, Number of EMI Coils = %d' % Nc, ylim=ylim_temp)
vis.absolute(datafft, name='Uncorrected', ylim=ylim_temp)

vis.absolute(editer_corr[0:400], name='Corrected with EDITER, Number of EMI Coils = %d' % Nc, ylim=ylim_temp)
vis.absolute(datafft[0:400], name='Uncorrected', ylim=ylim_temp)

vis.absolute(editer_corr[50000:50400], name='Corrected with EDITER, Number of EMI Coils = %d' % Nc)
vis.absolute(datafft[50000:50400], name='Uncorrected')

dt = 6e-6
vis.freq_plot(editer_corr, dt=dt, name='Corrected with EDITER, Number of EMI Coils = %d' % Nc, ifft=True)
vis.freq_plot(datafft, dt=dt, name='Uncorrected', ifft=True)
