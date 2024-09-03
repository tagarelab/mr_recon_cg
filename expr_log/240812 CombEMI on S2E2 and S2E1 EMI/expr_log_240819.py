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
# file_name = 'Step_10_Coil_Everything_Pol_Phantom_withPol_Phant_16avg.mat'
# file_name = 'Step_10_Coil_Everything_Pol_Phantom_withPol_Phant.mat'
# file_name = 'Step_9_Coil_Everything_Pol_withPol_noPhant_16avg.mat'
file_name = 'Step_9_Coil_Everything_Pol_withPol_noPhant.mat'
# file_name = 'Sig_3AxisEMI_Step1_Console.mat'
# file_name = 'Step_4_Console_Preamp_Coil.mat'
# file_name = 'Step_6_Console_Preamp_TRSwitch_Tx_Coil.mat'
data = sp.io.loadmat('sim_input/' + file_name)

sig_ch_name = 'ch2'
emi_ch_name = ['ch1', 'ch3', 'ch4']
# comb_sig_all = np.zeros(raw_sig_all.shape, dtype='complex')

# Load data from .mat file - Step 1 to 7
# # datafft = data[sig_ch_name][102000:, 0]
# # datanoise_fft_list = [data[name][102000:, 0] for name in emi_ch_name]
# datafft = data[sig_ch_name][:, 0]
# datanoise_fft_list = [data[name][:, 0] for name in emi_ch_name]
# Nc = len(datanoise_fft_list)

# Load data from .mat file
datafft = data[sig_ch_name]
datanoise_fft_list = [data[name] for name in emi_ch_name]
Nc = len(datanoise_fft_list)

# Process only the signal part
datafft = datafft[1020000:, :]
datanoise_fft_list = [datanoise_fft[1020000:, :] for datanoise_fft in datanoise_fft_list]

# %% EDITER correction
editer_corr = editer.editer_process_2D(datafft, datanoise_fft_list)

# %% Visualization
ylim_time = [0, 10000]
ylim_freq = [-800, 800]
vis.absolute(editer_corr, name='Corrected with EDITER, Number of EMI Coils = %d' % Nc, ylim=ylim_time)
vis.absolute(datafft, name='Uncorrected', ylim=ylim_time)

vis.absolute(editer_corr[0:400], name='Corrected with EDITER, Number of EMI Coils = %d' % Nc, ylim=ylim_time)
vis.absolute(datafft[0:400], name='Uncorrected', ylim=ylim_time)

# vis.absolute(editer_corr[50000:50400], name='Corrected with EDITER, Number of EMI Coils = %d' % Nc)
# vis.absolute(datafft[50000:50400], name='Uncorrected')

dt = 5e-6
vis.freq_plot(editer_corr, dt=dt, name='INACCURATE FFT, Corrected with EDITER, Number of EMI Coils = %d' % Nc,
              ifft=True,
              ylim=ylim_freq)
vis.freq_plot(datafft, dt=dt, name='INACCURATE FFT, Uncorrected', ifft=True, ylim=ylim_freq)

vis.freq_plot(editer_corr[120:240], dt=dt, name='Second Echo, Corrected with EDITER, Number of EMI Coils = %d' % Nc,
              ifft=True,
              ylim=ylim_freq)
vis.freq_plot(datafft[120:240], dt=dt, name='Second Echo, Uncorrected', ifft=True,
              ylim=ylim_freq)
