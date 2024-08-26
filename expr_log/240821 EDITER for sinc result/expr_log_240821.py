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
import mr_io
import visualization as vis

# Load data (replace with actual data loading)

# %% Load raw data
file_name = '30kHz_att90_16_att180_9.9794_64avg.mat'
# file_name = '30kHz_sinc_iter_.mat'
data = sp.io.loadmat('sim_input/' + file_name)

sig_ch_name = 'ch2'
emi_ch_name = ['ch1', 'ch3', 'ch4']
# comb_sig_all = np.zeros(raw_sig_all.shape, dtype='complex')

# Load data from .mat file
datafft = data[sig_ch_name]
datanoise_fft_list = [data[name] for name in emi_ch_name]
Nc = len(datanoise_fft_list)

# for i in [1,8,16,32,64]:
#     datafft_i = np.expand_dims(np.mean(datafft[:, :i], axis=1), axis=1)
#     datanoise_fft_list_i = [np.expand_dims(np.mean(datanoise_fft[:, :i], axis=1),axis=1) for datanoise_fft in
#                           datanoise_fft_list]
#
#     # %% Process the brain slice
#     editer_corr_i = editer.editer_process_2D(datafft_i, datanoise_fft_list_i)
#
#     # %% Save dict
#     dict = {'datafft': datafft_i, 'editer_corr': editer_corr_i}
#     mr_io.save_dict(dict, '30kHz_sinc_iter_EDITER_corrected_avg%d.mat'%i,path='sim_output/')

# %% Visualization
datafft = datafft / 64
editer_corr = editer.editer_process_2D(datafft, datanoise_fft_list)

# ylim_temp = [0, 41000]
ylim_temp = [0, 1000]
ylim_temp_freq = [-60, 60]
ylim_temp_freq2 = [-60, 60]
# ylim_temp_freq = [-2000, 2000]
# ylim_temp_freq2 = [-3000, 3000]
vis.absolute(editer_corr, name='Corrected with EDITER, Number of EMI Coils = %d' % Nc, ylim=ylim_temp)
vis.absolute(datafft, name='Uncorrected', ylim=ylim_temp)

region = [0, 500]
vis.absolute(editer_corr[region[0]:region[1]], name='Corrected with EDITER, Zoom in to %s' % region, ylim=ylim_temp)
vis.absolute(datafft[region[0]:region[1]], name='Uncorrected', ylim=ylim_temp)

region = [12500, 20000]
vis.absolute(editer_corr[region[0]:region[1]], name='Corrected with EDITER, Zoom in to %s' % region, ylim=ylim_temp)
vis.absolute(datafft[region[0]:region[1]], name='Uncorrected', ylim=ylim_temp)

dt = 6e-6

vis.freq_plot(editer_corr, dt=dt, name='Corrected with EDITER, Number of EMI Coils = %d' % Nc, ifft=True,
              ylim=ylim_temp_freq)
vis.freq_plot(datafft, dt=dt, name='Uncorrected', ifft=True, ylim=ylim_temp_freq)

vis.freq_plot(editer_corr[200:300], dt=dt, name='Second Echo, Corrected with EDITER, Number of EMI Coils = %d' % Nc,
              ifft=True,
              ylim=ylim_temp_freq2)
vis.freq_plot(datafft[200:300], dt=dt, name='Second Echo, Uncorrected', ifft=True, ylim=ylim_temp_freq2)
