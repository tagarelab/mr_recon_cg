"""
   Name: polar_period_240529.py
   Purpose:
   Created on: 6/3/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np

# plot data
ylim_time = [-15000, 15000]
ylim_freq = [-3e5, 3e5]
ylim_freq_zfilled = [-1e8, 1e8]
Disp_Intermediate = True


# %% support functions
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


# %% Load raw data
file_name = 'WithRO_16Scans'
mat_file = sp.io.loadmat('sim_input/' + file_name + '.mat')
raw_sig_all = mat_file['ch1'][1100:, :]  # the first 1000 points are polarization period
comb_sig_all = np.zeros(raw_sig_all.shape, dtype='complex')
