"""
   Name: polar_period_240529.py
   Purpose:
   Created on: 6/3/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import scipy as sp
import visualization as vis

# plot data
ylim_time = [-15000, 15000]
ylim_freq = [-5e6, 5e6]
ylim_freq_zfilled = [-1e8, 1e8]
Disp_Intermediate = True

# %% Data parameters
# N_echoes = 479
TE = 2e-3
dt = 6e-6
pre_drop = 0
post_drop = 0
pk_win = 0.33
pk_id = None  # None for auto peak detection
polar_time = 0

max_iter = 200
rho = 1
lambda_val = -1  # -1 for auto regularization
# auto lambda parameters
ft_prtct = 20


# echo_len = int(raw_sig_all.shape[0] / N_echoes)


# %% support functions
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


# %% Load raw data
file_name = 'NoRO_16Scans'
mat_file = sp.io.loadmat('sim_input/' + file_name + '.mat')
raw_pol = mat_file['ch1'][:1000, :]  # the first 1000 points are polarization period

vis.freq_plot(raw_pol[:, 0], dt=dt, name="Polarization Period, No Gradient", ylim=ylim_freq)
vis.freq_plot(raw_pol[:, 1], dt=dt, name="Polarization Period, No Gradient", ylim=ylim_freq)
vis.freq_plot(raw_pol[:, 2], dt=dt, name="Polarization Period, No Gradient", ylim=ylim_freq)
