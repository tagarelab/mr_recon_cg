"""
   Name: corr_analysis_240726.py
   Purpose:
   Created on: 7/26/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import scipy as sp
import visualization as vis
import mr_io
import algebra as algb
import matplotlib.pyplot as plt
from denoise import combEMI as cs

# plot data
ylim_time = [-15000, 15000]
ylim_freq = [0, 1e4]
ylim_freq_zfilled = [0, 200]
Disp_Intermediate = True
Perform_Comb = False


# %% support functions
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


def compare_unnormalized_freq(signal_list, legends, dt, name=None, ylim=None, log_scale=False):
    plt.figure()
    frag_len = signal_list.shape[1]
    freq_axis = sp.fft.fftshift(sp.fft.fftfreq(frag_len, dt)) / 1000
    for i in range(signal_list.shape[0]):
        signal = signal_list[i, :]
        signal_ft = sp.fft.fftshift(sp.fft.fft(signal))

        if log_scale:
            signal_ft = algb.linear_to_db(signal_ft)

        plt.plot(freq_axis, abs(signal_ft), label=legends[i])

    plt.legend()
    plt.xlabel('Frequency (kHz)')
    if log_scale:
        plt.ylabel('Magnitude (dB)')
    else:
        plt.ylabel('Magnitude')
    if ylim is not None:
        plt.ylim(ylim)
    if name is not None:
        plt.title(name)
    plt.show()

# %% Load raw data

file_name = 'sig_pol_all_steps_07302024'
dict = sp.io.loadmat('sim_output/' + file_name + '.mat')
channel_info = dict['channel_info']
file_name_stem = dict['file_name_stem']
file_name_steps = dict['file_name_steps']
pol_all_steps = dict['pol_all_steps']
segment_names = dict['segment_names']
sig_all_steps = dict['sig_all_steps']

# %% Data Attributes
# real TE part
N_echoes_real = 480
echo_len_real = 120
dt = 5e-6
TE_real = 2e-3

# polarization period
N_rep_pol = 170
rep_len_pol = 6000
# TODO: double check the log to see if it is 400us or 800 us gap - TNMR is not very clear
rep_time_pol = rep_len_pol * dt + 400e-6

pre_drop = 0
post_drop = 0
pk_win = 0
pk_id = 0  # None for auto peak detection
polar_time = 0
post_polar_gap_time = 0

max_iter = 200
rho = 1
lambda_val = -1  # -1 for auto regularization
# auto lambda parameters
ft_prtct = 5

# for j in [0]:
for j in range(len(file_name_stem)):
    for i in range(len(channel_info)):
        corr_i = algb.normalized_cross_correlation(sig_all_steps[i, j, 0, :], sig_all_steps[1, j, 0, :],
                                                   do_ifftshift=True)
        vis.complex(corr_i, name=channel_info[i]+channel_info[1]+file_name_stem[j], rect=True, ylim=[-1,1])
        vis.complex(corr_i[:150], name=channel_info[i] + channel_info[1] + file_name_stem[j], rect=True, ylim=[-1,1])
