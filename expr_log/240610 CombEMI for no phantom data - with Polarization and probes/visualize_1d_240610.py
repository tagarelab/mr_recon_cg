"""
   Name: visualize_2d_240520.py
   Purpose:
   Created on: 5/20/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import visualization as vis

# %% load data
file_name = 'WithRO_16Scans_comb_06032024'
mat_file = sp.io.loadmat('sim_output/' + file_name + '.mat')
# data = mat_file['data']
raw_sig_all = mat_file['raw_sig_all']
comb_sig_all = mat_file['comb_sig_all']

N_echoes = 479
echo_len = int(raw_sig_all.shape[0] / N_echoes)
plot_x_echoes = 10

ylim_time = [-15000, 15000]
ylim_freq = [-3e5, 3e5]
ylim_freq_zfilled = [-1e8, 1e8]


# %% averaged TR
def avg_first_k_TR(signal, k=None):
    if k is None:
        k = signal.shape[1]
    return np.mean(signal[:, :k], axis=1)


for k in [2, 4, 8, 16]:
    vis.complex(avg_first_k_TR(raw_sig_all[0:echo_len * plot_x_echoes, :], k), name="Raw Data %d TR Averaged" % k,
                ylim=ylim_time)
    vis.complex(avg_first_k_TR(comb_sig_all[0:echo_len * plot_x_echoes, :], k), name="Comb Data %d TR Averaged" % k,
                ylim=ylim_time)

vis.snr_tradeoff_compare(raw_sig_all, comb_sig_all, noi_range=[1, 20], sig_1_name='Raw Data', sig_2_name='Comb Data')


# %% averaged TE
def avg_first_k_TE(signal, echo_len, k):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)

# echo_len = int(data.shape[0] / N_echoes)
# sig_avg = np.zeros([echo_len, data.shape[1]], dtype='complex')
#
# for i in range(data.shape[1]):
#     sig_avg[:, i] = avg_first_k_peaks(data[:, i], echo_len, k=10)
