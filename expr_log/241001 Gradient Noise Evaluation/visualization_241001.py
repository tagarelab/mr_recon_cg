"""
   Name: visualization_240823.py
   Purpose:
   Created on: 8/23/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import matplotlib.pyplot as plt
import visualization as vis
from denoise import noise_analysis as na
import mr_io
from denoise import combEMI

# %% Load data
loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\241001_gradient_EMI_ctrl_w_Anja\\EDITER\\"

file_dict = {
    0: "noise_grad_off_RO_0_iter1.mat",
    1: "noise_grad_off_RO_0_iter2.mat",
    2: "noise_grad_off_RO_0_iter3.mat",
    3: "noise_grad_off_RO_0_iter4_postgrad.mat",
    4: "noise_grad_off_RO_0_iter5_postgrad.mat",
    5: "noise_grad_off_RO_0_iter6_postgrad.mat",
    6: "noise_grad_on_RO_0_PE_0_iter1.mat",
    7: "noise_grad_on_RO_0_PE_0_iter2.mat",
    8: "noise_grad_on_RO_0_PE_0_iter3.mat",
    9: "noise_grad_on_RO_0_PE_10_iter1.mat",
    10: "noise_grad_on_RO_0_PE_10_iter2.mat",
    11: "noise_grad_on_RO_0_PE_10_iter3.mat",
    12: "noise_grad_on_RO_4_PE_0_iter1.mat",
    13: "noise_grad_on_RO_4_PE_0_iter2.mat",
    14: "noise_grad_on_RO_4_PE_0_iter3.mat",
    15: "noise_grad_on_RO_4_PE_10_iter1.mat",
    16: "noise_grad_on_RO_4_PE_10_iter2.mat",
    17: "noise_grad_on_RO_4_PE_10_iter3.mat",
    18: "noise_grad_on_RO_8_PE_0_iter1.mat",
    19: "noise_grad_on_RO_8_PE_0_iter2.mat",
    20: "noise_grad_on_RO_8_PE_0_iter3.mat",
    21: "noise_grad_on_RO_8_PE_10_iter1.mat",
    22: "noise_grad_on_RO_8_PE_10_iter2.mat",
    23: "noise_grad_on_RO_8_PE_10_iter3.mat",
    24: "noise_grad_on_RO_12_PE_0_iter1.mat",
    25: "noise_grad_on_RO_12_PE_0_iter2.mat",
    26: "noise_grad_on_RO_12_PE_0_iter3.mat",
    27: "noise_grad_on_RO_12_PE_10_iter1.mat",
    28: "noise_grad_on_RO_12_PE_10_iter2.mat",
    29: "noise_grad_on_RO_12_PE_10_iter3.mat",
    30: "noise_grad_on_RO_16_PE_0_iter1.mat",
    31: "noise_grad_on_RO_16_PE_0_iter2.mat",
    32: "noise_grad_on_RO_16_PE_0_iter3.mat",
    33: "noise_grad_on_RO_16_PE_10_iter1.mat",
    34: "noise_grad_on_RO_16_PE_10_iter2.mat",
    35: "noise_grad_on_RO_16_PE_10_iter3.mat",
    36: "noise_grad_on_RO_20_PE_0_iter1.mat",
    37: "noise_grad_on_RO_20_PE_0_iter2.mat",
    38: "noise_grad_on_RO_20_PE_0_iter3.mat",
    39: "noise_grad_on_RO_20_PE_10_iter1.mat",
    40: "noise_grad_on_RO_20_PE_10_iter2.mat",
    41: "noise_grad_on_RO_20_PE_10_iter3.mat"
}
# # B0 originated noise
# names = ["B0 Disconnected", "B0 Connected to Power", "B0 Turned On",
#          "PA also Turned On", "RO 3%", "RO 6.3%"]
# ids = [41, 21, 25, 22, 26, 29] # position 1
# # ids = [20, 0, 4, 1, 5, 8]# position 2
# position = "Position 1"

# Prepol noise
# names = ["B0 Disconnected", "Prepol off, RO 0%", "Prepol off, RO 3%", "Prepol off RO 6.3%", "Prepol on, RO 0%",
#          "Prepol on, RO 3%", "Prepol on, RO 6.3%"]
# ids = [20, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # position 2
# # ids = [41, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]  # position 1
# position = "Position 2"

# Compare spectrum of with and without prepol
# names = ["B0 Disconnected", "Prepol off, RO 0%", "Prepol on, RO 0%", "Prepol off, RO 3%",
#          "Prepol on, RO 3%", "Prepol off RO 6.3%", "Prepol on, RO 6.3%"]
# ids = [41, 22, 32, 26, 35, 29, 38]  # position 1

# Compare different gradient dc vs. noise level
names = ["GA Off (pre-scan)", "GA Off (post-scan)", "G_RO 0%", "G_RO 4%", "G_RO 8%", "G_RO 12%",
         "G_RO 16%", "G_RO 20%"]
ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26, 30, 31, 32, 36, 37, 38]
title = "PE 0%"

# Compare different gradient dc vs. noise level
# names = ["GA Off (pre-scan)", "GA Off (post-scan)", "G_RO 0%", "G_RO 4%", "G_RO 8%", "G_RO 12%",
#          "G_RO 16%", "G_RO 20%"]
# ids = [0,1,2,3,4,5,9,10,11,15, 16, 17,21, 22, 23,27, 28,29,33, 34, 35, 39, 40, 41]
# title = "PE 10%"

Nc = 3
ylim_temp = [0, 1000]
ylim_temp_freq = [-60, 60]
ylim_temp_freq2 = None
dt = 6e-6  # 6 us
TE = 3.5e-3  # 3.5 ms

power_raw = np.zeros(len(ids))
power_corr = np.zeros(len(ids))

signal_raw = np.zeros((len(ids), 20000), dtype=complex)
signal_corr = np.zeros((len(ids), 20000), dtype=complex)

signal_raw_full = np.zeros((len(ids), 116600), dtype=complex)
signal_corr_full = np.zeros((len(ids), 116600), dtype=complex)

for i in range(len(ids)):
    dict_i = mr_io.load_single_mat(name=file_dict[ids[i]], path=loc)
    raw_sig_i = dict_i['raw_sig']
    editer_corr_i = dict_i['editer_corr']
    diff = editer_corr_i - raw_sig_i

    # power of pre and after noise cancellation
    power_raw[i] = na.power(raw_sig_i)
    power_corr[i] = na.power(editer_corr_i)

    signal_raw[i, :] = np.squeeze(raw_sig_i)
    signal_corr[i, :] = np.squeeze(editer_corr_i)

    signal_raw_full[i, :] = combEMI.sampled_to_full(signal=raw_sig_i, polar_time=0,
                                                    post_polar_gap_time=0, dt=dt,
                                                    acq_len=100, N_echoes=200, TE_len=int(TE / dt))
    signal_corr_full[i, :] = combEMI.sampled_to_full(signal=editer_corr_i, polar_time=0,
                                                     post_polar_gap_time=0, dt=dt,
                                                     acq_len=100, N_echoes=200, TE_len=int(TE / dt))

    # dt = 6e-6
    #
    # vis.freq_plot(editer_corr_i[200:300], dt=dt, name='Second Echo, %d Avgs first, then corrected with EDITER' % i,
    #               ifft=True,
    #               ylim=ylim_temp_freq)

# %% Plot
# tick_positions = range(len(names))
#
# plt.figure()
# plt.plot(tick_positions, power_raw, label='Raw')
# plt.plot(tick_positions, power_corr, label='Corrected')
# plt.legend()
# plt.ylim([0, 5.5e6])
# plt.xticks(tick_positions, names)
# plt.gca().tick_params(axis='x', rotation=15)
# plt.ylabel('Noise Power')
# # plt.title(position)
# plt.show()
#
# corr_diff = power_corr[-1] - power_corr[0]
# raw_diff = power_raw[-1] - power_raw[0]
# print(100 * (raw_diff - corr_diff) / raw_diff)

# Scatter plot
tick_positions = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]
tick_to_name = range(len(names))

plt.figure()
plt.scatter(tick_positions, power_raw, label='Raw')
plt.scatter(tick_positions, power_corr, label='Corrected')
plt.legend()
plt.ylim([1e5, 2e5])
plt.xticks(tick_to_name, names)
plt.gca().tick_params(axis='x', rotation=15)
plt.ylabel('Noise Power')
plt.title(title)
plt.show()

# %% Spectrum comparisons
# vis.compare_unnormalized_freq(signal_raw_full, names, dt, name="Raw Data", xlim=None, ylim=[0, 1e7],
#                               log_scale=False,
#                               rep_axis=1,
#                               subplot=True)
# vis.compare_unnormalized_freq(signal_corr_full, names, dt, name="Corrected Data", xlim=None, ylim=[0, 1e7],
#                               log_scale=False,
#                               rep_axis=1,
#                               subplot=True)
# vis.compare_unnormalized_freq(signal_raw_full - signal_corr_full, names, dt, name="Raw - Corrected", xlim=None,
#                               ylim=[0, 1e7],
#                               log_scale=False,
#                               rep_axis=1,
#                               subplot=True)
