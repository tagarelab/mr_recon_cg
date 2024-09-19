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
loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\240911_EMI_hunt_S2E5\\EDITER\\"

file_dict = {
    0: "Pos2_NoPhant_B0off_prepol_0_slice_0_PAoff.mat",
    1: "Pos2_NoPhant_B0on_prepol_0_slice_0_1.mat",
    2: "Pos2_NoPhant_B0on_prepol_0_slice_0_2.mat",
    3: "Pos2_NoPhant_B0on_prepol_0_slice_0_3.mat",
    4: "Pos2_NoPhant_B0on_prepol_0_slice_0_PAoff.mat",
    5: "Pos2_NoPhant_B0on_prepol_0_slice_3_1.mat",
    6: "Pos2_NoPhant_B0on_prepol_0_slice_3_2.mat",
    7: "Pos2_NoPhant_B0on_prepol_0_slice_3_3.mat",
    8: "Pos2_NoPhant_B0on_prepol_0_slice_6.3_1.mat",
    9: "Pos2_NoPhant_B0on_prepol_0_slice_6.3_2.mat",
    10: "Pos2_NoPhant_B0on_prepol_0_slice_6.3_3.mat",
    11: "Pos2_NoPhant_B0on_prepol_100_slice_0_1.mat",
    12: "Pos2_NoPhant_B0on_prepol_100_slice_0_2.mat",
    13: "Pos2_NoPhant_B0on_prepol_100_slice_0_3.mat",
    14: "Pos2_NoPhant_B0on_prepol_100_slice_3_1.mat",
    15: "Pos2_NoPhant_B0on_prepol_100_slice_3_2.mat",
    16: "Pos2_NoPhant_B0on_prepol_100_slice_3_3.mat",
    17: "Pos2_NoPhant_B0on_prepol_100_slice_6.3_1.mat",
    18: "Pos2_NoPhant_B0on_prepol_100_slice_6.3_2.mat",
    19: "Pos2_NoPhant_B0on_prepol_100_slice_6.3_3.mat",
    20: "Pos2_NoPhant_B0pwroff_prepol_0_slice_0_PAoff.mat",
    21: "Pos1_NoPhant_B0off_prepol_0_slice_0_PAoff.mat",
    22: "Pos1_NoPhant_B0on_prepol_0_slice_0_1.mat",
    23: "Pos1_NoPhant_B0on_prepol_0_slice_0_2.mat",
    24: "Pos1_NoPhant_B0on_prepol_0_slice_0_3.mat",
    25: "Pos1_NoPhant_B0on_prepol_0_slice_0_PAoff.mat",
    26: "Pos1_NoPhant_B0on_prepol_0_slice_3_1.mat",
    27: "Pos1_NoPhant_B0on_prepol_0_slice_3_2.mat",
    28: "Pos1_NoPhant_B0on_prepol_0_slice_3_3.mat",
    29: "Pos1_NoPhant_B0on_prepol_0_slice_6.3_1.mat",
    30: "Pos1_NoPhant_B0on_prepol_0_slice_6.3_2.mat",
    31: "Pos1_NoPhant_B0on_prepol_0_slice_6.3_3.mat",
    32: "Pos1_NoPhant_B0on_prepol_100_slice_0_1.mat",
    33: "Pos1_NoPhant_B0on_prepol_100_slice_0_2.mat",
    34: "Pos1_NoPhant_B0on_prepol_100_slice_0_3.mat",
    35: "Pos1_NoPhant_B0on_prepol_100_slice_3_1.mat",
    36: "Pos1_NoPhant_B0on_prepol_100_slice_3_2.mat",
    37: "Pos1_NoPhant_B0on_prepol_100_slice_3_3.mat",
    38: "Pos1_NoPhant_B0on_prepol_100_slice_6.3_1.mat",
    39: "Pos1_NoPhant_B0on_prepol_100_slice_6.3_2.mat",
    40: "Pos1_NoPhant_B0on_prepol_100_slice_6.3_3.mat",
    41: "Pos1_NoPhant_B0pwroff_prepol_0_slice_0_PAoff.mat",
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
names = ["B0 Disconnected", "Prepol off, RO 0%", "Prepol on, RO 0%", "Prepol off, RO 3%",
         "Prepol on, RO 3%", "Prepol off RO 6.3%", "Prepol on, RO 6.3%"]
ids = [41, 22, 32, 26, 35, 29, 38]  # position 1

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

    signal_raw_full[i, :] = combEMI.sampled_to_full(signal=raw_sig_i, polar_time=0, post_polar_gap_time=0, dt=dt,
                                                    acq_len=100, N_echoes=200, TE_len=int(TE / dt))
    signal_corr_full[i, :] = combEMI.sampled_to_full(signal=editer_corr_i, polar_time=0, post_polar_gap_time=0, dt=dt,
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
# plt.title(position)
# plt.show()

# corr_diff = power_corr[-1] - power_corr[0]
# raw_diff = power_raw[-1] - power_raw[0]
# print(100 * (raw_diff - corr_diff) / raw_diff)
#
# # Scatter plot
# tick_positions = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
# tick_to_name = range(len(names))
#
# plt.figure()
# plt.scatter(tick_positions, power_raw, label='Raw')
# plt.scatter(tick_positions, power_corr, label='Corrected')
# plt.legend()
# plt.ylim([0, 5.5e6])
# plt.xticks(tick_to_name, names)
# plt.gca().tick_params(axis='x', rotation=15)
# plt.ylabel('Noise Power')
# plt.title(position)
# plt.show()


# %% Spectrum comparisons
vis.compare_unnormalized_freq(signal_raw_full, names, dt, name="Raw Data", xlim=None, ylim=[0, 1e7],
                              log_scale=False,
                              rep_axis=1,
                              subplot=True)
vis.compare_unnormalized_freq(signal_corr_full, names, dt, name="Corrected Data", xlim=None, ylim=[0, 1e7],
                              log_scale=False,
                              rep_axis=1,
                              subplot=True)
vis.compare_unnormalized_freq(signal_raw_full - signal_corr_full, names, dt, name="Raw - Corrected", xlim=None,
                              ylim=[0, 1e7],
                              log_scale=False,
                              rep_axis=1,
                              subplot=True)
