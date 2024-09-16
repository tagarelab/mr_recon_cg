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

# %% Load data
dict = {
    0: "Pos1_NoPhant_B0off_prepol_0_slice_0_PAoff.mat",
    1: "Pos1_NoPhant_B0on_prepol_0_slice_0.mat",
    2: "Pos1_NoPhant_B0on_prepol_0_slice_0_PAoff.mat",
    3: "Pos1_NoPhant_B0on_prepol_0_slice_3.mat",
    4: "Pos1_NoPhant_B0on_prepol_0_slice_6.3.mat",
    5: "Pos1_NoPhant_B0on_prepol_100_slice_0.mat",
    6: "Pos1_NoPhant_B0on_prepol_100_slice_3.mat",
    7: "Pos1_NoPhant_B0on_prepol_100_slice_6.3.mat",
    8: "Pos1_NoPhant_B0on_prepol_100_slice_6.3_32avg.mat",
    9: "Pos1_NoPhant_B0pwroff_prepol_0_slice_0_PAoff.mat",
    10: "Pos1_WithPhant_B0on_prepol_0_slice_0.mat",
    11: "Pos1_WithPhant_B0on_prepol_0_slice_3.mat",
    12: "Pos1_WithPhant_B0on_prepol_0_slice_6.3.mat",
    13: "Pos1_WithPhant_B0on_prepol_100_slice_0.mat",
    14: "Pos1_WithPhant_B0on_prepol_100_slice_3.mat",
    15: "Pos1_WithPhant_B0on_prepol_100_slice_6.3.mat",
    16: "Pos1_WithPhant_B0on_prepol_100_slice_6.3_32avg.mat",
    17: "Pos2_WithPhant_B0on_prepol_0_slice_0.mat",
    18: "Pos2_WithPhant_B0on_prepol_0_slice_3.mat",
    19: "Pos2_WithPhant_B0on_prepol_0_slice_6.3.mat",
    20: "Pos2_WithPhant_B0on_prepol_100_slice_0.mat",
    21: "Pos2_WithPhant_B0on_prepol_100_slice_3.mat",
    22: "Pos2_WithPhant_B0on_prepol_100_slice_6.3.mat",
    23: "Pos2_WithPhant_B0on_prepol_100_slice_6.3_32avg.mat"
}
# B0 originated noise
# names = ["B0 Disconnected", "B0 Connected to Power", "B0 Turned On",
#          "PA also Turned On", "RO 3%", "RO 6.3%"]
# ids = [9, 0, 2, 1, 3, 4]

# Prepol noise
names = ["B0 Disconnected", "Prepol off, RO 0%", "Prepol off, RO 3%", "Prepol off RO 6.3%", "Prepol on, RO 0%",
         "Prepol on, RO 3%", "Prepol on, RO 6.3%"]
ids = [9, 1, 3, 4, 5, 6, 7]

Nc = 3
ylim_temp = [0, 1000]
ylim_temp_freq = [-60, 60]
ylim_temp_freq2 = None

power_raw = np.zeros(len(names))
power_corr = np.zeros(len(names))

for i in range(len(names)):
    dict_i = mr_io.load_single_mat('sim_input/' + dict[ids[i]])
    raw_sig_i = dict_i['raw_sig']
    editer_corr_i = dict_i['editer_corr']
    diff = editer_corr_i - raw_sig_i

    # power of pre and after noise cancellation
    power_raw[i] = na.power(raw_sig_i)
    power_corr[i] = na.power(editer_corr_i)

    # dt = 6e-6
    #
    # vis.freq_plot(editer_corr_i[200:300], dt=dt, name='Second Echo, %d Avgs first, then corrected with EDITER' % i,
    #               ifft=True,
    #               ylim=ylim_temp_freq)

tick_positions = range(len(names))

plt.figure()
plt.plot(tick_positions, power_raw, label='Raw')
plt.plot(tick_positions, power_corr, label='Corrected')
plt.legend()
plt.ylim([0, 5.5e6])
plt.xticks(tick_positions, names)
plt.gca().tick_params(axis='x', rotation=15)
plt.ylabel('Noise Power')
plt.title('Noise Power Before and After EDITER Correction')
plt.show()

corr_diff = power_corr[-1] - power_corr[0]
raw_diff = power_raw[-1] - power_raw[0]
print(100 * (raw_diff - corr_diff) / raw_diff)
