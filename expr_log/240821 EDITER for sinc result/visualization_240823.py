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
import mr_io


# %% Define functions
def easy_snr(data, noise_region):
    """
    Calculate the SNR of the data
    :param data:
    :param noise_region:
    :return:
    """
    signal = np.max(np.abs(data))
    noise = np.std(np.abs(data[noise_region[0]:noise_region[1]]))
    return signal / noise


# %% Load data
dict = mr_io.load_single_mat('sim_output/30kHz_sinc_iter_EDITER_corrected.mat')

number_of_avg = [1, 8, 16, 32, 64]
Nc = 3
ylim_temp = [0, 1000]
ylim_temp_freq = [-60, 60]
ylim_temp_freq2 = None

for i in number_of_avg:
    dict_i = mr_io.load_single_mat('sim_output/30kHz_sinc_iter_EDITER_corrected_avg%d.mat' % i)
    # datafft_i = np.mean(dict_i['datafft'], axis=1)
    editer_corr_i = np.mean(dict_i['editer_corr'], axis=1)
    editer_corr = np.mean(dict['editer_corr'][:, :i], axis=1)
    diff = editer_corr - editer_corr_i

    # vis.absolute([editer_corr, diff], name='Corrected with EDITER, %d Avgs' % i,
    #              legends=['Before Averages', 'difference with After Averages'], ylim=ylim_temp)
    #
    # region = [0, 500]
    # vis.absolute([editer_corr[region[0]:region[1]], diff[region[0]:region[1]]], name='Corrected with EDITER, %d Avgs, Zoom in to %s' % (i,
    #                                                                                                              region),
    #              legends=['Before Averages', 'difference with After Averages'], ylim=ylim_temp)

    dt = 6e-6

    vis.freq_plot(editer_corr_i[200:300], dt=dt, name='Second Echo, %d Avgs first, then corrected with EDITER' % i,
                  ifft=True,
                  ylim=ylim_temp_freq)

# for i in number_of_avg:
#     datafft = np.mean(dict['datafft'][:, :i], axis=1)
#     editer_corr = np.mean(dict['editer_corr'][:, :i], axis=1)
#
#     # visualization
#     # ylim_temp = [0, np.max(np.abs(datafft))]
#     # ylim_temp_freq = [-200, 200]
#     # vis.absolute(editer_corr, name='Corrected with EDITER, %d averages' % i, ylim=ylim_temp)
#     # vis.absolute(datafft, name='Uncorrected, %d averages' % i, ylim=ylim_temp)
#
#     region = [0, 500]
#     vis.absolute(editer_corr[region[0]:region[1]], name='Corrected with EDITER, Zoom in to %s, %d averages' % (
#         region, i),
#                  ylim=ylim_temp)
#     # vis.absolute(datafft[region[0]:region[1]], name='Uncorrected, %d averages' % i, ylim=ylim_temp)
#
#     # region = [12500, 20000]
#     # vis.absolute(editer_corr[region[0]:region[1]], name='Corrected with EDITER, Zoom in to %s, %d averages' % (
#     #     region, i), ylim=ylim_temp)
#     # vis.absolute(datafft[region[0]:region[1]], name='Uncorrected, %d averages' % i, ylim=ylim_temp)
#
#     dt = 6e-6
#
#     # vis.freq_plot(editer_corr, dt=dt, name='Corrected with EDITER, %d averages' % i, ifft=True,
#     #               ylim=ylim_temp_freq)
#     # vis.freq_plot(datafft, dt=dt, name='Uncorrected, %d averages' % i, ifft=True, ylim=ylim_temp_freq)
#
#     vis.freq_plot(editer_corr[200:300], dt=dt, name='Second Echo, Corrected with EDITER, %d averages' % i,
#                   ifft=True,
#                   ylim=ylim_temp_freq)
#     # vis.freq_plot(datafft[200:300], dt=dt, name='Second Echo, Uncorrected, %d averages' % i, ifft=True,
#     #               ylim=ylim_temp_freq)
