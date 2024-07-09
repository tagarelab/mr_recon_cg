"""
   Name: visualize_peaks.py
   Purpose:
   Created on: 6/18/2024
   Created by: Heng Sun
   Additional Notes: 
"""
import numpy as np
import mr_io
import visualization as vis
import scipy as sp
import matplotlib.pyplot as plt
import algebra as algb

# %% Define support functions


# %% Data Attributes

# peak recognization
peak_info = {"height": 30, "distance": 1000}

# Number of averages
N_avg = 4

# real TE part
N_echoes_real = 480
echo_len_real = 120
dt = 5e-6
TE_real = 2e-3

# polarization period
N_rep_pol = 17
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

# %% Load mat file
# data = mr_io.load_single_mat('sim_output/sig_pol_all_steps_06252024')
# data = mr_io.load_single_mat('sim_output/sig_pol_step_8_10_11_06272024')
data = mr_io.load_single_mat('sim_output/with_without_phant_06272024')
pol_all_steps = data['pol_all_steps']
sig_all_steps = data['sig_all_steps']
file_name_steps = data['file_name_steps']
channel_info = data['channel_info']
file_name_stem = ['No Avg', '16 Avgs']
# file_name_stem = ['60kHz Coil', 'Blank load']
segment_names = data['segment_names']

# %% First plot the signal and polarization for each step
# ylim_sig = [0, 5e7]
# ylim_prepol = [0, 1.5e9]
# for step_id in range(len(file_name_steps)):
#     # for step_id in [0]:
#     channel_id = 1
#     vis.compare_unnormalized_freq(signal_list=sig_all_steps[channel_id, :, step_id, :], legends=file_name_stem, dt=dt,
#                                   name=channel_info[channel_id] + 'TE Acquisition' + file_name_steps[step_id],
#                                   subplot=True)
#     vis.compare_unnormalized_freq(signal_list=pol_all_steps[channel_id, :, step_id, :], legends=file_name_stem, dt=dt,
#                                   name=channel_info[channel_id] + 'Pre Polarization' + file_name_steps[step_id],
#                                   subplot=True)

# %% Each step, plot the signal and polarization for each segment
# ylim_sig = [0, 5e7]
# ylim_prepol = [0, 1.5e9]
# step_ids = range(len(file_name_stem))
# stem_id = 0
# channel_id = 1
# vis.compare_unnormalized_freq(signal_list=sig_all_steps[channel_id, stem_id, step_ids, :], legends=file_name_steps[step_ids],
#                               dt=dt,
#                               name=channel_info[channel_id] + '_TE Acquisition' +file_name_stem[stem_id],
#                               subplot=True, ylim=ylim_sig)
# vis.compare_unnormalized_freq(signal_list=pol_all_steps[channel_id, stem_id, step_ids, :], legends=file_name_steps[step_ids],
#                               dt=dt,
#                               name=channel_info[channel_id] + '_Pre Polarization'+file_name_stem[stem_id],
#                               subplot=True, ylim=ylim_prepol)

# %% Each step, plot the signal and polarization for all Signal and EMI channels
for step_id in range(len(file_name_steps)):
    for stem_id in range(len(file_name_stem)):
        vis.compare_unnormalized_freq(signal_list=sig_all_steps[:, stem_id, step_id, :], legends=channel_info, dt=dt,
                                      name='TE Acquisition' + file_name_steps[step_id] + file_name_stem[stem_id],
                                      subplot=True)
        vis.compare_unnormalized_freq(signal_list=pol_all_steps[:, stem_id, step_id, :], legends=channel_info, dt=dt,
                                      name='Pre Polarization' + file_name_steps[step_id] + file_name_stem[stem_id],
                                      subplot=True)
