"""
   Name: visualize_peaks.py
   Purpose:
   Created on: 6/18/2024
   Created by: Heng Sun
   Additional Notes: 
"""
import numpy as np
import mr_io
import scipy as sp
import matplotlib.pyplot as plt
import algebra as algb


# %% Defind support functions
def compare_unnormalized_freq(signal_list, legends, dt, name=None, xlim=None, ylim=None, log_scale=True):
    plt.figure()
    frag_len = signal_list.shape[1]
    freq_axis = sp.fft.fftshift(sp.fft.fftfreq(frag_len, dt)) / 1000
    for i in range(signal_list.shape[0]):
        signal = signal_list[i, :]
        signal_ft = sp.fft.fftshift(sp.fft.fft(sp.fft.ifftshift(signal)))

        if log_scale:
            signal_ft = algb.linear_to_db(signal_ft)

        plt.plot(freq_axis, abs(signal_ft), label=legends[i])

    plt.legend()
    plt.xlabel('Frequency (kHz)')
    if xlim is not None:
        plt.xlim(xlim)
    if log_scale:
        plt.ylabel('Magnitude (dB)')
    else:
        plt.ylabel('Magnitude')
    if ylim is not None:
        plt.ylim(ylim)
    if name is not None:
        plt.title(name)
    plt.show()


# %% Data Attributes
channel_info_1 = ['EMI_X_', 'Signal_', 'EMI_Y_', 'EMI_Z_']
# file_name_stem = ['Sig_3AxisEMI_', 'SigOnly_']
file_name_stem = ['SigOnly_']
file_name_steps = ['Step1_Console', 'Step2_Console_Coil', 'Step3_Console_Preamp', 'Step4_Console_Preamp_Coil',
                   'Step5_Console_Preamp_TRSwitch_Coil',
                   'Step6_Console_Preamp_TRSwitch_Tx_Coil', 'Step7_Console_Preamp_TRSwitch_Tx_PA_Coil']
segment_names = ['Prepol_', 'TE_']

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
data = mr_io.load_single_mat('sim_output/sig_pol_all_steps_06182024')
pol_all_steps = data['pol_all_steps']
sig_all_steps = data['sig_all_steps']

compare_unnormalized_freq(sig_all_steps, file_name_steps, dt, name='TE Acquisition')
compare_unnormalized_freq(pol_all_steps, file_name_steps, dt, name='Pre Polarization')

# compare_unnormalized_freq(sig_all_steps, file_name_steps, dt, name='TE Acquisition', log_scale=False)
# compare_unnormalized_freq(pol_all_steps, file_name_steps, dt, name='Pre Polarization', log_scale=False)


# %%
compare_unnormalized_freq(pol_all_steps[0:4, :], file_name_steps[0:4], dt, name='Preamp')
compare_unnormalized_freq(pol_all_steps[3:5, :], file_name_steps[3:5], dt, name='TR Switch')
compare_unnormalized_freq(pol_all_steps[4:6, :], file_name_steps[4:6], dt, name='Tx')
compare_unnormalized_freq(pol_all_steps[5:7, :], file_name_steps[5:7], dt, name='PA')
