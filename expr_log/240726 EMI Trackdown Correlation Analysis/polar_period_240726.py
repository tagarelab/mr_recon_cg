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


# %% Data Attributes
channel_info = ['EMI_X_', 'Signal_', 'EMI_Y_', 'EMI_Z_']
# file_name_stem = ['', '_Blank']
# file_name_steps = ['Step_4_Console_Preamp_Coil', 'Step_6_Console_Preamp_TRSwitch_Tx_Coil', 'Step_7_Coil_Everything',
#                    'Step_7_Coil_Everything_PAon_nopulsing', 'Step_7_Coil_Everything_PAon_pulsing',
#                    'Step_8_Coil_Everything_Pol_withPol_noTx']

# file_name_stem = ['', '_Blank']
# file_name_steps = ['Step_8_Coil_Everything_Pol_withPol_noTx',
#                    'Step_10_Coil_Everything_Pol_Phantom_withPol_Phant',
#                    'Step_11_Coil_Everything_Pol_Phantom_withPol_Phant']

# file_name_steps = ['Step_9_Coil_Everything_Pol_withPol_noPhant',
#                    'Step_10_Coil_Everything_Pol_Phantom_withPol_Phant']
file_name_steps = ['Step_9_Coil_Everything_Pol_withPol_noPhant']
file_name_stem = ['', '_16avg']

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

sig_all_steps = np.zeros(
    (len(channel_info), len(file_name_stem), len(file_name_steps), int(np.ceil(N_echoes_real * TE_real / dt))),
    dtype='complex')
pol_all_steps = np.zeros(
    (len(channel_info), len(file_name_stem), len(file_name_steps), int(np.ceil(N_rep_pol * rep_time_pol / dt))),
    dtype='complex')
# sig_all_steps = []
# pol_all_steps = []

# %% Load raw data
for stem_id in range(len(file_name_stem)):
    for step_id in range(len(file_name_steps)):
        file_name = file_name_steps[step_id] + file_name_stem[stem_id]
        print("Processing File ", file_name)
        mat_file = sp.io.loadmat('sim_input/' + file_name + '.mat')
        for channel_id in range(len(channel_info)):
            raw_pol_all = mat_file['ch' + str(channel_id + 1)][:N_rep_pol * rep_len_pol, :]
            raw_sig_all = mat_file['ch' + str(channel_id + 1)][N_rep_pol * rep_len_pol:, :]
            if Perform_Comb:
                comb_pol_all = np.zeros(raw_pol_all.shape, dtype='complex')
                comb_sig_all = np.zeros(raw_sig_all.shape, dtype='complex')

            # %% Process data
            # for i in range(N_avg):
            for i in [0]:
                # print('Processing Repetition #', str(i + 1), " out of ", raw_pol_all.shape[1])
                for seg_id in range(len(segment_names)):
                    if seg_id == 0:
                        N_echoes = N_rep_pol
                        echo_len = rep_len_pol
                        TE = rep_time_pol
                        raw_sig = np.squeeze(raw_pol_all[:, i])
                    elif seg_id == 1:
                        N_echoes = N_echoes_real
                        echo_len = echo_len_real
                        TE = TE_real
                        raw_sig = np.squeeze(raw_sig_all[:, i])
                    # if Disp_Intermediate:
                    #     vis.complex(raw_sig, name='Raw Signal', rect=True, ylim=ylim_time)
                    #     vis.freq_plot(raw_sig, dt=dt, name='Raw Signal', ylim=ylim_freq)
                    #     vis.complex(raw_sig[0:400], name='Raw Signal', rect=True, ylim=ylim_time)

                    # Parameter formatting
                    polar_period = cs.calculate_polar_period(polar_time=polar_time, dt=dt)
                    post_polar_gap_period = cs.calculate_polar_period(polar_time=post_polar_gap_time, dt=dt)
                    signal_te = raw_sig[polar_period:].T
                    pre_drop = np.uint16(pre_drop)
                    post_drop = np.uint16(post_drop)

                    # Set binary mask for sampling window
                    TE_len = np.uint16(TE / dt)
                    rep_len = signal_te.shape[0]
                    acq_len = np.uint16(rep_len / N_echoes)

                    # Generate masks
                    samp_all = cs.gen_samp_mask(acq_len, N_echoes, TE_len, polar_time, post_polar_gap_time, dt,pre_drop=pre_drop, post_drop=post_drop)
                    zfilled_data = cs.sampled_to_full(raw_sig, polar_time, post_polar_gap_time, dt, acq_len, N_echoes,
                                                      TE_len)

                    if seg_id == 0:
                        pol_all_steps[channel_id, stem_id, step_id, :] = np.where(samp_all, zfilled_data, 0)

                    elif seg_id == 1:
                        sig_all_steps[channel_id, stem_id, step_id, :] = np.where(samp_all, zfilled_data, 0)

dict = {"sig_all_steps": sig_all_steps, "pol_all_steps": pol_all_steps, "file_name_steps": file_name_steps,
        "file_name_stem": file_name_stem, "channel_info": channel_info, "segment_names": segment_names}
# mr_io.save_dict(dict, 'sig_pol_all_steps', 'sim_output/', date=True)

# compare_unnormalized_freq(sig_all_steps, file_name_steps, dt, name='TE Acquisition', ylim=ylim_freq_zfilled)
# compare_unnormalized_freq(pol_all_steps, file_name_steps, dt, name='Pre Polarization', ylim=ylim_freq_zfilled)

# compare_unnormalized_freq(sig_all_steps, file_name_steps, dt, name='TE Acquisition', log_scale=False)
# compare_unnormalized_freq(pol_all_steps, file_name_steps, dt, name='Pre Polarization', log_scale=False)
