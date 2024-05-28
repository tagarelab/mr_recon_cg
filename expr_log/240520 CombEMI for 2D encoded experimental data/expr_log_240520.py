"""
   Name: expr_log_240513.py
   Purpose:
   Created on: 5/13/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import comb_test_240520 as cs
import mr_io
import visualization as vis

# plot data
ylim_time = [-200000, 200000]
ylim_freq = [-2.5e6, 2.5e6]


# %% support functions
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


# %% Load raw data
file_name = 'FirstTryPE#'
mat_file = sp.io.loadmat('sim_input/' + file_name + '.mat')
raw_sig_allPE = mat_file['ch1']
comb_sig_allPE = np.zeros(raw_sig_allPE.shape, dtype='complex')

# %% Data parameters
N_echoes = 720
TE = 1.5e-3
dt = 6e-6
pre_drop = 4
post_drop = 0
pk_win = 0.5
pk_id = None  # None for auto peak detection, 20 for peak at 17th PE
polar_time = 0

max_iter = 200
rho = 1
lambda_val = 92331432  # -1 for auto regularization
# auto lambda parameters
ft_prtct = 20
echo_len = int(raw_sig_allPE.shape[0] / N_echoes)

# %% Process data

# for i in range(raw_sig_allPE.shape[1]):
for i in [17]:
    print('Processing PE#', i, " out of ", raw_sig_allPE.shape[1])

    raw_sig = np.squeeze(raw_sig_allPE[:, i])
    vis.complex(raw_sig, name='Raw Signal', rect=True, ylim=ylim_time)

    vis.complex(raw_sig[0:400], name='Raw Signal', rect=True, ylim=ylim_time)

    # %% perform comb
    cancelled_comb_raw = cs.comb_optimized(signal=raw_sig, N_echoes=N_echoes, TE=TE, dt=dt, lambda_val=lambda_val,
                                           max_iter=max_iter, tol=0.1, pre_drop=pre_drop, post_drop=post_drop,
                                           pk_win=pk_win, pk_id=pk_id, polar_time=polar_time, rho=rho,
                                           ft_prtct=ft_prtct)

    samp_mask_w_pol = cs.gen_samp_mask(acq_len=echo_len, N_echoes=N_echoes, TE_len=np.uint16(TE / dt),
                                       polar_time=polar_time, dt=dt, pol=True)
    cancelled_comb = cancelled_comb_raw[samp_mask_w_pol]

    vis.complex(cancelled_comb, name='Comb Raw Output', rect=True, ylim=ylim_time)
    vis.complex(cancelled_comb[0:400], name='Comb Raw Output', rect=True, ylim=ylim_time)
    vis.freq_plot(cancelled_comb, dt=dt, name='Comb Raw Output')

    comb = np.squeeze(raw_sig) - cancelled_comb

    vis.complex(comb, name='Comb Cancelled Result', rect=True, ylim=ylim_time)
    vis.complex(comb[0:400], name='Comb Cancelled Result', rect=True, ylim=ylim_time)

    vis.freq_plot(raw_sig[0:echo_len], dt=dt, name='Raw Sig', ylim=ylim_freq)
    vis.freq_plot(comb[0:echo_len], dt=dt, name='Comb Cancelled Result', ylim=ylim_freq)

    vis.freq_plot(avg_first_k_peaks(raw_sig, echo_len, k=10), dt=dt, name='Raw Sig, Avg of First k Peaks',
                  ylim=ylim_freq)
    vis.freq_plot(avg_first_k_peaks(comb, echo_len, k=10), dt=dt, name='Comb Cancelled Result, Avg of First k Peaks',
                  ylim=ylim_freq)
    comb_sig_allPE[:, i] = comb

# %% Save data
# mr_io.save_single_mat(comb_sig_allPE, file_name + '_comb', 'sim_output/', date=True)
