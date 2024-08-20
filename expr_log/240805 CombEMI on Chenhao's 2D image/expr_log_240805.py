"""
   Name: expr_log_240701.py
   Purpose:
   Created on: 7/2/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from denoise import combEMI as cs
import mr_io
import visualization as vis

# plot data
ylim_time = None
ylim_freq = None
ylim_freq_zfilled = None
Disp_Intermediate = False


# %% support functions
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


# %% Load raw data
file_name = "SeventhTryAve#1PE#"
mat_file = sp.io.loadmat('sim_input/' + file_name + '.mat')
raw_sig_all = mat_file['ch2']
comb_sig_all = np.zeros(raw_sig_all.shape, dtype='complex')

# %% Data parameters
N_echoes = 120
TE = 3.5e-3
dt = 6e-6
pre_drop = 0
post_drop = 0
pk_win = 0.33
pk_id = 58  # None for auto peak detection
polar_period = 0
polar_time = dt * polar_period
post_polar_gap_time = 0

# polar_period = 1000
# polar_time = dt*polar_period  # 1000 points for polarization
# post_polar_gap_time = 200e-6+35e-3+100e-3+20e-6+80e-6   # maybe also add half of the 90 degree pulse length?

max_iter = 200
rho = 1
lambda_val = -1  # -1 for auto regularization
auto_corr_tol = 0.1
method = "peak_pick"
# auto lambda parameters
ft_prtct = 5
echo_len = int((raw_sig_all.shape[0] - polar_period) / N_echoes)

# %% Process data

# for i in range(raw_sig_all.shape[1]):
for i in [33]:
    print('Processing Repetition #', i + 1, " out of ", raw_sig_all.shape[1])

    raw_sig = np.squeeze(raw_sig_all[:, i])
    if Disp_Intermediate:
        vis.complex(raw_sig, name='Raw Signal', rect=True, ylim=ylim_time)
        vis.freq_plot(raw_sig, dt=dt, name='Raw Signal', ylim=ylim_freq_zfilled)
        vis.complex(raw_sig[0:400], name='Raw Signal', rect=True, ylim=ylim_time)

    # %% perform comb
    cancelled_comb_raw = cs.comb_optimized(signal=raw_sig, N_echoes=N_echoes, TE=TE, dt=dt, lambda_val=lambda_val,
                                           max_iter=max_iter, tol=0.1, pre_drop=pre_drop, post_drop=post_drop,
                                           pk_win=pk_win, pk_id=pk_id,
                                           polar_time=polar_time, post_polar_gap_time=post_polar_gap_time, rho=rho,
                                           ft_prtct=ft_prtct, Disp_Intermediate=Disp_Intermediate,
                                           ylim_time=ylim_time, ylim_freq=ylim_freq_zfilled,
                                           auto_corr_tol=auto_corr_tol, method=method)

    samp_mask_w_pol = cs.gen_samp_mask(acq_len=echo_len, N_echoes=N_echoes, TE_len=np.uint16(TE / dt),
                                       polar_time=polar_time, post_polar_gap_time=post_polar_gap_time, dt=dt,
                                       pol=True, pre_drop=pre_drop, post_drop=post_drop)
    cancelled_comb = cancelled_comb_raw[samp_mask_w_pol]

    if Disp_Intermediate:
        vis.complex(cancelled_comb_raw, name='Comb Raw Output', rect=True, ylim=ylim_time)
        vis.complex(cancelled_comb_raw[0:400], name='Comb Raw Output', rect=True, ylim=ylim_time)
        vis.freq_plot(cancelled_comb_raw, dt=dt, name='Comb Raw Output')

    comb = np.squeeze(raw_sig) - cancelled_comb

    if Disp_Intermediate:
        vis.complex(comb, name='Comb Cancelled Result', rect=True, ylim=ylim_time)
        vis.complex(comb[polar_period:polar_period + 400], name='Comb Cancelled Result', rect=True, ylim=ylim_time)

        vis.freq_plot(raw_sig[polar_period:polar_period + echo_len], dt=dt, name='Raw Sig', ylim=ylim_freq)
        vis.freq_plot(comb[polar_period:polar_period + echo_len], dt=dt, name='Comb Cancelled Result', ylim=ylim_freq)

        k = 10
        vis.freq_plot(avg_first_k_peaks(raw_sig[polar_period:], echo_len, k=k), dt=dt,
                      name="Raw Sig, Avg of First %d Peaks" % k,
                      ylim=ylim_freq)
        vis.freq_plot(avg_first_k_peaks(comb[polar_period:], echo_len, k=k), dt=dt,
                      name="Comb Cancelled Result, Avg of First %d Peaks" % k,
                      ylim=ylim_freq)
    comb_sig_all[:, i] = comb

# %% Save data
result = {"raw_sig_all": raw_sig_all, "comb_sig_all": comb_sig_all}
# mr_io.save_dict(result, file_name + '_comb', 'sim_output/', date=True)
# mr_io.save_single_mat(comb_sig_all, file_name + '_comb', 'sim_output/', date=True)
