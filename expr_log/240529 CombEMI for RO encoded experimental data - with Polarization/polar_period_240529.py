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
import comb_test_240529 as cs

# plot data
ylim_time = [-15000, 15000]
ylim_freq = [-5e6, 5e6]
ylim_freq_zfilled = [-1e8, 1e8]
Disp_Intermediate = True


# %% support functions
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


# %% Load raw data
file_name = 'NoRO_16Scans'
mat_file = sp.io.loadmat('sim_input/' + file_name + '.mat')
raw_pol_all = mat_file['ch1'][:1000, :]  # the first 1000 points are polarization period
comb_pol_all = np.zeros(raw_pol_all.shape, dtype='complex')
# vis.repetitions(abs(sp.fft.fft(raw_pol,axis=0)), name="Polarization Period")

# %% Data parameters
N_echoes = 1
echo_len = int(raw_pol_all.shape[0] / N_echoes)
dt = 6e-6
TE = echo_len * dt
pre_drop = 0
post_drop = 0
pk_win = 0
pk_id = 0  # None for auto peak detection
polar_time = 0

max_iter = 200
rho = 1
lambda_val = -1  # -1 for auto regularization
# auto lambda parameters
ft_prtct = 2

# %% Process data

for i in [0]:
    vis.freq_plot(raw_pol_all[:, i], dt=dt, name="Polarization Period", ylim=ylim_freq)
    print('Processing Repetition #', i, " out of ", raw_pol_all.shape[1])

    raw_sig = np.squeeze(raw_pol_all[:, i])
    if Disp_Intermediate:
        vis.complex(raw_sig, name='Raw Signal', rect=True, ylim=ylim_time)
        vis.freq_plot(raw_sig, dt=dt, name='Raw Signal', ylim=ylim_freq_zfilled)
        vis.complex(raw_sig[0:400], name='Raw Signal', rect=True, ylim=ylim_time)

    # %% perform comb
    cancelled_comb_raw = cs.comb_optimized(signal=raw_sig, N_echoes=N_echoes, TE=TE, dt=dt, lambda_val=lambda_val,
                                           max_iter=max_iter, tol=0.1, pre_drop=pre_drop, post_drop=post_drop,
                                           pk_win=pk_win, pk_id=pk_id, polar_time=polar_time, rho=rho,
                                           ft_prtct=ft_prtct, Disp_Intermediate=Disp_Intermediate,
                                           ylim_time=ylim_time, ylim_freq=ylim_freq_zfilled)

    samp_mask_w_pol = cs.gen_samp_mask(acq_len=echo_len, N_echoes=N_echoes, TE_len=np.uint16(TE / dt),
                                       polar_time=polar_time, dt=dt, pol=True)
    cancelled_comb = cancelled_comb_raw[samp_mask_w_pol]

    if Disp_Intermediate:
        vis.complex(cancelled_comb_raw, name='Comb Raw Output', rect=True, ylim=ylim_time)
        vis.complex(cancelled_comb_raw[0:400], name='Comb Raw Output', rect=True, ylim=ylim_time)
        vis.freq_plot(cancelled_comb_raw, dt=dt, name='Comb Raw Output')

    comb = np.squeeze(raw_sig) - cancelled_comb

    if Disp_Intermediate:
        vis.complex(comb, name='Comb Cancelled Result', rect=True, ylim=ylim_time)
        vis.complex(comb[0:400], name='Comb Cancelled Result', rect=True, ylim=ylim_time)

        vis.freq_plot(raw_sig[0:echo_len], dt=dt, name='Raw Sig', ylim=ylim_freq)
        vis.freq_plot(comb[0:echo_len], dt=dt, name='Comb Cancelled Result', ylim=ylim_freq)
    comb_pol_all[:, i] = comb
