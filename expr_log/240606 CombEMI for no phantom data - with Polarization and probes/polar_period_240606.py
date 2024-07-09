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
import comb_test_240606 as cs

# plot data
ylim_time = [-15000, 15000]
ylim_freq = [-5e7, 5e7]
ylim_freq_zfilled = [-5e8, 5e8]
Disp_Intermediate = True


# %% support functions
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


# %% Load raw data
file_name = '240606_noise_only'
mat_file = sp.io.loadmat('sim_input/' + file_name + '.mat')
raw_pol_all = mat_file['ch1'][:17 * 6000, :]
raw_sig_all = mat_file['ch1'][17 * 6000:, :]
comb_pol_all = np.zeros(raw_pol_all.shape, dtype='complex')
# vis.repetitions(abs(sp.fft.fft(raw_pol_all,axis=0)), name="Polarization Period")
# vis.absolute(raw_pol_all, name="Polarization Period")
# vis.absolute(raw_sig_all, name="Signal Period")

# %% Data parameters
# polarization period
N_echoes = 17
echo_len = 6000
dt = 5e-6
TE = echo_len * dt + 400  # TODO: double check the log to see if it is 400us or 800 us gap - TNMR is not very clear

# real TE part
# N_echoes = 480
# echo_len = 120
# dt = 5e-6
# TE = 2e-3

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

# %% Process data

for i in [0]:
    # vis.freq_plot(raw_pol_all[:, i], dt=dt, name="Polarization Period", ylim=ylim_freq)
    print('Processing Repetition #', i, " out of ", raw_pol_all.shape[1])

    raw_sig = np.squeeze(raw_pol_all[:, i])
    # raw_sig = np.squeeze(raw_sig_all[:, i])
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

    # Pick out the peak location
    sig_len_hf = np.uint16((acq_len - pre_drop - post_drop) * pk_win / 2)

    if pk_id is None:  # use input ID if otherwise
        if pk_id == "mid":
            pk_id = np.uint16((acq_len - pre_drop - post_drop) / 2)
        else:
            pks = np.abs(np.reshape(signal_te, (N_echoes, -1)))
            pks[:pre_drop, :] = 0  # sometimes there's a leak of signal at the beginning
            pks_val, pks_id = np.max(pks, axis=1), np.argmax(pks, axis=1)
            max_pks_id = np.argpartition(pks_val, -10)[-10:]
            pk_id = np.uint16(np.mean(pks_id[max_pks_id]))
        print("Auto-detected peak location: %d" % pk_id)

    # Generate masks
    samp_all = cs.gen_samp_mask(acq_len, N_echoes, TE_len, polar_time, post_polar_gap_time, dt)

    zfilled_data = cs.sampled_to_full(raw_sig, polar_time, post_polar_gap_time, dt, acq_len, N_echoes, TE_len)

    if Disp_Intermediate:
        vis.freq_plot(np.where(samp_all, zfilled_data, 0), dt=dt, name='Continuous Raw Signal (without Comb Shaped '
                                                                       'Mask)', ylim=ylim_freq_zfilled)

    # # %% perform comb
    # cancelled_comb_raw = cs.comb_optimized(signal=raw_sig, N_echoes=N_echoes, TE=TE, dt=dt, lambda_val=lambda_val,
    #                                        max_iter=max_iter, tol=0.1, pre_drop=pre_drop, post_drop=post_drop,
    #                                        pk_win=pk_win, pk_id=pk_id, polar_time=polar_time,
    #                                        post_polar_gap_time=post_polar_gap_time,
    #                                        rho=rho,
    #                                        ft_prtct=ft_prtct, Disp_Intermediate=Disp_Intermediate,
    #                                        ylim_time=ylim_time, ylim_freq=ylim_freq_zfilled)
    #
    # samp_mask_w_pol = cs.gen_samp_mask(acq_len=echo_len, N_echoes=N_echoes, TE_len=np.uint16(TE / dt),
    #                                    polar_time=polar_time, post_polar_gap_time=post_polar_gap_time, dt=dt, pol=True)
    # cancelled_comb = cancelled_comb_raw[samp_mask_w_pol]
    #
    # if Disp_Intermediate:
    #     vis.complex(cancelled_comb_raw, name='Comb Raw Output', rect=True, ylim=ylim_time)
    #     vis.complex(cancelled_comb_raw[0:400], name='Comb Raw Output', rect=True, ylim=ylim_time)
    #     vis.freq_plot(cancelled_comb_raw, dt=dt, name='Comb Raw Output')
    #
    # comb = np.squeeze(raw_sig) - cancelled_comb
    #
    # if Disp_Intermediate:
    #     vis.complex(comb, name='Comb Cancelled Result', rect=True, ylim=ylim_time)
    #     vis.complex(comb[0:400], name='Comb Cancelled Result', rect=True, ylim=ylim_time)
    #
    #     vis.freq_plot(raw_sig[0:echo_len], dt=dt, name='Raw Sig', ylim=ylim_freq)
    #     vis.freq_plot(comb[0:echo_len], dt=dt, name='Comb Cancelled Result', ylim=ylim_freq)
    # comb_pol_all[:, i] = comb
