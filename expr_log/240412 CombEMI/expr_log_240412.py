"""
   Name:
   Purpose:
   Created on:
   Created by: Heng Sun
   Additional Notes: 
"""

## Packages and Data
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import comb_test_240412 as cs
import visualization as vis

# %%
# imaging parameters
sample_num = 70  # image size
gyro_mag = 42.58 * 1e6  # gyro magnetic ratio
wrf = -(1 - 0.87) * 1e6  # Bloch-Siegert frequency (in units Hz)
polar_time = 0  # seconds

# Load .mat file
mat_file = sp.io.loadmat('sim_input/big_epfl_brain.mat')
phantom_img = sp.ndimage.zoom(mat_file['phantom'], sample_num / float(mat_file['phantom'].shape[0]), order=1)
# phantom_img = phantom_img[:, 0:60]  # for testing: otherwise the two dim have the same size
# phantom_img = np.zeros(phantom_img.shape, dtype=complex)  # ZERO PHANTOM FOR TESTING

# Convert image to frequency domain
phantom_fft = cs.im2freq(phantom_img)

# Prepare simulated signal
sim_sig = np.zeros((1, *phantom_fft.shape), dtype=complex)
sim_sig[0, :, :] = phantom_fft
echo_len = sim_sig.shape[2]
N_echoes = sim_sig.shape[1]
sim_sig = sim_sig.reshape((1, N_echoes * echo_len))

# Visualization of the phantom
# plt.figure()
# # Display image space
# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(cs.freq2im(phantom_fft)), vmin=0, vmax=1)
# plt.title('Image Space')
# plt.colorbar()
# # Display k-space
# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(phantom_fft), vmin=0, vmax=5e1)
# plt.title('K-space')
# plt.colorbar()
# plt.show()

## Start of the main simulation

# Simulation parameters
TE = 1.5e-3  # seconds
dt = 1e-5  # seconds
# TE = dt * 70  # just for testing, get rid of this
samp_freq = 1 / dt  # Hz, sampling freq
ctr_freq = 1e6  # Hz, coil center freq
# ctr_freq = 0  # Hz, coil center freq

# White noise
# wgn_snr = 10  # snr for Gaussian white noise
wgn_db = 10  # power for Gaussian white noise

# Pre-set structured noise
sn = np.array([[10, 24601, 30], [13, -10086, 20]]).T  # amplitude and Hz and phase for SN
# sn = np.array([0, 0, 0])  # amplitude and Hz for SN

# Random Structured Noise
N_sn = 1
amp_max = 35  # linear, not dB
amp_min = 30

# Comb params
lambda_val = 3000  # regularization term
rho = 1  # constant for the lagrange matrix, was 1.0 before
step = 0.1  # step size
max_iter = 100  # number of iterations
pre_drop = 0  # drop this many points at the front of each echo
post_drop = 0  # drop this many points at the end of each echo
pk_win = 0.33  # the window size for above-white-noise peak, should be within (0,1)

# Test params
N_rep = 1  # number of repetitions
rmse_org_freq = np.zeros(N_rep)
rmse_comb_freq = np.zeros(N_rep)
rmse_pro_freq = np.zeros(N_rep)
rmse_org_img = np.zeros(N_rep)
rmse_comb_img = np.zeros(N_rep)
rmse_pro_img = np.zeros(N_rep)

for k in range(N_rep):
    # generate random structured noise
    # sn = rand_single_sn(N_sn, amp_max, amp_min, ctr_freq, samp_freq, N_echoes * echo_len)

    # simulate noisy signal
    sim_noisy_sig = cs.add_polarization(sim_sig, time=polar_time, dt=dt)
    sim_noisy_sig = cs.sampled_to_full(signal=sim_noisy_sig, polar_time=polar_time, dt=dt, acq_len=echo_len,
                                       N_echoes=N_echoes,
                                       TE_len=np.uint16(TE / dt))

    samp_mask = cs.gen_samp_mask(acq_len=echo_len, N_echoes=N_echoes, TE_len=np.uint16(TE / dt),
                                 polar_time=polar_time, dt=dt, pol=False)
    samp_mask_w_pol = cs.gen_samp_mask(acq_len=echo_len, N_echoes=N_echoes, TE_len=np.uint16(TE / dt),
                                       polar_time=polar_time, dt=dt, pol=True)

    # img_fft = np.reshape(sim_noisy_sig, (N_echoes, -1))
    # plt.figure()
    # plt.imshow(np.abs(freq2im(img_fft)), vmin=0, vmax=1)
    # plt.colorbar()
    #
    sim_noisy_sig = sim_noisy_sig + np.random.normal(0, 10 ** (wgn_db / 20), sim_noisy_sig.shape) + \
                    1j * np.random.normal(0, 10 ** (wgn_db / 20), sim_noisy_sig.shape)

    # img_fft = np.reshape(sim_noisy_sig[samp_mask], (N_echoes, -1))
    # plt.figure()
    # plt.imshow(np.abs(cs.freq2im(img_fft)), vmin=0, vmax=1)
    # plt.colorbar()

    # str_noi = cs.gen_sn(sim_noisy_sig, N_echoes, TE, dt, sn, polar_time)
    str_noi = cs.gen_sn(sn=sn, length=len(sim_noisy_sig), dt=dt)
    sim_noisy_sig = sim_noisy_sig + str_noi
    # plt.title("With white noise")

    # plt.figure()
    # plt.plot(np.real(str_noi[:100]))
    # plt.plot(np.imag(str_noi[:100]))
    # plt.title("Noise added")

    # img_fft = np.reshape(sim_noisy_sig[samp_mask], (-1, N_echoes))
    # plt.figure()
    # plt.imshow(np.abs(cs.freq2im(img_fft)), vmin=0, vmax=1)
    # plt.colorbar()
    # plt.title("With white noise and structured noise")
    # plt.show()

    # perform comb
    signal = sim_noisy_sig[samp_mask_w_pol]
    cancelled_comb_raw = cs.comb_optimized(signal=signal, N_echoes=N_echoes, TE=TE, dt=dt, lambda_val=lambda_val,
                                           step=step,
                                           max_iter=max_iter, tol=0.1, pre_drop=pre_drop, post_drop=post_drop,
                                           pk_win=pk_win, polar_time=polar_time, rho=rho)

    # factor = np.abs(np.mean(np.abs(str_noi))/np.mean(np.abs(cancelled_comb)))
    # cancelled_comb = cancelled_comb * factor

    cancelled_comb = cancelled_comb_raw[samp_mask_w_pol]
    vis.complex(cancelled_comb, name='Comb Output after masking', rect=True)
    comb_scaling = 1.0099999999999991  # not related to lambda
    cancelled_comb = cancelled_comb * comb_scaling

    # perform simulated probe-based cancellation
    probe = str_noi[samp_mask_w_pol] + np.random.normal(0, 10 ** (wgn_db / 20), signal.shape) + \
            1j * np.random.normal(0, 10 ** (wgn_db / 20), signal.shape[0])
    signal_pro = signal - probe

    # calculate difference
    # signal_comb = signal - cancelled_comb

    # Generate imag
    sig_org = np.reshape(signal[cs.calculate_polar_period(polar_time=polar_time, dt=dt):], (N_echoes, -1))
    # sig_comb = np.reshape(signal_comb, (-1, N_echoes))
    sig_pro = np.reshape(signal_pro[cs.calculate_polar_period(polar_time=polar_time, dt=dt):], (N_echoes, -1))
    noi_comb = np.reshape(cancelled_comb[cs.calculate_polar_period(polar_time=polar_time, dt=dt):], (N_echoes, -1))
    sig_comb = sig_org - noi_comb

    sig_org_img = cs.freq2im(sig_org)
    sig_comb_img = cs.freq2im(sig_comb)
    sig_pro_img = cs.freq2im(sig_pro)
    noi_comb_img = cs.freq2im(noi_comb)

    # RMSE calculation
    rmse_org_freq[k] = cs.rmse(phantom_fft, sig_org)
    rmse_comb_freq[k] = cs.rmse(phantom_fft, sig_comb)
    rmse_pro_freq[k] = cs.rmse(phantom_fft, sig_pro)

    phantom_img = cs.freq2im(cs.im2freq(phantom_img))
    rmse_org_img[k] = cs.rmse(np.abs(phantom_img), np.abs(sig_org_img))
    rmse_comb_img[k] = cs.rmse(np.abs(phantom_img), np.abs(sig_comb_img))
    rmse_pro_img[k] = cs.rmse(np.abs(phantom_img), np.abs(sig_pro_img))

    # Visualize
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(sig_org_img), vmin=0, vmax=1)
    plt.title('Orig, RMSE: ' + "{:.2f}".format(rmse_org_img[k]))
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(sig_comb_img), vmin=0, vmax=1)
    plt.title('Comb, RMSE: ' + "{:.2f}".format(rmse_comb_img[k]))
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(sig_pro_img), vmin=0, vmax=1)
    plt.title('Probe, RMSE: ' + "{:.2f}".format(rmse_pro_img[k]))
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(noi_comb_img), vmin=0, vmax=1)
    plt.title('Original-Comb')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(sig_org_img), vmin=0, vmax=1)
    plt.title('Original')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(sig_comb_img), vmin=0, vmax=1)
    plt.title('Comb')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(phantom_img), vmin=0, vmax=1)
    plt.title('Simulated Phantom')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(sig_comb_img - phantom_img), vmin=0, vmax=1)
    plt.title('Comb-Phantom')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(sig_org))
    plt.title('Original')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(sig_comb))
    plt.title('Comb')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(sig_pro))
    plt.title('Probe')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(noi_comb))
    plt.title('Original-Comb')
    plt.colorbar()
    plt.show()

    # Visualize 1D signal
    # # Visualization of the first segment of 1D signal
    # vis.freq_plot(sig_org[0, :], dt=1e-5, name='Simulated Signal')
    # vis.freq_plot(str_noi[:echo_len], dt=1e-5, name='Simulated EMI')
    # vis.freq_plot(noi_comb[0, :], dt=1e-5, name='Comb Estimated EMI')
    # vis.complex(sig_org[0, :], name='Original Signal', rect=True)
    # vis.complex(str_noi[:echo_len], name='Simulated EMI', rect=True)
    # vis.complex(noi_comb[0, :], name='Comb Estimated EMI', rect=True)

    # Difference
    factor = str_noi / cancelled_comb_raw
    vis.complex(factor, name='True EMI/ Comb output', rect=True)
    vis.complex(factor[samp_mask_w_pol], name='True EMI/ Comb output, masked', rect=True)
    vis.freq_plot(factor, dt=1e-5, name='True EMI/ Comb output')

    vis.complex(cancelled_comb_raw, name='Comb Raw Output', rect=True)
    vis.freq_plot(cancelled_comb_raw, dt=1e-5, name='Comb Raw Output')

    vis.complex(str_noi, name='True EMI', rect=True)
    vis.freq_plot(str_noi, dt=1e-5, name='True EMI')

    vis.complex(signal, name='Signal', rect=True)
    vis.freq_plot(signal, dt=1e-5, name='Signal')
