"""
   Name: expr_log_240503.py
   Purpose: Investigate undersampling by TE
   Created on: 5/3/2024
   Created by: Heng Sun
   Additional Notes:
"""

# def phase_shift_by_segments(x, t):
#     """
#     Subtract the first t points by x[0], then the next t points by x[t], and so on.
#     """
#     # Ensure x is a numpy array
#     x = np.array(x)
#
#     # Calculate the number of segments
#     num_segments = len(x) // t
#
#     # Iterate over each segment
#     for i in range(num_segments):
#         x[i * t:(i + 1) * t] *= np.exp(-1j * np.angle(x[i * t]))
#         # x[i * t:(i + 1) * t] -= x[i * t]
#         # x[i * t:(i + 1) * t] -= x[0]
#
#     # Handle the last segment if it has less than t points
#     if len(x) % t != 0:
#         x[num_segments * t:] -= x[num_segments * t]
#
#     return x


## Packages and Data
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import comb_test_240422 as cs
import visualization as vis

# %%
# visualization parameters
short_sig_len = 200  # length of the signal to be visualized

# imaging parameters
sample_num = 70  # image size
gyro_mag = 42.58 * 1e6  # gyro magnetic ratio
wrf = -(1 - 0.87) * 1e6  # Bloch-Siegert frequency (in units Hz)
polar_time = 0  # seconds
theta = np.arange(0, 180, 180 / sample_num)  # degrees

# Load .mat file
mat_file = sp.io.loadmat('sim_input/big_epfl_brain.mat')
phantom_img = sp.ndimage.zoom(mat_file['phantom'], sample_num / float(mat_file['phantom'].shape[0]), order=1)
# phantom_img = phantom_img[:, 0:60]  # for testing: otherwise the two dim have the same size
phantom_img = np.zeros(phantom_img.shape, dtype=complex)  # ZERO PHANTOM FOR TESTING

# Convert image to frequency domain
phantom_fft = cs.im2freq(phantom_img, theta=theta)

# Prepare simulated signal
sim_sig = np.zeros((1, *phantom_fft.shape), dtype=complex)
sim_sig[0, :, :] = phantom_fft
echo_len = sim_sig.shape[2]
N_echoes = sim_sig.shape[1]
sim_sig = sim_sig.reshape((1, N_echoes * echo_len))

# Visualization of the phantom
plt.figure()
# Display image space
plt.subplot(1, 2, 1)
plt.imshow(np.abs(cs.freq2im(phantom_fft, theta=theta)), vmin=0, vmax=1)
plt.title('Image Space')
# Display k-space
plt.subplot(1, 2, 2)
plt.imshow(np.abs(phantom_fft), vmin=0, vmax=5e1)
plt.title('K-space')
plt.show()

## Start of the main simulation

# Simulation parameters
TE = 1.5e-3  # seconds
dt = 1e-5  # seconds
# TE = dt * 70  # just for testing, get rid of this
samp_freq = 1 / dt  # Hz, sampling freq
ctr_freq = 1e6  # Hz, coil center freq
# ctr_freq = 0  # Hz, coil center freq
freq_axis = cs.freq_axis(int(np.ceil(N_echoes * TE / dt)), dt)  # Hz, frequency axis

# White noise
# wgn_snr = 10  # snr for Gaussian white noise
# wgn_db = 10  # power for Gaussian white noise
wgn_lin = 0  # linear power for Gaussian white noise

# Pre-set structured noise
# sn = np.array([[50, freq_axis[3408], 30]]).T  # amplitude and
# Hz and phase
# for SN
# sn = np.array([[10, 300, 0]]).T  # amplitude and Hz for SN

# Random Structured Noise
N_sn = 1
amp_max = 100  # linear, not dB
amp_min = 10
amp_list = np.linspace(amp_min, amp_max, 10000)
phase_list = np.linspace(-np.pi, np.pi, 100)

# sn = cs.rand_sn_from_list(N_sn, amp_list, freq_axis, phase_list)
sn = np.array([[50, 24601, 0]]).T  # amplitude and Hz for SN

# Comb params
lambda_val = -1  # regularization term
rho = 1  # constant for the lagrange matrix, was 1.0 before
step = 0.1  # step size
max_iter = 1000  # number of iterations
pre_drop = 0  # drop this many points at the front of each echo
post_drop = 0  # drop this many points at the end of each echo
pk_win = 0.33  # the window size for above-white-noise peak, should be within (0,1)

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

img_fft = np.reshape(sim_noisy_sig, (N_echoes, -1))
plt.figure()
plt.imshow(np.abs(cs.freq2im(img_fft, theta=theta)), vmin=0, vmax=1)
plt.colorbar()
#
# sim_noisy_sig = sim_noisy_sig + np.random.normal(0, 10 ** (wgn_db / 20), sim_noisy_sig.shape) + \
#                 1j * np.random.normal(0, 10 ** (wgn_db / 20), sim_noisy_sig.shape)
sim_noisy_sig = sim_noisy_sig + cs.gen_white_noise(wgn_lin, sim_noisy_sig.shape)

# img_fft = np.reshape(sim_noisy_sig[samp_mask], (N_echoes, -1))
# plt.figure()
# plt.imshow(np.abs(cs.freq2im(img_fft, theta=theta)), vmin=0, vmax=1)
# plt.title("With white noise")
# plt.show()

# str_noi = cs.gen_sn(sim_noisy_sig, N_echoes, TE, dt, sn, polar_time)
str_noi = cs.gen_sn(sn=sn, length=len(sim_noisy_sig), dt=dt, bw=0)
sim_noisy_sig = sim_noisy_sig + str_noi

# plt.figure()
# plt.plot(np.real(str_noi[:100]))
# plt.plot(np.imag(str_noi[:100]))
# plt.title("Noise added")


# img_fft = np.reshape(sim_noisy_sig[samp_mask], (N_echoes, -1))
# plt.figure()
# # Display image space
# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(cs.freq2im(img_fft, theta=theta)), vmin=0, vmax=1)
# plt.title('Image Space')
# # Display k-space
# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(img_fft))
# plt.title('K-space')
# plt.show()
#
# # Display image space
# plt.subplot(1, 2, 1)
# plt.imshow(np.imag(cs.freq2im(img_fft, theta=theta)), vmin=0, vmax=1)
# plt.title('Image Space')
# # Display k-space
# plt.subplot(1, 2, 2)
# plt.imshow(np.imag(img_fft))
# plt.title('K-space')
# plt.show()
#
# # Display image space
# plt.subplot(1, 2, 1)
# plt.imshow(np.angle(cs.freq2im(img_fft, theta=theta)), vmin=0, vmax=1)
# plt.title('Image Space')
# # Display k-space
# plt.subplot(1, 2, 2)
# plt.imshow(np.angle(img_fft))
# plt.title('K-space')
# plt.show()

# plt.figure()
# plt.imshow(np.abs(cs.freq2im(img_fft, theta=theta)), vmin=0, vmax=1)
# plt.title("With white noise and structured noise")
# plt.show()

# undersample the signal
sim_us_by_TE = sim_noisy_sig[0::int(TE / dt)]
vis.complex(sim_us_by_TE, name='Undersampled by TE', rect=True)
vis.freq_plot(sim_us_by_TE, dt, name='Undersampled by TE')

vis.complex(sim_noisy_sig, name='Simulated Noisy Sig', rect=True)
vis.freq_plot(sim_noisy_sig, dt, name='Simulated Noisy Sig')

img_fft = np.reshape(sim_noisy_sig[samp_mask], (N_echoes, -1))
plt.subplot(1, 2, 1)
plt.imshow(np.angle(cs.freq2im(img_fft, theta=theta)), vmin=0, vmax=1)
plt.title('Image Space')
# Display k-space
plt.subplot(1, 2, 2)
plt.imshow(np.angle(img_fft))
plt.title('K-space')
plt.show()

temp = str_noi[samp_mask]
sim_noisy_corrected2 = cs.phase_shift_by_segments(sim_noisy_sig[samp_mask], echo_len, np.exp(-1j * np.angle(temp[
                                                                                                            ::echo_len])))

cs.plot_img_freq_spaces(sim_noisy_corrected2, N_echoes, theta, name='2')

sim_noisy_corrected3 = cs.phase_shift_by_segments(sim_noisy_corrected2, echo_len, temp[::echo_len])

cs.plot_img_freq_spaces(sim_noisy_corrected3, N_echoes, theta, name='3')
