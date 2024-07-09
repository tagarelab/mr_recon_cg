"""
   Name: visualize_2d_240520.py
   Purpose:
   Created on: 5/20/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# %% load data
file_name = 'FirstTryPE#_comb_05212024'
mat_file = sp.io.loadmat('sim_output/' + file_name + '.mat')
data = mat_file['data']


# file_name = 'FirstTryPE#'
# mat_file = sp.io.loadmat('sim_input/' + file_name + '.mat')
# data = mat_file['ch1']

# %% get averaged signal
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


echo_len = int(data.shape[0] / 720)
sig_avg = np.zeros([echo_len, data.shape[1]], dtype='complex')

for i in range(data.shape[1]):
    sig_avg[:, i] = avg_first_k_peaks(data[:, i], echo_len, k=10)

# %% plot data
plt.figure()
plt.imshow(np.abs(sig_avg), aspect='auto')
plt.colorbar()
plt.title('K space')
plt.show()

plt.figure()
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(sig_avg))), aspect='auto')
plt.colorbar()
plt.title('Image space')
plt.show()
