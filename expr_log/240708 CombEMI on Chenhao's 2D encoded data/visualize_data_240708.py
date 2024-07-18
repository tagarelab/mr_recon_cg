"""
   Name: visualize_data_240703.py
   Purpose:
   Created on: 7/3/2024
   Created by: Heng Sun
   Additional Notes: 
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import visualization as vis

# %% load data
file_name = 'FirstTryPE#_comb_07082024'
mat_file = sp.io.loadmat('sim_output/' + file_name + '.mat')
sig_comb = mat_file['comb_sig_all']
sig_raw = mat_file['raw_sig_all']


# %% get averaged signal
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


for data in [sig_raw, sig_comb]:
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
