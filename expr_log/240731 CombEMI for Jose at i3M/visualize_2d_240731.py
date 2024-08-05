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
file_name = 'ThirdTryPE#_comb_07182024'
mat_file = sp.io.loadmat('sim_output/' + file_name + '.mat')
comb_data = mat_file['comb_sig_all']
raw_data = mat_file['raw_sig_all']
data_names = ["Comb Corrected", "Raw"]
data_list = [comb_data, raw_data]


# file_name = 'ThirdTryPE#'
# mat_file = sp.io.loadmat('sim_input/' + file_name + '.mat')
# data = mat_file['ch1']

# %% get averaged signal
def avg_first_k_peaks(signal, echo_len, k=10):
    echoes = np.zeros((k, echo_len), dtype='complex')
    for i in range(k):
        echo = signal[i * echo_len:(i + 1) * echo_len]
        echoes[i, :] = np.squeeze(echo)
    return np.mean(echoes, axis=0)


echo_len = 100
k_list = [1, 5, 10, 50, 120]
# k_list = [1]

# %% plot k space and image space
# for j in range(len(data_names)):
#     data = data_list[j]
#     sig_avg = np.zeros([echo_len, data.shape[1]], dtype='complex')
#
#     for i in range(data.shape[1]):
#         sig_avg[:, i] = avg_first_k_peaks(data[:, i], echo_len, k=10)
#
#     # plot data
#     plt.figure()
#     plt.imshow(np.abs(sig_avg), aspect='auto')
#     plt.colorbar()
#     plt.title('%s K Space'%data_names[j])
#     plt.show()
#
#     plt.figure()
#     plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(sig_avg))), aspect='auto')
#     plt.colorbar()
#     plt.title('Image Space')
#     plt.clim(0, 6e5)
#     plt.show()

# %% compare
# v_max_list = [3e2,1e2,8e1,5e1,3e1]
v_max_list = [1e2, 1e2, 1e2, 1e2, 1e2]
for k in k_list:
    plt.figure()
    for j in range(len(data_names)):

        data = data_list[j]
        sig_avg = np.zeros([echo_len, data.shape[1]], dtype='complex')

        for i in range(data.shape[1]):
            sig_avg[:, i] = avg_first_k_peaks(data[:, i], echo_len, k=k)

        image = np.fft.ifftshift(np.fft.ifft2(sig_avg))
        # image = np.fft.fftshift(np.fft.fft2())

        ax = plt.subplot(1, len(data_names), j + 1)
        plt.imshow(np.abs(image), aspect='equal')
        # plt.plot(np.abs(np.sum(image,axis=1)), label='Abs')
        # plt.plot(np.real(np.sum(image, axis=1)), label='Real')
        # plt.plot(np.imag(np.sum(image, axis=1)), label='Imag')
        plt.title(data_names[j])
        plt.clim(0, v_max_list[k_list.index(k)])

    plt.suptitle('Average %d Echoes' % k)
    plt.show()
