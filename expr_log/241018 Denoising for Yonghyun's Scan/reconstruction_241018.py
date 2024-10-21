# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:06:58 2022

@author: Ha
Edited by Heng Sun
"""

import numpy as np
import matplotlib.pyplot as plt
import mr_io as mr_io

N_echo = 97

# loc = "C:\\Yale\\MRRC\\mr_recon_cg\\data\\241017_YongHyun_6min_scan\\EDITER\\"
# file_name = "test_256avg"
# compare = ['raw_sig', 'editer_corr']

loc = "sim_output/"
# file_name = "test_256avg_comb_10212024.mat"
# file_name = "test_256avg_editer_comb_10212024.mat"
# compare = ["raw_sig_all", "comb_sig_all"]
file_name = "all_combos_comb_editer_10212024.mat"
compare = ["raw", "editer", "comb", "editer_comb", "comb_editer"]

data_mat = mr_io.load_single_mat(name=file_name, path=loc)
for name in compare:
    data = data_mat[name][:]

    data_reshape = data.reshape(N_echo, 100)

    k_space = data_reshape[0, :]

    for i in range(N_echo - 1):
        if i % 2 == 0:
            k_space = np.vstack([k_space, data_reshape[i + 1, :]])
        else:
            k_space = np.vstack([data_reshape[i + 1, :], k_space])
    image = np.fft.ifftshift(np.fft.ifft2(k_space))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(k_space), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.flipud(np.abs(image)), cmap='gray', vmin=0, vmax=1000)
    plt.axis('off')
    plt.suptitle(name)
    plt.show()