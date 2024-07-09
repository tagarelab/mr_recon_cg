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
file_name = 'GLR_0_rftime80us_90ampl_3.6_200pts_full_comb_07032024.mat'
mat_file = sp.io.loadmat('sim_output/' + file_name)
# data = mat_file['data']
raw_sig_all = mat_file['raw_sig_all']
comb_sig_all = mat_file['comb_sig_all']

ylim = [0,15000]
vis.absolute(raw_sig_all[0:1000], name="Raw Data",ylim=ylim)
vis.absolute(comb_sig_all[0:1000], name="Comb Data",ylim=ylim)