"""
   Name: expr_log_240819.py
   Purpose:
   Created on: 8/19/2024
   Created by: Heng Sun
   Additional Notes: 
"""

from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from denoise import editer
import scipy as sp
import mr_io
import visualization as vis

# %% Load raw data
loc = "sim_input\\"
file_name = "Pos1_WithPhant_B0on_prepol_100_slice_6.3_32avg.mat"

actual_emi_ch_name = ['EMI-X', 'EMI-Y', 'EMI-Z']

# %% Apply EDITER to one data file
data_mat = mr_io.load_single_mat(name=file_name, path=loc)
datafft = data_mat["raw_sig"]
datanoise_fft_list = data_mat["raw_emi"]
editer_corr = editer.editer_process_2D(datafft, datanoise_fft_list)

# %% Visualization
region = [0, 500]
vis.absolute(datafft[region[0]:region[1]], name='Uncorrected')
vis.absolute(editer_corr[region[0]:region[1]], name='Corrected with EDITER, Zoom in to %s' % (
    region))

# %% Save the corrected data
data_mat['editer_corr'] = editer_corr
mr_io.save_dict(data_mat, name=file_name + '_EDITER', path=loc)
