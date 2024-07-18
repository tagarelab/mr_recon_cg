"""
   Name: sparse_cca.py
   Purpose:
   Created on: 7/09/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mr_io

from cca_zoo.datasets import JointData
from cca_zoo.linear import (
    CCA,
    PLS,
    SCCA_IPLS,
    SPLS,
    ElasticCCA,
    SCCA_Span,
)
from cca_zoo.model_selection import GridSearchCV

# %% Load data
# real TE part
N_echoes_real = 480
echo_len_real = 120
dt = 5e-6
TE_real = 2e-3

# polarization period
N_rep_pol = 17
rep_len_pol = 6000
# TODO: double check the log to see if it is 400us or 800 us gap - TNMR is not very clear
rep_time_pol = rep_len_pol * dt + 400e-6

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

# %% Load mat file
data = mr_io.load_single_mat('sim_output/with_without_phant_06272024')
pol_all_steps = data['pol_all_steps']
sig_all_steps = data['sig_all_steps']
file_name_steps = data['file_name_steps']
channel_info = data['channel_info']
file_name_stem = ['No Avg', '16 Avgs']
segment_names = data['segment_names']

# %% Load data
with_phant = sig_all_steps[1, 1, 1, :]
without_phant = sig_all_steps[1, 1, 0, :]
with_phant_pol = pol_all_steps[1, 1, 1, :]
without_phant_pol = pol_all_steps[1, 1, 0, :]

# %% Generate Training Data
# # Create synthetic data for two views
# train_view_1 = np.random.normal(size=(100, 10))
# train_view_2 = np.random.normal(size=(100, 10))
#
# # Normalize the data by removing the mean
# train_view_1 -= train_view_1.mean(axis=0)
# train_view_2 -= train_view_2.mean(axis=0)
#
# latent_dimensions = 3
# linear_cca = CCA(latent_dimensions=latent_dimensions)
#
# # Fit the model
# linear_cca.fit((train_view_1, train_view_2))
