"""
   Name: expr_log_240223.py
   Purpose:
   Created on: 2/22/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import gen_op
from sim import acquisition as acq
import mr_io
import numpy as np
import matplotlib.pyplot as plt
import visualization as vis

path = 'sim_inputs/magnetData.csv'
nubo_b0_raw, b0_X, b0_Y, b0_Z = mr_io.read_nubo_b0(path=path, intrp_x=30, intrp_y=30, intrp_z=30)

vis.scatter3d(nubo_b0_raw, b0_X, b0_Y, b0_Z)

# %% slice selection
# Constants
gamma = 42.58e6  # Hz/T
read_mag = 0.046  # T
DC = 0.1  # percent DC when polarizing
ctr_mag = read_mag  # slice selection gradient
slc_tkns_frq = 100e3 * 2  # Hz
slc_tkns_mag = slc_tkns_frq / gamma  # T

# Adjust nubo_b0
nubo_b0 = nubo_b0_raw * DC  # slice strength

# Call slice_select function
id = acq.slice_select(nubo_b0, ctr_mag, slc_tkns_mag)

vis.scatter3d(nubo_b0_raw, b0_X, b0_Y, b0_Z, mask=id)
