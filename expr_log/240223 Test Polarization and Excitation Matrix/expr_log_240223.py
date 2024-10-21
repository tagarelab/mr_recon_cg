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
import algebra as algb

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


# %% read B1
def read_b1(filename, path, intrp_x, intrp_y, intrp_z, FOV, scale=1):
    # Read the data
    B1_data = mr_io.load_single_mat(name=filename, path=path)

    # Extract coordinates and magnetic field components
    B1_X_coord = B1_data.iloc[:, 0].values
    B1_Y_coord = B1_data.iloc[:, 1].values
    B1_Z_coord = B1_data.iloc[:, 2].values
    X_data = B1_data.iloc[:, 3].values
    Y_data = B1_data.iloc[:, 4].values
    Z_data = B1_data.iloc[:, 5].values

    # Calculate the magnitude of the magnetic field and apply a scaling factor
    nubo_b1 = np.sqrt(X_data ** 2 + Y_data ** 2 + Z_data ** 2)
    nubo_b1 = scale * nubo_b1

    # Create a 3D grid for the magnetic field data
    x_M = np.linspace(B1_X_coord.min(), B1_X_coord.max(), 11)
    y_M = np.linspace(B1_Y_coord.min(), B1_Y_coord.max(), 11)
    z_M = np.linspace(B1_Z_coord.min(), B1_Z_coord.max(), 11)
    nubo_b1_mesh, _, _, _ = algb.vec2mesh(nubo_b1, B1_X_coord, B1_Y_coord, B1_Z_coord, 11, 11, 11)

    nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp = algb.interp_3dmat(nubo_b1_mesh, x_M, y_M, z_M, intrp_x,
                                                                          intrp_y, intrp_z)

    b1_X_intrp = b1_X_intrp / 200 * FOV
    b1_Y_intrp = b1_Y_intrp / 200 * FOV
    b1_Z_intrp = b1_Z_intrp / 200 * FOV

    return nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp


# %% read B1
path = 'sim_inputs/'
filename = 'B1_ROI_240mm_240mm_240mm'
intrp_x = 30
intrp_y = 30
intrp_z = 30
FOV = 0.24
scale = 1
# nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp = read_b1(filename=filename, path=path, intrp_x=30, intrp_y=30,
#                                                             intrp_z=30, FOV=0.24)
B1_data = mr_io.load_single_mat(name=filename, path=path)['B1']

# Extract coordinates and magnetic field components
X_data = B1_data[0, :, :, :]
Y_data = B1_data[1, :, :, :]
Z_data = B1_data[2, :, :, :]

# Calculate the magnitude of the magnetic field and apply a scaling factor
nubo_b1 = np.sqrt(X_data ** 2 + Y_data ** 2 + Z_data ** 2)
nubo_b1 = scale * nubo_b1

# Create a 3D grid for the magnetic field data
x_M = np.linspace(-0.12, +0.12, 31)
y_M = np.linspace(-0.12, +0.12, 31)
z_M = np.linspace(-0.12, +0.12, 31)

nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp = algb.interp_3dmat(nubo_b1, x_M, y_M, z_M, intrp_x,
                                                                      intrp_y, intrp_z)

# %% plot B1
vis.scatter3d(nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp)
vis.scatter3d(nubo_B1_intrp, b1_X_intrp, b1_Y_intrp, b1_Z_intrp, mask=id)
