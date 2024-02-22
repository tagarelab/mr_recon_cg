"""
   Name: algebra.py
   Purpose: Algebraic functions for the project
   Created on: 2/15/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import scipy as sp

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interp_3dmat(M, x_axis, y_axis, z_axis, x_pts, y_pts, z_pts, method='linear'):
    """
    Interpolate a 3D matrix to new dimensions x, y, and z.
    This function is adapted from a MATLAB function by Github Copilot and edited & tested by the author.

    Inputs:
    - M: Input 3D matrix.
    - x_pts: Number of desired points along the first dimension (X).
    - y_pts: Number of desired points along the second dimension (Y).
    - z_pts: Number of desired points along the third dimension (Z).

    Output:
    - interpolated_matrix: Interpolated 3D matrix with dimensions x, y, z.
    """

    # Dimensions of the expanded field taken from the raw data
    x_intrp = np.linspace(x_axis[0], x_axis[-1], x_pts)
    y_intrp = np.linspace(y_axis[0], y_axis[-1], y_pts)
    z_intrp = np.linspace(z_axis[0], z_axis[-1], z_pts)
    intrp_pts = np.array(np.meshgrid(x_intrp, y_intrp, z_intrp, indexing='ij')).T

    # Interpolate the expanded field
    interpolator = RegularGridInterpolator((x_axis, y_axis, z_axis), M, method=method)
    interpolated_matrix = interpolator(intrp_pts)

    return interpolated_matrix, x_intrp, y_intrp, z_intrp


def vec2mesh(mag, x_coord, y_coord, z_coord, x_dim, y_dim, z_dim):
    """
    Transform from vector representation to mesh representation,
    assuming even interpolation.
    This function is adapted from a MATLAB function by Github Copilot and edited & tested by the author.

    """
    if len(mag) != x_dim * y_dim * z_dim:
        raise ValueError('Error: target dimensions does not meet vector dimension.')

    x_M = np.linspace(np.min(x_coord), np.max(x_coord), x_dim)
    y_M = np.linspace(np.min(y_coord), np.max(y_coord), y_dim)
    z_M = np.linspace(np.min(z_coord), np.max(z_coord), z_dim)
    X_M, Y_M, Z_M = np.meshgrid(x_M, y_M, z_M, indexing='ij')

    mag_mesh = np.zeros((x_dim, y_dim, z_dim))
    for xx in range(x_dim):
        for yy in range(y_dim):
            for zz in range(z_dim):
                ind = np.argmax((x_coord == X_M[xx, yy, zz]) & (y_coord == Y_M[xx, yy, zz]) & (z_coord == Z_M[xx, yy,
                zz]))
                mag_mesh[xx, yy, zz] = mag[ind]

    return mag_mesh, X_M, Y_M, Z_M


def complex2array(x):
    """
    Convert a complex array to a real array.
    This function is drafted by Github Copilot and edited & tested by the author.

    Parameters:
    - x (numpy.ndarray): The input complex array.

    Returns:
    - numpy.ndarray: The real array.
    """
    return np.concatenate((np.real(x), np.imag(x)), axis=-1)


def array2complex(x):
    """
    Convert a real array to a complex array.
    This function is drafted by Github Copilot and edited & tested by the author.

    Parameters:
    - x (numpy.ndarray): The input real array.

    Returns:
    - numpy.ndarray: The complex array.
    """
    return x[..., :x.shape[-1] // 2] + 1j * x[..., x.shape[-1] // 2:]