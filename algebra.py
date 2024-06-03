"""
   Name: algebra.py
   Purpose: Algebraic functions for the project
   Created on: 2/15/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import warnings


def rot_mat(u, theta):
    """
    Generate matrix to perform rotation on a 3D vector.
    Reference: https://en.wikipedia.org/wiki/Rotation_matrix
    :param u: the fixed direction, with x,y,z component
    :param theta: rotate this much
    :return:
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[u[0] ** 2 * (1 - c) + c, u[0] * u[1] * (1 - c) - u[2] * s, u[0] * u[2] * (1 - c) + u[1] * s],
                     [u[0] * u[1] * (1 - c) + u[2] * s, u[1] ** 2 * (1 - c) + c, u[1] * u[2] * (1 - c) - u[0] * s],
                     [u[0] * u[2] * (1 - c) - u[1] * s, u[1] * u[2] * (1 - c) - u[0] * s, u[2] ** 2 * (1 - c) + c]])


def interp_by_pts(M, x_axis, y_axis, z_axis, intrp_pts, method='linear'):
    interpolator = RegularGridInterpolator((x_axis, y_axis, z_axis), M, method=method)
    # check if the interpolation points are within the range of the raw data
    if np.any(intrp_pts < np.array([x_axis[0], y_axis[0], z_axis[0]])) or np.any(
            intrp_pts > np.array([x_axis[-1], y_axis[-1], z_axis[-1]])):
        raise ValueError('Error: Interpolation points are out of range of the raw data.')
    interpolated_matrix = interpolator(intrp_pts)
    return interpolated_matrix


def gen_interp_pts(x_intrp, y_intrp, z_intrp):
    # Dimensions of the expanded field taken from the raw data
    intrp_pts = np.array(np.meshgrid(x_intrp, y_intrp, z_intrp, indexing='ij')).T
    return intrp_pts


def gen_interp_axis(x_axis, y_axis, z_axis, x_pts, y_pts, z_pts):
    # Dimensions of the expanded field taken from the raw data
    x_intrp = np.linspace(x_axis[0], x_axis[-1], x_pts)
    y_intrp = np.linspace(y_axis[0], y_axis[-1], y_pts)
    z_intrp = np.linspace(z_axis[0], z_axis[-1], z_pts)
    return x_intrp, y_intrp, z_intrp


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

    x_intrp, y_intrp, z_intrp = gen_interp_axis(x_axis, y_axis, z_axis, x_pts, y_pts, z_pts)
    intrp_pts = gen_interp_pts(x_intrp, y_intrp, z_intrp)
    interpolated_matrix = interp_by_pts(M, x_axis, y_axis, z_axis, intrp_pts, method=method)

    return interpolated_matrix, x_intrp, y_intrp, z_intrp


def vec2mesh(mag, x_coord, y_coord, z_coord, x_dim=None, y_dim=None, z_dim=None, empty_val=None):
    """
    Transform from vector representation to mesh representation,
    assuming even interpolation.
    This function is adapted from a MATLAB function by Github Copilot and edited & tested by the author.

    """
    if x_dim is not None:
        x_M = np.linspace(np.min(x_coord), np.max(x_coord), x_dim)
    else:
        x_M = np.unique(x_coord)
        x_dim = len(x_M)
    if y_dim is not None:
        y_M = np.linspace(np.min(y_coord), np.max(y_coord), y_dim)
    else:
        y_M = np.unique(y_coord)
        y_dim = len(y_M)
    if z_dim is not None:
        z_M = np.linspace(np.min(z_coord), np.max(z_coord), z_dim)
    else:
        z_M = np.unique(z_coord)
        z_dim = len(z_M)

    if mag.ndim == 1:
        mag = np.expand_dims(mag, axis=0)
    if mag.shape[1] != x_dim * y_dim * z_dim:
        if empty_val is None:
            raise ValueError('Error: target dimensions does not meet vector dimension.')
        else:
            warnings.warn("Warning: target dimensions does not meet vector dimension. Filling with empty_val: %.2f"
                          % empty_val)

    X_M, Y_M, Z_M = np.meshgrid(x_M, y_M, z_M, indexing='ij')

    mag_mesh = np.zeros((mag.shape[0], x_dim, y_dim, z_dim))
    for dd in range(mag.shape[0]):
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    ind = np.argmax(
                        (x_coord == X_M[xx, yy, zz]) & (y_coord == Y_M[xx, yy, zz]) & (z_coord == Z_M[xx, yy,
                        zz]))
                    try:
                        mag_mesh[dd, xx, yy, zz] = mag[dd, ind]
                    except ValueError:
                        print('ValueError: ', ind)
                        mag_mesh[dd, xx, yy, zz] = empty_val
    if mag.shape[0] == 1:
        mag_mesh = np.squeeze(mag_mesh, axis=0)

    return mag_mesh, x_M, y_M, z_M


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


def snr(signal, noi_range=None):
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)
    sig_abs = np.abs(signal)
    if noi_range is None:
        warnings.warn("Warning when calculating snr: Noise range is not specified. Using the whole signal.")
        noi_range = [0, len(sig_abs)]
    noi_std = np.std(sig_abs[noi_range[0]:noi_range[1]])
    noi_avg = np.mean(sig_abs[noi_range[0]:noi_range[1]])  # take off the noise floor when calculating SNR
    return (np.max(sig_abs) - noi_avg) / noi_std
