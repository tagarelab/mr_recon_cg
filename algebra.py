"""
   Name: algebra.py
   Purpose: Algebraic functions for the project
   Created on: 2/15/2024
   Created by: Heng Sun
   Additional Notes: 
"""

import numpy as np
import scipy as sp


def interpolate_3D(signal, target_shape=None, factor=1):
    """
    Interpolate the input 3D matrix to a higher resolution.
    This function is drafted by Github Copilot and edited & tested by the author.

    Parameters:
    - signal (numpy.ndarray): The input 3D matrix.
    - target_shape (tuple): The target shape of the interpolated matrix.
    - factor (int): The interpolation factor.

    Returns:
    - numpy.ndarray: The interpolated array.
    """
    if target_shape is None:
        target_shape = np.array(signal.shape) * factor

    if len(signal.shape) != 3 or len(target_shape) != 3:
        raise ValueError("The input and target shapes must both be 3D.")

    # Define the original grid
    x_orig = np.arange(signal.shape[0])
    y_orig = np.arange(signal.shape[1])
    z_orig = np.arange(signal.shape[2])

    # Define the new grid
    x_new = np.linspace(0, signal.shape[0], target_shape[0])
    y_new = np.linspace(0, signal.shape[1], target_shape[1])
    z_new = np.linspace(0, signal.shape[2], target_shape[2])

    # Create the interpolator function
    interp_func = sp.interpolate.RegularGridInterpolator((x_orig, y_orig, z_orig), signal)

    # Define the points where you want to interpolate
    points = np.mgrid[x_new[0]:x_new[-1]:target_shape[0] * 1j, y_new[0]:y_new[-1]:target_shape[1] * 1j,
             z_new[0]:z_new[-1]:target_shape[2] * 1j]
    points = points.reshape(3, -1).T

    # Get the interpolated values
    interpolated_values = interp_func(points)

    return interpolated_values.reshape(target_shape)




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